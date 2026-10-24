"""Run five-fold neural training, record metrics, and build one submission."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from swag_oct.preprocessing.one_hot import OneHotPreprocessor, TabularData, encode_one_hot, load_csv
from swag_oct.submission.write import validate_template, write_submission
from swag_oct.training.fit import Progress, TrainingConfig, train_fold
from swag_oct.validation.folds import roc_auc, stratified_folds


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "experiments"


def _logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("swag_oct.training")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _subset(data: TabularData, indices: np.ndarray) -> TabularData:
    return TabularData(
        ids=data.ids[indices],
        categorical_codes=data.categorical_codes[indices],
        numerical=data.numerical[indices],
        target=data.target[indices] if data.target is not None else None,
    )


def _record_experiment(
    path: Path,
    *,
    config: TrainingConfig,
    train_rows: int,
    test_rows: int,
    fold_results: list[tuple[int, float, int, int]],
    oof_auc: float,
    submission_path: Path,
    artifact_path: Path,
) -> None:
    fold_scores = np.asarray([result[1] for result in fold_results])
    lines = [
        f"# Нейросеть — {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}",
        "",
        "## Гипотеза и данные",
        "",
        "Первая MLP на one-hot оценках и категориях с числовыми признаками. Данные: `data/train.csv` и `data/test.csv`; фиксированная стратифицированная 5-фолдовая валидация. Аугментация не применялась.",
        "",
        f"Строк train/test: {train_rows:,} / {test_rows:,}. Seed: {config.seed}.",
        "",
        "## Настройки",
        "",
        f"PyTorch {torch.__version__}; NumPy {np.__version__}. Файл кода: `src/swag_oct/training/run.py`.",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in asdict(config).items())
    lines.extend(["", "## Результат", "", "Метрика: ROC AUC.", ""])
    lines.extend(
        f"- Фолд {number}: AUC {auc:.6f}, лучшая эпоха {best_epoch}, проведено эпох {epochs_run}."
        for number, auc, best_epoch, epochs_run in fold_results
    )
    lines.extend([
        "",
        f"- OOF ROC AUC: **{oof_auc:.6f}**.",
        f"- Среднее по фолдам: {fold_scores.mean():.6f}; стандартное отклонение: {fold_scores.std(ddof=1):.6f}.",
        f"- Текущий файл отправки: `{submission_path.relative_to(ROOT).as_posix()}`.",
        f"- Локальные OOF/test прогнозы: `{artifact_path.relative_to(ROOT).as_posix()}`.",
        "- Public ROC AUC: ожидается после загрузки на Kaggle.",
        "",
        "## Вывод и ограничение",
        "",
        "Это базовая нейросеть для дальнейших сравнений на тех же фолдах. Лучшая эпоха выбрана по той же валидационной части, на которой измерен AUC, поэтому оценка может быть немного оптимистичной; дальнейшие изменения проверять одинаковым протоколом.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: TrainingConfig, *, smoke: bool = False) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    log_path = EXPERIMENTS / f"{'smoke' if smoke else 'run'}_{run_id}.log"
    logger = _logger(log_path)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Start: mode=%s device=%s torch=%s seed=%d", "smoke" if smoke else "full", device, torch.__version__, config.seed)
    logger.info("Log file: %s", log_path)

    train = load_csv(ROOT / "data" / "train.csv", labeled=True)
    test = load_csv(ROOT / "data" / "test.csv", labeled=False)
    if smoke:
        generator = np.random.default_rng(config.seed)
        train = _subset(train, np.sort(generator.choice(train.row_count, size=min(8000, train.row_count), replace=False)))
        test = _subset(test, np.arange(min(512, test.row_count)))
        config = replace(config, epochs=1, patience=1)
    logger.info("Loaded: train=%d, test=%d, input_dim=%d", train.row_count, test.row_count, OneHotPreprocessor().output_dim)
    if not smoke:
        validate_template(test.ids)
        logger.info("Sample submission IDs and order verified before training")

    target = train.target
    assert target is not None
    n_splits = 2 if smoke else 5
    folds = stratified_folds(target, n_splits=n_splits, seed=config.seed)
    logger.info("Encoding fixed categories once for all %d folds", n_splits)
    categorical_train = encode_one_hot(train)
    categorical_test = encode_one_hot(test)
    logger.info("One-hot cache: %.1f MiB", (categorical_train.nbytes + categorical_test.nbytes) / (1024 ** 2))

    oof = np.full(train.row_count, np.nan, dtype=np.float32)
    test_sum = np.zeros(test.row_count, dtype=np.float64)
    fold_results: list[tuple[int, float, int, int]] = []
    progress = Progress(max_epochs=n_splits * config.epochs)
    for fold in folds:
        logger.info("Preparing fold %d/%d", fold.number, n_splits)
        preprocessor = OneHotPreprocessor().fit(train, fold.train_indices)
        train_matrix = preprocessor.transform(train, fold.train_indices, one_hot=categorical_train)
        valid_matrix = preprocessor.transform(train, fold.valid_indices, one_hot=categorical_train)
        test_matrix = preprocessor.transform(test, one_hot=categorical_test)
        result = train_fold(
            train_matrix, target[fold.train_indices],
            valid_matrix, target[fold.valid_indices], test_matrix,
            config=config, fold_number=fold.number, n_folds=n_splits,
            device=device, logger=logger, progress=progress,
        )
        oof[fold.valid_indices] = result.validation_predictions
        test_sum += result.test_predictions
        fold_results.append((fold.number, result.validation_auc, result.best_epoch, result.epochs_run))
        del train_matrix, valid_matrix, test_matrix, result, preprocessor
        gc.collect()

    if not np.isfinite(oof).all():
        raise RuntimeError("Не каждая строка train получила OOF-прогноз")
    oof_auc = roc_auc(target, oof)
    mean_fold_auc = float(np.mean([result[1] for result in fold_results]))
    logger.info("Finished CV: OOF AUC=%.6f, mean fold AUC=%.6f, total time=%.1f min", oof_auc, mean_fold_auc, (time.perf_counter() - started) / 60)

    if smoke:
        logger.info("Smoke check passed. Submission and experiment record were not created.")
        return

    test_predictions = (test_sum / n_splits).astype(np.float32)
    submission_path = write_submission(test.ids, test_predictions)
    artifact_path = EXPERIMENTS / f"run_{run_id}.npz"
    np.savez_compressed(
        artifact_path,
        train_id=train.ids, target=target, oof_prediction=oof,
        test_id=test.ids, test_prediction=test_predictions,
    )
    record_path = EXPERIMENTS / f"run_{run_id}.md"
    _record_experiment(
        record_path, config=config, train_rows=train.row_count, test_rows=test.row_count,
        fold_results=fold_results, oof_auc=oof_auc,
        submission_path=submission_path, artifact_path=artifact_path,
    )
    logger.info("Submission: %s", submission_path)
    logger.info("Experiment record: %s", record_path)
    logger.info("Public score is pending; add it to this record after Kaggle evaluates the file.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Quick 2-fold, 1-epoch check; does not replace the submission")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(256, 128))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None, help="PyTorch CPU threads; default is PyTorch's choice")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs, --batch-size и --patience должны быть положительными")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("--learning-rate должен быть > 0, --weight-decay должен быть >= 0")
    if not 0 <= args.dropout < 1 or any(width < 1 for width in args.hidden_dims):
        parser.error("--dropout должен быть в [0, 1), размеры скрытых слоёв должны быть > 0")
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads должен быть положительным")
        torch.set_num_threads(args.threads)
    config = TrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        dropout=args.dropout, hidden_dims=tuple(args.hidden_dims), seed=args.seed,
    )
    run(config, smoke=args.smoke)


if __name__ == "__main__":
    main()
