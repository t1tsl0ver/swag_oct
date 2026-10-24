"""Fit one tabular MLP fold with progress logs and AUC-based early stopping."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from swag_oct.neural_network.mlp import TabularMLP
from swag_oct.validation.folds import roc_auc


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    patience: int = 5
    seed: int = 42
    min_improvement: float = 1e-5


@dataclass
class Progress:
    max_epochs: int
    started_at: float = field(default_factory=time.perf_counter)
    completed_epochs: int = 0

    def finish_epoch(self) -> str:
        self.completed_epochs += 1
        elapsed = time.perf_counter() - self.started_at
        remaining = max(0, self.max_epochs - self.completed_epochs)
        estimate = elapsed / self.completed_epochs * remaining
        return _duration(estimate)


@dataclass(frozen=True)
class FoldResult:
    validation_predictions: np.ndarray
    test_predictions: np.ndarray
    validation_auc: float
    best_epoch: int
    epochs_run: int
    seconds: float


def _duration(seconds: float) -> str:
    minutes, remainder = divmod(int(max(0, seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{remainder:02d}"


def _predict(
    model: nn.Module,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    targets: torch.Tensor | None = None,
) -> tuple[np.ndarray, float | None]:
    model.eval()
    probabilities = np.empty(len(features), dtype=np.float32)
    total_loss = 0.0
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            x_batch = features[start:end].to(device)
            logits = model(x_batch)
            probabilities[start:end] = torch.sigmoid(logits).cpu().numpy()
            if targets is not None:
                y_batch = targets[start:end].to(device)
                total_loss += F.binary_cross_entropy_with_logits(logits, y_batch, reduction="sum").item()
    mean_loss = total_loss / len(features) if targets is not None else None
    return probabilities, mean_loss


def train_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    *,
    config: TrainingConfig,
    fold_number: int,
    n_folds: int,
    device: torch.device,
    logger: logging.Logger,
    progress: Progress,
) -> FoldResult:
    if config.epochs < 1 or config.batch_size < 1 or config.patience < 1:
        raise ValueError("epochs, batch_size и patience должны быть положительными")
    fold_started = time.perf_counter()
    torch.manual_seed(config.seed + fold_number)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + fold_number)

    train_x = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    train_y = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))
    valid_x = torch.from_numpy(np.ascontiguousarray(x_valid, dtype=np.float32))
    valid_y = torch.from_numpy(np.ascontiguousarray(y_valid, dtype=np.float32))
    test_x = torch.from_numpy(np.ascontiguousarray(x_test, dtype=np.float32))

    model = TabularMLP(train_x.shape[1], config.hidden_dims, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    logger.info(
        "Fold %d/%d: train=%d, valid=%d, test=%d, input=%d, parameters=%d",
        fold_number, n_folds, len(train_x), len(valid_x), len(test_x), train_x.shape[1],
        sum(parameter.numel() for parameter in model.parameters()),
    )

    best_auc = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: np.ndarray | None = None
    without_improvement = 0
    epochs_run = 0
    batch_count = (len(train_x) + config.batch_size - 1) // config.batch_size

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        epochs_run = epoch
        model.train()
        shuffled = torch.randperm(len(train_x), generator=generator)
        total_train_loss = 0.0
        log_every = max(1, batch_count // 4)
        for step, start in enumerate(range(0, len(train_x), config.batch_size), start=1):
            positions = shuffled[start:start + config.batch_size]
            x_batch = train_x[positions].to(device)
            y_batch = train_y[positions].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(positions)
            if step % log_every == 0 and step < batch_count:
                current_seconds = time.perf_counter() - epoch_started
                eta_epoch = current_seconds / step * (batch_count - step)
                logger.info(
                    "Fold %d/%d epoch %d/%d: batches %d/%d, epoch ETA ~%s",
                    fold_number, n_folds, epoch, config.epochs, step, batch_count, _duration(eta_epoch),
                )

        validation_probabilities, validation_loss = _predict(
            model, valid_x, device=device, batch_size=config.batch_size * 2, targets=valid_y,
        )
        auc = roc_auc(y_valid, validation_probabilities)
        improved = auc > best_auc + config.min_improvement
        if improved:
            best_auc = auc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_validation = validation_probabilities.copy()
            without_improvement = 0
        else:
            without_improvement += 1

        max_eta = progress.finish_epoch()
        logger.info(
            "Fold %d/%d epoch %d/%d: train_loss=%.5f val_loss=%.5f val_auc=%.6f best_auc=%.6f "
            "epoch_time=%s remaining_at_max_epochs~%s",
            fold_number, n_folds, epoch, config.epochs, total_train_loss / len(train_x),
            validation_loss, auc, best_auc, _duration(time.perf_counter() - epoch_started), max_eta,
        )
        if without_improvement >= config.patience:
            logger.info("Fold %d/%d: early stopping after %d epochs without AUC improvement", fold_number, n_folds, without_improvement)
            break

    assert best_state is not None and best_validation is not None
    model.load_state_dict(best_state)
    test_probabilities, _ = _predict(model, test_x, device=device, batch_size=config.batch_size * 2)
    duration = time.perf_counter() - fold_started
    logger.info("Fold %d/%d finished: best_epoch=%d AUC=%.6f time=%s", fold_number, n_folds, best_epoch, best_auc, _duration(duration))
    return FoldResult(best_validation, test_probabilities, best_auc, best_epoch, epochs_run, duration)
