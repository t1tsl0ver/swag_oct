r"""Fixed one-hot schema plus fold-specific numerical preprocessing.

Run a check from the repository root:
    .\.venv\Scripts\python.exe src\swag_oct\preprocessing\one_hot.py

CSV loading maps categories to compact integer codes once. The mapping is fixed
before cross-validation, so it can be shared by all folds. One-hot vectors are
materialized only for requested rows or batches. Medians, means, and scales
depend on data and are fitted on each fold's training indices.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]

TEXT_COLUMNS = (
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
)
RATING_COLUMNS = (
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
)
NUMERICAL_COLUMNS = (
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
)
CATEGORICAL_COLUMNS = (*TEXT_COLUMNS, *RATING_COLUMNS)
FEATURE_COLUMNS = (*CATEGORICAL_COLUMNS, *NUMERICAL_COLUMNS)
NUMERICAL_MISSING_INDICATORS = ("Arrival Delay in Minutes",)

# These possible values are a schema, not statistics learned from train/val.
# A dedicated 0 slot exists for every service rating, even when a fold has no 0.
RATING_VALUES = tuple(str(value) for value in range(6))
TEXT_VALUES = {
    "Gender": ("Female", "Male"),
    "Customer Type": ("Loyal Customer", "disloyal Customer"),
    "Type of Travel": ("Business travel", "Personal Travel"),
    "Class": ("Business", "Eco", "Eco Plus"),
}
CATEGORY_VALUES = tuple(TEXT_VALUES[name] if name in TEXT_VALUES else RATING_VALUES for name in CATEGORICAL_COLUMNS)
CATEGORY_MAPS = tuple({value: index for index, value in enumerate(values)} for values in CATEGORY_VALUES)
ONE_HOT_NAMES = tuple(
    feature_name
    for name, values in zip(CATEGORICAL_COLUMNS, CATEGORY_VALUES)
    for feature_name in (*[f"{name}={value}" for value in values], f"{name}=<unknown>")
)
ONE_HOT_WIDTH = len(ONE_HOT_NAMES)
NUMERICAL_NAMES = tuple(
    feature_name
    for name in NUMERICAL_COLUMNS
    for feature_name in (
        (f"{name} (standardized)", f"{name} (missing)")
        if name in NUMERICAL_MISSING_INDICATORS
        else (f"{name} (standardized)",)
    )
)
FEATURE_NAMES = (*ONE_HOT_NAMES, *NUMERICAL_NAMES)


@dataclass(frozen=True)
class TabularData:
    ids: np.ndarray
    categorical_codes: np.ndarray
    numerical: np.ndarray
    target: np.ndarray | None

    @property
    def row_count(self) -> int:
        return len(self.ids)


def _indices(data: TabularData, indices: Sequence[int] | np.ndarray | None) -> np.ndarray:
    if indices is None:
        return np.arange(data.row_count, dtype=np.int64)
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1:
        raise ValueError("Индексы должны быть одномерными")
    if len(selected) and (selected.min() < 0 or selected.max() >= data.row_count):
        raise IndexError("Индекс строки вне диапазона данных")
    return selected


def _count_csv_rows(path: Path) -> int:
    # Competition CSV records occupy one physical line each. The CSV pass
    # below checks that this fast line count matches the number of records.
    with path.open("rb") as stream:
        return sum(1 for _ in stream) - 1


def load_csv(path: Path, *, labeled: bool) -> TabularData:
    """Load the competition file; encode categories once with the fixed map."""
    if not path.is_file():
        raise FileNotFoundError(path)
    row_count = _count_csv_rows(path)
    if row_count <= 0:
        raise ValueError(f"Пустой файл данных: {path}")

    ids = np.empty(row_count, dtype=np.int64)
    codes = np.empty((row_count, len(CATEGORICAL_COLUMNS)), dtype=np.uint8)
    numerical = np.empty((row_count, len(NUMERICAL_COLUMNS)), dtype=np.float32)
    target = np.empty(row_count, dtype=np.uint8) if labeled else None

    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        expected = set(FEATURE_COLUMNS) | {"id"}
        if labeled:
            expected.add("satisfaction")
        if header is None or set(header) != expected or len(header) != len(expected):
            missing = expected - set(header or ())
            extra = set(header or ()) - expected
            raise ValueError(f"Неверные столбцы {path.name}: нет {sorted(missing)}, лишние {sorted(extra)}")
        positions = {name: index for index, name in enumerate(header)}
        categorical_positions = [positions[name] for name in CATEGORICAL_COLUMNS]
        numerical_positions = [positions[name] for name in NUMERICAL_COLUMNS]

        observed_rows = 0
        for index, row in enumerate(reader):
            if index >= row_count:
                raise ValueError(f"Число CSV-записей превышает число строк файла: {path}")
            if len(row) != len(header):
                raise ValueError(f"Неверное число столбцов в {path.name}, строка {index + 2}")
            observed_rows += 1
            ids[index] = int(row[positions["id"]])
            for column_index, position in enumerate(categorical_positions):
                raw = row[position].strip()
                codes[index, column_index] = CATEGORY_MAPS[column_index].get(raw, len(CATEGORY_VALUES[column_index]))
            for column_index, position in enumerate(numerical_positions):
                raw = row[position].strip()
                value = float(raw) if raw else np.nan
                if np.isinf(value):
                    raise ValueError(f"Бесконечное значение в {NUMERICAL_COLUMNS[column_index]}, строка {index + 2}")
                numerical[index, column_index] = value
            if target is not None:
                label = row[positions["satisfaction"]].strip().lower()
                if label not in ("true", "false", "1", "0"):
                    raise ValueError(f"Неизвестная метка satisfaction: {label!r}")
                target[index] = label in ("true", "1")

    if observed_rows != row_count:
        raise ValueError(f"Число CSV-записей не совпало с числом строк файла: {path}")
    return TabularData(ids=ids, categorical_codes=codes, numerical=numerical, target=target)


def encode_one_hot(data: TabularData, indices: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
    """Expand fixed category codes to compact uint8 one-hot rows or a batch."""
    selected = _indices(data, indices)
    result = np.zeros((len(selected), ONE_HOT_WIDTH), dtype=np.uint8)
    row_positions = np.arange(len(selected))
    offset = 0
    for column_index, values in enumerate(CATEGORY_VALUES):
        column_codes = data.categorical_codes[selected, column_index]
        if len(column_codes) and column_codes.max() > len(values):
            raise ValueError(f"Недопустимый код категории: {CATEGORICAL_COLUMNS[column_index]}")
        result[row_positions, offset + column_codes] = 1
        offset += len(values) + 1
    return result


class OneHotPreprocessor:
    """Join fixed one-hot categories with numerical values scaled per fold."""

    def __init__(self) -> None:
        self.medians_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.scales_: np.ndarray | None = None

    @property
    def feature_names_(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    @property
    def output_dim(self) -> int:
        return len(FEATURE_NAMES)

    def fit(self, data: TabularData, indices: Sequence[int] | np.ndarray) -> OneHotPreprocessor:
        """Fit only the numerical medians and scales on training-fold rows."""
        if data.target is None:
            raise ValueError("Обучать статистики можно только на размеченном train")
        selected = _indices(data, indices)
        if not len(selected):
            raise ValueError("Обучающая часть фолда пуста")

        medians = np.empty(len(NUMERICAL_COLUMNS), dtype=np.float32)
        means = np.empty(len(NUMERICAL_COLUMNS), dtype=np.float32)
        scales = np.empty(len(NUMERICAL_COLUMNS), dtype=np.float32)
        for column_index, name in enumerate(NUMERICAL_COLUMNS):
            values = data.numerical[selected, column_index]
            present = values[~np.isnan(values)]
            if not len(present):
                raise ValueError(f"В обучающем фолде все значения {name} пропущены")
            median = np.median(present)
            filled = np.where(np.isnan(values), median, values)
            scale = float(np.std(filled, dtype=np.float64))
            medians[column_index] = median
            means[column_index] = np.mean(filled, dtype=np.float64)
            scales[column_index] = scale if scale > 0 else 1.0

        self.medians_ = medians
        self.means_ = means
        self.scales_ = scales
        return self

    def transform_numerical(self, data: TabularData, indices: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        """Apply fold medians and scales; add a delay-missing indicator."""
        if self.medians_ is None or self.means_ is None or self.scales_ is None:
            raise RuntimeError("Сначала вызовите fit")
        selected = _indices(data, indices)
        result = np.empty((len(selected), len(NUMERICAL_NAMES)), dtype=np.float32)
        offset = 0
        for column_index, name in enumerate(NUMERICAL_COLUMNS):
            raw = data.numerical[selected, column_index]
            missing = np.isnan(raw)
            filled = np.where(missing, self.medians_[column_index], raw)
            result[:, offset] = (filled - self.means_[column_index]) / self.scales_[column_index]
            offset += 1
            if name in NUMERICAL_MISSING_INDICATORS:
                result[:, offset] = missing
                offset += 1
        return result

    def transform(
        self,
        data: TabularData,
        indices: Sequence[int] | np.ndarray | None = None,
        *,
        one_hot: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return float32 NN input; optionally reuse a cached one-hot matrix."""
        selected = _indices(data, indices)
        result = np.empty((len(selected), self.output_dim), dtype=np.float32)
        if one_hot is None:
            result[:, :ONE_HOT_WIDTH] = encode_one_hot(data, selected)
        elif one_hot.ndim != 2 or one_hot.shape[1] != ONE_HOT_WIDTH:
            raise ValueError("Неверная ширина кэшированной one-hot матрицы")
        elif len(one_hot) == data.row_count:
            # Advanced indexing creates a temporary array. Bounded chunks keep
            # memory use stable when transforming a whole training fold.
            for start in range(0, len(selected), 65536):
                end = min(start + 65536, len(selected))
                result[start:end, :ONE_HOT_WIDTH] = one_hot[selected[start:end]]
        elif len(one_hot) == len(selected):
            result[:, :ONE_HOT_WIDTH] = one_hot
        else:
            raise ValueError("Число строк one-hot матрицы не совпало с данными")
        result[:, ONE_HOT_WIDTH:] = self.transform_numerical(data, selected)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "test.csv")
    args = parser.parse_args()
    train = load_csv(args.train, labeled=True)
    test = load_csv(args.test, labeled=False)

    # Diagnostic slice only: no model is trained or scored here.
    holdout_size = min(1000, train.row_count // 5)
    held_out = np.arange(holdout_size)
    fitted = OneHotPreprocessor().fit(train, np.arange(holdout_size, train.row_count))
    validation_one_hot = encode_one_hot(train, held_out)
    test_one_hot = encode_one_hot(test, np.arange(min(1000, test.row_count)))
    validation_matrix = fitted.transform(train, held_out, one_hot=validation_one_hot)
    test_matrix = fitted.transform(test, np.arange(len(test_one_hot)), one_hot=test_one_hot)

    for matrix in (validation_matrix, test_matrix):
        if not np.isfinite(matrix).all():
            raise AssertionError("После преобразования остались нечисловые значения")
        if not np.all(matrix[:, :ONE_HOT_WIDTH].sum(axis=1) == len(CATEGORICAL_COLUMNS)):
            raise AssertionError("В каждом one-hot блоке должна быть ровно одна единица")
    if not all(f"{name}=0" in FEATURE_NAMES for name in RATING_COLUMNS):
        raise AssertionError("Для каждой оценки нужен отдельный one-hot столбец 0")

    unknowns = sum(
        np.count_nonzero(test.categorical_codes[:, index] == len(values))
        for index, values in enumerate(CATEGORY_VALUES)
    )
    print(f"Train: {train.row_count:,} строк; test: {test.row_count:,} строк")
    print(f"Категориальных полей: {len(CATEGORICAL_COLUMNS)}, one-hot столбцов: {ONE_HOT_WIDTH}")
    print(f"Числовых входов: {len(NUMERICAL_NAMES)}, всего входов сети: {fitted.output_dim}")
    print(f"Проверено строк validation/test: {len(validation_matrix)}/{len(test_matrix)}")
    print(f"Неизвестных категорий в test: {unknowns}")


if __name__ == "__main__":
    main()
