"""Validate probabilities and atomically replace the current submission."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SUBMISSION_DIR = ROOT / "submissions"
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"
SAMPLE_PATH = ROOT / "data" / "sample_submission.csv"


def validate_template(ids: np.ndarray, template_path: Path = SAMPLE_PATH) -> None:
    """Check test IDs and order before a long training run starts."""
    with template_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "satisfaction"]:
            raise ValueError("Неверный заголовок sample_submission.csv")
        observed = 0
        for observed, row in enumerate(reader, start=1):
            if observed > len(ids) or int(row["id"]) != int(ids[observed - 1]):
                raise ValueError("ID или порядок строк не совпадают с sample_submission.csv")
        if observed != len(ids):
            raise ValueError("Число строк не совпадает с sample_submission.csv")


def write_submission(
    ids: np.ndarray,
    probabilities: np.ndarray,
    *,
    destination: Path = SUBMISSION_PATH,
    template_path: Path = SAMPLE_PATH,
) -> Path:
    """Keep one current file; an invalid new run leaves the old file intact."""
    identifiers = np.asarray(ids)
    predictions = np.asarray(probabilities, dtype=np.float64)
    if identifiers.ndim != 1 or predictions.ndim != 1 or len(identifiers) != len(predictions):
        raise ValueError("ID и вероятности должны быть одномерными массивами одной длины")
    if not len(identifiers) or len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("ID должны быть непустыми и уникальными")
    if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("Вероятности должны лежать между 0 и 1 и не содержать NaN")
    validate_template(identifiers, template_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", delete=False,
            dir=destination.parent, prefix=".submission-", suffix=".tmp",
        ) as stream:
            temp_path = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(("id", "satisfaction"))
            writer.writerows((int(identifier), f"{score:.12f}") for identifier, score in zip(identifiers, predictions))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return destination
