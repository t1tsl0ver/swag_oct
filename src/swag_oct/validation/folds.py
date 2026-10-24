"""Deterministic stratified folds and exact ROC AUC with tie handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    number: int
    train_indices: np.ndarray
    valid_indices: np.ndarray


def stratified_folds(target: np.ndarray, *, n_splits: int = 5, seed: int = 42) -> list[Fold]:
    labels = np.asarray(target)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("Ожидается одномерная бинарная целевая переменная")
    if n_splits < 2:
        raise ValueError("Нужно не менее двух фолдов")
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    if any(len(group) < n_splits for group in groups):
        raise ValueError("В каждом классе должно быть не меньше строк, чем фолдов")

    generator = np.random.default_rng(seed)
    class_parts = [np.array_split(generator.permutation(group), n_splits) for group in groups]
    folds = []
    for number in range(n_splits):
        valid = np.sort(np.concatenate((class_parts[0][number], class_parts[1][number])))
        is_training = np.ones(len(labels), dtype=bool)
        is_training[valid] = False
        train = np.flatnonzero(is_training)
        folds.append(Fold(number=number + 1, train_indices=train, valid_indices=valid))
    return folds


def roc_auc(target: np.ndarray, scores: np.ndarray) -> float:
    """Probability that a positive ranks above a negative; ties count half."""
    labels = np.asarray(target)
    predictions = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or predictions.ndim != 1 or len(labels) != len(predictions):
        raise ValueError("Метка и прогноз должны быть одномерными массивами одинаковой длины")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(predictions).all():
        raise ValueError("Метки должны быть бинарными, прогнозы — конечными")
    positives = int(np.count_nonzero(labels))
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Для ROC AUC нужны оба класса")

    order = np.argsort(predictions, kind="stable")
    ordered_scores = predictions[order]
    ordered_labels = labels[order]
    starts = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1]
    group_positives = np.add.reduceat(ordered_labels.astype(np.int64), starts).astype(np.float64)
    group_sizes = np.diff(np.r_[starts, len(labels)]).astype(np.float64)
    group_negatives = group_sizes - group_positives
    negatives_below = np.cumsum(group_negatives) - group_negatives
    wins = np.sum(group_positives * negatives_below + 0.5 * group_positives * group_negatives)
    return float(wins / (positives * negatives))
