"""Check the validation contracts that determine reported competition scores."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swag_oct.validation.folds import roc_auc, stratified_folds


class ValidationTests(unittest.TestCase):
    def test_auc_counts_ranking_and_ties(self) -> None:
        target = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(roc_auc(target, np.array([0.1, 0.4, 0.35, 0.8])), 0.75)
        self.assertAlmostEqual(roc_auc(target, np.array([0.5, 0.5, 0.5, 0.5])), 0.5)
        self.assertAlmostEqual(roc_auc(target, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)

    def test_each_row_is_validated_once_without_train_overlap(self) -> None:
        target = np.array([0] * 55 + [1] * 45, dtype=np.uint8)
        folds = stratified_folds(target, n_splits=5, seed=42)
        validated = np.concatenate([fold.valid_indices for fold in folds])
        np.testing.assert_array_equal(np.sort(validated), np.arange(len(target)))
        for fold in folds:
            self.assertEqual(len(np.intersect1d(fold.train_indices, fold.valid_indices)), 0)
            self.assertEqual(int(target[fold.valid_indices].sum()), 9)


if __name__ == "__main__":
    unittest.main()
