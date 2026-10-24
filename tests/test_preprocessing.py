"""Check that category coding and fold statistics remain separate."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swag_oct.preprocessing.one_hot import (
    CATEGORICAL_COLUMNS, CATEGORY_MAPS, CATEGORY_VALUES, FEATURE_COLUMNS,
    NUMERICAL_COLUMNS, ONE_HOT_WIDTH, OneHotPreprocessor, TabularData,
    encode_one_hot, load_csv,
)


class PreprocessingTests(unittest.TestCase):
    def test_csv_loader_uses_column_names_and_keeps_target_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            row = {name: "1" for name in FEATURE_COLUMNS}
            row.update({"Gender": "Female", "Customer Type": "Loyal Customer", "Type of Travel": "Business travel", "Class": "Eco"})
            row["Arrival Delay in Minutes"] = ""
            row["id"] = "7"
            row["satisfaction"] = "True"
            header = ["satisfaction", *reversed(FEATURE_COLUMNS), "id"]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerow([row[name] for name in header])
            loaded = load_csv(path, labeled=True)
            self.assertEqual(int(loaded.ids[0]), 7)
            self.assertEqual(int(loaded.target[0]), 1)
            self.assertEqual(int(loaded.categorical_codes[0, CATEGORICAL_COLUMNS.index("Gender")]), CATEGORY_MAPS[0]["Female"])
            self.assertTrue(np.isnan(loaded.numerical[0, NUMERICAL_COLUMNS.index("Arrival Delay in Minutes")]))

    def test_one_hot_and_numeric_fit_use_the_right_rows(self) -> None:
        codes = np.ones((3, len(CATEGORICAL_COLUMNS)), dtype=np.uint8)
        gender_index = CATEGORICAL_COLUMNS.index("Gender")
        baggage_index = CATEGORICAL_COLUMNS.index("Baggage handling")
        codes[:, gender_index] = [
            CATEGORY_MAPS[gender_index]["Male"],
            CATEGORY_MAPS[gender_index]["Female"],
            len(CATEGORY_VALUES[gender_index]),
        ]
        codes[:, baggage_index] = [
            CATEGORY_MAPS[baggage_index]["1"],
            CATEGORY_MAPS[baggage_index]["2"],
            CATEGORY_MAPS[baggage_index]["0"],
        ]
        numerical = np.array(
            [[20, 100, 0, np.nan], [40, 200, 1, 5], [100, 150, 0, np.nan]],
            dtype=np.float32,
        )
        data = TabularData(np.arange(3), codes, numerical, np.array([0, 1, 0], dtype=np.uint8))
        fitted = OneHotPreprocessor().fit(data, [0, 1])
        cached = encode_one_hot(data)
        row = fitted.transform(data, [2], one_hot=cached)[0]
        names = fitted.feature_names_

        self.assertEqual(row.shape, (109,))
        self.assertEqual(row[:ONE_HOT_WIDTH].sum(), len(CATEGORICAL_COLUMNS))
        self.assertEqual(row[names.index("Gender=<unknown>")], 1)
        self.assertEqual(row[names.index("Baggage handling=0")], 1)
        self.assertEqual(row[names.index("Arrival Delay in Minutes (missing)")], 1)
        self.assertAlmostEqual(float(row[names.index("Age (standardized)")]), 7.0)


if __name__ == "__main__":
    unittest.main()
