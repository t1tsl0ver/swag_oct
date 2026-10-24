"""Check output order and safe replacement of the current Kaggle file."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swag_oct.submission.write import write_submission


class SubmissionTests(unittest.TestCase):
    def test_valid_file_replaces_old_and_invalid_run_keeps_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "sample.csv"
            template.write_text("id,satisfaction\n10,0\n11,0\n", encoding="utf-8")
            destination = root / "submission.csv"
            destination.write_text("old version", encoding="utf-8")
            write_submission(
                np.array([10, 11]), np.array([0.2, 0.8]),
                destination=destination, template_path=template,
            )
            with destination.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["id"] for row in rows], ["10", "11"])
            self.assertEqual([float(row["satisfaction"]) for row in rows], [0.2, 0.8])
            before = destination.read_bytes()
            with self.assertRaises(ValueError):
                write_submission(
                    np.array([11, 10]), np.array([0.3, 0.7]),
                    destination=destination, template_path=template,
                )
            self.assertEqual(destination.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
