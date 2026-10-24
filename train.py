"""Run the competition training pipeline from the repository root."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from swag_oct.training.run import main  # noqa: E402


if __name__ == "__main__":
    main()
