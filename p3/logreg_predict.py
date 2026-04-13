import sys
from pathlib import Path

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
	"""Load a CSV file and validate common error cases."""
	# Create a Path object from the provided path string.
	csv_path = Path(path)
	if not csv_path.exists():
		raise FileNotFoundError(f"File not found: {csv_path}")

	df = pd.read_csv(csv_path)
	if df.empty:
		raise ValueError(f"Empty CSV file: {csv_path}")

	return df


def parse_args() -> tuple[str, str]:
	"""Parse CLI args using sys.argv.

	Expected usage:
		python p3/logreg_predict.py <test_csv> <model_file>
	"""
	if len(sys.argv) != 3:
		raise ValueError(
			"Usage: python p3/logreg_predict.py <test_csv> <model_file>"
		)

	test_csv = sys.argv[1]
	model_file = sys.argv[2]
	return test_csv, model_file


def main() -> None:
	test_csv, model_file = parse_args()
	df = load_csv(test_csv)
	print(f"Loaded CSV: {test_csv}")
	print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
	print(f"Model file to load next: {model_file}")


if __name__ == "__main__":
	main()