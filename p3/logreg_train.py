import sys
from pathlib import Path

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
	"""Load a CSV file and validate common error cases.

	Args:
		path: Path to the CSV file.

	Returns:
		A pandas DataFrame containing the dataset.

	Raises:
		FileNotFoundError: if the file does not exist.
		ValueError: if the CSV is empty.
	"""
	csv_path = Path(path)
	if not csv_path.exists():
		raise FileNotFoundError(f"File not found: {csv_path}")

	df = pd.read_csv(csv_path)
	if df.empty:
		raise ValueError(f"Empty CSV file: {csv_path}")

	return df


def parse_args() -> str:
	"""Parse CLI args using sys.argv.

	Expected usage:
		python p3/logreg_train.py <train_csv>
	"""
	if len(sys.argv) != 2:
		raise ValueError("Usage: python p3/logreg_train.py <train_csv>")

	train_csv = sys.argv[1]
	return train_csv


def main() -> None:
	train_csv = parse_args()
	df = load_csv(train_csv)
	print(f"Loaded CSV: {train_csv}")
	print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")


if __name__ == "__main__":
	main()