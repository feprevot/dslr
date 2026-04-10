import argparse
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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Logistic regression training")
	parser.add_argument("train_csv", help="Path to dataset_train.csv")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	df = load_csv(args.train_csv)
	print(f"Loaded CSV: {args.train_csv}")
	print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")


if __name__ == "__main__":
	main()