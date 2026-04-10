import argparse
from pathlib import Path

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
	"""Load a CSV file and validate common error cases."""
	csv_path = Path(path) """create a Path object from the provided path string"""
	if not csv_path.exists():
		raise FileNotFoundError(f"File not found: {csv_path}")

	df = pd.read_csv(csv_path)
	if df.empty:
		raise ValueError(f"Empty CSV file: {csv_path}")

	return df


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Logistic regression prediction")
	parser.add_argument("test_csv", help="Path to dataset_test.csv")
	parser.add_argument("model_file", help="Path to trained model weights file")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	df = load_csv(args.test_csv)
	print(f"Loaded CSV: {args.test_csv}")
	print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
	print(f"Model file to load next: {args.model_file}")


if __name__ == "__main__":
	main()