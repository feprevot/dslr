import sys
from data_utils import load_csv


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