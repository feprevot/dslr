import sys
from data_utils import impute_median_train, load_csv, split_features_target


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
	X, y = split_features_target(df)
	missing_before = int(X.isna().sum().sum())
	X, medians = impute_median_train(X)
	missing_after = int(X.isna().sum().sum())
	print(f"Loaded CSV: {train_csv}")
	print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
	print(f"Selected features: {X.shape[1]} numeric columns")
	print(f"Target column: Hogwarts House ({y.shape[0]} labels)")
	print(f"Missing numeric values before imputation: {missing_before}")
	print(f"Missing numeric values after imputation: {missing_after}")
	print("Feature columns used by the model:")
	for col in X.columns:
		print(f"- {col}")
	print("First 3 train medians used for imputation:")
	for col, value in medians.head(3).items():
		print(f"- {col}: {value}")
	print("First 5 target values:")
	for value in y.head(5):
		print(f"- {value}")


if __name__ == "__main__":
	main()