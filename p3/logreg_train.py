import sys
import numpy as np
from binary_logreg import train_binary_logreg_gd
from io_utils import load_csv
from preprocessing import (
	apply_standardizer,
	fit_standardizer,
	impute_median_train,
	split_features_target,
)


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
	X, _medians = impute_median_train(X)
	missing_after = int(X.isna().sum().sum())
	means, stds = fit_standardizer(X)
	X = apply_standardizer(X, means, stds)

	print(f"Loaded CSV: {train_csv}")
	print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
	print(f"Selected features: {X.shape[1]} numeric columns")
	print(f"Missing numeric values before imputation: {missing_before}")
	print(f"Missing numeric values after imputation: {missing_after}")

	# First binary training step: Gryffindor vs all.
	y_binary = (y == "Gryffindor").astype(float).to_numpy()
	X_np = X.to_numpy(dtype=float)
	_w, _b, loss_history = train_binary_logreg_gd(
		X_np,
		y_binary,
		learning_rate=0.1,
		epochs=1000,
	)
	print("Binary training done: Gryffindor vs all")
	print(f"Initial loss: {loss_history[0]:.6f}")
	print(f"Final loss:   {loss_history[-1]:.6f}")


if __name__ == "__main__":
	main()