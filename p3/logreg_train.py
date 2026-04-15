import sys
import json
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


def preprocess_train_data(df):
	"""Split, impute missing values, and standardize training features."""
	X, y = split_features_target(df)
	missing_before = int(X.isna().sum().sum())
	X, medians = impute_median_train(X)
	missing_after = int(X.isna().sum().sum())
	means, stds = fit_standardizer(X)
	X = apply_standardizer(X, means, stds)
	return X, y, medians, means, stds, missing_before, missing_after


def build_model_dict(features, houses, medians, means, stds):
	"""Build the serializable model container."""
	return {
		"features": features,
		"houses": houses,
		"weights": {},
		"bias": {},
		"train_medians": medians.tolist(),
		"train_means": means.tolist(),
		"train_stds": stds.tolist(),
	}


def train_one_vs_all(X_np, y_np, houses, model_dict):
	"""Train one binary classifier per house and store parameters."""
	for house in houses:
		print(f"Training {house} vs all...")
		y_binary = (y_np == house).astype(float)

		w, b, loss_history = train_binary_logreg_gd(
			X_np,
			y_binary,
			learning_rate=0.1,
			epochs=1000,
		)

		model_dict["weights"][house] = w.tolist()
		model_dict["bias"][house] = float(b)

		print(f"  Initial loss: {loss_history[0]:.6f}")
		print(f"  Final loss:   {loss_history[-1]:.6f}")


def save_model(model_dict, output_path="model.json"):
	"""Save model dictionary to a JSON file."""
	with open(output_path, "w") as f:
		json.dump(model_dict, f, indent=2)


def main() -> None:
	train_csv = parse_args()
	df = load_csv(train_csv)
	X, y, medians, means, stds, missing_before, missing_after = preprocess_train_data(df)

	print(f"Loaded CSV: {train_csv}")
	print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
	print(f"Selected features: {X.shape[1]} numeric columns")
	print(f"Missing numeric values before imputation: {missing_before}")
	print(f"Missing numeric values after imputation: {missing_after}")
	print()

	# Get unique houses and convert to numpy.
	X_np = X.to_numpy(dtype=float)
	y_np = y.to_numpy()
	houses = sorted(np.unique(y_np))
	features = list(X.columns)

	# Train one-vs-all classifiers.
	model_dict = build_model_dict(features, houses, medians, means, stds)
	train_one_vs_all(X_np, y_np, houses, model_dict)

	print()
	print("All classifiers trained. Saving model...")
	save_model(model_dict, "model.json")
	print("Model saved to model.json")


if __name__ == "__main__":
	main()