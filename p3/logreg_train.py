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
		print(f"[TRAIN] {house:<10} vs all")
		y_binary = (y_np == house).astype(float)

		w, b, loss_history = train_binary_logreg_gd(
			X_np,
			y_binary,
			learning_rate=0.1,
			epochs=1000,
		)

		model_dict["weights"][house] = w.tolist()
		model_dict["bias"][house] = float(b)

		print(f"        initial_loss = {loss_history[0]:.6f}")
		print(f"        final_loss   = {loss_history[-1]:.6f}")


def compute_training_accuracy(X_np, y_np, model_dict):
	"""Compute training accuracy for the full one-vs-all model."""
	houses = model_dict["houses"]
	weights = model_dict["weights"]
	bias = model_dict["bias"]

	proba_list = []
	for house in houses:
		w = np.array(weights[house], dtype=float)
		b = float(bias[house])
		z = X_np @ w + b
		p = 1.0 / (1.0 + np.exp(-z))
		proba_list.append(p)

	proba_matrix = np.column_stack(proba_list)
	best_idx = np.argmax(proba_matrix, axis=1)
	predictions = np.array([houses[i] for i in best_idx])
	accuracy = float(np.mean(predictions == y_np))
	return accuracy


def save_model(model_dict, output_path="model.json"):
	"""Save model dictionary to a JSON file."""
	with open(output_path, "w") as f:
		json.dump(model_dict, f, indent=2)


def main() -> None:
	train_csv = parse_args()
	df = load_csv(train_csv)
	X, y, medians, means, stds, missing_before, missing_after = preprocess_train_data(df)

	print("=" * 66)
	print("LOGISTIC REGRESSION TRAINING".center(66))
	print("=" * 66)
	print("[DATA]")
	print(f"  csv_path          : {train_csv}")
	print(f"  shape             : {df.shape[0]} rows x {df.shape[1]} cols")
	print(f"  selected_features : {X.shape[1]} numeric")
	print(f"  missing_before    : {missing_before}")
	print(f"  missing_after     : {missing_after}")
	print("-" * 66)

	# Get unique houses and convert to numpy.
	X_np = X.to_numpy(dtype=float)
	y_np = y.to_numpy()
	houses = sorted(np.unique(y_np))
	features = list(X.columns)

	# Train one-vs-all classifiers.
	model_dict = build_model_dict(features, houses, medians, means, stds)
	train_one_vs_all(X_np, y_np, houses, model_dict)
	train_accuracy = compute_training_accuracy(X_np, y_np, model_dict)

	print("-" * 66)
	print("[SUMMARY]")
	print(f"  training_accuracy : {train_accuracy * 100:.2f}%")
	print("  save_model        : model.json")
	save_model(model_dict, "model.json")
	print("  status            : done")
	print("=" * 66)


if __name__ == "__main__":
	main()