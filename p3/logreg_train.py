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


def main() -> None:
	train_csv = parse_args()
	df = load_csv(train_csv)
	X, y = split_features_target(df)
	missing_before = int(X.isna().sum().sum())
	X, medians = impute_median_train(X)
	missing_after = int(X.isna().sum().sum())
	means, stds = fit_standardizer(X)
	X = apply_standardizer(X, means, stds)

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
	model_dict = {
		"features": features,
		"houses": houses,
		"weights": {},
		"bias": {},
		"train_medians": medians.tolist(),
		"train_means": means.tolist(),
		"train_stds": stds.tolist(),
	}

	for house in houses:
		print(f"Training {house} vs all...")
		# Create binary label: 1 if house, 0 otherwise.
		y_binary = (y_np == house).astype(float)
		
		# Train binary classifier.
		w, b, loss_history = train_binary_logreg_gd(
			X_np,
			y_binary,
			learning_rate=0.1,
			epochs=1000,
		)
		
		# Store weights and bias.
		model_dict["weights"][house] = w.tolist()
		model_dict["bias"][house] = float(b)
		
		print(f"  Initial loss: {loss_history[0]:.6f}")
		print(f"  Final loss:   {loss_history[-1]:.6f}")

	print()
	print("All classifiers trained. Saving model...")
	
	# Save model to JSON.
	with open("model.json", "w") as f:
		json.dump(model_dict, f, indent=2)
	
	print("Model saved to model.json")


if __name__ == "__main__":
	main()