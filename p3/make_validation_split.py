import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> tuple[str, str, str, int]:
	"""Parse CLI args using sys.argv.

	Usage:
		python p3/make_validation_split.py <input_csv> [seed]

	Defaults:
		train_out = datasets/validation/dataset_train_80.csv
		val_out = datasets/validation/dataset_val_20.csv
	"""
	if len(sys.argv) < 2 or len(sys.argv) > 3:
		raise SystemExit(
			"Usage: python p3/make_validation_split.py <input_csv> [seed]\n"
			"Outputs are fixed to: datasets/validation/dataset_train_80.csv and "
			"datasets/validation/dataset_val_20.csv"
		)

	input_csv = sys.argv[1]
	train_out = "datasets/validation/dataset_train_80.csv"
	val_out = "datasets/validation/dataset_val_20.csv"
	if len(sys.argv) == 3:
		try:
			seed = int(sys.argv[2])
		except ValueError as exc:
			raise SystemExit(
				"Error: seed must be an integer.\n"
				"Usage: python p3/make_validation_split.py <input_csv> [seed]"
			) from exc
	else:
		seed = 42
	return input_csv, train_out, val_out, seed


def stratified_split_80_20(df: pd.DataFrame, target_col: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Create a stratified 80/20 split while preserving class proportions."""
	if target_col not in df.columns:
		raise ValueError(f"Missing target column: {target_col}")

	rng = np.random.default_rng(seed)
	train_idx: list[int] = []
	val_idx: list[int] = []

	for house in sorted(df[target_col].dropna().unique()):
		house_indices = np.array(df.index[df[target_col] == house].to_numpy(), copy=True)
		rng.shuffle(house_indices)

		n_total = len(house_indices)
		n_val = int(round(n_total * 0.2))
		if n_val <= 0:
			n_val = 1
		if n_val >= n_total:
			n_val = n_total - 1

		val_idx.extend(house_indices[:n_val].tolist())
		train_idx.extend(house_indices[n_val:].tolist())

	rng.shuffle(train_idx)
	rng.shuffle(val_idx)

	train_df = df.loc[train_idx].copy()
	val_df = df.loc[val_idx].copy()

	if "Index" in train_df.columns:
		train_df = train_df.sort_values(by="Index").reset_index(drop=True)
	else:
		train_df = train_df.reset_index(drop=True)

	if "Index" in val_df.columns:
		val_df = val_df.sort_values(by="Index").reset_index(drop=True)
	else:
		val_df = val_df.reset_index(drop=True)
	return train_df, val_df


def print_distribution(df: pd.DataFrame, target_col: str, title: str) -> None:
	counts = df[target_col].value_counts().sort_index()
	print(title)
	for house, count in counts.items():
		print(f"  {house}: {int(count)}")


def main() -> None:
	input_csv, train_out, val_out, seed = parse_args()
	try:
		df = pd.read_csv(input_csv)
	except FileNotFoundError as exc:
		raise SystemExit(f"Error: File not found: {input_csv}") from exc

	if "Hogwarts House" not in df.columns:
		raise SystemExit(
			"Error: input CSV must contain 'Hogwarts House' labels (use dataset_train.csv)."
		)

	if df["Hogwarts House"].dropna().empty:
		raise SystemExit(
			"Error: input CSV has no non-empty 'Hogwarts House' labels; cannot build stratified split."
		)

	train_df, val_df = stratified_split_80_20(df, target_col="Hogwarts House", seed=seed)

	Path(train_out).parent.mkdir(parents=True, exist_ok=True)
	Path(val_out).parent.mkdir(parents=True, exist_ok=True)

	train_df.to_csv(train_out, index=False)
	val_df.to_csv(val_out, index=False)

	print(f"Input: {input_csv}")
	print(f"Total rows: {len(df)}")
	print(f"Train rows (80%): {len(train_df)} -> {train_out}")
	print(f"Val rows (20%):  {len(val_df)} -> {val_out}")
	print(f"Seed: {seed}")
	print_distribution(train_df, "Hogwarts House", "Train class distribution:")
	print_distribution(val_df, "Hogwarts House", "Validation class distribution:")


if __name__ == "__main__":
	main()
