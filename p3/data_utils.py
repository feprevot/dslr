from pathlib import Path

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
	"""Load a CSV file and validate common error cases."""
	csv_path = Path(path)
	if not csv_path.exists():
		raise FileNotFoundError(f"File not found: {csv_path}")

	df = pd.read_csv(csv_path)
	if df.empty:
		raise ValueError(f"Empty CSV file: {csv_path}")

	return df


def split_features_target(
	df: pd.DataFrame,
	target_col: str = "Hogwarts House",
	exclude_cols: tuple[str, ...] = ("Index",),
) -> tuple[pd.DataFrame, pd.Series]:
	"""Split dataset into numeric features X and target labels y.

	Rules:
	- Keep only numeric feature columns.
	- Exclude columns listed in exclude_cols from features.
	- Target must exist in train dataset.
	"""
	if target_col not in df.columns:
		raise ValueError(f"Missing target column: {target_col}")

	y = df[target_col].copy()

	numeric_cols = df.select_dtypes(include="number").columns.tolist()
	feature_cols = [col for col in numeric_cols if col not in exclude_cols]

	if not feature_cols:
		raise ValueError("No numeric feature columns found after exclusions")

	X = df[feature_cols].copy()
	return X, y


def impute_median_train(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
	"""Fit median values on train features and impute missing values.

	Returns:
		X_imputed: train features with NaN replaced by column medians.
		medians: median per feature column, to reuse on test/predict.
	"""
	medians = X.median(numeric_only=True)
	X_imputed = X.fillna(medians)
	return X_imputed, medians


def impute_with_train_medians(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
	"""Impute missing values using medians learned from train data."""
	X_imputed = X.fillna(medians)
	return X_imputed
