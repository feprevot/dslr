import pandas as pd


def split_features_target(
	df: pd.DataFrame,
	target_col: str = "Hogwarts House",
	exclude_cols: tuple[str, ...] = ("Index",),
) -> tuple[pd.DataFrame, pd.Series]:
	"""Split dataset into numeric features X and target labels y."""
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
	"""Fit train medians and impute missing values in train features."""
	medians = X.median(numeric_only=True)
	X_imputed = X.fillna(medians)
	return X_imputed, medians


def impute_with_train_medians(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
	"""Impute missing values using medians learned from train data."""
	X_imputed = X.fillna(medians)
	return X_imputed


def fit_standardizer(X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
	"""Fit z-score standardization parameters on train features."""
	means = X.mean(numeric_only=True)
	stds = X.std(numeric_only=True)
	stds = stds.replace(0, 1.0)
	return means, stds


def apply_standardizer(
	X: pd.DataFrame,
	means: pd.Series,
	stds: pd.Series,
) -> pd.DataFrame:
	"""Apply z-score standardization using train means/stds."""
	X_scaled = (X - means) / stds
	return X_scaled
