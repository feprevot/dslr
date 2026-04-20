import pandas as pd
from utils import mean, median, std


SELECTED_FEATURES = (
	"Astronomy",
	"Herbology",
	"Divination",
	"Muggle Studies",
	"Ancient Runes",
	"History of Magic",
	"Transfiguration",
	"Potions",
	"Charms",
	"Flying",
    # "Arithmancy", # homogenous repartition between houses, don't keep it
    # "Defense Against the Dark Arts", # same as Astronomy, don't keep it
    # "Care of Magical Creatures", # homogenous repartition between houses, don't keep it
)


def split_features_target(
	df: pd.DataFrame,
	target_col: str = "Hogwarts House",
	exclude_cols: tuple[str, ...] = ("Index",),
	feature_cols: tuple[str, ...] | None = SELECTED_FEATURES,
) -> tuple[pd.DataFrame, pd.Series]:
	"""Split dataset into numeric features X and target labels y."""
	if target_col not in df.columns:
		raise ValueError(f"Missing target column: {target_col}")

	y = df[target_col].copy()

	numeric_cols = df.select_dtypes(include="number").columns.tolist()

	if feature_cols is None:
		feature_cols_final = [col for col in numeric_cols if col not in exclude_cols]
	else:
		feature_cols_final = [
			col
			for col in feature_cols
			if col in df.columns and col in numeric_cols and col not in exclude_cols
		]

	if not feature_cols_final:
		raise ValueError("No numeric feature columns found after exclusions")

	X = df[feature_cols_final].copy()
	return X, y


def impute_median_train(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
	"""Fit train medians and impute missing values in train features."""
	median_map: dict[str, float] = {}
	for col in X.columns:
		values = [v for v in X[col] if pd.notna(v)]
		median_map[col] = median(values)
	medians = pd.Series(median_map)
	X_imputed = X.fillna(medians)
	return X_imputed, medians


def impute_with_train_medians(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
	"""Impute missing values using medians learned from train data."""
	X_imputed = X.fillna(medians)
	return X_imputed


def fit_standardizer(X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
	"""Learn scaling params from train data: compute mean and std per feature.
	These are saved in model.json and reused at predict time to apply x' = (x - mean) / std."""
	mean_map: dict[str, float] = {}
	std_map: dict[str, float] = {}
	for col in X.columns:
		values = [v for v in X[col] if pd.notna(v)]
		col_mean = mean(values)
		col_std = std(values)
		if col_std == 0 or col_std != col_std:  # zero or nan
			col_std = 1.0
		mean_map[col] = col_mean
		std_map[col] = col_std
	means = pd.Series(mean_map)
	stds = pd.Series(std_map)
	return means, stds


def apply_standardizer(
	X: pd.DataFrame,
	means: pd.Series,
	stds: pd.Series,
) -> pd.DataFrame:
	"""Apply z-score standardization using train means/stds."""
	X_scaled = (X - means) / stds
	return X_scaled
