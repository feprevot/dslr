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
