import pandas as pd


def print_prediction_summary(
	test_csv: str,
	n_rows: int,
	n_cols: int,
	model_file: str,
	n_features: int,
	missing_before: int,
	missing_after: int,
	pred_counts: pd.Series,
	n_predictions: int,
) -> None:
	"""Print concise prediction run information."""
	print(f"Loaded CSV: {test_csv}")
	print(f"Shape: {n_rows} rows, {n_cols} columns")
	print(f"Loaded model: {model_file}")
	print(f"Selected features: {n_features}")
	print(f"Missing values (selected features) before imputation: {missing_before}")
	print(f"Missing values (selected features) after imputation:  {missing_after}")
	print("Prediction distribution:")
	for house, count in pred_counts.items():
		print(f"  {house}: {int(count)}")
	print(f"Predictions written to houses.csv ({n_predictions} rows)")


def print_evaluation_report(report: dict) -> None:
	"""Print a concise, readable evaluation report."""
	print("=" * 66)
	print("EVALUATION REPORT".center(66))
	print("=" * 66)
	print(f"Overall accuracy: {report['accuracy'] * 100:.2f}% ({report['correct']}/{report['rows']})")
	print("-" * 66)
	print("Per-house metrics:")
	for item in report["per_house"]:
		print(
			f"  {item['house']:<10} "
			f"P={item['precision']:.4f}  "
			f"R={item['recall']:.4f}  "
			f"F1={item['f1']:.4f}  "
			f"S={item['support']}"
		)
	print("-" * 66)
	print("Confusion matrix (rows=true, cols=pred):")
	cm = report["confusion_matrix"]
	print(cm.to_string())
