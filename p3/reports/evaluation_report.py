import pandas as pd


def evaluate_with_truth(predictions_path: str, truth_csv: str) -> dict:
	"""Build evaluation metrics using Index + Hogwarts House from truth and predictions."""
	pred_df = pd.read_csv(predictions_path)
	truth_df = pd.read_csv(truth_csv)

	required = {"Index", "Hogwarts House"}
	if not required.issubset(pred_df.columns):
		raise ValueError(f"Predictions CSV must contain columns: {required}")
	if not required.issubset(truth_df.columns):
		raise ValueError(f"Truth CSV must contain columns: {required}")

	merged = truth_df[["Index", "Hogwarts House"]].merge(
		pred_df[["Index", "Hogwarts House"]],
		on="Index",
		how="inner",
		suffixes=("_true", "_pred"),
	)

	if merged.empty:
		raise ValueError("No matching Index values between truth and predictions")

	correct = merged["Hogwarts House_true"] == merged["Hogwarts House_pred"]
	accuracy = float(correct.mean())
	correct_count = int(correct.sum())
	row_count = int(len(merged))

	houses = sorted(set(merged["Hogwarts House_true"]).union(set(merged["Hogwarts House_pred"])))
	cm = pd.crosstab(
		merged["Hogwarts House_true"],
		merged["Hogwarts House_pred"],
		dropna=False,
	).reindex(index=houses, columns=houses, fill_value=0)

	per_house: list[dict] = []
	for house in houses:
		tp = int(cm.loc[house, house])
		fp = int(cm[house].sum() - tp)
		fn = int(cm.loc[house].sum() - tp)
		support = int(cm.loc[house].sum())

		precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
		recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
		f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

		per_house.append({
			"house": house,
			"precision": precision,
			"recall": recall,
			"f1": f1,
			"support": support,
		})

	return {
		"accuracy": accuracy,
		"correct": correct_count,
		"rows": row_count,
		"per_house": per_house,
		"confusion_matrix": cm,
	}


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
