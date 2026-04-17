import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import mean, std, minimum, maximum, percentile, value_range

import pandas as pd

def describe(dataset, max_features=None):
    df = pd.read_csv(dataset)
    num_df = df.select_dtypes(include=["number"]).drop(columns=["Index"], errors="ignore")

    if num_df.shape[1] == 0:
        print("No numeric columns found")
        return

    columns = list(num_df.columns)
    if max_features is not None:
        columns = columns[:int(max_features)]

    max_col_display = 16
    cell_width = max(13, min(max_col_display, max(len(c) for c in columns)) + 2)

    def fmt(x):
        return ("nan" if isinstance(x, float) and x != x else f"{float(x):.6f}").rjust(cell_width)

    def trunc(s):
        return s[:max_col_display - 2] + ".." if len(s) > max_col_display else s

    total_rows = len(df)

    rows = [
        ("Missing",  [total_rows - len(num_df[c].dropna()) for c in columns]),
        ("Missing%", [((total_rows - len(num_df[c].dropna())) / total_rows * 100.0) if total_rows else float('nan') for c in columns]),
        ("Count",[len(num_df[c].dropna())              for c in columns]),
        ("Mean", [mean(num_df[c].dropna())             for c in columns]),
        ("Std",  [std(num_df[c].dropna())              for c in columns]),
        ("Min",  [minimum(num_df[c].dropna())          for c in columns]),
        ("25%",  [percentile(num_df[c].dropna(), 0.25) for c in columns]),
        ("50%",  [percentile(num_df[c].dropna(), 0.50) for c in columns]),
        ("75%",  [percentile(num_df[c].dropna(), 0.75) for c in columns]),
        ("Max",  [maximum(num_df[c].dropna())          for c in columns]),
        ("Range",[value_range(num_df[c].dropna())      for c in columns]),
    ]

    print("".ljust(8) + " " + " ".join(trunc(c).rjust(cell_width) for c in columns))
    for label, vals in rows:
        print(label.ljust(8) + " " + " ".join(fmt(v) for v in vals))


def main():
    if len(sys.argv) < 2:
        print("Usage: python describe.py <dataset_path>")
        return
    max_features = sys.argv[2] if len(sys.argv) >= 3 else None
    describe(sys.argv[1], max_features=max_features)


if __name__ == "__main__":
    main()
