import sys
import math
import pandas as pd


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float('nan')


def std(values):
    values = list(values)
    n = len(values)
    if n < 2:
        return float('nan')
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1))


def minimum(values):
    values = list(values)
    if not values:
        return float('nan')
    m = values[0]
    for v in values[1:]:
        if v < m:
            m = v
    return float(m)


def maximum(values):
    values = list(values)
    if not values:
        return float('nan')
    m = values[0]
    for v in values[1:]:
        if v > m:
            m = v
    return float(m)


def percentile(values, p):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return float('nan')
    pos = p * (n - 1)
    lo, hi = int(pos), math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))


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
        return ("nan" if isinstance(x, float) and math.isnan(x) else f"{float(x):.6f}").rjust(cell_width)

    def trunc(s):
        return s[:max_col_display - 2] + ".." if len(s) > max_col_display else s

    rows = [
        ("Count",[len(num_df[c].dropna())              for c in columns]),
        ("Mean", [mean(num_df[c].dropna())             for c in columns]),
        ("Std",  [std(num_df[c].dropna())              for c in columns]),
        ("Min",  [minimum(num_df[c].dropna())              for c in columns]),
        ("25%",  [percentile(num_df[c].dropna(), 0.25) for c in columns]),
        ("50%",  [percentile(num_df[c].dropna(), 0.50) for c in columns]),
        ("75%",  [percentile(num_df[c].dropna(), 0.75) for c in columns]),
        ("Max",  [maximum(num_df[c].dropna())              for c in columns]),
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
