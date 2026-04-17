import numpy as np

def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float('nan')

def std(values):
    values = list(values)
    n = len(values)
    if n < 2:
        return float('nan')
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / (n - 1)) ** 0.5

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
    lo, hi = int(pos), int(pos) + (1 if pos != int(pos) else 0)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))


def value_range(values):
    values = list(values)
    if not values:
        return float('nan')
    return maximum(values) - minimum(values)


def median(values) -> float:
	"""Median helper reused in P3, built on top of P1 percentile implementation."""
	return percentile(values, 0.5)


def array_mean(arr: np.ndarray) -> float:
	"""Mean for numpy arrays without using np.mean."""
	size = arr.size
	if size == 0:
		return float("nan")
	return float(np.sum(arr) / size)
	