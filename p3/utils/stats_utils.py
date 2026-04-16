import os
import sys

import numpy as np

# Reuse manual stats from P1 to avoid duplicated implementations.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from p1.describe import mean, percentile, std  # noqa: E402


def median(values) -> float:
	"""Median helper reused in P3, built on top of P1 percentile implementation."""
	return percentile(values, 0.5)


def array_mean(arr: np.ndarray) -> float:
	"""Mean for numpy arrays without using np.mean."""
	size = arr.size
	if size == 0:
		return float("nan")
	return float(np.sum(arr) / size)
