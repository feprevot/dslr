import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from utils import array_mean


def sigmoid(z: np.ndarray) -> np.ndarray:
	"""Convert raw linear scores into probabilities in [0, 1]."""
	return 1.0 / (1.0 + np.exp(-z))


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	"""Measure prediction error for binary classification."""
	eps = 1e-15
	y_prob = np.clip(y_prob, eps, 1 - eps)
	term = y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)
	loss = -array_mean(term)
	return float(loss)


def train_binary_logreg_gd(
	X: np.ndarray,
	y: np.ndarray,
	learning_rate: float = 0.1,
	epochs: int = 1000,
) -> tuple[np.ndarray, float, list[float]]:
	"""Train binary logistic regression with gradient descent."""
	m, n = X.shape
	w = np.zeros(n, dtype=float)
	b = 0.0
	history: list[float] = []

	for _ in range(epochs):
		z = X @ w + b              # score for each student
		y_prob = sigmoid(z)        # convert score to probability [0, 1]
		loss = binary_log_loss(y, y_prob)  # measure error on probabilities
		history.append(loss)       # saved for final display

		error = y_prob - y         # gap between prediction and truth → used to adjust w and b
		grad_w = (X.T @ error) / m
		grad_b = array_mean(error)

		w -= learning_rate * grad_w
		b -= learning_rate * grad_b

	return w, b, history
