import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
	"""Compute sigmoid probabilities from linear scores."""
	return 1.0 / (1.0 + np.exp(-z))


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	"""Compute binary cross-entropy loss with numeric stability."""
	eps = 1e-15
	y_prob = np.clip(y_prob, eps, 1 - eps)
	loss = -np.mean(
		y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)
	)
	return float(loss)


def train_binary_logreg_gd(
	X: np.ndarray,
	y: np.ndarray,
	learning_rate: float = 0.1,
	epochs: int = 1000,
) -> tuple[np.ndarray, float, list[float]]:
	"""Train a binary logistic regression model with gradient descent."""
	m, n = X.shape
	w = np.zeros(n, dtype=float)
	b = 0.0
	history: list[float] = []

	for _ in range(epochs):
		z = X @ w + b
		y_prob = sigmoid(z)
		loss = binary_log_loss(y, y_prob)
		history.append(loss)

		error = y_prob - y
		grad_w = (X.T @ error) / m
		grad_b = float(np.mean(error))

		w -= learning_rate * grad_w
		b -= learning_rate * grad_b

	return w, b, history
