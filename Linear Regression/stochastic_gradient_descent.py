import random

import numpy as np


class StochasticGradientDescent:
    """Implement stochastic gradient descent for linear regression."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        weight_difference_norm_between_iterations = 1

        cost = []

        while weight_difference_norm_between_iterations > 0.000001:
            i = random.randint(0, n_samples - 1)
            y_predicted = np.dot(X[i], self.weights)
            dw = np.dot(X[i].T, (y_predicted - y[i]))

            weight_difference_norm_between_iterations = np.linalg.norm(
                self.weights - (self.weights - self.learning_rate * dw))

            self.weights -= self.learning_rate * dw

            error = 0.5 * np.sum((y_predicted - y[i]) ** 2)
            cost.append(error)

        return cost

    def get_error(self, X: pd.DataFrame, y: pd.DataFrame):
        y_predicted = np.dot(X, self.weights)
        error = 0.5 * np.sum((y_predicted - y) ** 2)
        return error
