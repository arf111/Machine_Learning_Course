import random

import numpy as np
import pandas as pd


class StochasticGradientDescent:
    """Implement stochastic gradient descent for linear regression."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape  # get number of samples and features
        self.weights = np.zeros(n_features)  # initialize weights
        weight_difference_norm_between_iterations = 1

        cost = []

        while weight_difference_norm_between_iterations > 0.000001:
            random_index = random.randint(0, n_samples - 1)
            y_predicted = np.dot(X, self.weights)
            error = 0.5 * np.sum((y_predicted - y) ** 2)

            y_random_predicted = np.dot(X[random_index], self.weights)
            dw = np.dot(X[random_index].T, (y_random_predicted - y[random_index]))

            weight_difference_norm_between_iterations = np.linalg.norm(
                self.weights - (self.weights - self.learning_rate * dw))

            self.weights -= self.learning_rate * dw

            cost.append(error)

        return cost

    def get_error(self, X: pd.DataFrame, y: pd.DataFrame):
        y_predicted = np.dot(X, self.weights)
        error = 0.5 * np.sum((y_predicted - y) ** 2)
        return error
