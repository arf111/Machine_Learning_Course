import numpy as np


class AnalyticalGradientDescent:
    """Analytical Solution of linear regression"""

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.weights = np.dot(np.linalg.inv(np.dot(X.T, X)),
                              np.dot(X.T, y))  # compute weights using the analytical solution
        self.bias = 0

        error = 0

        for idx, x_i in enumerate(X):  # for each sample
            y_predicted = np.dot(x_i, self.weights) + self.bias  # predict
            error += np.mean(np.abs(y_predicted - y))  # compute error

        return error
