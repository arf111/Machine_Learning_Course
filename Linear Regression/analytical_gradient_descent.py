import numpy as np


class AnalyticalGradientDescent:
    """Analytical Solution of linear regression"""

    def __init__(self,  X, y):
        self.weights = np.dot(np.linalg.inv(np.dot(X.T, X)),
                              np.dot(X.T, y))  # compute weights using the analytical solution

    def get_error(self, X, y):
        y_predicted = np.dot(X, self.weights)
        error = 0.5 * np.sum((y_predicted - y) ** 2)
        return error
