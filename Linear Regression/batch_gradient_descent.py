import numpy as np


class BatchGradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # compute batch gradient descent until error is below than 0.001
        n_samples, n_features = X.shape  # get number of samples and features
        self.weights = np.zeros(n_features)  # initialize weights
        self.bias = 0  # initialize bias

        weight_difference_norm_between_iterations = 1

        cost = []

        while weight_difference_norm_between_iterations > 0.000001:
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = np.dot(X.T, (y_predicted - y))

            weight_difference_norm_between_iterations = np.linalg.norm(
                self.weights - (self.weights - self.learning_rate * dw))

            self.weights -= self.learning_rate * dw

            error = 0.5 * np.sum((y_predicted - y) ** 2)
            cost.append(error)

        return cost

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated
