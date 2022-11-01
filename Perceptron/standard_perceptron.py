import numpy as np


class StandardPerceptron:
    def __init__(self, num_features: int, learning_rate: float = 1):
        self.weights = np.zeros(num_features)
        self.num_features = num_features
        self.learning_rate = learning_rate

    def predict(self, x: np.ndarray):
        return np.sign(np.dot(x, self.weights))

    def update(self, x: np.ndarray, y: int):
        self.weights = self.weights + self.learning_rate * y * x

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        for _ in range(epochs):
            # shuffle the data
            idx = np.random.permutation(len(x))
            x = x[idx]
            y = y[idx]
            # update the weights
            for i in range(len(x)):
                if y[i] * np.dot(x[i], self.weights) <= 0:
                    self.update(x[i], y[i])

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        return np.mean(self.predict(x) != y)
