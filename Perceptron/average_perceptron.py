import numpy as np


class AveragePerceptron:
    def __init__(self, num_features, learning_rate=1):
        self.weights = np.zeros(num_features)
        self.a = np.zeros(num_features)
        self.learning_rate = learning_rate

    def train(self, x, y, epochs):
        for _ in range(epochs):
            idx = np.random.permutation(len(x))
            x = x[idx]
            y = y[idx]
            for i in range(len(x)):
                if y[i] * np.dot(x[i], self.weights) <= 0:
                    self.weights = self.weights + self.learning_rate * y[i] * x[i]
                self.a = self.a + self.weights

    def predict(self, x):
        return np.sign(np.dot(x, self.a))

    def evaluate(self, x, y):
        return np.mean(self.predict(x) != y)

