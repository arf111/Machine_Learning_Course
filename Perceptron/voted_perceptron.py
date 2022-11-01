import numpy as np


class VotedPerceptron:
    def __init__(self, num_features: int, learning_rate: float = 1):
        self.weights = np.zeros(num_features, dtype=float)
        self.weights_list = [self.weights]
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.c = 0
        self.c_list = [self.c]
        self.voted_weights = [self.weights]
        self.voted_weights_array = None
        self.c_array = None

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        for _ in range(epochs):
            # update the weights
            for i in range(len(x)):
                if y[i] * np.dot(x[i], self.weights) <= 0:
                    self.weights = self.weights + self.learning_rate * y[i] * x[i]
                    self.weights_list.append(self.weights)
                    self.c = 1
                else:
                    self.c += 1
            self.voted_weights.append(self.weights)
            self.c_list.append(self.c)

        self.c_array = np.array(self.c_list)
        self.voted_weights_array = np.array(self.voted_weights)

    def predict(self, x: np.ndarray):
        return np.sign(np.dot(self.c_array, np.sign(np.dot(x, self.voted_weights_array.T)).T))

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        return np.mean(self.predict(x) != y)
