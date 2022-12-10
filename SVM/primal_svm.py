import numpy as np


class Primal_SVM:
    def __init__(self, lr_type, a, bias=0, lr=0.001, C=1.0):
        self.C = C  # regularization parameter
        self.w = None
        self.a = a  #
        self.bias = bias
        if lr_type == "lr_a":
            self.lr_inc = self.learning_rate_increase_on_a
        elif lr_type == "lr_epoch":
            self.lr_inc = self.learning_rate_increase_on_epoch

        self.learning_rate = lr

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=100):
        # add bias
        if self.bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        else:
            X = np.hstack((X, np.zeros((X.shape[0], 1))))

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # initialize weights

        for epoch in range(epochs):
            lr_epoch = self.lr_inc(epoch)
            # shuffle the data
            idx = np.random.permutation(n_samples)
            X = X[idx]
            y = y[idx]

            for i in range(n_samples):
                xi = X[i]
                yi = y[i]
                if yi * np.dot(xi, self.w) <= 1:
                    dw = np.append(self.w[:len(self.w) - 1], 0) - self.C * n_samples * yi * xi
                    self.w = self.w - lr_epoch * dw
                else:
                    self.w[:len(self.w) - 1] = (1 - lr_epoch) * self.w[:len(self.w) - 1]

    def learning_rate_increase_on_epoch(self, epoch):
        return self.learning_rate / (1 + epoch)

    def learning_rate_increase_on_a(self, epoch):
        return self.learning_rate / (1 + (self.learning_rate * epoch) / self.a)

    def predict(self, X: np.ndarray):
        if self.bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        else:
            X = np.hstack((X, np.zeros((X.shape[0], 1))))

        return np.sign(np.dot(X, self.w))

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        return np.mean(self.predict(x) != y)
