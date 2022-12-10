import numpy as np

class LogisticRegression:
    "Implement Logistic Regression using stochastic gradient descent and Maximum Likelihood Estimation"
    def __init__(self, learning_rate=0.01, iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        
    def fit(self, variance, initial_gamma, d_val, estimation_type: str, X: np.ndarray, Y: np.ndarray, n_samples=872):
        "Fit the model to the data"
        n_samples, n_features = X.shape
        loss = []
        # init parameters
        self.weights = np.array([[0., 0., 0., 0., 0.]])
        self.bias = 0

        for epoch in range(self.iterations):
            learning_rate = self.update_learning_rate(initial_gamma, d_val, epoch)

            [y, data] = self.shuffle_data(Y, X) # shuffle the data

            for i in range(0, n_samples):
                self.weights[0, -1] = 0.

                if estimation_type == "MAP":
                    self.weights -= learning_rate * self.calculate_logistic_map_gradient(y[i], self.weights, data[i], variance, n_samples)
                    l = self.calculate_map_loss(y[i], self.weights, data[i], variance, n_samples)
                else:
                    self.weights -= learning_rate * self.calculate_logistic_mle_gradient(y[i], self.weights, data[i], n_samples)
                    l = self.calculate_mle_loss(y[i], self.weights, data[i], n_samples)

                loss.append(l)

    def shuffle_data(self, y, data):
        "Shuffle the data"
        idx = np.random.permutation(len(y))
        return [y[idx], data[idx]]

    def sigmoid(self, x):
        "Calculate the sigmoid function"
        return 1 / (1 + np.exp(-x))
    
    # Maximum A Posteriori Estimation
    def calculate_logistic_map_gradient(self, y, weights, data, variance, n):
        "Calculate the gradient for MAP"
        loss_grad = self.calculate_logistic_mle_gradient(y, weights, data, n)
        regularization_grad = weights/variance
        return loss_grad + regularization_grad

    def calculate_map_loss(self, y, weights, data, variance, n):
        "Calculate the loss for MAP"
        z = np.dot(-(data * weights.T).T, y)
        loss = np.log(1 + np.exp(z))
        regularization = np.dot(weights, weights.T) / variance
        return (n * loss + regularization)[0, 0]

    # Maximum Likelihood Estimation
    def calculate_logistic_mle_gradient(self, y, weights, data, n):
        "Calculate the gradient for MLE"
        z = np.dot(data * weights.T, y)
        sigmoid_res = self.sigmoid(z[0, 0])
        return - n * np.dot(y.T, data) * sigmoid_res

    def calculate_mle_loss(self, y, weights, data, n):
        "Calculate the loss for MLE"
        z = np.dot(-(data * weights.T).T, y)
        return (n * np.log(1 + np.exp(z)))[0, 0]

    def update_learning_rate(self, gamma, d, epoch):
        "Update the learning rate"
        return gamma / (1.0 + epoch * (gamma / d))

    def get_smallest_hyperparameter(self, variance, estimation_type, train_X, train_Y):
        "Find the smallest hyperparameter"
        gammas = [1, 0.1, 0.5, 0.01]
        ds_values = [1, 0.1, 0.05, 0.01, 0.005]

        smallest_values = [0, 100.0, 100.0]

        for gamma in gammas:
            for d_val in ds_values:
                smallest_values = self.get_smallest_error(variance, gamma, d_val, smallest_values, estimation_type, train_X, train_Y)
    
        return smallest_values
    
    def get_smallest_error(self, variance, gamma, d_val, smallest_values, estimation_type, train_X, train_Y):
        "Find the smallest error"
        self.fit(variance = variance, initial_gamma=gamma, d_val=d_val, estimation_type=estimation_type, X=train_X, Y=train_Y)

        predictions = np.sign(train_X.dot(self.weights.T))
        error = self.calculate_error(train_Y, predictions)
        # print("GAMMA:", gamma, " D:", d_val, " ERROR:", error)

        if error < smallest_values[2]: 
            smallest_values = [gamma, d_val, error]
        return smallest_values
    
    def calculate_error(self, actual, predicted):
        "Calculate the error"
        return 1 - np.count_nonzero(np.multiply(actual, predicted) == 1) / len(actual)
    
    def predict(self, X, Y):
        "Predict the output"
        linear_model = np.dot(X, self.weights.T)
        y_predicted = np.sign(linear_model)

        return self.calculate_error(Y, y_predicted)