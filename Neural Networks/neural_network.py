from typing import List
import numpy as np


class NeuralNetwork:
    """Build Neural Network with 2 hidden layers using mean squared loss as error function."""

    def __init__(self, num_features: int, no_of_node: int, weights_initialization: str, d: float, weights: List[np.ndarray], learning_rate: float):
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.no_of_nodes = no_of_node
        self.d = d
        if weights_initialization == "random":
            self.weights = [np.random.randn(num_features + 1, no_of_node), np.random.randn(no_of_node + 1, no_of_node), 
                            np.random.randn(no_of_node + 1, 1)]
        elif weights_initialization == "zeros":
            self.weights = [np.zeros((num_features + 1, no_of_node)), np.zeros((no_of_node + 1, no_of_node)), 
                            np.zeros((no_of_node + 1, 1))]
        else:
            self.weights = weights


        self.lr_inc = self.learning_rate_increase_on_a

    def sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        for epoch in range(epochs):
            lr_a = self.lr_inc(epoch)

            idx = np.random.permutation(len(x))  # shuffle the data
            train_x = x[idx]
            train_y = y[idx]
            for i in range(len(x)):
                # forward pass
                activation_outputs = [train_x[i]]  # input layer
                z = [] # z = wx for each layer except input layer
                for j in range(len(self.weights)):  # 2 hidden layers + output layer = 3 layers in total
                    input_x = np.hstack((activation_outputs[j], np.ones(1)))  # add bias, shape = (no_of_nodes, )
                    z.append(np.dot(input_x, self.weights[j]))  # z = wx for each layer except input layer, shape = (1, no_of_node - 1)
                    activation_outputs.append(self.sigmoid(z[j]))  # a[0] is x[i] (input layer) and a[1] is the output of the first hidden layer and so on

                # backward pass
                delta = [activation_outputs[-1] - train_y[i]]  # delta for output layer is a - y
                for j in range(len(self.weights) - 1, 0, -1):  # 2 hidden layers and 1 output layer so we start from 1 and go to 0 (not including 0) with step -1 (decreasing)
                    delta.append(np.dot(delta[-1], self.weights[j][:-1, :].T) * self.sigmoid_derivative(activation_outputs[j+1]))  # delta for hidden layers is delta * weights * sigmoid_derivative
                delta.reverse()  # reverse the list to match the order of the weights

                for j in range(len(self.weights)):
                    input_x = np.hstack((activation_outputs[j], np.ones(1)))  # add bias, shape = (no_of_nodes, )
                    self.weights[j] -= lr_a * np.dot(input_x[:, np.newaxis], delta[j][np.newaxis, :])  # update weights
            # loss
            loss = 0
            for i in range(len(x)):
                a = [x[i]]
                for j in range(len(self.weights)):
                    input_x = np.hstack((a[j], np.ones(1)))
                    a.append(self.sigmoid(np.dot(input_x, self.weights[j])))
                loss += (a[-1] - y[i]) ** 2
            loss /= len(x)
            print(f"Epoch: {epoch}, Loss: {loss}")


    def learning_rate_increase_on_a(self, epoch):
        #  lr_0/(1+(lr_0/d)*t)
        return self.learning_rate / (1 + (self.learning_rate / self.d) * epoch)
        
    def predict(self, x: np.ndarray):
        for i in range(len(x)):
            a = [x[i]]  # input layer
            for j in range(len(self.weights)):  # 2 hidden layers + output layer = 3 layers in total
                # stack bias in each layer
                input_x = np.hstack((a[j], np.ones(1)))  # add bias, shape = (no_of_nodes, )
                a.append(self.sigmoid(np.dot(input_x, self.weights[j])))  # a[0] is x[i] (input layer) and a[1] is the output of the first hidden layer and so on
        if a[-1] >= 0.5:
            return 1
        else:
            return 0

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        return np.mean(self.predict(x) != y)
