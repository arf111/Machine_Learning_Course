import numpy as np
import pandas as pd
from tqdm import tqdm
from neural_network import NeuralNetwork
from nn_pytorch import NeuralNetworkPyTorch
import torch

bank_note_train_dataframe = pd.read_csv('./Neural Networks/bank-note/train.csv', header=None)
bank_note_test_dataframe = pd.read_csv('./Neural Networks/bank-note/test.csv', header=None)

bank_note_train_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
bank_note_test_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

trainX = bank_note_train_dataframe.iloc[:, :-1].values
trainY = bank_note_train_dataframe.iloc[:, -1].values

lr = 0.001
d = 0.01
T = 100
weights = []


testX = bank_note_test_dataframe.iloc[:, :-1].values
testY = bank_note_test_dataframe.iloc[:, -1].values


# (2b)
no_of_nodes = [5, 10, 25, 50, 100]

print("------ Neural Network with random initialization--------")
for no_of_node in tqdm(no_of_nodes):
    print("Number of nodes: " + str(no_of_node))
    nn_model = NeuralNetwork(trainX.shape[1], no_of_node, "random", d, weights, lr)
    nn_model.train(trainX, trainY, T)

    print('Train Error: ' + str(nn_model.evaluate(trainX, trainY)))
    print('Test Error: ' + str(nn_model.evaluate(testX, testY)))
    print()

# (2c)
print("------ Neural Network with 0 initialization--------")
for no_of_node in tqdm(no_of_nodes):
    print("Number of nodes: " + str(no_of_node))
    nn_model = NeuralNetwork(trainX.shape[1], no_of_node, "zeros", d, weights, lr)
    nn_model.train(trainX, trainY, T)

    print('Train Error: ' + str(nn_model.evaluate(trainX, trainY)))
    print('Test Error: ' + str(nn_model.evaluate(testX, testY)))
    print()

# 2(e) Bonus
print("------ Neural Network with Xavier initialization--------")
no_of_layers = [3, 5, 9]

for no_of_layer in tqdm(no_of_layers):
    for no_of_node in no_of_nodes:
        print("Number of nodes: " + str(no_of_node))
        nn_model = NeuralNetworkPyTorch(trainX.shape[1], no_of_layer, "tanh", no_of_node, lr)
        torch_trainX = torch.from_numpy(trainX).float()
        torch_trainY = torch.from_numpy(trainY).float()
        torch_textX = torch.from_numpy(testX).float()
        torch_textY = torch.from_numpy(testY).float()
        nn_model.train(torch_trainX, torch_trainY, T)

        print('Train Error: ' + str(nn_model.evaluate(torch_trainX, torch_trainY).item()))
        print('Test Error: ' + str(nn_model.evaluate(torch_textX, torch_textY).item()))
        print()

print("------ Neural Network with He initialization--------")

for no_of_layer in tqdm(no_of_layers):
    for no_of_node in no_of_nodes:
        print("Number of nodes: " + str(no_of_node))
        # NeuralNetworkPytorch(input_size, no_of_layers, act_func, hidden_size, lr = 0.001):
        nn_model = NeuralNetworkPyTorch(trainX.shape[1], no_of_layer, "relu", no_of_node, lr)
        torch_trainX = torch.from_numpy(trainX).float()
        torch_trainY = torch.from_numpy(trainY).float()
        torch_textX = torch.from_numpy(testX).float()
        torch_textY = torch.from_numpy(testY).float()
        nn_model.train(torch_trainX, torch_trainY, T)

        print('Train Error: ' + str(nn_model.evaluate(torch_trainX, torch_trainY).item()))
        print('Test Error: ' + str(nn_model.evaluate(torch_textX, torch_textY).item()))
        print()