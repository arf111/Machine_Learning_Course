import numpy as np
from tqdm import tqdm
from logistic_regression import LogisticRegression
import pandas as pd


def import_data(path, num_examples):
    data = np.empty((num_examples, 4), dtype="float")
    y = np.empty((num_examples, 1), dtype="float")

    with open(path, 'r') as f:
        i = 0
        for line in f:
            example = []
            terms = line.strip().split(',')
            for j in range(len(terms)):
                if j == 4:
                    y[i] = 2 * float(terms[j]) - 1
                else:
                    example.append(float(terms[j]))
            data[i] = example
            i += 1

    bias = np.tile([1.], (num_examples, 1))
    data = np.append(np.asmatrix(data), bias, axis=1)

    return [data, np.asmatrix(y)]

[trainX, trainY] = import_data("./Neural Networks/bank-note/train.csv", 872)
[testX, testY] = import_data("./Neural Networks/bank-note/test.csv", 500)

lr = 0.001
d = 0.01
T = 100
weights = None

variance = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
linear_regression = LogisticRegression()

[gamma, d, _] = linear_regression.get_smallest_hyperparameter(variance=1, estimation_type="MAP", train_X=trainX, train_Y=trainY)
print("GAMMA:", gamma, "D:", d)
errors = []

print("------ Logistic Regression with MAP--------")
for var in tqdm(variance):
    print("Variance: " + str(var))
    linear_regression.fit(var, gamma, d, "MAP", trainX, trainY)
    # predict(self, X, Y):
    print('Train Error: ' + str(linear_regression.predict(trainX, trainY)))
    print('Test Error: ' + str(linear_regression.predict(testX, testY)))

linear_regression = LogisticRegression()

[gamma, d, _] = linear_regression.get_smallest_hyperparameter(variance=1, estimation_type="MLE", train_X=trainX, train_Y=trainY)
print("GAMMA:", gamma, "D:", d)
errors = []

print("------ Logistic Regression with MLE--------")
for var in tqdm(variance):
    print("Variance: " + str(var))
    linear_regression.fit(var, gamma, d, "MLE", trainX, trainY)
    # predict(self, X, Y):
    print('Train Error: ' + str(linear_regression.predict(trainX, trainY)))
    print('Test Error: ' + str(linear_regression.predict(testX, testY)))