import pandas as pd

from Perceptron.average_perceptron import AveragePerceptron
from Perceptron.standard_perceptron import StandardPerceptron
from Perceptron.voted_perceptron import VotedPerceptron

bank_note_train_dataframe = pd.read_csv('bank-note/train.csv', header=None)
bank_note_test_dataframe = pd.read_csv('bank-note/test.csv', header=None)

bank_note_train_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
bank_note_test_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

standard_perceptron = StandardPerceptron(num_features=len(bank_note_train_dataframe.columns) - 1, learning_rate=0.1)

trainX = bank_note_train_dataframe.iloc[:, :-1].values
trainY = bank_note_train_dataframe.iloc[:, -1].values

# make (-1, 1) labels
trainY[trainY == 0] = -1

testX = bank_note_test_dataframe.iloc[:, :-1].values
testY = bank_note_test_dataframe.iloc[:, -1].values

# make (-1, 1) labels
testY[testY == 0] = -1

standard_perceptron.train(x=trainX, y=trainY, epochs=10)

print("------ Standard Perceptron -------")
print("Learned weight vector: {}".format(standard_perceptron.weights))
print("Error rate on test set: ", standard_perceptron.evaluate(x=testX, y=testY))

# voted perceptron

voted_perceptron = VotedPerceptron(num_features=len(bank_note_train_dataframe.columns) - 1, learning_rate=0.1)

voted_perceptron.train(x=trainX, y=trainY, epochs=10)

print("\n------ Voted Perceptron -------")
print("Learned weight vector: {}".format(voted_perceptron.voted_weights_array))
print("Error rate on test set: ", voted_perceptron.evaluate(x=testX, y=testY))
print("Learned count vector: {}".format(voted_perceptron.c_list))

# average perceptron

average_perceptron = AveragePerceptron(num_features=len(bank_note_train_dataframe.columns) - 1, learning_rate=0.1)

average_perceptron.train(x=trainX, y=trainY, epochs=10)

print("\n------ Average Perceptron -------")
print("Learned weight vector: {}".format(average_perceptron.weights))
print("Error rate on test set: ", average_perceptron.evaluate(x=testX, y=testY))


