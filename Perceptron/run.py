import pandas as pd

from Perceptron.standard_perceptron import StandardPerceptron

bank_note_train_dataframe = pd.read_csv('bank-note/train.csv', header=None)
bank_note_test_dataframe = pd.read_csv('bank-note/test.csv', header=None)

bank_note_train_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
bank_note_test_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

standard_perceptron = StandardPerceptron(num_features=len(bank_note_train_dataframe.columns) - 1)

trainX = bank_note_train_dataframe.iloc[:, :-1].values
trainY = bank_note_train_dataframe.iloc[:, -1].values

testX = bank_note_test_dataframe.iloc[:, :-1].values
testY = bank_note_test_dataframe.iloc[:, -1].values

standard_perceptron.train(x=trainX, y=trainY, epochs=10)

print("Learned weight vector: {}".format(standard_perceptron.weights))
print("Error rate on test set: ", standard_perceptron.evaluate(x=testX, y=testY))

