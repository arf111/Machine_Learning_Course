import numpy as np
import pandas as pd
from tqdm import tqdm
from dual_svm import Dual_SVM

from primal_svm import Primal_SVM

bank_note_train_dataframe = pd.read_csv('./SVM/bank_note/train.csv', header=None)
bank_note_test_dataframe = pd.read_csv('./SVM/bank_note/test.csv', header=None)

bank_note_train_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
bank_note_test_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

trainX = bank_note_train_dataframe.iloc[:, :-1].values
trainY = bank_note_train_dataframe.iloc[:, -1].values

lr = 0.001
a = 0.001
T = 100
C_list = [100/873, 500/873, 700/873]
gamma_list = [0.1, 0.5, 1, 5, 100]
weights = []

# make (-1, 1) labels
trainY[trainY == 0] = -1

testX = bank_note_test_dataframe.iloc[:, :-1].values
testY = bank_note_test_dataframe.iloc[:, -1].values

# make (-1, 1) labels
testY[testY == 0] = -1

print("------ Primal SVM --------")
print('Using learning rate change with a:')
for C in tqdm(C_list):
    prim = Primal_SVM("lr_a", a, lr, C)
    prim.fit(trainX, trainY, T)
    pred_train = prim.predict(trainX)
    pred_test = prim.predict(testX)

    print('C: ' + str(C))
    print('Train Error: ' + str(prim.evaluate(trainX, trainY)))
    print('Test Error: ' + str(prim.evaluate(testX, testY)))
    weights.append(prim.w)
    print()

print('Using learning rate change with epoch:')
for C in tqdm(C_list):
    prim = Primal_SVM("lr_epoch", a, lr, C)
    prim.fit(trainX, trainY)
    pred_train = prim.predict(trainX)
    pred_test = prim.predict(testX)

    print('C: ' + str(C))
    print('Train Error: ' + str(prim.evaluate(trainX, trainY)))
    print('Test Error: ' + str(prim.evaluate(testX, testY)))
    weights.append(prim.w)
    print()

print("------ Dual SVM --------")
for C in tqdm(C_list):
    dual = Dual_SVM("linear", C=C)
    dual.fit(trainX, trainY)
    pred_train = dual.predict(trainX)
    pred_test = dual.predict(testX)
    print('C: ' + str(C))
    print('Train Error: ' + str(dual.evaluate(trainX, trainY)))
    print('Test Error: ' + str(dual.evaluate(testX, testY)))
    weights.append(np.append(dual.w, dual.b))
    # print support vectors
    print("Number of support vectors: " + str(len(dual.support_vectors)))
    print()


for gamma in tqdm(gamma_list):
    for C in C_list:
        dual = Dual_SVM("gaussian", C=C, gamma=gamma)
        dual.fit(trainX, trainY)
        pred_train = dual.predict(trainX)
        pred_test = dual.predict(testX)
        print('C: ' + str(C) + ', gamma: ' + str(gamma))
        print('Train Error: ' + str(dual.evaluate(trainX, trainY)))
        print('Test Error: ' + str(dual.evaluate(testX, testY)))
        weights.append(np.append(dual.w, dual.b))
        # print support vectors
        print("Number of support vectors: " + str(len(dual.support_vectors)))

        if C == 500/873:
            # print overlapping support vectors
            print("Number of overlapping support vectors: " + str(len(dual.overlapping_support_vectors)))
            
        print()