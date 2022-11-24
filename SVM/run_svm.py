from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from dual_svm import Dual_SVM
from collections import defaultdict

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
gamma_list = [0.01, 0.1, 0.5, 1, 5, 100]
weights = []

# make (-1, 1) labels
trainY[trainY == 0] = -1

testX = bank_note_test_dataframe.iloc[:, :-1].values
testY = bank_note_test_dataframe.iloc[:, -1].values

# make (-1, 1) labels
testY[testY == 0] = -1
# declare a dictionary to store the results
primal_svm_parameters = defaultdict(list)
primal_svm_train_test_error = defaultdict(list)

print("------ Primal SVM --------")
print('Using learning rate change with a:')
for C in tqdm(C_list):
    prim = Primal_SVM("lr_a", a, lr=lr, C=C)
    prim.fit(trainX, trainY, T)
    pred_train = prim.predict(trainX)
    pred_test = prim.predict(testX)

    print('C: ' + str(C))
    print('Train Error: ' + str(prim.evaluate(trainX, trainY)))
    print('Test Error: ' + str(prim.evaluate(testX, testY)))
    primal_svm_train_test_error['a'].append({'train': prim.evaluate(trainX, trainY), 'test': prim.evaluate(testX, testY)})
    weights.append(prim.w)
    primal_svm_parameters["a"].append(prim.w)
    print()

print('Using learning rate change with epoch:')
for C in tqdm(C_list):
    prim = Primal_SVM("lr_epoch", a, lr=lr, C=C)
    prim.fit(trainX, trainY)
    pred_train = prim.predict(trainX)
    pred_test = prim.predict(testX)

    print('C: ' + str(C))
    print('Train Error: ' + str(prim.evaluate(trainX, trainY)))
    print('Test Error: ' + str(prim.evaluate(testX, testY)))
    primal_svm_train_test_error['epoch'].append({'train': prim.evaluate(trainX, trainY), 'test': prim.evaluate(testX, testY)})
    weights.append(prim.w)
    primal_svm_parameters["epoch"].append(prim.w)
    print()

# difference between the two methods
print('Difference of weights between the two methods of learning rate schedule:')
for i in range(len(primal_svm_parameters["a"])):
    print('C: ' + str(C_list[i]))
    print('Difference: ' + str(np.linalg.norm(primal_svm_parameters["a"][i] - primal_svm_parameters["epoch"][i])))
print()

# difference between the two methods
print('Difference of train and test error between the two methods of learning rate schedule:')
for i in range(len(primal_svm_train_test_error["a"])):
    print('C: ' + str(C_list[i]))
    print('Difference: ' + str(primal_svm_train_test_error["a"][i]['train'] - primal_svm_train_test_error["epoch"][i]['train']))
    print('Difference: ' + str(primal_svm_train_test_error["a"][i]['test'] - primal_svm_train_test_error["epoch"][i]['test']))
print()

dual_svm_parameters = defaultdict(list)
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
    dual_svm_parameters["linear"].append(np.append(dual.w, dual.b))
    # print support vectors
    print("Number of support vectors: " + str(len(dual.support_vectors)))
    print()

# difference between the two methods
print('Difference of weights between the two methods of learning rate schedule:')
for i in range(len(primal_svm_parameters["a"])):
    print('C: ' + str(C_list[i]))
    print('a Difference: ' + str(np.linalg.norm(primal_svm_parameters["a"][i] - dual_svm_parameters["linear"][i])))
    print("epoch Difference: " + str(np.linalg.norm(primal_svm_parameters["epoch"][i] - dual_svm_parameters["linear"][i])))
print()


C_dict_for_500_873 = {}

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
        print("Weights and bias: ", dual.w, dual.b)
        # print support vectors
        print("Number of support vectors: " + str(len(dual.support_vectors)))

        if C == 500/873:
            C_dict_for_500_873[gamma] = dual.support_vectors
            
        print()

# print no of support vectors that are same for C = 500/873 and gamma = 0.1, and 0.5
print("Number of support vectors that are same for C = 500/873 and gamma = 0.1, and 0.5: " + str(len(set(C_dict_for_500_873[0.1]).intersection(set(C_dict_for_500_873[0.5])))))

# print no of support vectors that are same for C = 500/873 and gamma = 0.1, and 0.01
print("Number of support vectors that are same for C = 500/873 and gamma = 0.1, and 0.01: " + str(len(set(C_dict_for_500_873[0.1]).intersection(set(C_dict_for_500_873[0.01])))))