import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from bagged_tree import BaggedTree

# 3

data_path = 'credit_card/data.csv'
credit_card_colnames = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4',
                        'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                        'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
                        'y']  # y: default payment next month

data = pd.read_csv(data_path, header=None, names=credit_card_colnames)

continuous_columns = ["LIMIT_BAL", "AGE", 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                      'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
thresholds = data[continuous_columns].median()


def preprocessing(df):
    for column in continuous_columns:
        df.loc[df[column] <= thresholds[column], column] = 0
        df.loc[df[column] > thresholds[column], column] = 1
        df[column] = df[column].map({0: "low", 1: "high"})

    return df


data = preprocessing(data)
shuffle = np.random.choice(np.arange(30000), size=30000, replace=False)
data = data.iloc[shuffle]

# get 24000 rows for training
train_data = data.iloc[:24000]
# get 6000 rows for testing
test_data = data.iloc[24000:]

# Bagging
T = 500

bgt = BaggedTree(train_data, test_data, list(train_data.columns[:-1]),
                 train_data['y'], 16, T, subset_size=len(train_data))

x = range(1, T+1)

fig1 = plt.figure(1)
ax1 = plt.axes()
ax1.plot(x, bgt.train_error, c='b', label='Train Error')
ax1.plot(x, bgt.test_error, c='r', label='Test Error')
ax1.set_title("Bagged Tree Error")
plt.legend()
plt.savefig("credit_bagged_tree.png")
plt.show()
