import numpy as np
from tqdm import tqdm

from bagged_tree import BaggedTree
import pandas as pd

from DecisionTree.preprocessing_data import converting_numerical_to_binary

# 2 (c)

train_data = "bank/train.csv"
test_data = "bank/test.csv"


bank_column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                     'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

# bank data numerical columns
bank_numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

train_df = pd.read_csv(train_data, names=bank_column_names)
test_df = pd.read_csv(test_data, names=bank_column_names)

train_df['y'] = train_df['y'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)
test_df['y'] = test_df['y'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)

# median of numerical attributes of bank train data
train_numerical_thresholds = train_df[bank_numerical_columns].median()
test_numerical_thresholds = test_df[bank_numerical_columns].median()

#  consider unknown as category
preprocessed_bank_train_df = converting_numerical_to_binary(train_df, train_numerical_thresholds,
                                                            bank_numerical_columns)
preprocessed_bank_test_df = converting_numerical_to_binary(test_df, test_numerical_thresholds, bank_numerical_columns)

print("Bank Dataset Evaluation (with unknown considered as value):")
T = 500


def bias_squared(y, y_hat):
    return np.square(np.subtract(np.mean(y_hat, axis=0), y)).mean()


def variance(y_hat):
    return np.var(y_hat)


y_pred_single_tree = []
y_pred_bagged_tree = []

for _ in tqdm(range(100)):
    bgt = BaggedTree(preprocessed_bank_train_df, preprocessed_bank_test_df,
                     list(preprocessed_bank_train_df.columns[:-1]),
                     preprocessed_bank_train_df['y'], 16, T, subset_size=1000, replacement=False)
    single_tree = bgt.trees[0]
    y_hat = single_tree.predictions(preprocessed_bank_test_df)
    y_pred_single_tree.append(y_hat)

    y_hat = bgt.predictions(preprocessed_bank_test_df)
    y_pred_bagged_tree.append(y_hat[499])

y_pred_single_tree = np.array(y_pred_single_tree)
y_pred_bagged_tree = np.array(y_pred_bagged_tree)
y_true = preprocessed_bank_test_df['y'].to_numpy()

bias_single_tree = bias_squared(preprocessed_bank_test_df['y'], y_pred_single_tree)
bias_bagged_tree = bias_squared(preprocessed_bank_test_df['y'], y_pred_bagged_tree)
variance_single_tree = variance(y_pred_single_tree)
variance_bagged_tree = variance(y_pred_bagged_tree)

print("Bias of single tree: ", bias_single_tree)
print("Bias of bagged tree: ", bias_bagged_tree)
print("Variance of single tree: ", variance_single_tree)
print("Variance of bagged tree: ", variance_bagged_tree)
print("Total error of single tree: ", bias_single_tree + variance_single_tree)
print("Total error of bagged tree: ", bias_bagged_tree + variance_bagged_tree)

f = open("random_forest_2e.txt", "a")
f.write(f"Bias of single tree:  {bias_single_tree}\n")
f.write(f"Bias of bagged tree:  {bias_bagged_tree}\n")
f.write(f"Variance of single tree:  {variance_single_tree}\n")
f.write(f"Variance of bagged tree:  {variance_bagged_tree}\n")
f.write(f"Total error of single tree:  {bias_single_tree + variance_single_tree}\n")
f.write(f"Total error of bagged tree:  {bias_bagged_tree + variance_bagged_tree}\n")
f.close()
