from adaboost import Adaboost
import pandas as pd
import matplotlib.pyplot as plt

from DecisionTree.preprocessing_data import converting_numerical_to_binary

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

ada = Adaboost(preprocessed_bank_train_df, preprocessed_bank_test_df, list(preprocessed_bank_train_df.columns[:-1]),
               preprocessed_bank_train_df['y'], 1, T)

x = range(1, T+1)
fig1 = plt.figure(1)
ax1 = plt.axes()
ax1.plot(x, ada.train_error_decision_tree, c='b', label='Train Error')
ax1.plot(x, ada.test_error_decision_tree, c='r', label='Test Error')
ax1.set_title("Tree Error")
plt.legend()

fig2 = plt.figure(2)
ax2 = plt.axes()
ax2.plot(x, ada.train_error, c='b', label='Train Error')
ax2.plot(x, ada.test_error, c='r', label='Test Error')
ax2.set_title("Adaboost Error")
plt.legend()
plt.show()

