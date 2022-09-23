import pandas as pd

from DecisionTree.preprocessing_data import preprocessing_bank_dataset, fill_unknown_data
from decision_tree import DecisionTree

# car data training and testing

car_train_data = pd.read_csv('car/train.csv')

car_train_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

# build the decision tree
tree = DecisionTree(car_train_data, list(car_train_data.columns[:-1]), car_train_data['label'], max_depth=6)

# evaluate the test data
car_test_data = pd.read_csv('car/test.csv')
car_test_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

print("Car Dataset Evaluation:")
print(f"Avg prediction error: {tree.evaluate(car_test_data, 'label')}\n")

# bank data training and testing

bank_train_data = pd.read_csv('bank/train.csv')
bank_test_data = pd.read_csv('bank/test.csv')
bank_column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                     'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

bank_train_data.columns = bank_column_names
bank_test_data.columns = bank_column_names

# bank data numerical columns
bank_numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
# median of numerical attributes of bank train data
numerical_thresholds = bank_train_data[bank_numerical_columns].median()

#  consider unknown as category
preprocessed_bank_train_df = preprocessing_bank_dataset(bank_train_data, numerical_thresholds, bank_numerical_columns)
preprocessed_bank_test_df = preprocessing_bank_dataset(bank_test_data, numerical_thresholds, bank_numerical_columns)

# build the decision tree
bank_decision_tree_for_considering_unknown_values = DecisionTree(preprocessed_bank_train_df,
                                                                 list(preprocessed_bank_train_df.columns[:-1]),
                                                                 preprocessed_bank_train_df['y'], max_depth=1)

# evaluate the test data
print("Bank Dataset Evaluation (with unknown considered as value):")
print(f"Avg prediction error: {bank_decision_tree_for_considering_unknown_values.evaluate(preprocessed_bank_test_df, 'y')}\n")

# categorical columns with value unknown
categorical_columns_with_unknown_values = ['job', 'education', 'contact', 'poutcome']

# replace unknown by most frequent value
preprocessed_bank_train_df = fill_unknown_data(bank_train_data, categorical_columns_with_unknown_values)
preprocessed_bank_test_df = fill_unknown_data(bank_test_data, categorical_columns_with_unknown_values)

# build the decision tree
bank_decision_tree_for_replaced_unknown_values = DecisionTree(preprocessed_bank_train_df,
                                                              list(preprocessed_bank_train_df.columns[:-1]),
                                                              preprocessed_bank_train_df['y'], max_depth=1)

# evaluate the test data
print("Bank Dataset Evaluation (replacing unknown value):")
print(
    f"Avg prediction error: {bank_decision_tree_for_replaced_unknown_values.evaluate(preprocessed_bank_test_df, 'y')}")
