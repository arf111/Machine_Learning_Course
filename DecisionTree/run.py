import numpy as np
import pandas as pd

from DecisionTree.preprocessing_data import preprocessing_bank_dataset, fill_unknown_data
from decision_tree import DecisionTree

# car data training and testing

car_train_data = pd.read_csv('car/train.csv')
car_test_data = pd.read_csv('car/test.csv')

car_test_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
car_train_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

car_error_table = np.zeros((6, 6))
# build the decision tree
for depth in range(1, 7):
    for criteria in ['entropy', 'gini', 'majority']:
        car_decision_tree = DecisionTree(car_train_data, list(car_train_data.columns[:-1]), car_train_data['label'],
                                         max_depth=depth)

        training_error = car_decision_tree.training_error('label')

        car_error_table[depth - 1, 2 * ['entropy', 'gini', 'majority'].index(criteria)] = training_error

        test_error = car_decision_tree.evaluate(car_test_data, 'label')

        car_error_table[depth - 1, 2 * ['entropy', 'gini', 'majority'].index(criteria) + 1] = test_error

car_report = pd.DataFrame(car_error_table, columns=['entropy_train', 'entropy_test', 'gini_train', 'gini_test', 'majority_train', 'majority_test'])
car_report.insert(0, 'depth', value=np.arange(1, 7))
print(car_report.to_string(index=False))
# print(car_report.to_latex(index=False))
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

print("Bank Dataset Evaluation (with unknown considered as value):")
bank_error_table = np.zeros((16, 6))
# build the decision tree
for depth in range(1, 17):
    for criteria in ['entropy', 'gini', 'majority']:
        bank_decision_tree = DecisionTree(preprocessed_bank_train_df, list(preprocessed_bank_train_df.columns[:-1]),
                                          preprocessed_bank_train_df['y'], max_depth=depth)

        training_error = bank_decision_tree.training_error('y')

        bank_error_table[depth - 1, 2 * ['entropy', 'gini', 'majority'].index(criteria)] = training_error

        test_error = bank_decision_tree.evaluate(preprocessed_bank_test_df, 'y')

        bank_error_table[depth - 1, 2 * ['entropy', 'gini', 'majority'].index(criteria) + 1] = test_error

bank_report = pd.DataFrame(bank_error_table, columns=['entropy_train', 'entropy_test', 'gini_train', 'gini_test', 'majority_train', 'majority_test'])
bank_report.insert(0, 'depth', value=np.arange(1, 17))
print(bank_report.to_string(index=False))
# print(bank_report.to_latex(index=False))
# categorical columns with value unknown
categorical_columns_with_unknown_values = ['job', 'education', 'contact', 'poutcome']

# replace unknown by most frequent value
preprocessed_bank_train_df = fill_unknown_data(bank_train_data, categorical_columns_with_unknown_values)
preprocessed_bank_test_df = fill_unknown_data(bank_test_data, categorical_columns_with_unknown_values)

print("Bank Dataset Evaluation (with unknown replaced by most frequent value):")
bank_error_table = np.zeros((16, 6))
# build the decision tree
for depth in range(1, 17):
    for criteria in ['entropy', 'gini', 'majority']:
        bank_decision_tree_for_replaced_unknown_values = DecisionTree(preprocessed_bank_train_df,
                                                                      list(preprocessed_bank_train_df.columns[:-1]),
                                                                      preprocessed_bank_train_df['y'], max_depth=depth)

        training_error = bank_decision_tree_for_replaced_unknown_values.training_error('y')

        bank_error_table[depth - 1, 2 * ['entropy', 'gini', 'majority'].index(criteria)] = training_error

        test_error = bank_decision_tree_for_replaced_unknown_values.evaluate(preprocessed_bank_test_df, 'y')

        bank_error_table[depth - 1, 2 * ['entropy', 'gini', 'majority'].index(criteria) + 1] = test_error

bank_report = pd.DataFrame(bank_error_table, columns=['entropy_train', 'entropy_test', 'gini_train', 'gini_test', 'majority_train', 'majority_test'])
bank_report.insert(0, 'depth', value=np.arange(1, 17))
print(bank_report.to_string(index=False))
# print(bank_report.to_latex(index=False))
