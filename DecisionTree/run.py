import pandas as pd
import numpy as np

from DecisionTree import DecisionTree

train_data = pd.read_csv('car/train.csv')

train_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

# build the decision tree
tree = DecisionTree(train_data, list(train_data.columns[:-1]), train_data['label'], max_depth=6)

# evaluate the test data
test_data = pd.read_csv('car/test.csv')
test_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

print(tree.evaluate(test_data, 'label'))
