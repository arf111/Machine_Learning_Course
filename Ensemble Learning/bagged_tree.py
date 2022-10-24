import numpy as np
import pandas as pd
from tqdm import tqdm

from DecisionTree.decision_tree import DecisionTree


class BaggedTree:
    def __init__(self, train_data, test_data, attributes, labels, max_depth, num_trees, criterion='entropy'):
        self.train_data = train_data
        self.test_data = test_data
        self.attributes = attributes
        self.labels = labels
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.criteria = criterion
        self.trees = []
        self.train_error, self.test_error = self.build_trees()

    def build_trees(self):
        """Build a forest of decision trees"""
        train_error = []
        test_error = []
        subset_size = len(self.train_data)

        for _ in tqdm(range(self.num_trees)):
            # Create a bootstrap sample
            bootstrap = self.train_data.sample(n=subset_size, replace=True)
            # Create a decision tree
            tree = DecisionTree(bootstrap, self.attributes, bootstrap['y'], self.max_depth, self.criteria)
            # Add the tree to the forest
            self.trees.append(tree)
            # Calculate the training and testing error
            train_error.append(self.evaluate(self.train_data, 'y'))
            test_error.append(self.evaluate(self.test_data, 'y'))

        return train_error, test_error

    def predict(self, row):
        """Predict the label of a row"""
        predictions = []

        for tree in self.trees:
            predictions.append(tree.predict(row))

        return max(set(predictions), key=predictions.count)

    def predictions(self, data):
        """Predict the labels of a dataset"""
        return data.apply(self.predict, axis=1)

    def evaluate(self, data: pd.DataFrame, label: str):
        """Evaluate the accuracy of the bagged tree"""
        predictions = self.predictions(data)
        actual = data[label]
        # average prediction error
        return np.mean(predictions != actual)
