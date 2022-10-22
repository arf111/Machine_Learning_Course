import numpy as np
import pandas as pd

from DecisionTree.decision_tree import DecisionTree


class BaggedTree:
    def __init__(self, data, attributes, labels, max_depth, num_trees, criterion='entropy'):
        self.data = data
        self.attributes = attributes
        self.labels = labels
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.criteria = criterion
        self.trees = self.build_trees()

    def build_trees(self):
        """Build a forest of decision trees"""
        trees = []

        for i in range(self.num_trees):
            # Create a bootstrap sample
            bootstrap = self.data.sample(frac=1, replace=True)
            # Create a decision tree
            tree = DecisionTree(bootstrap, self.attributes, self.labels, self.max_depth, self.criteria)
            # Add the tree to the forest
            trees.append(tree)

        return trees

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

    def training_error(self, label: str):
        """Calculate the training error of the bagged tree"""
        return self.evaluate(self.data, label)
