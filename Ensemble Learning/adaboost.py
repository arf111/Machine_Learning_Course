import numpy as np
import pandas as pd

from DecisionTree.decision_tree import DecisionTree


class Adaboost:
    def __init__(self, train_data, test_data, attributes, labels, max_depth, num_trees, criterion='entropy'):
        self.train_data = train_data
        self.test_data = test_data
        self.attributes = attributes
        self.labels = labels
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.criteria = criterion
        self.trees = self.build_trees(train_data, test_data, attributes, labels, num_trees)

    def build_trees(self, train_data: pd.DataFrame, test_data: pd.DataFrame, attributes: list, labels: pd.Series,
                    num_trees: int):
        # Initialize the weights of each example to 1/m
        weights = np.ones(len(train_data)) / len(train_data)
        trees = []
        train_error = []
        test_error = []
        train_error_decision_tree = []
        test_error_decision_tree = []

        for _ in range(num_trees):
            # sample the data with replacement according to the weights
            sample = train_data.sample(len(train_data), replace=True, weights=weights, ignore_index=True)
            # Build a decision tree with the weights
            tree = DecisionTree(sample, attributes, labels, self.max_depth, self.criteria)
            # Calculate the predictions of the tree
            predictions = tree.predictions(train_data)
            # calculate training error for the tree
            train_error_decision_tree.append(tree.evaluate(train_data, 'label'))
            # calculate testing error for the tree
            test_error_decision_tree.append(tree.evaluate(test_data, 'label'))
            # Calculate the error of the tree
            error = np.sum(weights[predictions != labels])
            # Calculate the weight of the tree
            tree_weight = 0.5 * np.log((1 - error) / error)  # alpha_t
            # Update the weights of the examples
            weights *= np.exp(-tree_weight * labels * predictions)
            # Normalize the weights
            weights /= np.sum(weights)
            # Add the tree to the list of trees
            trees.append((tree, tree_weight))
            # train error
            train_error.append(self.evaluate(train_data, 'y'))
        #     text error
            test_error.append(self.evaluate(test_data, 'y'))

        return train_error_decision_tree, test_error_decision_tree, train_error, test_error

    def predict(self, row):
        """Predict the label of a row"""
        # Calculate the weighted sum of the predictions of the trees
        weighted_sum = np.sum(tree.predict(row) * weight for tree, weight in self.trees)

        # Return the sign of the weighted sum
        return np.sign(weighted_sum)

    def predictions(self, data):
        """Predict the labels of a dataset"""
        return data.apply(self.predict, axis=1)

    def evaluate(self, data: pd.DataFrame, label: str):
        """Evaluate the accuracy of the decision tree"""
        predictions = self.predictions(data)
        actual = data[label]
        # average prediction error
        return np.mean(predictions != actual)
