import numpy as np
from collections import Counter

# here we are creating our own DecisionTree, so we starting by creating the Node class
class Node:
    def __init__(self, feature = None, treshold = None, left = None, right = None, *, value = None):
        self.feature = feature
        self.treshold = treshold
        self.left = left
        self.right = right
        self.value = None

    def is_leaf_node():
        return self.value is not None


# Tree class
class DecisionTree:
    def __init__(self, min_samples_split = 2, max_depth = 100, n_features = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)


    def _grow_tree(self, X, y, depth = 0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)

        feature_indexes = np.random.choice(n_features, self.n_features, replace = False)

        # Find the best split
        best_feature, best_tresh = self._best_split(X, y, feature_indexes)

        

        #Create child nodes
    def _best_split(self, X, y, feature_indexes):
        best_gain = -1
        split_idx, split_treshold = None, None

        for feature_index in feature_indexes:
            X_column = X[:, feature_index]
            tresholds = np.unique(X_column)

            for thr in tresholds:
                # Calculate the information gain
                gain = self._information_gain(y, X_column, thr):

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_index
                    split_treshold = thr

        return split_idx, split_treshold



    def _information_gain(self, y, X_column, treshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, treshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate the weighted avreage entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self.entropy(y[left_idxs]), self.entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n)

        # Calculate the Information Gain (IG)

    def _split(self, X_column, split_tresh):
        left_indxs = np.argwhere(X_column <= split_tresh).flatten()
        right_indxs = left_indxs = np.argwhere(X_column > split_tresh).flatten()

        return left_indxs, right_indxs

    def _entropy(self, y):
        hist = np.bitcount(y)
        ps = hits / len(y)

        return -np.sum([p * np.log(p) for p in ps if p > 0])

        
    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value


    def predict():