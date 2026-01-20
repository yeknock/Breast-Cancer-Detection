

# here we are creating our own DecisionTree, so we starting by creating the Node class
class Node:
    def __init__(self, feature=None, treshhold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.treshhold = treshhold
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


    def _grow_tree(self, X, y):

        # Check the stopping criteria


        # Find the best split


        #Create child nodes


    def predict():