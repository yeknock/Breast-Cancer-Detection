

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
    def __init__():