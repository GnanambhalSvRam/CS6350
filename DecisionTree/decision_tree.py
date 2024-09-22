import math
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth, heuristic='information_gain'):
        self.max_depth = max_depth
        self.heuristic = heuristic
        self.tree = None

    def fit(self, X, y):
        attributes = list(range(len(X[0])))
        self.tree = self._id3(X, y, attributes, depth=0)

    def predict(self, X):
        return [self._predict_example(x, self.tree) for x in X]

    def _id3(self, X, y, attributes, depth):
        if len(set(y)) == 1:  # Pure node
            return y[0]
        if not attributes or depth == self.max_depth:  # Max depth reached
            return self._majority_class(y)

        best_attr = self._choose_best_attribute(X, y, attributes)
        tree = {best_attr: {}}
        attribute_values = set([x[best_attr] for x in X])

        for value in attribute_values:
            subset_X, subset_y = self._split(X, y, best_attr, value)
            if not subset_X:
                tree[best_attr][value] = self._majority_class(y)
            else:
                tree[best_attr][value] = self._id3(subset_X, subset_y, [a for a in attributes if a != best_attr], depth + 1)

        return tree

    def _split(self, X, y, attribute, value):
        subset_X = [x for x in X if x[attribute] == value]
        subset_y = [y[i] for i in range(len(X)) if X[i][attribute] == value]
        return subset_X, subset_y

    def _choose_best_attribute(self, X, y, attributes):
        if self.heuristic == 'information_gain':
            return self._best_info_gain(X, y, attributes)
        elif self.heuristic == 'gini_index':
            return self._best_gini_index(X, y, attributes)
        elif self.heuristic == 'majority_error':
            return self._best_majority_error(X, y, attributes)

    def _best_info_gain(self, X, y, attributes):
        base_entropy = self._entropy(y)
        best_gain = -1
        best_attr = None

        for attr in attributes:
            values = set([x[attr] for x in X])
            attr_entropy = 0.0
            for value in values:
                subset_X, subset_y = self._split(X, y, attr, value)
                weight = len(subset_y) / len(y)
                attr_entropy += weight * self._entropy(subset_y)

            info_gain = base_entropy - attr_entropy
            if info_gain > best_gain:
                best_gain = info_gain
                best_attr = attr

        return best_attr

    def _entropy(self, y):
        counts = Counter(y)
        entropy = 0.0
        for label in counts:
            prob = counts[label] / len(y)
            entropy -= prob * math.log2(prob)
        return entropy

    def _best_gini_index(self, X, y, attributes):
        best_gini = float('inf')
        best_attr = None

        for attr in attributes:
            values = set([x[attr] for x in X])
            gini = 0.0
            for value in values:
                subset_X, subset_y = self._split(X, y, attr, value)
                weight = len(subset_y) / len(y)
                gini += weight * self._gini(subset_y)

            if gini < best_gini:
                best_gini = gini
                best_attr = attr

        return best_attr

    def _gini(self, y):
        counts = Counter(y)
        gini = 1.0
        for label in counts:
            prob = counts[label] / len(y)
            gini -= prob ** 2
        return gini

    def _best_majority_error(self, X, y, attributes):
        best_error = float('inf')
        best_attr = None

        for attr in attributes:
            values = set([x[attr] for x in X])
            error = 0.0
            for value in values:
                subset_X, subset_y = self._split(X, y, attr, value)
                weight = len(subset_y) / len(y)
                error += weight * self._majority_error(subset_y)

            if error < best_error:
                best_error = error
                best_attr = attr

        return best_attr

    def _majority_error(self, y):
        counts = Counter(y)
        majority = max(counts.values())
        return 1 - majority / len(y)

    def _majority_class(self, y):
        counts = Counter(y)
        return counts.most_common(1)[0][0]

    def _predict_example(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        attribute = list(tree.keys())[0]
        value = x[attribute]
        if value in tree[attribute]:
            return self._predict_example(x, tree[attribute][value])
        else:
            return None
