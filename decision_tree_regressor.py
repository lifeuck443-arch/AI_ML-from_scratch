import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeRegressor:
    def __init__(self, max_depth=3, criterion="mse"):
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None

    # -----------------------------
    # 1. Impurity Measure (Variance or MSE)
    # -----------------------------
    def mse(self, y):
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)

    def variance(self, y):
        return np.var(y)

    # -----------------------------
    # 2. Information Gain
    # -----------------------------
    def information_gain(self, y, y_left, y_right):
        impurity_func = self.mse if self.criterion == "mse" else self.variance

        parent_impurity = impurity_func(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        child_impurity = (n_left / n) * impurity_func(y_left) + (n_right / n) * impurity_func(y_right)
        return parent_impurity - child_impurity

    # -----------------------------
    # 3. Best Split
    # -----------------------------
    def best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feat, best_thresh, best_ig = None, None, -1

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = X[:, feature] > t

                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue

                ig = self.information_gain(y, y[left_mask], y[right_mask])
                if ig > best_ig:
                    best_feat, best_thresh, best_ig = feature, t, ig

        return best_feat, best_thresh, best_ig

    # -----------------------------
    # 4. Build Tree
    # -----------------------------
    def build_tree(self, X, y, depth=0):
        if len(y) == 0:
            return None
        if depth == self.max_depth:
            return Node(value=np.mean(y))
        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        feature, threshold, ig = self.best_split(X, y)
        if feature is None or ig <= 0:
            return Node(value=np.mean(y))

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature, threshold, left, right)

    # -----------------------------
    # 5. Fit
    # -----------------------------
    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    # -----------------------------
    # 6. Predict
    # -----------------------------
    def predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])


# -----------------------------
# 7. Test it
# -----------------------------
if __name__ == "__main__":
    # simple regression dataset
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5]
    ])
    y = np.array([1.2, 1.9, 3.2, 3.9, 5.1])

    tree = DecisionTreeRegressor(max_depth=2, criterion="mse")
    tree.fit(X, y)

    preds = tree.predict(np.array([[1,4,5,6,6]]))
    print("Predictions:", preds)
    print("MSE:", np.mean((preds - y) ** 2))
    # generating tree visually
    
    
   