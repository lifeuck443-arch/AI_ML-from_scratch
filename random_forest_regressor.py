import numpy as np


# ==============================
# Tree Node Class
# ==============================
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # prediction value if leaf


# ==============================
# Decision Tree Regressor
# ==============================
class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def variance(self, y):
        return np.var(y)  # impurity measure for regression

    def variance_reduction(self, y, y_left, y_right):
        parent_var = self.variance(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        if n_left == 0 or n_right == 0:
            return 0
        child_var = (n_left / n) * self.variance(y_left) + (n_right / n) * self.variance(y_right)
        return parent_var - child_var

    def best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feat, best_thresh, best_vr = None, None, -1

        if self.max_features is not None:
            n_feat_subset = min(self.max_features, n_features)
            feature_indices = np.random.choice(n_features, n_feat_subset, replace=False)
        else:
            feature_indices = range(n_features)

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = X[:, feature] > t

                if len(y[left_mask]) < self.min_samples_split or len(y[right_mask]) < self.min_samples_split:
                    continue

                vr = self.variance_reduction(y, y[left_mask], y[right_mask])
                if vr > best_vr:
                    best_feat, best_thresh, best_vr = feature, t, vr

        return best_feat, best_thresh, best_vr

    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return Node(value=np.mean(y))

        feature, threshold, vr = self.best_split(X, y)
        if feature is None or vr <= 0:
            return Node(value=np.mean(y))

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature, threshold, left, right)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)
        return self

    def predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])


# ==============================
# Random Forest Regressor
# ==============================
class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=3, max_features='sqrt',
                 bootstrap=True, random_state=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.trees = []
        self.feature_importances_ = None

    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return self.max_features if self.max_features else n_features

    def _update_feature_importance(self, tree, feature_importances):
        def traverse(node):
            if node.feature is not None:
                feature_importances[node.feature] += 1
            if node.left:
                traverse(node.left)
            if node.right:
                traverse(node.right)
        traverse(tree.root)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)
        feature_importances = np.zeros(n_features)

        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot, y_boot = X[indices], y[indices]
            else:
                X_boot, y_boot = X, y

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            self._update_feature_importance(tree, feature_importances)

        if np.sum(feature_importances) > 0:
            self.feature_importances_ = feature_importances / np.sum(feature_importances)
        else:
            self.feature_importances_ = feature_importances
        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot  # R² score


# ==============================
# Example Usage
# ==============================

import pandas as pd

from sklearn.model_selection import train_test_split
if __name__ == "__main__":
    
    dataset = pd.read_csv("/home/hamza/python/ml_projects/bank_customer_churn.csv")

    # Target
    target = dataset["Churn"]

    # Drop and encode features
    data = dataset.drop("Churn", axis=1)
    data = pd.get_dummies(data, drop_first=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    print(f"R² Score: {rf.score(X_test, y_test):.4f}")
    print(f"Feature Importances: {rf.feature_importances_}")
