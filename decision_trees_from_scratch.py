import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, max_depth=3, criterion="gini"):
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None

    # -----------------------------
    # 1. Impurity Measures
    # -----------------------------
    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        print()
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p + 1e-9))  # avoid log(0)

    # -----------------------------
    # 2. Information Gain
    # -----------------------------
    def information_gain(self, y, y_left, y_right):
        impurity_func = self.gini if self.criterion == "gini" else self.entropy

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
                # update  information gain
                if ig > best_ig:
                    best_feat, best_thresh, best_ig = feature, t, ig

        return best_feat, best_thresh, best_ig










    # -----------------------------
    # 4. Build Tree (Recursive)
    # -----------------------------
    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return Node(value=np.bincount(y).argmax())

        feature, threshold, ig = self.best_split(X, y)
        if feature is None or ig <= 0:
            return Node(value=np.bincount(y).argmax())

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
    # 6. Predict (single + batch)
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
    X = np.array([
        [2.7, 2.5],
        [1.3, 1.8],
        [3.6, 3.0],
        [4.4, 1.9],
        [3.9, 4.2]
    ])
    y = np.array([0, 0, 1, 1, 1])

    # print("=== Using GINI ===")
    tree_gini = DecisionTreeClassifier(max_depth=3, criterion="gini")
    tree_gini.fit(X, y)
    preds_gini = tree_gini.predict(X)
    # print("Predictions:", preds_gini)
    # print("Accuracy:", np.mean(preds_gini == y))

    # print("\n=== Using ENTROPY ===")
    tree_entropy = DecisionTreeClassifier(max_depth=3, criterion="entropy")
    tree_entropy.fit(X, y)
    preds_entropy = tree_entropy.predict(X)
    # print("Predictions:", preds_entropy)
    # print("Accuracy:", np.mean(preds_entropy == y))
