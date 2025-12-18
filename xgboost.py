import numpy as np

class XGBMiniRegressor:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=2, min_samples_split=2, reg_lambda=1, gamma=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda  # L2 regularization
        self.gamma = gamma            # complexity penalty
        self.trees = []
        self.initial_prediction = None

    class Node:
        def __init__(self, left=None, right=None, feature=None, threshold=None, value=None):
            self.left = left
            self.right = right
            self.feature = feature
            self.threshold = threshold
            self.value = value

    def _gradient(self, y, y_pred):
        # MSE: g = dL/dy_pred, h = d²L/dy_pred²
        g = y_pred - y
        h = np.ones_like(y)  # second derivative of MSE
        return g, h

    def _calc_leaf_value(self, g, h):
        return -np.sum(g) / (np.sum(h) + self.reg_lambda)

    def _split_gain(self, g_left, h_left, g_right, h_right):
        gain = 0.5 * (
            (np.sum(g_left)**2 / (np.sum(h_left) + self.reg_lambda)) +
            (np.sum(g_right)**2 / (np.sum(h_right) + self.reg_lambda)) -
            (np.sum(g_left + g_right)**2 / (np.sum(h_left + h_right) + self.reg_lambda))
        ) - self.gamma
        return gain

    def _build_tree(self, X, g, h, depth=0):
        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            value = self._calc_leaf_value(g, h)
            return self.Node(value=value)

        best_gain = -np.inf
        best_feature, best_threshold = None, None
        best_splits = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = X[:, feature] > t

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                g_left, h_left = g[left_mask], h[left_mask]
                g_right, h_right = g[right_mask], h[right_mask]
                gain = self._split_gain(g_left, h_left, g_right, h_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = t
                    best_splits = (left_mask, right_mask)

        if best_gain <= 0:
            value = self._calc_leaf_value(g, h)
            return self.Node(value=value)

        left = self._build_tree(X[best_splits[0]], g[best_splits[0]], h[best_splits[0]], depth+1)
        right = self._build_tree(X[best_splits[1]], g[best_splits[1]], h[best_splits[1]], depth+1)
        return self.Node(left=left, right=right, feature=best_feature, threshold=best_threshold)

    def _predict_tree(self, node, X):
        if node.value is not None:
            return np.full(X.shape[0], node.value)
        left_pred = self._predict_tree(node.left, X)
        right_pred = self._predict_tree(node.right, X)
        mask = X[:, node.feature] <= node.threshold
        return np.where(mask, left_pred, right_pred)

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            g, h = self._gradient(y, y_pred)
            tree = self._build_tree(X, g, h)
            self.trees.append(tree)
            y_pred += self.learning_rate * self._predict_tree(tree, X)

    def predict(self, X):
        y_pred = np.full((X.shape[0],), self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * self._predict_tree(tree, X)
        return y_pred

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.5, 3.7, 3.9, 6.2, 7.1])

    model = XGBMiniRegressor(n_estimators=3, learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    preds = model.predict(X)
    print("Predictions:", preds)
