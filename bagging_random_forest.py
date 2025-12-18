import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=3, criterion="gini", 
                 max_features=None, min_samples_split=2):
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.root = None

    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p + 1e-9))

    def information_gain(self, y, y_left, y_right):
        impurity_func = self.gini if self.criterion == "gini" else self.entropy
        parent_impurity = impurity_func(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        child_impurity = (n_left / n) * impurity_func(y_left) + (n_right / n) * impurity_func(y_right)
        return parent_impurity - child_impurity

    def best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feat, best_thresh, best_ig = None, None, -1

        # Random feature selection
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

                ig = self.information_gain(y, y[left_mask], y[right_mask])
                if ig > best_ig:
                    best_feat, best_thresh, best_ig = feature, t, ig

        return best_feat, best_thresh, best_ig

    def build_tree(self, X, y, depth=0):
        # Stopping conditions
        if (len(np.unique(y)) == 1 or 
            depth >= self.max_depth or 
            len(y) < self.min_samples_split):
            return Node(value=np.bincount(y).argmax())

        feature, threshold, ig = self.best_split(X, y)
        if feature is None or ig <= 0:
            return Node(value=np.bincount(y).argmax())

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

    def predict_proba(self, X):
        predictions = self.predict(X)
        n_classes = len(np.unique(predictions))
        proba = np.zeros((len(X), n_classes))
        for i, pred in enumerate(predictions):
            proba[i, pred] = 1.0
        return proba

class RandomForestClassifier:
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
        self.classes_ = None
        
    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return self.max_features if self.max_features else n_features
    
    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        n_samples, n_features = X.shape

        max_features = self._get_max_features(n_features)

        self.classes_ = np.unique(y)
        
        feature_importances = np.zeros(n_features)
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X[indices]
                y_boot = y[indices]
            else:
                X_boot, y_boot = X, y
            
            # Train tree with random features
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                criterion='gini',
                max_features=max_features,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
        
        return self
    
    
    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        y_pred = []
        for sample_predictions in tree_predictions.T:
            most_common = Counter(sample_predictions).most_common(1)[0][0]
            y_pred.append(most_common)
        return np.array(y_pred)
    
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    
   

# Example of complete usage without scikit-learn
if __name__ == "__main__":
    # Create sample data without scikit-learn
    def make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42):
        np.random.seed(random_state)
        X = np.random.randn(n_samples, n_features)
        # Create some informative features
        X[:, :3] += np.random.randn(n_samples, 3) * 2
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        return X, y
    
    # Manual train-test split
    def train_test_split(X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        indices = np.random.permutation(n_samples)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    
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

    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    print(f"R² Score: {rf.score(X_test, y_test):.4f}")
    print(f"Feature Importances: {rf.feature_importances_}")
