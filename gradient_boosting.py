import numpy as np
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class DecisionStump:
    """A simple 1-level decision tree regressor."""
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_error = float('inf')
        
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] >threshold
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_value = np.mean(y[left_mask])
                right_value = np.mean(y[right_mask])
                
                predictions = np.where(left_mask, left_value, right_value)
                error = np.mean((y - predictions) ** 2)
                
                if error < best_error:
                    best_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_value = left_value
                    self.right_value = right_value

    def predict(self, X):
        left_mask = X[:, self.feature_index] <= self.threshold
        return np.where(left_mask, self.left_value, self.right_value)


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.initial_prediction = None
        self.models = []

    def fit(self, X, y):
        # Initialize with the mean of y
        self.initial_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.initial_prediction)
        
        for m in range(self.n_estimators):
            # Compute residuals (negative gradients)
            residuals = y - y_pred
            
            # Fit a weak learner to residuals
            stump = DecisionStump()
            stump.fit(X, residuals)
            self.models.append(stump)
            
            # Update predictions
            y_pred += self.learning_rate * stump.predict(X)

    def predict(self, X):
        y_pred = np.full((X.shape[0]), self.initial_prediction)
        for stump in self.models:
            y_pred += self.learning_rate * stump.predict(X)
        return y_pred


# --- Example usage ---
if __name__ == "__main__":
    # Generate simple data
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

    # Train model
    model = GradientBoostingRegressor(n_estimators=50, 
                                      learning_rate=0.1)
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # Evaluate
    mse = np.mean((y - y_pred) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")
