import math
import itertools
from collections import Counter

# --- Your base binary Logistic Regression ---
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def predict_proba(self, x):
        z = self.w * x + self.b
        return self.sigmoid(z)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0

    def compute_gradients(self, X, y):
        n = len(X)
        dw, db = 0, 0
        for i in range(n):
            z = self.w * X[i] + self.b
            y_hat = self.sigmoid(z)
            dw += (y_hat - y[i]) * X[i]
            db += (y_hat - y[i])
        dw /= n
        db /= n
        return dw, db

    def fit(self, X, y):
        for _ in range(self.epochs):
            dw, db = self.compute_gradients(X, y)
            self.w -= self.learning_rate * dw    # <--- FIXED
            self.b -= self.learning_rate * db

# --- OvO Multiclass Wrapper ---
class OvOClassifier:
    def __init__(self, base_model_class):
        self.base_model_class = base_model_class
        self.models = {}  # (class_i, class_j): model

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        for (c1, c2) in itertools.combinations(self.classes_, 2):
            # Filter samples belonging to c1 or c2
            X_pair = [X[i] for i in range(len(X)) if y[i] in (c1, c2)]
            y_pair = [1 if y[i] == c1 else 0 for i in range(len(X)) if y[i] in (c1, c2)]

            model = self.base_model_class()
            model.fit(X_pair, y_pair)
            self.models[(c1, c2)] = model

    def predict(self, x):
        votes = []
        for (c1, c2), model in self.models.items():
            pred = model.predict(x)
            votes.append(c1 if pred == 1 else c2)
        # Majority vote
        return Counter(votes).most_common(1)[0][0]

# --- Example usage ---
X = [1, 2, 3, 10, 11, 12, 20, 21, 22]
y = [0, 0, 0, 1, 1, 1, 2, 2, 2]  # Three classes: 0, 1, 2

ovo = OvOClassifier(LogisticRegression)
ovo.fit(X, y)

for x in [2, 11, 21.1]:
    print(f"x={x} → Predicted class: {ovo.predict(x)}")
