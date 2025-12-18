import math

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
        for epoch in range(self.epochs):
            dw, db = self.compute_gradients(X, y)
            # selfUse precision, recall, and F1 score to evaluate your model..w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            self.w -= self.learning_rate * dw
    def evaluate(self, X, y):
        preds = [self.predict(x) for x in X]
        accuracy = sum([preds[i] == y[i] for i in range(len(y))]) / len(y)
        return accuracy
# Training data (simple example: OR gate)
X = [10, 10, 1, 10]
y = [0, 1, 1, 1]

model = LogisticRegression(learning_rate=0.1, epochs=1000)
model.fit(X, y)

print("Weight:", model.w)
print("Bias:", model.b)

# Predictions
for x in X:
    print(f"x={x} → Predicted={model.predict(x)} (prob={model.predict_proba(x):.4f})")

print("Accuracy:", model.evaluate(X, y))
