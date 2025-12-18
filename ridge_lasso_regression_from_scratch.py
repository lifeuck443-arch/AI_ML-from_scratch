import math

class LassoRegression:
    def __init__(self, learning_rate=0.01, lamda=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.epochs = epochs
        self.w = 0.0  # single weight for 1D case
        self.b = 0.0  # bias term

    def predict(self, x):
        """Predict y for a given x"""
        return self.w * x + self.b

    def compute_gradients(self, X, y):
        """Compute gradients for weight and bias"""
        n = len(X)
        dw, db = 0.0, 0.0
        for i in range(n):
            y_pred = self.predict(X[i])
            error = y_pred - y[i]

            # Gradient for w includes L1 penalty term (sign(w))
            dw += (2 / n) * (X[i] * error) + (self.lamda * math.copysign(1, self.w))
            db += (2 / n) * error

        return dw, db

    def update(self, X, y):
        """Update weights using computed gradients"""
        dw, db = self.compute_gradients(X, y)
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def mean_squared_error(self, y_true, y_pred):
        """Compute MSE with L1 penalty"""
        n = len(y_true)
        mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
        mse += self.lamda * abs(self.w)
        return mse

    def fit(self, X, y):
        """Train the model"""
        for epoch in range(self.epochs):
            self.update(X, y)
            if epoch % 200 == 0:
                y_pred = [self.predict(x) for x in X]
                mse = self.mean_squared_error(y, y_pred)
                print(f"Epoch {epoch}: MSE = {mse:.4f}")

# ---------------------------------------
# Example usage
# ---------------------------------------
X = [2, 4, 6, 8, 9]
y = [4, 8, 12, 11, 14]

model = LassoRegression(learning_rate=0.01, lamda=0.1, epochs=1000)
model.fit(X, y)

print("\nFinal Parameters:")
print("Weight (w):", round(model.w, 4))
print("Bias (b):", round(model.b, 4))
print("Prediction for x=9:", round(model.predict(9), 2))
