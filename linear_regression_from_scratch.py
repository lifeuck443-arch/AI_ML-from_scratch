class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w1 = 0.0
        self.b = 0.0
        self.w2 = 0.0
        
    def predict(self, x1, x2):
        return self.w1 * x1 + self.w2 * x2 + self.b

    def compute_gradients(self, X1, X2, y):
        n = len(y)
        dw1, dw2, db = 0.0, 0.0, 0.0
        
        for i in range(n):
            y_pred = self.predict(X1[i], X2[i])
            error = y[i] - y_pred
            dw1 += -2 * X1[i] * error
            dw2 += -2 * X2[i] * error
            db += -2 * error
        
        return dw1 / n, dw2 / n, db / n

    def update(self, X1, X2, y):
        dw1, dw2, db = self.compute_gradients(X1, X2, y)
        self.w1 -= self.learning_rate * dw1
        self.w2 -= self.learning_rate * dw2
        self.b -= self.learning_rate * db

    def mean_squared_error(self, y, y_pred):
        n = len(y)
        mse = sum((y[i] - y_pred[i]) ** 2 for i in range(n)) / n
        return mse

    def fit(self, X1, X2, y):
        for epoch in range(self.epochs):
            self.update(X1, X2, y)
            
            if epoch % 200 == 0:
                y_pred = [self.predict(X1[i], X2[i]) for i in range(len(y))]
                mse = self.mean_squared_error(y, y_pred)
                print(f"Epoch {epoch}: MSE = {mse:.4f}")

# Example use
X1 = [2, 4, 6, 8]
X2 = [2, 4, 6, 8]
y = [4, 8, 12, 16]

model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X1, X2, y)
print("Prediction for 9, 9:", model.predict(2, 2))