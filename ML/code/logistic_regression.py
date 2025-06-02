import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

# -----------------------------
# Logistic Regression from Scratch
# -----------------------------
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=1000, reg_lambda=0.0):
        self.lr = lr
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # add bias column
        self.theta = np.zeros(X.shape[1])
        
        for _ in range(self.epochs):
            z = X @ self.theta
            h = self.sigmoid(z)
            gradient = (X.T @ (h - y)) / len(y)
            reg_term = self.reg_lambda * np.r_[[0], self.theta[1:]]  # no reg for bias
            self.theta -= self.lr * (gradient + reg_term)

    def predict_proba(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X @ self.theta)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# -----------------------------
# Step 1: Generate and Visualize Data
# -----------------------------
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')
plt.title("Training Data")
plt.show()

# -----------------------------
# Step 2: Base Logistic Regression (No Regularization)
# -----------------------------
print("===> Logistic Regression without Polynomial Features:")
model = LogisticRegressionScratch(lr=0.1, epochs=1000, reg_lambda=0.0)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, train_preds))
print("Test Accuracy:", accuracy_score(y_test, test_preds))

# -----------------------------
# Step 3: Vary Polynomial Degree to Cause Overfitting
# -----------------------------
print("\n===> Logistic Regression with Polynomial Features (no regularization):")
for degree in [1, 2, 4, 6, 8]:
    poly = PolynomialFeatures(degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    model = LogisticRegressionScratch(lr=0.1, epochs=2000, reg_lambda=0.0)
    model.fit(X_poly_train[:, 1:], y_train)  # remove bias column

    train_acc = accuracy_score(y_train, model.predict(X_poly_train[:, 1:]))
    test_acc = accuracy_score(y_test, model.predict(X_poly_test[:, 1:]))

    print(f"Degree {degree} - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")

# -----------------------------
# Step 4: Apply Regularization to Reduce Overfitting
# -----------------------------
print("\n===> Regularization Effect (degree = 6):")
degree = 6
poly = PolynomialFeatures(degree)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

for reg_lambda in [0.0, 0.1, 1.0, 10.0, 100.0]:
    model = LogisticRegressionScratch(lr=0.1, epochs=2000, reg_lambda=reg_lambda)
    model.fit(X_poly_train[:, 1:], y_train)

    train_acc = accuracy_score(y_train, model.predict(X_poly_train[:, 1:]))
    test_acc = accuracy_score(y_test, model.predict(X_poly_test[:, 1:]))

    print(f"Lambda {reg_lambda:<5} - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
