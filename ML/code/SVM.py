import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#Generate synthetic dataset
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
y = 2 * y - 1  # Convert labels to {-1, 1}

# Visualization function
def plot_decision_boundary(X, y, w, b, title="Decision Boundary"):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b)
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[0], linewidths=2)
    plt.title(title)
    plt.show()

#Implement Perceptron loss and subgradient
def perceptron_ls(X, y, w, b):
    loss = 0
    dw = np.zeros_like(w)
    db = 0
    for i in range(len(X)):
        margin = y[i] * (np.dot(X[i], w) + b)
        if margin <= 0:  # Misclassified, apply update
            dw -= y[i] * X[i]
            db -= y[i]
            loss += 1
    return loss, dw, db

#Implement Hinge loss (SVM loss) and subgradient
def hinge_ls(X, y, w, b):
    loss = 0
    dw = np.zeros_like(w)
    db = 0
    for i in range(len(X)):
        margin = y[i] * (np.dot(X[i], w) + b)
        if margin < 1:  # If margin is less than 1, update
            dw -= y[i] * X[i]
            db -= y[i]
            loss += max(0, 1 - margin)
    return loss, dw, db

# Step 4: Subgradient Descent Algorithm
def subgradient_descent(X, y, loss_function, w_init, b_init, lr=0.01, epochs=100):
    w = w_init
    b = b_init
    for epoch in range(epochs):
        loss, dw, db = loss_function(X, y, w, b)
        w -= lr * dw
        b -= lr * db
    return w, b

# Step 5: Initialize weights and bias
w_init = np.zeros(X.shape[1])
b_init = 0.0

# Step 6: Train Perceptron using subgradient descent
w_perceptron, b_perceptron = subgradient_descent(X, y, perceptron_ls, w_init, b_init, lr=0.01, epochs=100)

# Step 7: Train SVM (Hinge Loss) using subgradient descent
w_svm, b_svm = subgradient_descent(X, y, hinge_ls, w_init, b_init, lr=0.01, epochs=100)

# Step 8: Plot decision boundaries for both models
plot_decision_boundary(X, y, w_perceptron, b_perceptron, title="Perceptron")
plot_decision_boundary(X, y, w_svm, b_svm, title="SVM (Hinge Loss)")
