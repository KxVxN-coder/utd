import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from collections import defaultdict

class KNN:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

# Load dataset
data = load_wine()
X, y = data.data, data.target

def train_test_split_manual_stratified(X, y, test_size=0.3, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)
    
    train_indices, test_indices = [], []
    
    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - test_size))
        train_indices.extend(indices[:split_idx])
        test_indices.extend(indices[split_idx:])
    
    # Shuffle final train/test indices to avoid any order bias
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
X_train, X_test, y_train, y_test = train_test_split_manual_stratified(X, y, test_size=0.3, random_state=42)

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.3)

knn = KNN(n_neighbors=100)
knn.fit(X_train, y_train)

# Predict on train and test sets
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Compute accuracy
train_accuracy = np.mean(y_train_pred == y_train)
test_accuracy = np.mean(y_test_pred == y_test)

print(f"Train Accuracy: {train_accuracy:.5f}")
print(f"Test Accuracy: {test_accuracy:.5f}")
