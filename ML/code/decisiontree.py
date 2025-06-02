import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from collections import defaultdict

class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None, min_samples_leaf=5, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree = None
        if random_state is not None:
            np.random.seed(random_state)

    def entropy(self, y):
        probability = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in probability if p > 0])

    def gini(self, y):
        probability = np.bincount(y) / len(y)
        return 1 - np.sum([p ** 2 for p in probability])
    
    def best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        criterion_func = self.entropy if self.criterion == 'entropy' else self.gini
        current_impurity = criterion_func(y)
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                if np.sum(left_idx) < self.min_samples_leaf or np.sum(right_idx) < self.min_samples_leaf:
                    continue
                
                left_impurity = criterion_func(y[left_idx])
                right_impurity = criterion_func(y[right_idx])
                n, n_left, n_right = len(y), np.sum(left_idx), np.sum(right_idx)
                
                weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
                gain = (current_impurity - weighted_impurity) / (current_impurity + 1e-10)  # Normalized gain
                
                #print(f"Feature: {feature}, Threshold: {threshold}, Gain: {gain:.4f}")
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        #print(f"Best Split Selected - Feature: {best_feature}, Threshold: {best_threshold}, Gain: {best_gain:.4f}")
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return np.bincount(y).argmax()
        
        feature, threshold = self.best_split(X, y)
        if feature is None:
            return np.bincount(y).argmax()
        
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self.build_tree(X[left_idx], y[left_idx], depth + 1),
            'right': self.build_tree(X[right_idx], y[right_idx], depth + 1)
        }
    
    def fit(self, X, y):
        self.tree = self.build_tree(np.array(X), np.array(y))
    
    def predict_one(self, x, node):
        if isinstance(node, dict):
            if x[node['feature']] <= node['threshold']:
                return self.predict_one(x, node['left'])
            else:
                return self.predict_one(x, node['right'])
        return node
    
    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in np.array(X)])

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
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate the decision tree
for criterion in ['gini', 'entropy']:
    custom_tree = DecisionTree(criterion=criterion, max_depth=3, min_samples_leaf=10, random_state=42)
    custom_tree.fit(X_train, y_train)
    y_train_pred = custom_tree.predict(X_train)
    y_test_pred = custom_tree.predict(X_test)
    
    def accuracy(y_true, y_pred):
        return sum(y_true == y_pred) / len(y_true)
    
    train_acc = accuracy(y_train, y_train_pred)
    test_acc = accuracy(y_test, y_test_pred)
    
    print(f'Custom {criterion} Decision Tree - Test Accuracy: {test_acc:.4f}, Training Accuracy: {train_acc:.4f}')
