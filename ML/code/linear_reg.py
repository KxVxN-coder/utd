import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, loss_type='mse'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_type = loss_type
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent
        for epoch in range(self.epochs):
            # Compute the model's predictions
            y_pred = self.predict(X)
            
            # Calculate the loss and gradients
            if self.loss_type == 'mse':
                loss = self.mse_loss(y, y_pred)
                dw, db = self.mse_gradient(X, y, y_pred)
            elif self.loss_type == 'mae':
                loss = self.mae_loss(y, y_pred)
                dw, db = self.mae_gradient(X, y, y_pred)

            # Update weights and bias using gradients
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # Optionally print the loss during training
            #print(f'Epoch ({self.epochs-1}), Loss: {loss}')

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    def mse_gradient(self, X, y_true, y_pred):
        dw = (2 / len(y_true)) * np.dot(X.T, (y_pred - y_true))
        db = (2 / len(y_true)) * np.sum(y_pred - y_true)
        return dw, db

    def mae_loss(self, y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true))

    def mae_gradient(self, X, y_true, y_pred):
        dw = (1 / len(y_true)) * np.dot(X.T, np.sign(y_pred - y_true))
        db = (1 / len(y_true)) * np.sum(np.sign(y_pred - y_true))
        return dw, db

# Load the Boston Housing Dataset
class BostonHousingDataset:
    def __init__(self):
        self.url = "http://lib.stat.cmu.edu/datasets/boston"
        self.feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

    def load_dataset(self):
        # Fetch data from URL
        raw_df = pd.read_csv(self.url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]

        # Create the dictionary in sklearn format
        dataset = {
            'data': [],
            'target': [],
            'feature_names': self.feature_names,
            'DESCR': 'Boston House Prices dataset'
        }

        dataset['data'] = data
        dataset['target'] = target

        return dataset
    
# Load the Boston Housing Dataset
boston_housing = BostonHousingDataset()
boston_dataset = boston_housing.load_dataset()

# Convert to pandas DataFrame
boston = pd.DataFrame(boston_dataset['data'], columns=boston_dataset['feature_names'])
boston['MEDV'] = boston_dataset['target']

# Prepare the data
X = boston.drop(columns=['MEDV']).values  # Features
y = boston['MEDV'].values  # Target variable

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train the model with MSE
model_mse = LinearRegression(learning_rate=0.1, epochs=1000, loss_type='mse')
model_mse.fit(X_scaled, y)

# Initialize and train the model with MAE
model_mae = LinearRegression(learning_rate=0.1, epochs=1000, loss_type='mae')
model_mae.fit(X_scaled, y)

# Make predictions
predictions_mse = model_mse.predict(X_scaled)
predictions_mae = model_mae.predict(X_scaled)

mse_value = model_mse.mse_loss(y, predictions_mse)
mae_value = model_mae.mae_loss(y, predictions_mae)

print(f"Final MSE Value: {mse_value:.4f}")
print(f"Final MAE Value: {mae_value:.4f}")

# Display a comparison of predictions
plt.figure(figsize=(12, 6))

# Plot MSE Model predictions
plt.subplot(1, 2, 1)
plt.scatter(y, predictions_mse, color='blue', label="MSE Predictions")
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.title('MSE Model Predictions')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.legend()

# Plot MAE Model predictions
plt.subplot(1, 2, 2)
plt.scatter(y, predictions_mae, color='green', label="MAE Predictions")
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.title('MAE Model Predictions')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.legend()

plt.tight_layout()
plt.show()


