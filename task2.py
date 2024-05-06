import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Function to split data into training and testing sets
def data_split(features, target, ratio):
    split_index = int(len(features) * (1 - ratio))
    
    features_train = features[:split_index]
    features_test = features[split_index:]

    target_train = target[:split_index]
    target_test = target[split_index:]
    
    return features_train, features_test, target_train, target_test

# Load data
data = pd.read_csv('/Users/aliihsangungoren/Downloads/predictive_maintenance.csv')
feature1 = data['Air temperature [K]'].values
feature2 = data['Process temperature [K]'].values
feature3 = data['Rotational speed [rpm]'].values
feature4 = data['Torque [Nm]'].values
feature5 = data['Tool wear [min]'].values
features = np.stack((feature1, feature2, feature3, feature4, feature5), axis=-1)
features = StandardScaler().fit_transform(features)

target = data['Target'].values

# Split data
features_train, features_test, target_train, target_test = data_split(features, target, 0.2)

# Parameters
learning_rates = [0.001, 0.01, 0.1]
iterations = 50000
reg_param = 0.00
weights = np.zeros(5)
bias = 0
losses = []

# Training loop
for lr in learning_rates:
    for _ in range(iterations):
        num_samples = len(target)
        linear_model = np.dot(features_train, weights) + bias
        predictions = 1 / (1 + np.exp(-linear_model))
        loss = -np.mean(target_train * np.log(predictions) + (1 - target_train) * np.log(1 - predictions)) + (reg_param * np.sum(weights**2) / (2 * num_samples))
        losses.append(loss)
        grad = np.dot(features_train.T, (predictions - target_train)) / num_samples + (reg_param * weights**2 / num_samples)
        weights -= lr * grad
        bias -= lr * np.sum(predictions - target_train) / num_samples

    # Predictions
    target_pred = 1 / (np.exp(-features_test.dot(weights) - bias) + 1)
    target_pred = pd.Series(np.where(target_pred > 0.16, 1, 0))

    # Accuracy
    error_count = np.count_nonzero(target_test)
    accuracy = np.sum(np.abs(target_test - target_pred))

    print(f"Number of Errors for Learning Rate {lr}: {error_count}")
    print(f"Number of False Predictions for Learning Rate {lr}: {accuracy}")

    # Plotting
    plt.plot(target_pred, label=f'Predicted (Learning rate: {lr})')
    plt.plot(target_test, label='Actual') 
    plt.legend()

    plt.figure()
    plt.plot(losses)
    plt.title(f'Loss vs. Iterations (Learning rate: {lr})')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()