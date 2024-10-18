import numpy as np
from os import makedirs
import matplotlib.pyplot as plt

def calculate_mse(predictions, targets) -> float:
    assert len(predictions) == len(targets), "Prediction and target lists must be of the same length."
    predictions, targets = np.array(predictions), np.array(targets)
    return np.sum((targets - predictions) ** 2) / 2


class LinearModelWeights:
    
    def __init__(self, initial_weights: list):
        self.weights = initial_weights

    def __str__(self) -> str:
        return f"Weights: {self.weights}"

    def predict(self, features) -> list:
        return list(map(lambda row: np.dot(self.weights, row), features))


def batch_gradient_descent(X, Y, learning_rate: float = 1, num_epochs: int = 10, convergence_threshold=1e-6):
    weights = np.ones_like(X[0])  # Initialize weights with ones

    losses, previous_loss, loss_diff = [], 9999, 1
    for epoch in range(num_epochs):
        if loss_diff <= convergence_threshold:
            break  # Stop if converged
        grad = np.zeros(weights.shape[0])  # Gradient initialization

        # Calculate gradient
        for j in range(len(grad)):
            for feature, target in zip(X, Y):
                grad[j] -= (target - np.dot(weights, feature)) * feature[j]

        # Update weights using gradient and learning rate
        weights -= learning_rate * grad

        # Compute loss
        current_loss = np.sum([(target - np.dot(weights, feature)) ** 2 for feature, target in zip(X, Y)]) / 2

        # Calculate loss difference for convergence check
        loss_diff = abs(current_loss - previous_loss)
        previous_loss = current_loss
        losses.append(current_loss)

    print(f"Converged at epoch {epoch}, with change: {loss_diff}")
    return LinearModelWeights(weights), losses


def stochastic_gradient_descent(X, Y, learning_rate: float = 1, num_epochs: int = 10, convergence_threshold=1e-6):
    
    weights = np.ones_like(X[0], dtype=np.float64)  # Ensure weights are a numpy array

    losses, previous_loss, loss_diff = [], 9999, 1
    for epoch in range(num_epochs):
        if loss_diff <= convergence_threshold:
            break

        for feature, target in zip(X, Y):
            # Update weights one sample at a time
            weights += learning_rate * (target - np.dot(weights, feature)) * np.array(feature)  # Ensure feature is a numpy array

            # Calculate current loss
            current_loss = np.sum([(target - np.dot(weights, feature)) ** 2 for feature, target in zip(X, Y)]) / 2

            # Check for convergence
            loss_diff = abs(current_loss - previous_loss)
            previous_loss = current_loss
            losses.append(current_loss)

    print(f"Converged at epoch {epoch}, with change: {loss_diff}")
    return LinearModelWeights(weights), losses


def solve_least_squares(X, Y):
    X_transposed = np.transpose(np.array(X))
    Y_array = np.array(Y)

    weights = np.linalg.inv(X_transposed @ X_transposed.T) @ (X_transposed @ Y_array)
    return LinearModelWeights(weights)


# Load training data
train_features, train_targets = [], []
with open("concrete/train.csv", "r") as file:
    for line in file:
        data = list(map(float, line.strip().split(",")))
        train_features.append(data[:-1])
        train_targets.append(data[-1])

# Load test data
test_features, test_targets = [], []
with open("concrete/test.csv", "r") as file:
    for line in file:
        data = list(map(float, line.strip().split(",")))
        test_features.append(data[:-1])
        test_targets.append(data[-1])

print("Linear Model with Batch Gradient Descent")
bgd_weights, bgd_loss = batch_gradient_descent(train_features, train_targets, learning_rate=1e-3, num_epochs=500)

print(f"Final weights (BGD): {bgd_weights}")

print("Linear Model with Stochastic Gradient Descent")
sgd_weights, sgd_loss = stochastic_gradient_descent(train_features, train_targets, learning_rate=1e-3, num_epochs=500)

print(f"Final weights (SGD): {sgd_weights}")

# Plotting the loss for both gradient descent methods
plt.figure()
plt.plot(np.array(range(len(bgd_loss))) * len(train_features), bgd_loss, color='tab:blue', label="Batch Gradient Descent")
plt.plot(sgd_loss, color='tab:orange', label="Stochastic Gradient Descent")
plt.legend()
plt.title("Gradient Descent Comparison")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.savefig("MSE_comparison.png")
plt.clf()

print("Linear Model with Analytic Least Squares Method")
analytic_weights = solve_least_squares(train_features, train_targets)

print(f"Final weights (Analytic): {analytic_weights}")