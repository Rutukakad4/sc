'''Write a program to implement perceptron learning algorithm on the MINST
handwritten digit dataset, describe the output when the learning rate is too small and
too high. Test your model with 0.0001, 0.01, 0.5, 1.0 learning rates'''

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Convert labels to integers
y = y.astype(np.uint8)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled = scaler.transform(X_test.astype(np.float64))

# Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def train(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)
        self.errors = []

        for _ in range(self.n_iterations):
            error = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                error += int(update != 0.0)
            self.errors.append(error)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Test different learning rates
learning_rates = [0.0001, 0.01, 0.5, 1.0]
results = []

for lr in learning_rates:
    ppn = Perceptron(learning_rate=lr, n_iterations=100)
    ppn.train(X_train_scaled, y_train)
    accuracy = (ppn.predict(X_test_scaled) == y_test).mean()
    results.append((lr, accuracy))

# Print results
for lr, accuracy in results:
    print(f"Learning rate: {lr}, Accuracy: {accuracy}")

# Plot learning curves
plt.figure(figsize=(10, 6))
for lr, _ in results:
    plt.plot(range(1, 101), ppn.errors, label=f'LR={lr}')
plt.xlabel('Iterations')
plt.ylabel('Number of misclassifications')
plt.title('Perceptron Learning Algorithm on MNIST')
plt.legend()
plt.grid(True)
plt.show()

