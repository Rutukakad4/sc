
'''Write a program to implement gradient descent learning algorithm, describe the
output when the learning rate is too small and too high. Test your model with
0.0001, 0.01, 0.5, 1.0 learning rates. Generate a linear data with some random
Gaussian noise. Plot the cost history over iterations '''
import numpy as np
import matplotlib.pyplot as plt

# Generate linear data with noise
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term to X
X_b = np.c_[np.ones((100, 1)), X]

# Gradient Descent Algorithm
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    m = len(y)
    cost_history = np.zeros(n_iterations)

    for iteration in range(n_iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        theta -= learning_rate * gradient
        cost_history[iteration] = compute_cost(X, y, theta)

    return theta, cost_history

# Test different learning rates
learning_rates = [0.0001, 0.01, 0.5, 1.0]
n_iterations = 1000
initial_theta = np.random.randn(2, 1)

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    theta = initial_theta
    theta, cost_history = gradient_descent(X_b, y, theta, lr, n_iterations)
    plt.plot(range(1, n_iterations + 1), cost_history, label=f'lr={lr}')

plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost History over Iterations for Different Learning Rates')
plt.legend()
plt.grid(True)
plt.show()

