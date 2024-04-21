'''Write a program to implement basic genetic algorithm with the following parameters.
Population size = 500, Target = ‘AGT’ and initial population is the string
' TTCGATGAGTCCATAATTGGCAGT'. Assume a 1 bit mutation or
0.1%probability'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def train(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
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

# Load IRIS dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = np.where(iris.target == 0, -1, 1)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Standardize features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Plotting function
def plot_learning(iterations, learning_rates, errors):
    plt.figure(figsize=(10, 6))
    for i, lr in enumerate(learning_rates):
        plt.plot(iterations, errors[i], label=f'LR={lr}')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    plt.title('Perceptron Learning Algorithm')
    plt.legend()
    plt.grid(True)
    plt.show()

# Test different learning rates
learning_rates = [0.0001, 0.01, 0.5, 1.0]
iterations = range(1, 101)
errors = []

for lr in learning_rates:
    ppn = Perceptron(learning_rate=lr, n_iterations=100)
    ppn.train(X_train_std, y_train)
    errors.append(ppn.errors)

# Plot graphs
plot_learning(iterations, learning_rates, errors)

