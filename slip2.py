'''Write a program to implement gradient descent learning algorithm, describe the
output when the learning rate is too small and too high. Test your model with
0.0001, 0.01, 0.5, 1.0 learning rates [11]
Data set having 2 values and target variable (0/1) is as follows:
dataset = [[2.7810836,2.550537003,0],
[1.465489372,2.362125076,0],
[3.396561688,4.400293529,0],
[1.38807019,1.850220317,0],
[3.06407232,3.005305973,0],
[7.627531214,2.759262235,1],
[5.332441248,2.088626775,1],
[6.922596716,1.77106367,1],
[8.675418651,-0.242068655,1],
[7.673756466,3.508563011,1]]
Initial weights = [-0.1, 0.20653640140000007, -0.23418117710000003]'''

# Gradient Descent Algorithm
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

def train_weights(train, learning_rate, n_epochs):
    weights = [0.0 for _ in range(len(train[0]))]
    for epoch in range(n_epochs):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error ** 2
            weights[0] = weights[0] + learning_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + learning_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
    return weights

# Dataset
dataset = [
    [2.7810836, 2.550537003, 0],
    [1.465489372, 2.362125076, 0],
    [3.396561688, 4.400293529, 0],
    [1.38807019, 1.850220317, 0],
    [3.06407232, 3.005305973, 0],
    [7.627531214, 2.759262235, 1],
    [5.332441248, 2.088626775, 1],
    [6.922596716, 1.77106367, 1],
    [8.675418651, -0.242068655, 1],
    [7.673756466, 3.508563011, 1]
]

# Hyperparameters
learning_rates = [0.0001, 0.01, 0.5, 1.0]
n_epochs = 5
initial_weights = [-0.1, 0.20653640140000007, -0.23418117710000003]

# Test each learning rate
for lr in learning_rates:
    print("\nLearning Rate:", lr)
    weights = train_weights(dataset, lr, n_epochs)
    print("Final Weights:", weights)

