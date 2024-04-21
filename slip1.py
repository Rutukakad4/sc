''' Write a program to implement AND and OR gate using perceptron learning
algorithm. Show that the model does not learn the XOR nonlinear relationship '''

import numpy as np

def unitstep(v):
    return 1 if v > 0 else 0

def perceptron(x, w, b):
    v = np.dot(w, x) + b
    return unitstep(v)

def andfunc(x):
    w = np.array(x)
    b = -1.5
    return perceptron(x, w, b)

def orfunc(x):
    w = np.array(x)
    b = -0.5
    return perceptron(x, w, b)

def test_gate(gate, inputs):
    print(f"{gate} Gate:")
    for input_pair in inputs:
        prediction = gate(input_pair)
        print(f"{input_pair}: {prediction}")

def xorfunc(x):
    a = andfunc(x)
    b = orfunc(x)
    c = np.array([a, b])
    return andfunc(c)

# Define input pairs for testing
inputs = [(0, 1), (1, 1), (0, 0), (1, 0)]

# Test AND gate
test_gate(andfunc, inputs)

# Test OR gate
test_gate(orfunc, inputs)

# Test XOR gate
test_gate(xorfunc, inputs)
