'''
Simple 3 Layer Neural Network with 4 input vectors
'''
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivation(x):
    return x * (1-x)


training_inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

weights = 2 * np.random.random((3, 1)) - 1

print("weights")

print(weights)

for iteration in range(200000):

    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, weights))

    error = training_outputs - outputs

    adjustments = error*sigmoid_derivation(outputs)

    weights += np.dot(input_layer.T, adjustments)

print("synaptic_weights")
print(weights)

print('outputs after training')
print(outputs)



