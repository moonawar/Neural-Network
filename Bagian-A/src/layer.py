import numpy as np
from activation import activation

class Layer:
    def __init__(self, neuron: int, activation_function: str, weights: np.array, bias: np.array):
        self.neuron = neuron
        self.weights = weights
        self.bias = bias
        if activation_function not in activation:
            raise Exception('Invalid activation function')
        else:
            self.activation_function = activation_function
            if activation_function == 'sigmoid':
                self.function = lambda net: 1 / (1 + np.exp(-net))
            elif activation_function == 'relu':
                self.function = lambda net: np.maximum(0, net)
            elif activation_function == 'linear':
                self.function = lambda net: net
            elif activation_function == 'softmax':
                self.function = lambda net: np.exp(net) / np.sum(np.exp(net))

    def forward(self, input: np.array):
        output = self.function(np.dot(input, self.weights) + self.bias)
        return output