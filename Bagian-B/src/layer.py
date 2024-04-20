import numpy as np
from constant import activation
from activation_function import ActivationFunction

class Layer:
    def __init__(self, neuron: int, activation_function: str, weights: np.array, bias: np.array):
        self.neuron = neuron
        self.weights = weights
        self.bias = bias
        if activation_function not in activation:
            raise Exception('Invalid activation function')
        else:
            self.activation_function = ActivationFunction(activation_function)
            self.function = self.activation_function.get_activation_function()
            self.error_term_output = self.activation_function.get_error_term_output()
            self.error_term_hidden = self.activation_function.get_error_term_hidden()

    def forward(self, input: np.array):
        output = self.function(np.dot(input, self.weights) + self.bias)
        return np.round(output, decimals=7)