import numpy as np
from constant import activation
from activation_function import ActivationFunction

class Layer:
    def __init__(self, neuron: int, activation_function: str, weights: np.array, bias: np.array, out: bool = False):
        self.neuron = neuron
        self.weights = weights
        self.bias = bias
        self.out = out
        if activation_function not in activation:
            raise Exception('Invalid activation function')
        else:
            self.activation_function = ActivationFunction(activation_function)
            self.function = self.activation_function.get_activation_function()
            if out:
                self.error_term = self.activation_function.get_error_term_output()
            else:
                self.error_term = self.activation_function.get_error_term_hidden()
            self.loss = self.activation_function.get_loss()

    def forward(self, input: np.array):
        self.last_activation = self.function(np.dot(input, self.weights) + self.bias)
        return self.last_activation

    def update_weight(self, input: np.array, error_term: np.array, learning_rate: float):
        self.weights += learning_rate * np.dot(input.T, error_term)
        self.bias += learning_rate * error_term.sum(axis=0)