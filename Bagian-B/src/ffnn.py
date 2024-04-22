import numpy as np
from layer import Layer

class FFNN:
    def __init__(self):
        self.layers = []
        self.node_weights = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, input: np.array):
        output = input
        batch_count = len(output)
        for i in range(batch_count):
            self.node_weights.append([output[i]])
        for layer in self.layers:
            output = layer.forward(output)
            for i in range(batch_count):
                temp_nodes = output[i].tolist()
                self.node_weights[i].append(temp_nodes)
        return output

    def fit(self, input: np.array, target: np.array, learning_rate: float, batch_size: int, max_iteration: int, error_threshold: float):
        for iter in range(max_iteration):
            total_err = 0
            for i in range(0, len(input), batch_size):
                input_batch = input[i:i+batch_size]
                target_batch = target[i:i+batch_size]
                output_batch = self.forward(input_batch)
                error_batch = np.mean(self.layers[-1].loss(output_batch, target_batch))
                total_err += error_batch

                self.backward(input_batch, output_batch, target_batch, learning_rate)
            total_err /= len(input) / batch_size
            if np.abs(total_err) < error_threshold:
                return "error_threshold"
        return "max_iteration"

    def backward(self, input : np.array, output: np.array, target: np.array, learning_rate: float):
        first_input = input
        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]
            input = first_input if i == 0 else self.layers[i-1].last_activation
            if i == len(self.layers)-1:
                error_term = layer.error_term(output, target)
                layer.update_weight(input, error_term, learning_rate)
            else:
                error_term_output = self.layers[i+1].error_term(self.layers[i+1].last_activation, target)
                error_term = layer.error_term(self.layers[i].last_activation, error_term_output, self.layers[i+1].weights)
                layer.update_weight(input, error_term, learning_rate)
        self.node_weights = []