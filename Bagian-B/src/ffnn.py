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
        self.input_data = input
        for _ in range(max_iteration):
            total_err = 0
            last_layer : Layer = self.layers[-1]
            for i in range(0, len(input), batch_size):
                input_batch = input[i:i+batch_size]
                target_batch = target[i:i+batch_size]
                output = self.forward(input_batch)
                error_batch = np.mean(np.sum(target_batch - output, axis=1) / 2)
                total_err += error_batch

                error_term = last_layer.error_term(output, target_batch)
                self.backward(error_term, learning_rate)
            total_err /= len(input) / batch_size
            if np.abs(total_err) < error_threshold:
                return "error_threshold"
        return "max_iteration"

    def backward(self, error_term, learning_rate: float):
        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]
            input = self.input_data if i == 0 else self.layers[i-1].last_activation
            if i == len(self.layers)-1:
                layer.update_weight(input, error_term, learning_rate)
            else:
                error_term = layer.error_term(self.layers[i+1].last_activation, error_term, self.layers[i+1].weights)
                layer.update_weight(input, error_term, learning_rate)
        self.node_weights = []