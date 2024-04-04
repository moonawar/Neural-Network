import numpy as np
from layer import Layer

class FFNN:
    def __init__(self):
        self.layers = []
        self.node_bobots = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, input: np.array):
        output = input
        batch_count = len(output)
        for i in range(batch_count):
            self.node_bobots.append([output[i]])
        for layer in self.layers:
            output = layer.forward(output)
            for i in range(batch_count):
                temp_nodes = output[i].tolist()
                self.node_bobots[i].append(temp_nodes)
        return output