import numpy as np
from layer import Layer

class FFNN:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)
    
    def forward(self, input: np.array):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output