import json
import numpy as np
from layer import Layer

model = open(f'Bagian-A/test/relu.json', 'r')
model = json.load(model)

neuron = model['case']['model']['layers'][0]['number_of_neurons']
activation_function = model['case']['model']['layers'][0]['activation_function']
weights = np.array(model['case']['weights'][0][1:])
bias = np.array(model['case']['weights'][0][0])

layer = Layer(neuron, activation_function, weights, bias)
input = model['case']['input']

output = layer.forward(input)
expected_output = model['expect']['output']

print(f'output: {output}')
print(f'expected output: {expected_output}')