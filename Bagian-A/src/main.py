import json
import numpy as np
from ffnn import FFNN
from layer import Layer

model = open(f'../../Bagian-A/test/relu.json', 'r')
model = json.load(model)

layers = model['case']['model']['layers']
weights = model['case']['weights']

ffnn = FFNN()
for i in range (len(layers)):
    layer = layers[i]
    weight = weights[i]
    ffnn.add_layer(Layer(layer["number_of_neurons"], layer["activation_function"], np.array(weight[1:]), np.array(weight[0])))

input = model["case"]["input"]

output = ffnn.forward(input)
expected_output = model['expect']['output']

print(f'output: {output}')
print(f'expected output: {expected_output}')