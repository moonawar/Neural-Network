import json
import numpy as np
from ffnn import FFNN
from layer import Layer

model = open(f'../../Bagian-A/test/softmax.json', 'r')
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
expected_output = np.array(model['expect']['output'])

for i in range(len(output)):
    print(f'output {i+1}: {output[i]}')
    print(f'expected_output {i+1}: {expected_output[i]}')
    print(f'output == expected_output: {np.allclose(output[i], expected_output[i], atol=1e-4)}')

    print() if i != len(output)-1 else None