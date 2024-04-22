import json
import numpy as np
from ffnn import FFNN
from layer import Layer

# model = open(f'../../Bagian-A/test/softmax.json', 'r')
# model = json.load(model)

# layers = model['case']['model']['layers']
# weights = model['case']['weights']

# ffnn = FFNN()
# for i in range (len(layers)):
#     layer = layers[i]
#     weight = weights[i]
#     ffnn.add_layer(Layer(layer["number_of_neurons"], layer["activation_function"], np.array(weight[1:]), np.array(weight[0])))

# input = model["case"]["input"]

# output = ffnn.forward(input)
# expected_output = np.array(model['expect']['output'])

# for i in range(len(output)):
#     print(f'output {i+1}: {output[i]}')
#     print(f'expected_output {i+1}: {expected_output[i]}')
#     print(f'output == expected_output: {np.allclose(output[i], expected_output[i], atol=model["expect"]["max_sse"])}')

#     print() if i != len(output)-1 else None

# region BAGIAN A
BAGIAN_A_TEST_CASES = {
    '1': '../../Bagian-B/test/linear.json',
    '2': '../../Bagian-B/test/reLU.json',
    '3': '../../Bagian-B/test/sigmoid.json',
    '4': '../../Bagian-B/test/softmax.json',
    '5': '../../Bagian-B/test/multilayer.json',
    '6': '../../Bagian-B/test/multilayer_softmax.json',
}

def start_test_case_a(test_case):
    model = open(test_case, 'r')
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

    print(f'\nResults ------------------\n')
    for i in range(len(output)):
        print(f'output {i+1}: {output[i]}')
        print(f'expected_output {i+1}: {expected_output[i]}')
        print(f'output == expected_output: {np.allclose(output[i], expected_output[i], atol=model["expect"]["max_sse"])}')

        print() if i != len(output)-1 else None
    print()
# endregion

# region BAGIAN B
BAGIAN_B_TEST_CASES = {
    '1': '../../Bagian-B/test/linear_small_lr.json',
    '2': '../../Bagian-B/test/linear_two_iteration.json',
    '3': '../../Bagian-B/test/linear.json',
    '4': '../../Bagian-B/test/mlp.json',
    '5': '../../Bagian-B/test/relu_b.json',
    '6': '../../Bagian-B/test/softmax.json',
    '7': '../../Bagian-B/test/softmax_two_layer.json',
}

def start_test_case_b(test_case):
    model = open(test_case, 'r')
    model = json.load(model)

    layers = model['case']['model']['layers']
    init_weights = model['case']['initial_weights']

    ffnn = FFNN()
    for i in range (len(layers)):
        layer = layers[i]
        weight = init_weights[i]
        if (i == len(layers) - 1):
            ffnn.add_layer(Layer(layer["number_of_neurons"], layer["activation_function"], np.array(weight[1:]), np.array(weight[0]), out=True))
        else:
            ffnn.add_layer(Layer(layer["number_of_neurons"], layer["activation_function"], np.array(weight[1:]), np.array(weight[0])))


    input = np.array(model["case"]["input"])
    target = np.array(model["case"]["target"])

    params = model["case"]["learning_parameters"]

    stop_cond = ffnn.fit(input, target, params["learning_rate"], params["batch_size"], params["max_iteration"], params["error_threshold"])

    excepted_weights = model["expect"]["final_weights"]
    excepted_stop_cond = model["expect"]["stopped_by"]

    print(f'\nResults ------------------')
    print(f'Stop condition: {stop_cond}')
    print(f'Expected stop condition: {excepted_stop_cond}')

    for i in range(len(ffnn.layers)):
        print(f'Layer {i+1} weights:')
        bias_weights = np.array([ffnn.layers[i].bias] + ffnn.layers[i].weights.tolist())
        print(bias_weights)
        print(f'Expected Layer {i+1} weights:')
        print(np.array(excepted_weights[i]))
    print()
# endregion

def main():
    print("\nArtificial Neural Network Trainer (˵ •̀ ᴗ •́ ˵ ) ✧ ")
    print("Credits : ")
    print("1. Diky Restu Maulana - 135210017")
    print("2. Muhammad Gilang Ramadhan - 13520137")
    print("3. Louis Caesa Kesuma - 13521069")
    print("4. Addin Munawwar Yusuf - 13521085\n")

    print("What are we going to do today?")
    print("1. Load FFNN model")
    print("2. Train FFNN model")

    # choice = input("Choose your path: ")
    choice = '2'
    while choice != '1' and choice != '2':
        print("Invalid choice. Please choose again.")
        choice = input("Choose your path: ")
    
    print()
    if choice == '1':
        l = len(BAGIAN_A_TEST_CASES) + 1
        print("Select from the following tests or input test file path manually")
        for key, value in BAGIAN_A_TEST_CASES.items():
            print(f"{key}. {value}")
        print(f"{l}. Input test file path manually")

        test_case = input("Choose test case: ")

        while test_case not in BAGIAN_A_TEST_CASES.keys() and test_case != str(l):
            print("Invalid choice. Please choose again.")
            test_case = input("Choose test case: ")

        if test_case == str(l):
            test_case = input("Input test file path: ")
            start_test_case_a(test_case)
            return

        start_test_case_a(BAGIAN_A_TEST_CASES[test_case])

    elif choice == '2':
        l = len(BAGIAN_B_TEST_CASES) + 1
        print("Select from the following tests or input test file path manually")
        for key, value in BAGIAN_B_TEST_CASES.items():
            print(f"{key}. {value}")
        print(f"{l}. Input test file path manually")

        test_case = input("Choose test case: ")
        # test_case = '3'

        while test_case not in BAGIAN_B_TEST_CASES.keys() and test_case != str(l):
            print("Invalid choice. Please choose again.")
            test_case = input("Choose test case: ")

        if test_case == str(l):
            test_case = input("Input test file path: ")
            start_test_case_b(test_case)
            return
        
        start_test_case_b(BAGIAN_B_TEST_CASES[test_case])
        
if __name__ == '__main__':
    main()