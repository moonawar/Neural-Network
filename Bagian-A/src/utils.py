import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt
from neural_network_graph import NeuralNetworkGraph
from ffnn import *

# for visualizing the model
def visualizeModel(ffnn, weights):
    # Initialize a directed graph for visualization
    node_bobots = ffnn.node_bobots
    batch_count = len(node_bobots)
    for batch_num in range(batch_count):
        neural_network_graph = NeuralNetworkGraph()

        neural_network_graph.add_all_nodes(node_bobots[batch_num])

        neural_network_graph.add_all_edges(node_bobots[batch_num])

        # Assuming neural_network_graph.graphs is your graph object
        neural_network_graphs = neural_network_graph.graphs  # Assuming this is a correct reference

        # Add 'layer' attribute to nodes if it's missing
        for node in neural_network_graphs.nodes():
            if 'layer' not in neural_network_graphs.nodes[node]:
                neural_network_graphs.nodes[node]['layer'] = 0  # Set a default layer if needed

        # Plot the neural network structure
        pos = nx.multipartite_layout(neural_network_graphs, subset_key="layer", align='horizontal')
        nx.draw(neural_network_graphs, pos, with_labels=True, labels=neural_network_graph.node_labels, node_size=2000, node_color="lightblue", font_size=7, font_weight="bold")

        # Add edge labels for better understanding
        edge_labels = {}
        index = 0
        sub_index = 0
        sub_sub_index = 0
        for u, v in neural_network_graphs.edges():
            edge_labels[(u,v)] = "W: "+str(weights[index][sub_index][sub_sub_index])
            if(sub_sub_index+1<len(weights[index][sub_index])):
                sub_sub_index += 1
            else:
                sub_sub_index = 0
                if(sub_index+1<len(weights[index])):
                    sub_index += 1
                else:
                    sub_index = 0
                    index += 1
        print("==========================================================================================================================================================================================================================")
        print("Edge labels: ",edge_labels)
        print("==========================================================================================================================================================================================================================")
        nx.draw_networkx_edge_labels(neural_network_graphs, pos, edge_labels=edge_labels, font_size=7, font_color='red')

        plt.title("Neural Network Structure (Input to Output)")
        plt.axis('off')
        plt.show()


# with saved model
def calculateWithSavedModel(fileName):
    model = open(f'../../Bagian-A/test/{fileName}.json', 'r')
    model = json.load(model)

    # loading the saved models (it contains the layers and weights)
    savedModel = open(f'../../Bagian-A/model/{fileName}_latest_weights_and_structures.json', 'r')
    savedModel = json.load(savedModel)

    layers = savedModel['case']['model']['layers']
    weights = savedModel['case']['weights']

    ffnn = FFNN()
    for i in range (len(layers)):
        layer = layers[i]
        weight = weights[i]
        ffnn.add_layer(Layer(layer["number_of_neurons"], layer["activation_function"], np.array(weight[1:]), np.array(weight[0])))

    input = model["case"]["input"]

    output = ffnn.forward(input).tolist()
    expected_output = model['expect']['output']
    print("==========================================================================================================================================================================================================================")
    print("Node values: ", ffnn.node_bobots)
    print(f'output: {output}')
    print(f'expected output: {expected_output}')
    print("==========================================================================================================================================================================================================================")

    visualizeModel(ffnn, weights)

# without using saved model (contains weights and structure)
def calculateWithoutSavedModel(fileName):
    model = open(f'../../Bagian-A/test/{fileName}.json', 'r')
    model = json.load(model)

    layers = model['case']['model']['layers']
    weights = model['case']['weights']

    # for saving model
    layers_dict = []
    weights_dict = []

    ffnn = FFNN()
    for i in range (len(layers)):
        layer = layers[i]
        weight = weights[i]

        # add the layers and weights
        layers_dict.append(layer)
        weights_dict.append(weight)

        ffnn.add_layer(Layer(layer["number_of_neurons"], layer["activation_function"], np.array(weight[1:]), np.array(weight[0])))

    saveModel(weights_dict, layers_dict, fileName) # save the model

    input = model["case"]["input"]

    output = ffnn.forward(input).tolist()
    expected_output = model['expect']['output']
    print("==========================================================================================================================================================================================================================")
    print("Node values: ", ffnn.node_bobots)

    print(f'output: {output}')
    print(f'expected output: {expected_output}')
    print("==========================================================================================================================================================================================================================")

    visualizeModel(ffnn, weights)

# for saving the model
def saveModel(weights, layers, fileName):
    case_dict = {'case': {'weights': weights, 'model': {'layers': layers}}}
    with open(f"../../Bagian-A/model/{fileName}_latest_weights_and_structures.json", 'w') as outfile:
        json.dump(case_dict, outfile, indent=4)