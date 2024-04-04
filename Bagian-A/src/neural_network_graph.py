import networkx as nx

class NeuralNetworkGraph:
    def __init__(self):
        self.graphs = nx.DiGraph()
        self.node_labels = {}

    def add_nodes_layer(self, nodes, layer):
        self.graphs.add_nodes_from(nodes, layer=layer)

    def add_edge(self, node_source, node_goal):
        self.graphs.add_edge(node_source, node_goal)

    def add_all_nodes(self, node_bobots):

        # Add input bias label
        self.node_labels[0] = "Input"+str(0)+": "+str(1)

        node_number = 1

        self.add_nodes_layer([i for i in range(len(node_bobots[0])+1)], 0)

        # Add input label
        for i, node_value in enumerate(node_bobots[0]):
            self.node_labels[node_number] = "Input"+str(i+1)+": "+str(node_value)
            node_number += 1

        # Add nodes for each layer with the subset key
        for i in range(1, len(node_bobots)-1):
            self.node_labels[node_number] = "H"+str(i)+str(0)+": "+str(1)
            self.add_nodes_layer([node_number+j for j in range(len(node_bobots[i])+1)], i)
            node_number += 1
            for j in range(len(node_bobots[i])):
                self.node_labels[node_number] = "H"+str(i)+str(j+1)+": "+str(node_bobots[i][j])
                node_number += 1

        self.add_nodes_layer([node_number+j for j in range(len(node_bobots[-1]))], len(node_bobots))

        # Add output nodes
        for i, node_value in enumerate(node_bobots[-1]):
            self.node_labels[node_number] = "Output"+str(i+1)+": "+str(node_value)
            node_number += 1

    def add_all_edges(self, node_bobots):
        number_of_prev_neuron = 0
        curr_number_neuron = 0

        # Add edge for each node of input layer and hidden layer
        for layer_number in range(len(node_bobots)-2):
            curr_number_neuron += len(node_bobots[layer_number])+1
            for i in range(len(node_bobots[layer_number])+1):
                for j in range(len(node_bobots[layer_number+1])):
                    self.add_edge(number_of_prev_neuron, curr_number_neuron+j+1)
                number_of_prev_neuron += 1

        # Add edge for each node on output layer
        curr_number_neuron += len(node_bobots[-2])+1
        for i in range(len(node_bobots[-2])+1):
            for j in range(len(node_bobots[-1])):
                self.add_edge(number_of_prev_neuron, curr_number_neuron+j)
            number_of_prev_neuron += 1