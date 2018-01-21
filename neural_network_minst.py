import numpy as np
from scipy import special

# neural network class definition
class NeuralNetwork:

    # initialise the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # set the number of nodes in layers
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # learning rate
        self.learning_rate = learning_rate

        # setup weight matrices for links between layers, w_input_hidden and w_hidden_output
        # weights inside arrays are w_i_j where link is from node i to node j in the next layer
        # example: w_2_1 is weight moderating signal from node 2 to node 1 in the next layer
        self.w_input_hidden = np.random.uniform(low=-1.0, high=1.0, size=(self.hidden_nodes, self.input_nodes))
        self.w_output_hidden = np.random.uniform(low=-1.0, high=1.0, size=(self.hidden_nodes, self.input_nodes))


    # train the neural network
    def train(self):
        pass

    # query the neural network
    def query(self):
        pass

# neural network hyperparameters
input_nodes, hidden_nodes, output_nodes = 3, 3, 3
alpha = 0.5

# create neural network
model = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)