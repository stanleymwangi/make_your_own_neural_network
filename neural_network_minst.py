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