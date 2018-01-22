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
        self.w_hidden_output = np.random.uniform(low=-1.0, high=1.0, size=(self.hidden_nodes, self.input_nodes))

        # setup the activation function
        self.activation_function = lambda x: special.expit(x) # using the sigmoid function

    # train the neural network
    def train(self):
        pass

    # query the neural network
    def query(self, input_list):
        # convert input list to a 2d array
        inputs = np.array(input_list, ndmin=2).T

        # calculate signals going into hidden layer
        hidden_inputs = np.dot(self.w_input_hidden, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals going into output layer
        final_inputs = np.dot(self.w_hidden_output, hidden_inputs)
        # calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# neural network hyperparameters
input_nodes, hidden_nodes, output_nodes = 3, 3, 3
alpha = 0.5

# create neural network
model = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)