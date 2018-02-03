import numpy as np
from scipy import special

train_data_loc = "mnist_dataset/mnist_train.csv"
test_data_loc = "mnist_dataset/mnist_test.csv"

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
        self.w_input_hidden = np.random.normal(0.0, pow(self.input_nodes, -0.5), size=(self.hidden_nodes, self.input_nodes))
        self.w_hidden_output = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), size=(self.output_nodes, self.hidden_nodes))

        # setup the activation function
        self.activation_function = lambda x: special.expit(x) # using the sigmoid function

    # train the neural network
    def train(self, inputs, targets):
        # convert the inputs and targets into  2d arrays
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        # work out the output for training examples
        # calculate signals going into hidden layer
        hidden_inputs = np.dot(self.w_input_hidden, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals going into final layer
        final_inputs = np.dot(self.w_hidden_output, hidden_outputs)
        # calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # update weights based on error between prediction and target
        # calculate error i.e. desired target output - calculated prediction
        output_errors = targets - final_outputs

        # hidden layer error derived by taking output errors split by weights and recombined at hidden nodes
        hidden_errors = np.dot(self.w_hidden_output.T, output_errors)

        # update the hidden weights for links between hidden and output layers
        self.w_hidden_output += self.learning_rate * np.dot(output_errors * final_outputs * (1.0 - final_outputs),
                                                           np.transpose(hidden_outputs))
        # update the weights for links between input and hidden layer
        self.w_input_hidden += self.learning_rate * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs),
                                                           np.transpose(inputs))
        
    # query the neural network
    def query(self, input_list):
        # convert input list to a 2d array
        inputs = np.array(input_list, ndmin=2).T

        # calculate signals going into hidden layer
        hidden_inputs = np.dot(self.w_input_hidden, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals going into output layer
        final_inputs = np.dot(self.w_hidden_output, hidden_outputs)
        # calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# neural network hyperparameters
input_nodes, hidden_nodes, output_nodes = 784, 100, 10
alpha = 0.1

# create neural network
mnist_neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, alpha)

# load MNIST training data from CSV
with open(train_data_loc) as f:
    training_data = f.readlines()

# train the neural network
epochs = 7 # epochs is number of times training data set is used for training
for e in range(epochs):

    # training run
    for record in training_data:
        # split record strings into individual values
        all_values = record.split(',')
        # scale data to avoid network large weights and saturated network
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create target output values (0.01 for all except desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is target label for current record
        targets[int(all_values[0])] = 0.99
        mnist_neural_network.train(inputs, targets)

# load MNIST test data from CSV
with open(test_data_loc) as f:
    test_data = f.readlines()

# test the neural network

# score_card keeps track of neural network performance
score_card = []

# loop over all test data records
for record in test_data:
    # split record strings into individual values
    test_values = record.split(',')
    # extract correct value which is first value in record
    correct_label = int(test_values[0])
    # scale data to avoid network large weights and saturated network
    test_inputs = (np.asfarray(test_values[1:]) / 255.0 * 0.99) + 0.01

    # get predictions based on test_input values from trained neural network
    predicted_outputs = mnist_neural_network.query(test_inputs)
    # get predicted label which is the maximum of predicted_outputs
    predicted_label = np.argmax(predicted_outputs)

    # classify each prediction as correct/incorrect by comparing correct_label (actual) to predicted_label
    if predicted_label == correct_label:
        score_card.append(1) # network answer matches correct answer
    else:
        score_card.append(0) # network answer does not match correct answer

    # all_values[0] is target label for current record
    targets[int(all_values[0])] = 0.99
    mnist_neural_network.train(inputs, targets)

# calculate performance
score_card_array = np.asfarray(score_card)
print("\nperformance = ", score_card_array.sum() / score_card_array.size)