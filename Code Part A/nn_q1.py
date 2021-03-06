import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import time
import datetime

DECAY_PARAM = 1e-6
LEARNING_RATE = 0.01
no_of_epochs = 1000
batch_size = 32
no_of_input_nodes = 36
no_of_hidden_layer_neurons = 10
no_of_output_neurons = 6
input_file_training = 'sat_train.txt'
input_file_testing = 'sat_test.txt'


def init_bias(no_of_neurons=1):
    """ Set initial values for the bias """

    return theano.shared(np.zeros(no_of_neurons), theano.config.floatX)


def get_initial_weights(no_of_inputs=1, no_of_neurons=1, is_logistic=True):
    """ Get the initial values for the weights """

    # At initialization it is desirable for the weights to be uniformly distributed
    # within the given limits
    weight_values = np.asarray( np.random.uniform(
                          low = -np.sqrt(6.0 / (no_of_inputs + no_of_neurons)),
                          high = np.sqrt(6.0 / (no_of_inputs + no_of_neurons)),
                          size=(no_of_inputs, no_of_neurons)
                        ),
                        dtype=theano.config.floatX
                    )

    # The limits are different in case of the logistic activation function 
    if is_logistic:
        weight_values *= 4

    return weight_values


def init_weights(no_of_inputs=1, no_of_neurons=1, is_logistic=True):
    """ Set initial values for the weights """

    weights = get_initial_weights(no_of_inputs, no_of_neurons)
    return theano.shared(value=weights, name='W', borrow=True)


def normalize_inputs(x, x_min, x_max):
    """ Return the normalized inputs """

    # Scaling to achieve a better approximation of inputs
    return (x - x_min)/(x_max - x_min)


def update_params(cost, params, learning_rate=0.01):
    """ Return the updated weights and biases """

    # In Gradient descent learning the parameters are updated 
    # proportional to the negative gradient of the cost function 
    gradients = T.grad(cost=cost, wrt=params)
    updated_params = []
    for p, g in zip(params, gradients):
        updated_params.append([p, p - g*learning_rate])

    return updated_params


def shuffle_data(input_samples, labels):
    """ Shuffle and return the data for training """

    indexes = np.arange(input_samples.shape[0])
    np.random.shuffle(indexes)

    return input_samples[indexes], labels[indexes]


def get_data(filepath):
    # Read the training data
    input_arr = np.loadtxt(filepath, delimiter=' ')
    # Get all the input patterns i.e. the first 36 elements of each vector (indexes 0 to 35)
    input_patterns = input_arr[:, :36]
    # Get the normalized input array
    normalized_input_matrix = normalize_inputs(input_patterns, np.min(input_patterns, axis=0), 
                                         np.max(input_patterns, axis=0))

    # Get the class labels
    class_labels = input_arr[:, -1].astype(int)
    # Wherever the class label is 7, change it to 6
    class_labels[class_labels == 7] = 6
    # Initialize the desired output matrix
    desired_output_matrix = np.zeros((class_labels.shape[0], 6))
    # Create the desired output matrix, with each row a one-hot vector
    desired_output_matrix[np.arange(class_labels.shape[0]), class_labels - 1] = 1

    return normalized_input_matrix, desired_output_matrix


x = T.matrix() # Input matrix
y = T.matrix() # Output

# Initial weights and biases from the input layer to the hidden layer
w1 = init_weights(no_of_input_nodes, no_of_hidden_layer_neurons)
b1 = init_bias(no_of_hidden_layer_neurons) 

# Initial weights and biases from the hidden layer to the output layer
w2 = init_weights(no_of_hidden_layer_neurons, no_of_output_neurons, is_logistic=False)
b2 = init_bias(no_of_output_neurons)

# Given that the hidden layer has a logistic activation function and
# the output layer is a softmax layer, find the outputs
output_hidden_layer = T.nnet.sigmoid(T.dot(x, w1) + b1)
output_output_layer = T.nnet.softmax(T.dot(output_hidden_layer, w2) + b2)

output_class = T.argmax(output_output_layer, axis=1)

# Compute the training cost
cost = T.mean(T.nnet.categorical_crossentropy(output_output_layer, y)) \
              + DECAY_PARAM*(T.sum(T.sqr(w1)+T.sum(T.sqr(w2))))
params = [w1, b1, w2, b2]
updated_params = update_params(cost, params, LEARNING_RATE)

# Compile the functions
train = theano.function(inputs=[x, y], outputs=cost, updates=updated_params, allow_input_downcast=True)
predict = theano.function(inputs=[x], outputs=output_class, allow_input_downcast=True)

train_x, train_y = get_data(input_file_training)
test_x, test_y = get_data(input_file_testing)
no_input_patterns = len(train_x) # 4435
    
training_costs = []
test_accuracy = []

# Set the weights and biases to initial values
w1.set_value(get_initial_weights(no_of_input_nodes, no_of_hidden_layer_neurons))
b1.set_value(np.zeros(no_of_hidden_layer_neurons), theano.config.floatX)
w2.set_value(get_initial_weights(no_of_hidden_layer_neurons, no_of_output_neurons, is_logistic=False))
b2.set_value(np.zeros(no_of_output_neurons), theano.config.floatX)

for x in range(no_of_epochs):
    # Shuffle train_x (inputs) and train_y (desired outputs) correspondingly
    train_x, train_y = shuffle_data(train_x, train_y)
    training_cost = 0.0

    # Train the neural network using mini batch gradient descent
    for batch_start_idx, batch_end_idx in zip(range(0, no_input_patterns, batch_size), range(batch_size, no_input_patterns, batch_size)):
        training_cost += train( train_x[batch_start_idx:batch_end_idx], train_y[batch_start_idx:batch_end_idx] )

    # Calculate the average training cost for each epoch
    training_costs = np.append(training_costs, training_cost/(no_input_patterns // batch_size))
    print('Epoch {}'.format(str(x+1)))
 
    # Calculate the mean test accuracy for each epoch
    test_accuracy = np.append(test_accuracy, np.mean(np.argmax(test_y, axis=1) == predict(test_x)))


# Plot the graphs
plt.figure(1)
plt.plot(range(no_of_epochs), training_costs)
plt.xlabel('No. of Iterations')
plt.ylabel('Training Error')
plt.title('Training Error vs Iterations for 4-layer network')
plt.savefig('q1a_training_error_vs_iterations.png')

plt.figure(2)
plt.plot(range(no_of_epochs), test_accuracy)
plt.xlabel('No. of Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy vs No. of Iterations for 4-layer network')
plt.savefig('q1a_test_accuracy_vs_iterations.png')