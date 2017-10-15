import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

DECAY_PARAM = 1e-6
LEARNING_RATE = 0.01
EPOCHS = 1000
batch_size = 32
NO_OF_INPUT_NODES = 36
NO_OF_HIDDEN_LAYER_NEURONS = 10
NO_OF_OUTPUT_LAYER_NEURONS = 6
TRAINING_INPUT_FILE = 'sat_train.txt'


def init_bias(no_of_neurons=1):
	""" Set initial values for the bias """

	return theano.shared(np.zeros(no_of_neurons), theano.config.floatX)


def init_weights(no_of_inputs=1, no_of_neurons=1, is_logistic=True):
    """ Set initial values for the weights """

    # At initialization it is desirable for the weights to be uniformly distributed
    # within the given limits
    weights = np.asarray( np.random.uniform(
        				  low = -np.sqrt(6.0 / (no_of_inputs + no_of_neurons)),
        				  high = np.sqrt(6.0 / (no_of_inputs + no_of_neurons)),
        				  size=(no_of_inputs, no_of_neurons)
        				),
        				dtype=theano.config.floatX
        			)

    # The limits are different in case of the logistic activation function 
    if is_logistic:
        weights *= 4

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


# Define Theano symbolic variables
X = T.matrix() # Input matrix
Y = T.matrix() # Output

# Initial weights and biases from the input layer to the hidden layer
w1 = init_weights(NO_OF_INPUT_NODES, NO_OF_HIDDEN_LAYER_NEURONS)
b1 = init_bias(NO_OF_HIDDEN_LAYER_NEURONS) 

# Initial weights and biases from the hidden layer to the output layer
w2 = init_weights(NO_OF_HIDDEN_LAYER_NEURONS, NO_OF_OUTPUT_LAYER_NEURONS, is_logistic=False)
b2 = init_bias(NO_OF_OUTPUT_LAYER_NEURONS)

''' Given that the hidden layer has a logistic activation function and
	the output layer is a softmax layer, find the outputs '''
output_hidden_layer = T.nnet.sigmoid(T.dot(X, w1) + b1)
output_output_layer = T.nnet.softmax(T.dot(output_hidden_layer, w2) + b2)

output_class = T.argmax(output_output_layer, axis=1)

# understand this
cost = T.mean(T.nnet.categorical_crossentropy(output_output_layer, Y)) \
			  + DECAY_PARAM*(T.sum(T.sqr(w1)+T.sum(T.sqr(w2))))
params = [w1, b1, w2, b2]
updated_params = update_params(cost, params, LEARNING_RATE)

# Compile the functions
train = theano.function(inputs=[X, Y], outputs=cost, updates=updated_params, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=output_class, allow_input_downcast=True)

# Read the training data
train_input = np.loadtxt(TRAINING_INPUT_FILE, delimiter=' ')
# Get all the input patterns i.e. the first 36 elements of each vector (indexes 0 to 35)
trainX = train_input[:,:36]
# Get the input pattern (row) having the minimum input pattern values
trainX_min = np.min(trainX, axis=0)
# Get the input pattern (row) having the maximum input pattern values
trainX_max = np.max(trainX, axis=0)
# Get the normalized input array
trainX = normalize_inputs(trainX, trainX_min, trainX_max)

# Get the class labels
train_Y = train_input[:,-1].astype(int)
# Wherever the class label is 7, change it to 6
train_Y[train_Y == 7] = 6
# Initialize the desired output matrix
trainY = np.zeros((train_Y.shape[0], 6))
# Create the desired output matrix, with each row a one-hot vector
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1

# Read the test data
test_input = np.loadtxt('sat_test.txt',delimiter=' ')
testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)

testX_min, testX_max = np.min(testX, axis=0), np.max(testX, axis=0)
testX = normalize_inputs(testX, testX_min, testX_max)

test_Y[test_Y == 7] = 6
testY = np.zeros((test_Y.shape[0], 6))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1

# Train and test the neural network
no_input_patterns = len(trainX) # 4435
test_accuracy = []
train_cost = []

for _ in range(EPOCHS):
    # Shuffle trainX (inputs) and trainY (desired outputs) correspondingly
    trainX, trainY = shuffle_data(trainX, trainY)
    cost = 0.0
    # Train the neural network using mini batch gradient descent
    for start, end in zip(range(0, no_input_patterns, batch_size), range(batch_size, no_input_patterns, batch_size)):
        cost += train(trainX[start:end], trainY[start:end])
    
    # Calculate the average training cost for each epoch
    train_cost = np.append(train_cost, cost/(no_input_patterns // batch_size))
 
    # Calculate the mean test accuracy for each epoch
    test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))

print('%.1f accuracy at %d iterations'%(np.max(test_accuracy)*100, np.argmax(test_accuracy)+1))

#Plots
plt.figure()
plt.plot(range(EPOCHS), train_cost)
plt.xlabel('iterations')
plt.ylabel('cross-entropy')
plt.title('training cost')
plt.savefig('p1a_sample_cost.png')

plt.figure()
plt.plot(range(EPOCHS), test_accuracy)
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.title('test accuracy')
plt.savefig('p1a_sample_accuracy.png')

plt.show()


