
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(10)

epochs = 1000
batch_size = 32
no_hidden1 = 60 #num of neurons in hidden layer 1

learning_rate =1e-5


floatX = theano.config.floatX

# scale and normalize input data
def normalize(X, X_mean, X_std):
    return (X - X_mean)/X_std

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

#read and divide data into test and train sets 
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

X_data, Y_data = shuffle_data(X_data, Y_data)

#separate train and test data
m = 3*X_data.shape[0] // 10
testX, testY = X_data[:m],Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

# scale and normalize data


trainX_mean, trainX_std = np.mean(trainX, axis=0), np.std(trainX, axis=0)
testX_mean, testX_std = np.mean(testX, axis=0), np.std(testX, axis=0)

trainX = normalize(trainX, trainX_mean, trainX_std)
testX = normalize(testX, testX_mean, testX_std)

    

no_features = trainX.shape[1] 
x = T.matrix('x') # data sample
d = T.matrix('d') # desired output
no_samples = T.scalar('no_samples')




# initialize the biases
b_o=theano.shared(np.zeros(1),floatX)
b_h1=theano.shared(np.zeros(no_hidden1),floatX)


#initialize the weights
w_o = theano.shared(np.random.randn(no_hidden1,1)*.01, floatX )
w_h1 = theano.shared(np.random.randn(no_features,no_hidden1)*.01, floatX )




# learning rate
alpha = theano.shared(learning_rate, floatX) 


#Define mathematical expression:
h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
y = T.dot(h1_out, w_o) + b_o

cost = T.abs_(T.mean(T.sqr(d - y)))
accuracy =T.mean(d - y)

#define gradients
dw_o, db_o, dw_h1, db_h1 = T.grad(cost, [w_o, b_o, w_h1, b_h1])


#Compile train and test functions
train = theano.function(
        inputs = [x, d],
        outputs =cost,
        updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h1],
                   [b_h1, b_h1 - alpha*db_h1]],
        allow_input_downcast=True
        )

test = theano.function(
    inputs = [x, d],
    outputs = [y, cost, accuracy],
    allow_input_downcast=True
    )

# matrices for training and test errors
train_cost = np.zeros(epochs)
test_cost = np.zeros(epochs)
test_accuracy = np.zeros(epochs)
validation_cost = np.zeros(epochs)
validation_accuracy = np.zeros(epochs)
trainX, trainY = shuffle_data(trainX, trainY)


min_error = 1e+15
best_iter = 0
best_w_o = np.zeros(no_hidden1)
best_w_h1 = np.zeros([no_features, no_hidden1])
best_b_o = 0
best_b_h1 = np.zeros(no_hidden1)

for fold in range(no_folds):
    print("fold="+str(fold))
    factor=trainX.shape[0]//5
    start,end=fold*factor, (fold+1)*factor
    validation_x,validation_y = trainX[start:end], trainY[start:end]
    train_x, train_y = np.append(trainX[:start], trainX[end:], axis=0), np.append(trainY[:start], trainY[end:], axis=0)

    for iter in range(epochs):
        if iter % 100 == 0:
            print(iter)
        
        
        train_cost[iter] = train(train_x,train_y)
        pred, test_cost[iter], test_accuracy[iter] = test(testX,testY)

        # This is to find out the best weights and biases
        
        if test_cost[iter] < min_error:
            best_iter = iter
            min_error = test_cost[iter]
            best_w_o = w_o.get_value()
            best_w_h1 = w_h1.get_value()
            best_b_o = b_o.get_value()
            best_b_h1 = b_h1.get_value()

#set weights and biases to values at which performance was best
w_o.set_value(best_w_o)
b_o.set_value(best_b_o)
w_h1.set_value(best_w_h1)
b_h1.set_value(best_b_h1)

# Equation for prediction
best_pred, best_cost, best_accuracy = test(testX,testY)

print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d'%(best_cost, best_accuracy, best_iter))

#Plots
plt.figure()
plt.plot(range(epochs), train_cost, label='train error')
plt.plot(range(epochs), test_cost, label = 'test error')
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Error')
plt.title('Training and validation error')
plt.legend()
plt.savefig('3a.png')
plt.show()

plt.figure()
plt.plot(range(epochs), test_accuracy)
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.title('accuracy')
plt.savefig('3b.png')
plt.show()



