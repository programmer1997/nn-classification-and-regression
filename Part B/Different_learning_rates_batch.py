import time
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(10)

epochs = 1000
batch_size = 32
no_hidden1 = 30 #num of neurons in hidden layer 1
learning_rates =[1e-3,0.5*1e-3,1e-4,0.5*1e-4,1e-5]
no_folds=5

floatX = theano.config.floatX

# scale and normalize input data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max - X_min)
 
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
trainX_max, trainX_min =  np.max(trainX, axis=0), np.min(trainX, axis=0)
testX_max, testX_min =  np.max(testX, axis=0), np.min(testX, axis=0)

trainX = scale(trainX, trainX_min, trainX_max)
testX = scale(testX, testX_min, testX_max)

trainX_mean, trainX_std = np.mean(trainX, axis=0), np.std(trainX, axis=0)
testX_mean, testX_std = np.mean(testX, axis=0), np.std(testX, axis=0)

trainX = normalize(trainX, trainX_mean, trainX_std)
testX = normalize(testX, testX_mean, testX_std)
#Function to set weights and biases
def set_bias(b, n = 1):
    b.set_value(np.zeros(n))

def set_weights(w, n_in=1, n_out=1, logistic=True):
 
    W_values = np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                 high=np.sqrt(6. / (n_in + n_out)),
                                 size=(n_in, n_out))
    if logistic == True:
        W_values *= 4
    w.set_value(W_values)

def bias_and_weight():
        w_o.set_value(np.random.uniform(low=-np.sqrt(6. / (no_hidden1 + 1)),
                                 high=np.sqrt(6. / (no_hidden1 + 1)),
                                 size=(no_hidden1,1)))
        b_o.set_value(theano.shared(np.random.randn()*.01, floatX))
        w_h1.set_value(np.random.uniform(low=-np.sqrt(6. / (8+no_hidden1)),
                                 high=np.sqrt(6. / (no_hidden1 + 8)),
                                 size=(8,no_hidden1)))
        b_h1.set_value(theano.shared(np.random.randn(no_hidden1)*0.01, floatX))
    

no_features = trainX.shape[1] 
x = T.matrix('x') # data sample
d = T.matrix('d') # desired output
no_samples = T.scalar('no_samples')


no_features = trainX.shape[1] 
x = T.matrix('x') # data sample
d = T.matrix('d') # desired output
no_samples = T.scalar('no_samples')

# initialize weights and biases for hidden layer(s) and output layer
w_o = theano.shared(np.random.randn(no_hidden1)*.01, floatX ) 
b_o = theano.shared(np.random.randn()*.01, floatX)
w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX )
b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)

# learning rate
alpha = theano.shared(1e-3, floatX) 


#Define mathematical expression:
h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
y = T.dot(h1_out, w_o) + b_o

cost = T.abs_(T.mean(T.sqr(d - y)))
accuracy = T.mean(d - y)

#define gradients
dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])

train = theano.function(
        inputs = [x, d],
        outputs = cost,
        updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h],
                   [b_h1, b_h1 - alpha*db_h]],
        allow_input_downcast=True
        )

test = theano.function(
    inputs = [x, d],
    outputs = [y, cost, accuracy],
    allow_input_downcast=True
    )


train_cost = np.zeros(epochs)
validation_cost = np.zeros(epochs)
test_accuracy = np.zeros(epochs)


min_error = 1e+15
best_iter = 0
best_w_o = np.zeros(no_hidden1)
best_w_h1 = np.zeros([no_features, no_hidden1])
best_b_o = 0
best_b_h1 = np.zeros(no_hidden1)

trainX,trainY=shuffle_data(trainX,trainY)
for fold in range(no_folds):
    print(fold)
    fold_cost=[]
 
    factor=trainX.shape[0]//5
    start,end=fold*factor, (fold+1)*factor
    validation_x,validation_y = trainX[start:end], trainY[start:end]
    train_x, train_y = np.append(trainX[:start], trainX[end:], axis=0), np.append(trainY[:start], trainY[end:], axis=0)
    plt.figure(1)
    plt.figure(2)
    for learning_rate in learning_rates:
        alpha.set_value(learning_rate)
        print(alpha.get_value())
        '''
        bias_and_weight()

        
        w_o = np.random.uniform(low=-np.sqrt(6. / (no_hidden1 + 1)),
                                 high=np.sqrt(6. / (no_hidden1 + 1)),
                                 size=(no_hidden1,1))
        b_o = theano.shared(np.random.randn()*.01, floatX)
        w_h1 = np.random.uniform(low=-np.sqrt(6. / (8+no_hidden1)),
                                 high=np.sqrt(6. / (no_hidden1 + 8)),
                                 size=(8,no_hidden1))
        b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)
        
        w_o = theano.shared(np.random.randn(no_hidden1)*.01, floatX ) 
        b_o = theano.shared(np.random.randn()*.01, floatX)
        w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX )
        b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)
        
        set_weights(w_o,no_hidden1,1)
        set_weights(w_h1,8,no_hidden1)
        set_bias(b_o)
        set_bias(b_h1,no_hidden1)
        '''
        w_o.set_value(np.random.randn(no_hidden1)*.01, floatX )
        w_h1.set_value(np.random.randn(no_features,no_hidden1)*.01, floatX )
        b_h1.set_value(np.random.randn(no_hidden1)*0.01, floatX)
        b_o.set_value(np.random.randn()*.01, floatX)
        idd = range(len(train_x))
        for iter in range(epochs):
            if iter%100:
                print(iter)
            train_x = train_x[idd,:]
            train_y = train_y[idd,:]
            no_batch=train_x.shape[0]//32
            batch_cost=np.empty((no_batch,1))
            for start, end in zip(range(0, len(train_x), 32), range(32, len(train_x), 32)):
               batch_cost[start//32] = train(train_x[start:end],np.transpose(train_y[start:end]))
            train_cost[iter]=np.mean(batch_cost)
            pred,validation_cost[iter],test_accuracy[iter]=test(validation_x,np.transpose(validation_y))
        plt.figure(1)
        plt.plot(range(epochs),train_cost)
        plt.xlabel("epoch")
        plt.ylabel("train cost")
        plt.title('Training error')
        plt.figure(2)
        plt.plot(range(epochs),validation_cost)
        plt.xlabel("epoch")
        plt.ylabel("validation cost")
        plt.title('validation error')
        
    plt.figure(1)
    plt.savefig('training_fold'+str(fold))
    plt.figure(2)
    plt.savefig('validation_fold'+str(fold))
'''
alpha.set_value(learning_rate)
print(alpha.get_value())

t = time.time()
for iter in range(epochs):
    if iter % 100 == 0:
        print(iter)
    
    trainX, trainY = shuffle_data(trainX, trainY)
    train_cost[iter] = train(trainX, np.transpose(trainY))
    pred, test_cost[iter], test_accuracy[iter] = test(testX, np.transpose(testY))

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
    
best_pred, best_cost, best_accuracy = test(testX, np.transpose(testY))

print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d'%(best_cost, best_accuracy, best_iter))

#Plots
plt.figure()
plt.plot(range(epochs), train_cost, label='train error')
plt.plot(range(epochs), test_cost, label = 'test error')
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Error')
plt.title('Training and Test Errors at Alpha = %.3f'%learning_rate)
plt.legend()
plt.savefig('p_1b_sample_mse.png')
plt.show()

plt.figure()
plt.plot(range(epochs), test_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.savefig('p_1b_sample_accuracy.png')
plt.show()
'''
