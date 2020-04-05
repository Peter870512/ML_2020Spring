import numpy as np
import pandas as pd
import os
import sys

np.random.seed(0)
X_train_fpath = sys.argv[3]   #'./X_train'
Y_train_fpath = sys.argv[4]   #'./Y_train'
X_test_fpath = sys.argv[5]    #'./X_test'
output_fpath = sys.argv[6]    #'./output.csv'

with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

train = pd.DataFrame(X_train)
binary = list()
for i in range(X_train.shape[1]):
    value_counts = train[i].value_counts().tolist()
    if len(value_counts) == 2:
        binary.append(i)
continuous = []
for i in range(510):
    if i not in binary:
        continuous.append(i)
print(continuous)

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]
# preprocessing
X_train = X_train[:,:508]
X_test = X_test[:,:508]
# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)
'''  
# Split data into training set and development set
dev_ratio = 0.15
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)
'''
# Continuous index (not one-hot encoding)
s_index = [0,126,210,211,212,358,507]
c_index = [126,210,211,212,358,507]
X_square = X_train[:,s_index] ** 2
#X_square = X_train ** 2
X_cubic = X_train[:,c_index] ** 3
X_train_new = np.concatenate((X_train,X_square,X_cubic),axis=1)
#print(X_train_new.shape[1])
'''
X_dev_square = X_dev[:,s_index] ** 2
#X_dev_square = X_dev ** 2
X_dev_cubic = X_dev[:,c_index] ** 3
X_dev_new = np.concatenate((X_dev,X_dev_square,X_dev_cubic),axis=1)
'''
X_test_square = X_test[:,s_index] ** 2
#X_test_square = X_test ** 2
X_test_cubic = X_test[:,c_index] ** 3
X_test_new = np.concatenate((X_test,X_test_square,X_test_cubic),axis=1) 


train_size = X_train_new.shape[0]
#dev_size = X_dev_new.shape[0]
test_size = X_test.shape[0]
data_dim = X_train_new.shape[1]


def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    #print(w.shape)
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)
    
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad


# Zero initialization for weights ans bias
w = np.zeros((data_dim,)) 
b = np.zeros((1,))

# Some parameters for training    
max_iter = 20000
batch_size = 50
learning_rate = 0.0001

# Keep the loss and accuracy at every iteration for plotting
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

m = np.zeros((data_dim,))
v = np.zeros((data_dim,))
m_b = np.zeros((1,))
v_b = np.zeros((1,))
beta1 = 0.9
beta2 = 0.999
eps = 0.00000001
lamda = 0.01 
# Calcuate the number of parameter updates
step = 1

# Iterative training
for epoch in range(max_iter):
    # Random shuffle at the begging of each epoch
    X_train_new, Y_train = _shuffle(X_train_new, Y_train)
    print("Epoch:",epoch)
    '''
    if epoch <= 70:
        learning_rate = 0.001
    elif epoch > 70 and epoch <= 600:
        learning_rate = 0.00015
    elif epoch > 600 and epoch <= 2000:
        learning_rate = 0.00003
    elif epoch > 2000 and epoch <= 9000:
        learning_rate = 0.00001
    else:
        learning_rate = 0.000001
    '''
    # Mini-batch training
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train_new[idx*batch_size:(idx+1)*batch_size]
        Y = Y_train[idx*batch_size:(idx+1)*batch_size]

        # Compute the gradient
        w_grad, b_grad = _gradient(X, Y, w, b)

        #w_grad += 2*np.sum(w) * lamda
        #b_grad += 2*np.sum(b) * lamda
        
        m = beta1 * m + (1 - beta1) * w_grad
        v = beta2 * v + (1 - beta2) * np.power(w_grad, 2)
        m_t = m / (1 - beta1)
        v_t = v / (1 - beta2)
        w = w - learning_rate * m_t / (np.sqrt(v_t) + eps)

        m_b = beta1 * m_b + (1 - beta1) * b_grad
        v_b = beta2 * v_b + (1 - beta2) * np.power(b_grad, 2)
        m_b_t = m_b / (1 - beta1)
        v_b_t = v_b / (1 - beta2)
        b = b - learning_rate * m_b_t / (np.sqrt(v_b_t) + eps)
        '''
        # gradient descent update
        # learning rate decay with time
        w = w - learning_rate/np.sqrt(step) * w_grad
        b = b - learning_rate/np.sqrt(step) * b_grad

        step = step + 1
        '''   
    # Compute loss and accuracy of training set and development set
    y_train_pred = _f(X_train_new, w, b)
    Y_train_pred = np.round(y_train_pred)
    '''
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)
    #print('Accuracy :',_accuracy(Y_train_pred, Y_train))
    '''
    print('Training Loss :',_cross_entropy_loss(y_train_pred, Y_train) / train_size)
    
    '''
    y_dev_pred = _f(X_dev_new, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)
    print('Validation Loss :',_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)
    '''
np.save('weight.npy',w)
np.save('bias.npy',b)
'''
print('Training loss: {}'.format(train_loss[-1]))
#print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
#print('Development accuracy: {}'.format(dev_acc[-1]))
'''
