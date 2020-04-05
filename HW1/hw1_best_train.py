
import sys
import pandas as pd
import numpy as np
import math 

train_path = sys.argv[1] #  './hw1_data/train.csv'
#train_path = './hw1_data/train.csv'

data = pd.read_csv(train_path, encoding = 'big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
        #sample[14, day * 24 : (day + 1) * 24] = sample[14, day * 24 : (day + 1) * 24] *6.28 / 360
        #sample[15, day * 24 : (day + 1) * 24] = sample[15, day * 24 : (day + 1) * 24] *6.28 / 360
    month_data[month] = sample
x = np.empty([12 * 471, 8 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            #x[month * 471 + day * 24 + hour, :1*9] = month_data[month][2:3,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1)
            x[month * 471 + day * 24 + hour, :5*9] = month_data[month][5:10,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1)
            x[month * 471 + day * 24 + hour, 5*9:] = month_data[month][12:15,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) 
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value
mean_x = np.mean(x, axis = 0) #14 * 9 
std_x = np.std(x, axis = 0) #14 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #14 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
np.save('best_train_mean.npy',mean_x)
np.save('best_train_std.npy',std_x)


train_x = math.floor(len(x) * 0.8)
train_y = math.floor(len(y) * 0.8)

x_train_set = x[: train_x, :]
y_train_set = y[: train_y, :]
x_validation = x[ train_x: , :]
y_validation = y[ train_y: , :]

dim = 4 * 8 * 9 + 1
#x_train_set = np.concatenate((x_train_set**2, x_train_set), axis = 1).astype(float)
x = np.concatenate((x, x ** 2, x ** 3, x ** 4), axis = 1).astype(float)
x = np.concatenate((np.ones([471 * 12, 1]), x), axis = 1).astype(float)
w = np.zeros([dim, 1])
learning_rate = 0.002
iter_time = 50000
m = np.zeros([dim, 1])
v = np.zeros([dim, 1])
beta1 = 0.9
beta2 = 0.999
eps = 0.00000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12) #rmse 
    if(t%1000==0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #+ lamda* 2 * np.sum(w) #dim*1
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * np.power(gradient, 2)
    m_t = m / (1 - beta1)
    v_t = v / (1 - beta2)
    w = w - learning_rate * m_t / np.sqrt(v_t) + eps
'''
# Validation
#x_validation = np.concatenate((x_validation**2, x_validation), axis = 1).astype(float)
x_validation = np.concatenate((np.ones([ len(x)-train_x, 1]), x_validation), axis = 1).astype(float)
ans_validation = np.dot(x_validation, w)
rmse = np.sqrt( np.sum(np.power(ans_validation - y_validation,2)) / (len(x)-train_x) )
print(rmse)
'''
np.save('weight.npy', w)




