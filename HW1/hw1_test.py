import sys
import pandas as pd
import numpy as np
import os

test_path = sys.argv[1] # './test.csv'
output_path = sys.argv[2] # './output.csv'
output_dir = os.path.split(output_path)[0]
if os.path.exists(output_dir):
    print("Exist")
else:
    os.mkdir(output_dir)

'''
test_path = './hw1_data/test.csv'
output_path = './ans.csv'
'''
mean_x = np.load('train_mean.npy') 
std_x = np.load('train_std.npy') 

testdata = pd.read_csv(test_path, header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 8*9], dtype = float)
for i in range(240):
    #test_x[i, : 1 * 9] = test_data[18 * i + 2: 18* i + 3, :].reshape(1, -1)
    test_x[i, : 5 * 9] = test_data[18 * i + 5: 18* i + 10, :].reshape(1, -1)
    test_x[i, 5 * 9:] = test_data[18 * i + 12: 18* i + 15, :].reshape(1, -1)
    #test_x[i, 9 * 12:] = test_x[i, 9 * 12:] * 6.28/360
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]

#test_x = np.concatenate((test_x, test_x ** 2, test_x ** 3), axis = 1).astype(float)
test_x = np.concatenate((test_x, test_x ** 2), axis = 1).astype(float)
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
w = np.load('train_weight.npy')
ans_y = np.dot(test_x, w)
import csv
with open(output_path, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        if ans_y[i][0] < 0:
            ans_y[i][0] = 0
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
