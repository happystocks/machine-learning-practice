import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.common import global_object
from src.common.constants import *


def get_x_y(data, N, offset=0):
    """
    Split data into x (features) and y (target)
    """
    x, y = [], []
    for i in range(offset, len(data)):
        x.append(data[i-N:i])
        y.append(data[i])
    x = np.array(x)
    y = np.array(y)

    return x, y


def pre_process():
    num_cv = int(cv_size*len(global_object.stock_data))
    num_test = int(test_size*len(global_object.stock_data))
    num_train = len(global_object.stock_data) - num_test - num_cv
    print("num_train = " + str(num_train))
    print("num_cv = " + str(num_cv))
    print("num_test = " + str(num_test))

    # Split into train, cv, and test
    train = global_object.stock_data[:num_train][['Date', 'High']]
    # cv = global_object.stock_data[num_train:num_train+num_cv][['Date', 'High']]
    # train_cv = global_object.stock_data[:num_train+num_cv][['Date', 'High']]
    test = global_object.stock_data[num_train+num_cv:][['Date', 'High']]

    print("train.shape = " + str(train.shape))
    # print("cv.shape = " + str(cv.shape))
    # print("train_cv.shape = " + str(train_cv.shape))
    print("test.shape = " + str(test.shape))

    # Number of input time-steps
    N = 3
    # Converting dataset into x_train and y_train
    # Here we only scale the train dataset, and not the entire dataset to prevent information leak
    global_object.scalar = StandardScaler()
    train_scaled = np.array(train['High']).reshape(-1, 1)
    # print("scaler.mean_ = " + str(global_object.scalar.mean_))
    # print("scaler.var_ = " + str(global_object.scalar.var_))
    print("Length of trained data :: ", len(train_scaled))

    # Split train into x and y
    x_train_scaled, y_train_scaled = get_x_y(train_scaled, N, offset=N)

    # test_transformed = global_object.scalar.transform(np.array(test['High']).reshape(-1, 1))
    test_transformed = np.array(test['High']).reshape(-1, 1)
    x_test_scaled, y_test = [], []
    for i in range(N, len(test_transformed)):
        x_test_scaled.append(test_transformed[i-N: i])
        y_test.append(np.array(test['High']).reshape(-1, 1)[i])

    x_test_scaled = np.array(x_test_scaled)
    y_test = np.array(y_test)

    print("Shape of x_test::: ", x_test_scaled.shape)

    return x_train_scaled, y_train_scaled, x_test_scaled, y_test
