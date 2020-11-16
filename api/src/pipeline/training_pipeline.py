import pandas as pd
import numpy as np
from datetime import date
from matplotlib import pyplot as plt
from numpy.random import seed
from pylab import rcParams
from sklearn.metrics import mean_squared_error
# from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
# from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import plot_model
import tensorflow
import plotly
import math

from src.common.constants import *
from src.common import global_object
from src.data_processing.pre_process import *


def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_x_y(data, N, offset):
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


def get_x_scaled_y(data, N, offset):
    """
    Split data into x (features) and y (target)
    We scale x to have mean 0 and std dev 1, and return this.
    We do not scale y here.
    Inputs
        data     : pandas series to extract x and y
        N
        offset
    Outputs
        x_scaled : features used to predict y. Scaled such that each element has mean 0 and std dev 1
        y        : target values. Not scaled
        mu_list  : list of the means. Same length as x_scaled and y
        std_list : list of the std devs. Same length as x_scaled and y
    """
    x_scaled, y, mu_list, std_list = [], [], [], []
    for i in range(offset, len(data)):
        mu_list.append(np.mean(data[i-N:i]))
        std_list.append(np.std(data[i-N:i]))
        x_scaled.append((data[i-N:i]-mu_list[i-offset])/std_list[i-offset])
        y.append(data[i])
    x_scaled = np.array(x_scaled)
    y = np.array(y)

    return x_scaled, y, mu_list, std_list


def train_pred_eval_model(x_train_scaled, \
                          y_train_scaled, \
                          x_cv_scaled, \
                          y_cv, \
                          mu_cv_list, \
                          std_cv_list, \
                          lstm_units=50, \
                          dropout_prob=0.5, \
                          optimizer='adam', \
                          epochs=1, \
                          activation='tanh', \
                          recurrent_activation='sigmoid', \
                          batch_size=1):
    '''
    Train model, do prediction, scale back to original range and do evaluation
    Use LSTM here.
    Returns rmse, mape and predicted values
    Inputs
        x_train_scaled  : e.g. x_train_scaled.shape=(451, 9, 1). Here we are using the past 9 values to predict the next value
        y_train_scaled  : e.g. y_train_scaled.shape=(451, 1)
        x_cv_scaled     : use this to do predictions
        y_cv            : actual value of the predictions
        mu_cv_list      : list of the means. Same length as x_scaled and y
        std_cv_list     : list of the std devs. Same length as x_scaled and y
        lstm_units      : lstm param
        dropout_prob    : lstm param
        optimizer       : lstm param
        epochs          : lstm param
        batch_size      : lstm param
    Outputs
        rmse            : root mean square error
        mape            : mean absolute percentage error
        est             : predictions
    '''
    # Create the LSTM network
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=(x_train_scaled.shape[1],1),
                   activation=activation, recurrent_activation=recurrent_activation))
    model.add(Dropout(dropout_prob)) # Add dropout with a probability of 0.5
#     model.add(LSTM(units=lstm_units))
#     model.add(Dropout(dropout_prob)) # Add dropout with a probability of 0.5
    model.add(Dense(1))

    # Compile and fit the LSTM network
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.fit(x_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

    # Do prediction
    est_scaled = model.predict(x_cv_scaled)
    est = (est_scaled * np.array(std_cv_list).reshape(-1,1)) + np.array(mu_cv_list).reshape(-1,1)

    # Calculate RMSE and MAPE
#     print("x_cv_scaled = " + str(x_cv_scaled))
#     print("est_scaled = " + str(est_scaled))
#     print("est = " + str(est))
    rmse = math.sqrt(mean_squared_error(y_cv, est))
    mape = get_mape(y_cv, est)

    return rmse, mape, est


def training_pipeline():
    # num_cv = int(cv_size*len(global_object.stock_data))
    # num_test = int(test_size*len(global_object.stock_data))
    # num_train = len(global_object.stock_data) - num_cv - num_test
    # print("num_train = " + str(num_train))
    # print("num_cv = " + str(num_cv))
    # print("num_test = " + str(num_test))
    #
    # # Split into train, cv, and test
    # train = global_object.stock_data[:num_train][['Date', 'High']]
    # cv = global_object.stock_data[num_train:num_train+num_cv][['Date', 'High']]
    # train_cv = global_object.stock_data[:num_train+num_cv][['Date', 'High']]
    # test = global_object.stock_data[num_train+num_cv:][['Date', 'High']]
    #
    # print("train.shape = " + str(train.shape))
    # print("cv.shape = " + str(cv.shape))
    # print("train_cv.shape = " + str(train_cv.shape))
    # print("test.shape = " + str(test.shape))
    #
    # # Number of input time-steps
    # N = 3
    # # Converting dataset into x_train and y_train
    # # Here we only scale the train dataset, and not the entire dataset to prevent information leak
    # scaler = StandardScaler()
    # train_scaled = scaler.fit_transform(np.array(train['High']).reshape(-1,1))
    # print("scaler.mean_ = " + str(scaler.mean_))
    # print("scaler.var_ = " + str(scaler.var_))
    #
    # # Split train into x and y
    # x_train_scaled, y_train_scaled = get_x_y(train_scaled, N, N)
    #
    # # Split cv into x and y
    # x_cv_scaled, y_cv, mu_cv_list, std_cv_list = get_x_scaled_y(np.array(train_cv['High']).reshape(-1,1), N, num_train)

    x_train, y_train, x_test, y_test = pre_process()

    # Set seeds to ensure same output results
    model_seed = 100
    seed(101)
    # set_random_seed(model_seed)
    tensorflow.random.set_seed(model_seed)

    model = Sequential()
    model.add(LSTM(units=75, input_shape=(x_train.shape[1], 1), activation='tanh',
                   recurrent_activation='sigmoid'))
    model.add(Dropout(0.0))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=50, batch_size=8, verbose=0)

    # # Split test into x and y
    # x_test_scaled, y_test, mu_test_list, std_test_list = get_x_scaled_y(
    #     np.array(global_object.stock_data['High']).reshape(-1,1), N, num_train+num_cv)

    print("Shape of test data set :: ", x_test.shape)

    est_scaled = model.predict(x_test)
    # est = (est_scaled * np.array(std_test_list).reshape(-1,1)) + np.array(mu_test_list).reshape(-1,1)
    # est = global_object.scalar.inverse_transform(est_scaled)

    print("Data type of y_test and estimated :: ", [y_test.shape, est_scaled.shape])
    rmse = math.sqrt(mean_squared_error(y_test, est_scaled))
    mape = get_mape(y_test, est_scaled)

    print("RMSE :: ", rmse)
    print('MAPE :: ', mape)
    return model
