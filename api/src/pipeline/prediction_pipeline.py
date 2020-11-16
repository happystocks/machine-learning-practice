import pandas as pd
import numpy as np

from src.common import global_object
from src.common.constants import *


# def get_x_prediction(df):
#     x_scaled = global_object.scalar.transform(np.array(df).reshape(-1, 1))
#     x = []
#     x.append =

def start_prediction(date):
    """
        Here, the input is taken from the json and input data is processed.
        After that output is predicted. The predicted value is denormalized and returned as JSON
    """
    print("Starting prediction")
    global_object.stock_data['Date'] = pd.to_datetime(global_object.stock_data['Date'])
    prediction_df = global_object.stock_data[global_object.stock_data['Date'] <= pd.to_datetime(date)]

    print("----Prediction dataframe ----")
    print(prediction_df.tail(3))
    print("Prediction data is selected")
    x_pred = global_object.scalar.transform(np.array(prediction_df['High'][-3:]).reshape(-1, 1))

    print("X_prd.shape ::: ", x_pred.shape)
    x_pred_scaled = list()
    x_pred_scaled.append(x_pred[0:])
    x_pred_scaled = np.array(x_pred_scaled)
    print("X_pred_scaled_shape ::: ", x_pred_scaled.shape)

    prediction = global_object.lstm_model.predict(x_pred_scaled)
    print(prediction)
    prediction = global_object.scalar.inverse_transform(prediction)
    print(prediction)

    return prediction

