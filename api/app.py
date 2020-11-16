import pandas as pd
from flask import Flask, render_template, request
import json
import os, sys

from src.common.common import *
from src.read_data.read_data import *
from src.pipeline.training_pipeline import *
from src.pipeline.prediction_pipeline import *

print(2)
app = Flask(__name__)


def app_initialization():
    """
        Here, the dataset will be loaded along with the trained model.
        Then api's will be called.
    """
    # Loading data set
    read_data()

    # Loading trained model
    load_model()
    # print("Trained model is :: ", model)

    @app.route('/index/', methods=['GET'])
    def home_page():
        # Here we return readme file
        status = {'home page': "Welcome to happy stocks"}
        return json.dumps(status)

    @app.route('/train/', methods=['GET'])
    def train():
        # here model will be trained after data processing
        model = training_pipeline()
        status = {"status": "Training successful"}
        return json.dumps(status)

    @app.route('/predict/', methods=['POST'])
    def prediction():
        """
            Inputs are symbol, candle size, and exchange
        """
        if request.method == 'POST':
            print(request.data)
            content = json.loads(request.data)
            print(content)
            date = content['stock']

            prediction = start_prediction(date)
            status = {"status": "Training successful"}

        return json.dumps(status)

    # backtesting API

    # get strategies api

    # NIFTY 50 stock selection

    # Get all NIFTY 50, 100 stocks

    ###    User specific API
    # create, read, update, and delete strategy
    # Asset distribution
    # Place trade

    return app


app_initialization()


if __name__ == "__main__":
    app.run(debug=False, threaded=False)

