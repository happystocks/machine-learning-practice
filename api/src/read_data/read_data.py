import pandas as pd
import pickle

from src.common.common import *
from src.common import global_object


def read_data():
    # We will data from local csv file
    global_object.stock_data = pd.read_csv(os.path.join(data_dir, 'IBM.csv'))


def load_model():
    # Trained model will be loaded here
    open_obj = open(os.path.join(model_dir, 'LSTM.pickle'), 'rb')
    global_object.lstm_model = pickle.load(open_obj)
    open_obj.close()
