from predictMyStocks import *
import pandas as pd

import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.svm import SVR
from pandas.tseries.offsets import DateOffset



def test_create_ML_whenRun():
    # Create a sample DataFrame to pass as input
    data = {'Date': ['2022-01-01', '2022-01-02', '2022-01-03'],
            'Close': [10, 20, 30]}
    df = pd.DataFrame(data)

    # Call the function
    result = create_ML_whenRun(df)

    # Check if the returned value is a tuple
    assert isinstance(result, tuple)

    # Check if the first element of the tuple is a DataFrame
    assert isinstance(result[0], pd.DataFrame)

    # Check if the second element of the tuple is a numpy array
    assert isinstance(result[1], np.ndarray)

    # Check if the first element of the numpy array has the correct shape
    assert result[1].shape == (1, 7)

def test_make_all_before_new_prediction_returns_model():
    # Arrange
    df = pd.DataFrame({'Close': [1, 2, 3, 4, 5]})
    expected_type = SVR

    # Act
    model = make_all_before_new_prediction(df)

    # Assert
    assert isinstance(model, expected_type)

def test_make_all_before_new_prediction_model_fit():
    # Arrange
    df = pd.DataFrame({'Close': [1, 2, 3, 4, 5]})
    expected_coef_shape = (1,)

    # Act
    model = make_all_before_new_prediction(df)
    coef_shape = model.dual_coef_.shape

    # Assert
    assert coef_shape == expected_coef_shape

def test_create_df_from_predictions():
    last_day = pd.to_datetime("2022-01-01")
    pred = [[10, 20, 30, 40, 50, 60, 70]]
    cols = ['Close']
    
    df_predictions = create_df_from_predictions(last_day, pred, cols)
    
    # Check if the dataframe has the correct number of rows
    assert len(df_predictions) == 7
    
    # Check if the first prediction is correct
    first_prediction = df_predictions.iloc[0]['Close']
    assert first_prediction == 10
    
    # Check if the last prediction is correct
    last_prediction = df_predictions.iloc[-1]['Close']
    assert last_prediction == 70
    
    # Check if the dates are correct
    expected_dates = [last_day + DateOffset(days=i+1) for i in range(7)]
    for i, date in enumerate(expected_dates):
        assert df_predictions.index[i] == date


