import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
from datetime import datetime
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from fastapi import FastAPI ,Request
import tensorflow as tf

app = FastAPI()

def plot_history_vs_predictions(df_history,predictions):

    df_history =df_history.set_index("Date")
    df_history.index = pd.DatetimeIndex(df_history.index)
    df_history.index = df_history.index.strftime("%d-%m-%y")
    df_history = df_history["Close"]

    predictions.index = pd.DatetimeIndex(predictions.index)
    predictions.index = predictions.index.strftime("%d-%m-%y")


    plt.plot(df_history.index, df_history,color='blue', label="History Stock's Price")
    plt.plot(predictions.index, predictions,color='red', label="Predictions Price")

    first_point = [predictions.index[0], predictions.iloc[0]]
    last_point = [df_history.index[-1], df_history.iloc[-1]]
    x = [first_point[0], last_point[0]]
    y = [first_point[1], last_point[1]]
    plt.plot(x, y, color='red')

    # Add labels and title to the plot
    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.title("Scatter Plot of Data")

    # Show the plot
    plt.legend()
    plt.show()

def make_all_before_new_prediction(df):
    # Use the model to predict the close price for the next day
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(df_scaled, test_size=0.2)

    # Create and fit the support vector regression model
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    model.fit(train_data, train_data.reshape(-1,))

    return model

def create_df_from_predictions(last_day,pred,cols):
    df_predictions =pd.DataFrame(columns=cols)
    print("last Day: ",last_day)
    next_day = last_day + DateOffset(days=1)
    next_day_timestamp = next_day.timestamp()
    for i in range(0,7):
    # Add the predicted close price for the next day to the DataFrame
        df_predictions.loc[next_day] = pd.Series({'Close': pred[0][i]})
        next_day = next_day + DateOffset(days=1)
        next_day_timestamp = next_day.timestamp()
    return df_predictions

def create_ML_whenRun(df):
    df['Date'] = pd.to_datetime(df['Date'])

    # Get the last date in the data
    last_date = df['Date'].iloc[-1]
    df = df.set_index('Date')
    df = df[['Close']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    # Create the training data
    n_input = 30 # number of days to use as input
    n_out = 7 # number of days to predict
    X_train = []
    y_train = []
    for i in range(n_input, len(df_scaled) - n_out + 1):
        X_train.append(df_scaled[i-n_input:i, 0])
        y_train.append(df_scaled[i:i+n_out, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the data for the LSTM model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(n_out))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Make predictions for tomorrow
    x_input = df_scaled[-n_input:]
    x_input = np.reshape(x_input, (1, x_input.shape[0], 1))
    pred = model.predict(x_input)

    # Invert the scaling to get the actual stock price
    pred = scaler.inverse_transform(pred)

    # Print the prediction
    print("Predicted stock price for tomorrow:", pred[0][0])
    x_input = df_scaled[-n_input:]
    x_input = np.reshape(x_input, (1, x_input.shape[0], 1))
    pred = model.predict(x_input)

    # Invert the scaling to get the actual stock price
    pred = scaler.inverse_transform(pred)
    # Print the prediction
    print("Predicted stock price for tomorrow:", pred[0][0])


    return pred





@app.get("/Prediction")
async def make_prediction(df_stocks):
    try:
        #json_body = await df_stocks.json()
        #print(json_body)
        #df_stocks1 = pd.read_json(df_stocks.json())
        print(df_stocks)
        #print(df_stocks1)
        #print(type(df_stocks1))
        df_stocks = pd.read_json(df_stocks)
        print(type(df_stocks))
        #df_stocks = pd.read_json(json_body)
        df_stocks['Date'] = pd.to_datetime(df_stocks['Date'])
        #print(df_stocks)
        #df_stocks = df_stocks.set_index('Date')
        #df_stocks = df_stocks.drop(['Date'],axis=1)
        print(df_stocks)
    except Exception as e:
        return "Problem with Machine Learning ---> problem in converting the data index: Error:  " + str(e)
    
    try:
        df_predictions = create_ML_whenRun(df_stocks.copy())
        print(df_stocks.copy())
        print(df_predictions)
    except Exception as e:
        return "Problem with Machine Learning ----> in 'create_ML_whenRun function':  Error: " + str(e)

    try :   

        df_predictions = create_df_from_predictions(df_stocks.copy()['Date'].iloc[-1], df_predictions, df_stocks.copy().columns)
        print(df_stocks.copy())
        print(df_predictions)
    except Exception as e:
        return "Problem with Machine Learning ----> in 'create_df_from_predictions Error': " + str(e)
    
    try:
        df_predictions['Date'] = df_predictions.index
        df_predictions['Date'] = df_predictions['Date'].astype(str)
        df_predictions = df_predictions[["Date","Close"]]
        df_predictions.reset_index(drop=True, inplace=True)
        #plot_history_vs_predictions(df_history=df, predictions=df_predictions)
        print(df_predictions)
        return df_predictions.to_json()
    except Exception as e:
        return "Problem with Machine Learning -----> problem in convertthe data index second time Error: " + str(e)
    

@app.post("/Prediction")
async def make_prediction(request):
    try:
        #json_body = await df_stocks.json()
        #print(json_body)
        #print(type(df_stocks))

        json_body = await request.json()
        df_stocks = pd.DataFrame(json_body)
        df_stocks['Date'] = pd.to_datetime(df_stocks['Date'])
        df_stocks = df_stocks.set_index('Date')
        df_stocks = df_stocks.drop(columns=['Date'])

        print(df_stocks)
    except Exception as e:
        return "Problem with Machine Learning ---> problem in converting the data index: Error:  " + str(e)
    
    try:
        df_predictions = create_ML_whenRun(df_stocks.copy())
    except Exception as e:
        return "Problem with Machine Learning ----> in 'create_ML_whenRun function':  Error: " + str(e)

    try :   
        df_predictions = create_df_from_predictions(df_stocks.index[-1], df_predictions, df_stocks.columns)
        print(df_predictions)
    except Exception as e:
        return "Problem with Machine Learning ----> in 'create_df_from_predictions Error': " + str(e)
    
    try:
        df_predictions['Date'] = df_predictions.index
        df_predictions['Date'] = df_predictions['Date'].astype(str)
        df_predictions.reset_index(drop=True, inplace=True)
        #plot_history_vs_predictions(df_history=df, predictions=df_predictions)
        return df_predictions.to_json()
    except Exception as e:
        return "Problem with Machine Learning -----> problem in convertthe data index second time Error: " + str(e)
    
