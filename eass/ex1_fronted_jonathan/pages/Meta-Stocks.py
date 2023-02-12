from main import *
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="Data Science In Finance Stocks",
                #page_icon=":bar_chart:",
                layout="wide"
                )
st.title("Meta's Stocks:")

start = st.date_input('Start', value=pd.to_datetime('2022-01-01'))
end = st.date_input('End', pd.datetime.now())
#df = yfinance.download("META",start,end)['Adj Close']

#r = httpx.get("http://backend-service:8080/selector/5")

r1 = httpx.get(f"http://backend-service:8080/Meta?s={start}&e={end}")
print(type(r1.json()))
df2 = pd.read_json(r1.json())
df_stocks = df2.copy()
st.line_chart(df2["Adj Close"])
st.dataframe(data=df2)

#------------------
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
#------------------

def plot_history_vs_predictions(df_history,predictions):

    #df_history =df_history.set_index("Date")
    #df_history.index = pd.DatetimeIndex(df_history.index)
    #df_history.index = df_history.index.strftime("%d-%m-%y")
    df_history = df_history["Close"]

    #predictions.index = pd.DatetimeIndex(predictions.index)
    #predictions.index = predictions.index.strftime("%d-%m-%y")

    st.line_chart(df_history, label="History Stock's Price", color='blue')
    st.line_chart(predictions, label="Predictions Price", color='red')

    first_point = [predictions.index[0], predictions.iloc[0]]
    last_point = [df_history.index[-1], df_history.iloc[-1]]
    x = [first_point[0], last_point[0]]
    y = [first_point[1], last_point[1]]
    st.line_chart(pd.DataFrame({'x': x, 'y': y}), label="", color='red')

    # Add labels and title to the plot
    st.write("Date", "Stock Price ($)")
    st.title("Scatter Plot of Data")

    # Show the plot
    st.legend()



def predict_function(df_stocks):
    # Do something when the button is clicked
    headers = {'Content-Type': 'application/json'}
    df_stocks['Date'] = df_stocks.index
    df_stocks['Date'] = df_stocks['Date'].astype(str)
    df_stocks.reset_index(drop=True, inplace=True)
   
#    response = requests.get(f"http://ML-service:81/Prediction?df_stocks={df_stocks.to_json()}")
    response = requests.get(f"http://ML-service:81/Prediction?df_stocks={df_stocks.to_json()}")

    
    if response.status_code == 200:
        data = response.json()
        st.write('the Json:', response.json())
        #data = pd.DataFrame(data)
        #data['Date'] = pd.to_datetime(data['Date'])
        #data = data.set_index('Date')
        #data = data.drop(columns=['Date'])
        df_pred = pd.read_json(response.json())
        df_pred = df_pred.set_index('Date')
        st.dataframe(data=df_pred)
        #st.dataframe(data)
    else:
        st.write('Unable to retrieve data. Response status code:', response.status_code)
        st.write(response.text)
    return df_pred





#Button to send the data to ML for prediction:
if st.button('Make A Prediction'):
    df_pred = predict_function(df_stocks)
    st.title("Meta's Predictions:")

    new_date = pd.to_datetime(df2.index[-1])
    new_row = pd.DataFrame([[df2["Adj Close"].iloc[-1]]], index=[new_date], columns=["Close"])
    df_pred = pd.concat([new_row, df_pred]).sort_index()

    #plot_history_vs_predictions(df2,df_pred)
    #st.line_chart(df2["Adj Close"])
    history = df2[["Adj Close"]]
    merged = pd.concat([history, df_pred], axis=1)
    print(merged)
    chart_data = merged
    st.line_chart(chart_data)


    





