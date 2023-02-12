from fastapi import FastAPI ,Request
#import datetime as dt
#import pandas_datareader as web
import json
#import DB 
#import os, sys
#import module 
#import numpy as np
#import matplotlib.pyplot as plt
import requests
import pandas as pd
import yfinance 

#from sklearn.preprocessing import MinMaxScaler

#TODO --->>> real time data ---- פתאום הפסיק לעבוד, מקווה שאצליח למצוא פתרון

async def getData(num):
    dataDict = {
        1 : "META",
        2 : "BTC-USD",
        3 : "TSLA",
        4 : "KO",
        5 : "GC=F"
    }
    

    base = r"./DBs/"
    df = pd.DataFrame(pd.read_csv(base + dataDict[num]+".csv"))
    print("data of "+ dataDict[num])
    print(type(df))
    print(df)
    return df



app = FastAPI()
df = []

@app.get("/")
async def home():

    return "Welcome Home, Select Your Choice To Make a Prediction:"
#---------------------------------------------------------------
#---------------------------------------------------------------


 #-----------------------------------------------CocaCola function:----------------------
@app.get("/CocaCola") 
async def getCocaColaData(s,e):
    #df = getData(4)
# TODO sending the DataFrame To ML microservice to process the data

    df = yfinance.download('KO',s,e) #['Adj Close']
    df_To_UI = df['Adj Close']
    return df.to_json()   


# TODO Updating the csv file in new real time data ----> לא כלכך עובד עכשיו 
    return "Updating new CocaCola's data to DB"



#-----------------------------------------------TESLA function:--------------------

#select silver data
@app.get("/TESLA") 
async def getTeslaData(s,e):
    #df = yfinance.download(dropdown,start,end)['Adj Close']

    df = yfinance.download('TSLA',s,e) #['Adj Close']
    df_To_UI = df['Adj Close']
# TODO sending the DataFrame To ML microservice to process the data
#    x = DataToML(df,3)

    return df.to_json()  
#------------------------------------------------------------------------------------

#-----------------------------------------------BitCoin function----------------
#select bitcoin data
@app.get("/BitCoin") 
async def getBitCoinData(s,e):
    
    #df = getData(2)
# TODO sending the DataFrame To ML microservice to process the data
#    x = DataToML(df,2)

    df = yfinance.download('BTC-USD',s,e) #['Adj Close']
    df_To_UI = df['Adj Close']

    return df.to_json() 



# TODO Updating the csv file in new real time data ----> לא כלכך עובד עכשיו     
    return "Updating BitCoins' data to DB"

#-------------------------------------------------------------------------------


#-----------------------------------------------Meta function-------------------
@app.get("/Meta") 
async def GetMetaData(s,e):
    #df = getData(1)
#    start = pd.to_datetime('2018-01-01')
#    end = pd.datetime.now()
    df = yfinance.download('META',s,e) #['Adj Close']
    
# TODO sending the DataFrame To ML microservice to process the data
#    x = DataToML(df,1)
    

    return df.to_json()




#--------------------------------------------------------------------------------
@app.get("/Gold") 
async def getGoldData(s,e):

    df = yfinance.download('GC=F',s,e) #['Adj Close']

    print(df)

# TODO sending the DataFrame To ML microservice to process the data
#    x = DataToML(df,1)

    return df.to_json()


        #df = getData(1)


#--------------------------------------------------------------------------------

###(conector with UI)
###funcUI_Selector  -> getting the chice of dropbox and fit the right backend func

@app.get("/selector/{choice_id}") 
async def funcUI_Selector(choice_id):
    #something i get from DropBox of UI
    #TODO finish the function...
    dataDict = {
    "1" : GetMetaData(),
    "2" : getBitCoinData(),
    "3" : getTeslaData(),
    "4" : getCocaColaData(),
    "5" : getGoldData()
}

@app.get("/MultiGraph") 
async def getMultigraphData(stocks,start,end):
    #df = getData(5)
    #start = pd.to_datetime('2018-01-01')
    #end = pd.datetime.now()
    #start = pd.to_datetime(lst[1])
    #end = pd.to_datetime(lst[2])
    #df = yfinance.download(stocks,start,end) #['Adj Close']
    df = yfinance.download(stocks[0],start,end)
    #df_To_UI = df['Adj Close']
    print(df)
# TODO sending the DataFrame To ML microservice to process the data
#    x = DataToML(df,1)

    return df.to_json()


        #df = getData(1)

######################################


#@app.get("/Prediction")
#def get_stock_prediction(ticker):
#    response = requests.get(f'http://ML-service:80/Prediction?ticker={ticker}')
#    if response.status_code != 200:
#        raise Exception(f'Unable to retrieve prediction for ticker {ticker}: {response.text}')
#    prediction_data = response.json()
#    # Use the prediction data as needed in your backend code

@app.post("/Prediction")
async def receive_prediction_data(request: Request):
    try:
        df_json = pd.read_json(await request.body(), orient='records')
    except:
        return "problem in back before send it to ml"
    response = requests.post("http://ML-service:80/Prediction", json=df_json.to_json(orient='records'))
    if response.status_code != 200:
        raise Exception("Unable to get prediction from ML service")
    return response


@app.get("/Prediction2")
async def bdika(df_stocks):
    try:
        #json_body = await df_stocks.to_json()
        #response = requests.post("http://ML-service:81/Prediction", headers=headers, json=json_body)
        response = requests.get(f"http://ML-service:81/Prediction?df_stocks={df_stocks.to_json()}")    
    except Exception as e:
        return "problem in back before send it to ml "+ str(e)

    return response
    