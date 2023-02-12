#from fastapi import FastAPI
import datetime as dt
import matplotlib.pyplot as plt
#import pandas_datareader as web
##import json
#import DB 
#import os, sys
#import module 
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import plotly.express as px
import yfinance 
import httpx
import requests
import altair as alt




##import httpx 
##import plotly.figure_factory as ff
#from sklearn.preprocessing import MinMaxScaler

 
#app = FastAPI()
 

st.set_page_config(page_title="Data Science In Finance Stocks",
                page_icon=":bar_chart:",
                layout="wide"
                )

#res = httpx.get("https://172.20.0.2")



st.title('Lit Finance Dashboard')
tickers = ('META','BTC-USD','TSLA','KO','GC=F')
dropdown = st.multiselect('Pick your assets',tickers)
start = st.date_input('Start', value=pd.to_datetime('2021-01-01'))
end = st.date_input('End', pd.datetime.now())

#st.sidebar.success("select a page above")

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

if len(dropdown) > 0:
    df = yfinance.download(dropdown,start,end)['Adj Close']
    st.line_chart(df)
    print(df)


    ##r = httpx.get("http://backend-service:8080/Meta")
    ##data = r.json()
    ##print(type(data))
    ##df2 = pd.read_json(data)
    ##st.dataframe(data=df2)
    st.title(dropdown)  # ->רשימה



##### צריך להבין למה הוא כן מצליח להגיע לאתר וכן מצליח להוריד את כל הדאטה אבל כשהוא מחזיר את זה לפה זה ריק והוא לא יכול לעשות גרף
#    lst_choosen_data = [dropdown,start,end]
#    r1 = httpx.get("http://backend-service:8080/MultiGraph?stocks={dropdown}&start={start}&end={end}")
#    if r1.status_code == 200:
#        #st.title(r1)
#        #print(type(r1.json()))
#        try:
#            df = pd.read_json(r1.json())
#            st.line_chart(df["Adj Close"])
#        except:
#            st.title("problem in print the chart...")
#        #st.dataframe(data=df)
#        st.dataframe(data=df)

