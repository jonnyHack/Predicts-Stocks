import main
import pandas as pd

import pytest
from fastapi import FastAPI, Request
import requests
import pandas as pd
import yfinance


def test_CocaCola():
    response = requests.get('http://localhost:8000/CocaCola')
    assert response.status_code == 200
    df = pd.read_json(response.text)
    assert isinstance(df, pd.DataFrame)
    assert 'Adj Close' in df.columns
    assert 'KO' in df.columns

def test_Tesla():
    response = requests.get('http://localhost:8000/TESLA')
    assert response.status_code == 200
    df = pd.read_json(response.text)
    assert isinstance(df, pd.DataFrame)
    assert 'Adj Close' in df.columns
    assert 'TSLA' in df.columns

def test_BitCoin():
    response = requests.get('http://localhost:8000/BitCoin')
    assert response.status_code == 200
    df = pd.read_json(response.text)
    assert isinstance(df, pd.DataFrame)
    assert 'Adj Close' in df.columns
    assert 'BTC-USD' in df.columns

def test_Meta():
    response = requests.get('http://localhost:8000/Meta?s=2018-01-01&e=2023-02-12')
    assert response.status_code == 200
    df = pd.read_json(response.text)
    assert isinstance(df, pd.DataFrame)
    assert 'META' in df.columns

def test_Gold():
    response = requests.get('http://localhost:8000/Gold')
    assert response.status_code == 200
    df = pd.read_json(response.text)
    assert isinstance(df, pd.DataFrame)
    assert 'Adj Close' in df.columns
    assert 'GC=F' in df.columns


