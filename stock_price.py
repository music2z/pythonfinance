import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import numpy as np
import pandas_datareader.data as web


class StockPrice:

    @staticmethod
    def get_price(symbol, timeframe, count):
        """네이버 주가가져오기"""
        url = "https://fchart.stock.naver.com/sise.nhn?symbol={0}&timeframe={1}&count={2}&requestType=0"
        url = url.format(symbol, timeframe, count)
        response = requests.get(url)
        bs = BeautifulSoup(response.content, "html.parser")

        # 종목명
        chartdata = bs.select('chartdata')
        name = chartdata[0].attrs['name']

        # 주가데이터
        items = bs.select("item")
        columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.DataFrame([], columns=columns, index=range(len(items)))

        for i in range(len(items)):
            df.iloc[i] = str(items[i]['data']).split('|')

        df.index = pd.to_datetime(df['Date'])

        return df.drop('Date', axis=1).astype(float)

    @staticmethod
    def get_price_yahoo(code, start, end):
        """야후에서 주가얻어오기"""
        web.DataReader(code, 'yahoo', start, end)

    @staticmethod
    def get_cagr(sr_data, period_type):
        """복리수익률(CAGR) 구하기"""
        sr_data = sr_data.dropna()
        data_count = sr_data.count() - 1
        if period_type == 'M':
            period = data_count / 12
        elif period_type == 'D':
            period = data_count / 250

        last = sr_data[-1]
        cagr = (last ** (1 / period) - 1) * 100
        return cagr

    @staticmethod
    def get_mdd(sr_data):
        """MDD 구하기"""
        sr_max = sr_data.expanding(1).max()
        sr_mdd = sr_data / sr_max - 1
        return sr_mdd.min() * 100


