import pandas as pd
import pandas_datareader.data as web

"""
일단위 가격데이터 => 월단위 가격데이터
"""
df = web.DataReader('078930.KS', 'yahoo', '2018-01-01')
df.to_csv('time_test.csv')


def get_month(data):
    name = data.name
    ret = None
    if name == 'High':
        ret = data.max()
    elif name == 'Low':
        ret = data.min()
    elif name == 'Open':
        ret = data[0]
    elif name == 'Close':
        ret = data[-1]
    elif name == 'Volume':
        ret = data.sum()
    elif name == 'Adj Close':
        ret = data[-1]
    return ret


# rule은 Time siries Offset aliases 문서참조
df_m = df.resample(rule='M').apply(get_month)
df_m['2019':].Close.plot()