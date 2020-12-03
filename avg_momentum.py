import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import numpy as np


def get_price(symbol, timeframe, count):
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

    return df.drop('Date', axis=1)
    # return df.drop('Date', axis=1).astype(float)


# tiger200 = pd.DataFrame(get_price('102110', 'month', '1000')['Close'])
# treasury10 = pd.DataFrame(get_price('148070', 'month', '1000')['Close'])
# treasury3 = pd.DataFrame(get_price('114470', 'month', '1000')['Close'])
#
# portfolio = pd.concat([tiger200, treasury10, treasury3], axis=1)
# portfolio.columns = ['tiger200', 'treasury10', 'treasury3']

# portfolio.to_excel('portfolio.xlsx')

# 데이터 불러오기
df_price = pd.read_excel('portfolio.xlsx', index_col='Date', parse_dates=True, sheet_name='Sheet1')

# 월간수익률 계산
df_month_return = df_price / df_price.shift(1)

# 평균모템텀 계산
df_momentum = 0
for i in range(1, 13):
    df_momentum = df_price / df_price.shift(i) + df_momentum
# print(df_momentum)


# 평균모멘텀 스코어
def get_avg_momentum(sr_data):
    score = 0
    for i in range(1, 13):
        # sr_diff = sr_diff.apply(lambda x: 1 if x >= 0 else 0)
        score = np.where(sr_data - sr_data.shift(i) >= 0, 1, 0) + score
    return score


# 평균모템텀 스코어 계산
df_momentum = pd.DataFrame([], index=df_price.index, columns=df_price.columns)
df_momentum = df_momentum.fillna(0)

for i in range(1, 13):
    # 이렇게 하면 12개 데이터가 있을때만 계산이 되고 (if:1 elif:0 else:NaN)
    df_tmp = df_price / df_price.shift(i)
    df_tmp[df_tmp >= 1] = 1
    df_tmp[df_tmp < 1] = 0
    df_momentum += df_tmp
    # 이렇게 하면 데이터가 2개만 있어도 계산이 된다. (if:1 else:0)
    # df_momentum += np.where(df_price / df_price.shift(i) >= 1, 1, 0)

# 투자비중 계산
df_momentum = df_momentum.mul([1, 1, 0.5])
s_sum = df_momentum.sum(axis=1)
df_weight = df_momentum.divide(s_sum, axis=0)
print(df_momentum)

# df_weight[df_weight.notnull()] = 1

# 통합수익률
df_total = df_month_return * df_weight.shift(1)
df_total['sum'] = df_total.sum(axis=1)
# df_total.loc[df_total['sum'] == 0, 'sum'] = np.nan
df_total['sum'].replace(0, np.nan, inplace=True)
df_total['cumprod'] = df_total['sum'].cumprod()

# 마지막 값
last = df_total.iloc[-1]['cumprod']

# CAGR 계산
period = df_total['cumprod'].count() / 12
print('Last: {0:.6f}'.format(last))
print('CAGR: {0:.2f}%'.format((last ** (1 / period) - 1) * 100))

# MDD 구하기
sr_max = df_total['cumprod'].expanding(1).max()
df_total['mdd'] = df_total['cumprod'] / sr_max - 1
print('MDD: {0:.2f}%'.format(df_total['mdd'].min() * 100))

# 월별수익률
df_total['return'] = df_total['cumprod'].rolling(min_periods=1, window=2).apply(lambda x: (x[1] / x[0] - 1) * 100)


x = [
    [10, 22],
    [11, 23],
    [12, 24],
    [13, 25],
    [0, 0]
]
df = pd.DataFrame(x, columns=['price', 'price2'])


# 컬럼명 바꾸기
# print(df.columns)
df.columns = ['p1', 'p2']
# print(df.columns)
df.rename(columns={'p1': 'p1111', 'p2': 'p2222'}, inplace=True)
# print(df.columns)

# df['mean'] = df['price'].rolling(window=12, min_periods=1).mean()

# def avgmom12(x):
#     df_momentum = 0
#     for i in range(len(x)-1):
#         if x[i] < x[-1]:
#             df_momentum += 1
#     return df_momentum
#
# df['aaa'] = df.rolling(window=7).apply(avgmom12)













