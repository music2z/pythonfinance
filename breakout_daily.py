import pandas as pd
import numpy as np
from stock_price import StockPrice
import matplotlib.pyplot as plt

pd.set_option('display.width', 400)

# 주가 가져오기
df_price = StockPrice.get_price('233740', 'day', '5000')

# 변동폭 구하기
sr_range = df_price['High'] - df_price['Low']
sr_range = sr_range * 0.4
df_price['range'] = sr_range.shift(1)

# 돌파여부
df_price['breakout'] = np.where(df_price['High'] >= df_price['Open'] + sr_range.shift(1), 1, 0)

# 컬럼추가
df_price.insert(len(df_price.columns), 'buy', np.nan)
df_price.insert(len(df_price.columns), 'sell', np.nan)
df_price.insert(len(df_price.columns), 'ret', 1)

# 첫번째 방법
for idx, row in df_price.iterrows(): #iteritems
    if row['breakout'] == 1:
        buy = row['Open'] + row['range']
        sell = df_price.shift(-1).loc[idx, 'Open']
        ret = sell / buy

        df_price.loc[idx, 'buy'] = buy
        df_price.loc[idx, 'sell'] = sell
        df_price.loc[idx, 'ret'] = ret
    # 첫행은 스킵
    # if df_price.index[0] == idx:
    #     continue
    #
    # index = df_price.shift(1).loc[idx, 'index'] * df_price.loc[idx, 'ret']
    # df_price.loc[idx, 'index'] = index

# 전체 수익률 누적합계
df_price['index'] = df_price['ret'].cumprod()
df_price['index'] = df_price['index'].fillna(method='ffill')
# 컬럼추가
# df_price['index'] = np.nan
# df_price.assign(index=np.nan)
# df_price.iloc[0, df_price.columns.get_loc('index')] = 1
# df_price.loc[df_price.index[0], 'index'] = 1

print('CAGR:', StockPrice.get_cagr(df_price['index'], 'D'))
print('MDD:', StockPrice.get_mdd(df_price['index']))