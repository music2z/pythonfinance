import pandas as pd
import numpy as np
from stock_price import StockPrice
import matplotlib.pyplot as plt

# 가격정보 가져오기
assets = {'122630': 'kodex2x',
          '167860': 'treasury2x',
          '114470': 'cash'}
# assets = dict(zip(['122630','167860','114470'],
#                   ['kodex2x','treasury2x','cash']))
prices = []
for symbol in assets.keys():
    prices.append(StockPrice.get_price(symbol, 'month', '1000')['Close'])

df_price = pd.concat(prices, axis=1)
df_price.columns = assets.values()
df_price.dropna(inplace=True)

# 월 수익률
df_pct = df_price.pct_change(periods=1)

# 변동성 구하기
df_std_r = 1 / df_pct.rolling(window=12).std()
sr_std_r_sum = df_std_r.sum(axis=1)
sr_std_r_sum = sr_std_r_sum.replace(0, np.nan)

# 투자비중
df_weight = df_std_r.divide(sr_std_r_sum, axis=0)
df_index = df_pct * df_weight.shift(1)
df_index['sum'] = df_index.sum(axis=1).replace(0, np.nan)
df_index['index'] = (df_index['sum'] + 1).cumprod()

# 마지막 값
last = df_index.iloc[-1]['index']

# CAGR 계산
period = df_index['index'].count() / 12
print('Last: {0:.6f}'.format(last))
print('CAGR: {0:.2f}%'.format((last ** (1 / period) - 1) * 100))

# MDD 구하기
sr_max = df_index['index'].expanding(1).max()
df_index['mdd'] = df_index['index'] / sr_max - 1
print('MDD: {0:.2f}%'.format(df_index['mdd'].min() * 100))

# 챠트 그리기
df_index['index'].plot(figsize=[6, 4])
# df_index['mdd'].plot(figsize=[6, 2], linestyle='dotted')
plt.show()




