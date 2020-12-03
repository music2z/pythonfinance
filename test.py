import pandas as pd
import pandas_datareader.data as web
from stock_price import StockPrice
import matplotlib.pyplot as plt

pd.set_option('display.width', 400)

# df = web.DataReader(['010130.KS', '132030.KS'], 'yahoo', '2000-01-01')

# 월단위 쉬프트
# df.tshift(freq='M', periods=1)

# 132030 : KODEX 골드선물(H)
sr_gold = StockPrice.get_price('132030', 'month', '1000')['Close']
sr_gold.name = 'Gold'


def get_corr(code):
    sr_stock = StockPrice.get_price(code, 'month', '1000')['Close']
    if sr_stock.count() < 60:
        return 0
    df = pd.concat([sr_gold, sr_stock], axis=1)
    return df.corr().iloc[0, 1]


df = pd.read_excel('stock_list.xls', dtype={'종목코드': str})
df.index = df['종목코드']
df.drop('종목코드', axis=1, inplace=True)

# 종목별 상관계수 구하기
# df_corr = pd.DataFrame([], columns=['code', 'corr'])
# for code in df['종목코드']:
#     corr = get_corr(code)
#     df_corr.loc[len(df_corr)] = [code, corr]

# df_corr.to_excel('corr.xls')

df_corr = pd.read_excel('corr.xls', dtype={'code': str})
df_corr.index = df_corr['code']
df_corr.drop('code', axis=1, inplace=True)

# df_merge = pd.merge(df, df_corr, how='right', left_on='종목코드', right_on='code')
df_merge = pd.merge(df, df_corr, how='right', left_index=True, right_index=True)
df_sort = df_merge.sort_values('corr', ascending=False)

# print(df_sort[['기업명', 'corr']].head(20))

sr_stock = StockPrice.get_price('131970', 'month', '1000')['Close']
sr_stock.name = 'stock'
sr_stock.plot(subplots=True, figsize=(9, 7))
sr_gold.plot(subplots=True, figsize=(9, 7))
plt.legend()
plt.show()

# tot_cnt = df_corr.count()
# cnt = df_corr[df_corr['corr'] > 0.8].count()
# print(tot_cnt, cnt)
# print(df_sort[df_sort['corr'] > 0.8])