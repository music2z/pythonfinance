import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.width', 400)
# pd.set_option('display.max_columns', 10)


'''
1. Object Creation (객체 생성)
'''
print('1. Object Creation (객체 생성)')
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

dates = pd.date_range('20130101', periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)

df2 = pd.DataFrame({
    'A': 1.,
    'B': pd.Timestamp('20130102'),
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'D': np.array([3] * 4, dtype='float32'),
    'E': pd.Categorical(["test", "train", "test", "train"]),
    'F': 'foo'
})
print(df2)
print(df2.dtypes)

'''
2. Viewing Data (데이터 확인하기)
'''
print('2. Viewing Data (데이터 확인하기)')
df.tail(3)
df.head()

print(df.index)
print(df.columns)
print(df.values)
print(df.describe())
print(df.T)
print(df.sort_index(axis=1, ascending=True))
print(df.sort_values(by='B', ascending=True))

'''
3. Selection (선택)
'''
# Getting
print('3. Selection (선택)')
print(df['A'])
print(df[0:3])
print(df['20130102':'20130104'])

# Selection by Label
df.loc[dates[0]]
df.loc[:, ['A', 'B']]
df.loc['20130102':'20130104', ['A', 'B']]
df.loc['20130102', ['A', 'B']]
df.loc[dates[0], 'A']

# Selection by Position
df.iloc[3]
df.iloc[3:5, 0:2]
df.iloc[[1, 2, 4], [0, 2]]
df.iloc[1:3, :]
df.iloc[:, 1:3]
df.iloc[1, 1]
df.iat[1, 1]

# Boolean Indexing
df[df.A > 0]
df[df > 0]
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
df2
df2[df2['E'].isin(['two', 'four'])]

# Setting
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
df['F'] = s1
df.at[dates[0], 'A'] = 0
df.iat[0, 1] = 0
df.loc[:, 'D'] = np.array([5] * len(df))

df2 = df.copy()
df2[df2 > 0] = -df2
df2

'''
4. Missing Data (결측치)
'''
print('4. Missing Data (결측치)')

df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
df1.dropna(how='any')
df1.fillna(value=5)

pd.isna(df1)

'''
5. Operation (연산)
'''
print('5. Operation (연산)')

df.mean()
df.mean(1)

s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
s
df
df.sub(s, axis='index')

# Apply
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())

# Histogramming
s = pd.Series(np.random.randint(0, 7, size=10))
s.value_counts()

# String Methods (문자열 메소드)
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()

'''
6. Merge (병합)
'''
print('6. Merge (병합)')

# Concat
df = pd.DataFrame(np.random.randn(10, 4))
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)

# Join
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
pd.merge(left, right, on='key')

left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
pd.merge(left, right, on='key')

# Append
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
s = df.iloc[3]
df.append(s, ignore_index=True)

'''
7. Grouping (그룹화)
'''
print('7. Grouping (그룹화)')
df = pd.DataFrame(
    {
        'A' : ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B' : ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
        'C' : np.random.randn(8),
        'D' : np.random.randn(8)
    })

df.groupby('A').sum()
df.groupby(['A', 'B']).sum()

'''
8. Reshaping (변형)
'''
print('8. Reshaping (변형)')
# Stack (스택)
# Pivot Tables (피봇 테이블)

'''
9. Time Series (시계열)
'''
print('9. Time Series (시계열)')
rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample('5Min').sum()

rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts_utc = ts.tz_localize('UTC')
ts_utc.tz_convert('US/Eastern')

rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ps = ts.to_period()
ps.to_timestamp()

prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
ts.head()

'''
10. Categoricals (범주화)
'''
print('10. Categoricals (범주화)')

df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
df["grade"] = df["raw_grade"].astype("category")
df['grade'].cat.categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
df["grade"]
df.sort_values(by="grade")
df.groupby("grade").size()

'''
11. Plotting (그래프)
'''
print('11. Plotting (그래프)')
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
# ts.plot()

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
# plt.figure(); df.plot(); plt.legend(loc='best')

'''
12. Getting Data In / Out (데이터 입 / 출력)
'''
print('12. Getting Data In / Out (데이터 입 / 출력)')
# CSV
df.to_csv('foo.csv')
pd.read_csv('foo.csv')

# Excel
df.to_excel('foo.xlsx', sheet_name='Sheet1')
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])

'''
13. Gotchas (잡았다!)
'''
print('13. Gotchas (잡았다!)')
if pd.Series([False, True, False]) is not None:
    print("I was true")
