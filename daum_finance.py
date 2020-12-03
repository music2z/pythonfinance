from urllib.request import urlopen, Request
from fake_useragent import UserAgent
import json
import pandas

# fake_useragent 모듈을 통한 User-Agent 정보 생성
useragent = UserAgent()
print(useragent.chrome)

# 헤더 선언 및 referrer, User-Agent 전송
headers = {
    'referer': 'https://finance.daum.net/domestic/',
    'User-Agent': useragent.chrome,
}

# 주식 데이터 요청 URL
#url = 'https://finance.daum.net/api/sectors/?includedStockLimit=2&page=1&perPage=30&fieldName=changeRate&order=desc&market=KOSPI&change=RISE&includeStocks=true&pagination=true'
url1 = 'http://finance.daum.net/api/market_index/days?page='
page = 1
url2 = '&perPage=100&market=KOSDAQ&pagination=true'
url = url1 + str(page) + url2

# 주식 데이터 요청
response = urlopen(Request(url, headers=headers)).read().decode('utf-8')

# 응답 데이터 str 타입을 json 포맷으로 변환 및 data 저장
rank_json = json.loads(response)
totalPages = rank_json['totalPages']

data = []
for i in range(1, totalPages+1):
    url = url1 + str(i) + url2
    response = urlopen(Request(url, headers=headers)).read().decode('utf-8')
    json_data = json.loads(response)
    data = data + json_data['data']

print(len(data))
print(data[0])

writer = pandas.ExcelWriter('./daum_finance.xlsx')
df = pandas.DataFrame(data)
df.to_excel(writer)
writer.save()

# 더 간단히
# df.to_excel('./daum_finance.xlsx')