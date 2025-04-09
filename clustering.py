# 경상북도 시/군 군집화
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 설정하고 시작
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# golf.info()
# golf['개방자치단체코드'].unique() # 5050000: 경주, 5220000: 칠곡군

# 스포츠 시설 데이터
golf = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/골프장.csv', encoding='euc-kr')
swim = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/수영장업.csv', encoding='euc-kr')
ski = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/스키장.csv', encoding='euc-kr')
ssul = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/썰매장업.csv', encoding='euc-kr')
bing = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/빙상장업.csv', encoding='euc-kr')

# 문화 시설 데이터
zul = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/전통사찰.csv', encoding='euc-kr')
art = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/박물관,미술관.csv', encoding='euc-kr')
hanok = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/한옥체험업.csv', encoding='euc-kr')

# 관광 휴양시설 데이터
circle = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/시내순환관광업.csv', encoding='euc-kr')
yayeong = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/일반야영장업.csv', encoding='cp949')
uone = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/일반유원시설업.csv', encoding='euc-kr')
car_ya = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/자동차야영장업.csv', encoding='euc-kr')
pro_hue = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/전문휴양업.csv', encoding='euc-kr')
multi_hue = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/종합휴양업.csv', encoding='euc-kr')

# 숙박 시설 데이터
travel_suk = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/관광숙박업.csv', encoding='euc-kr')
travel_pen = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/관광펜션업.csv', encoding='euc-kr')
sigol_suk = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/농어촌민박업.csv', encoding='cp949')
foreign_suk = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/외국인관광도시민박업.csv', encoding='euc-kr')


# 음식점 데이터
gen_food = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/일반음식점.csv', encoding='cp949', low_memory=False)
rest_food = pd.read_csv('C:/Users/USER/Desktop/TIL/yeongchun/asset/data/travel/휴게음식점.csv', encoding='cp949')

