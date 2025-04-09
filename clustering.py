# 경상북도 시/군 군집화
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 한글 설정하고 시작
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 경로 설정
base_path = Path.cwd()
file_path = base_path / 'asset' / 'data' / 'travel' / '골프장.csv'

# golf['개방자치단체코드'].unique() # 5050000: 경주, 5220000: 칠곡군

# 스포츠 시설 데이터
golf = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '골프장.csv', encoding='euc-kr')
swim = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '수영장업.csv', encoding='euc-kr')
ssul = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '썰매장업.csv', encoding='euc-kr')
bing = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '빙상장업.csv', encoding='euc-kr')

# 시설 종류 컬럼 추가
golf = golf.iloc[:, :28]
swim = swim.iloc[:, :28]
ssul = ssul.iloc[:, :28]
bing = bing.iloc[:, :28]
golf['시설종류'] = '골프장'
swim['시설종류'] = '수영장'
ssul['시설종류'] = '썰매장'
bing['시설종류'] = '빙상장'

# 스포츠 데이터 결합
dfs = [golf, swim, ssul, bing]
sports_data = pd.concat(dfs, ignore_index=True)  # 인덱스 무시하고 행 방향으로 연결

# 문화 시설 데이터
zul = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '전통사찰.csv', encoding='euc-kr')
art = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '박물관,미술관.csv', encoding='euc-kr')
hanok = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '한옥체험업.csv', encoding='euc-kr')

# 문화시설 데이터 결합
zul = zul.iloc[:, :28]
art = art.iloc[:, :28]
hanok = hanok.iloc[:, :28]
zul['시설종류'] = '사찰'
art['시설종류'] = '박물관/미술관'
hanok['시설종류'] = '한옥체험'
dfs = [zul.iloc[:, :29], art.iloc[:, :29], hanok.iloc[:, :29]]
culture_data = pd.concat(dfs, ignore_index=True)



# 관광 휴양시설 데이터
circle = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '시내순환관광업.csv', encoding='euc-kr')
yayeong = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '일반야영장업.csv', encoding='cp949')
uone = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '일반유원시설업.csv', encoding='euc-kr')
car_ya = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '자동차야영장업.csv', encoding='euc-kr')
pro_hue = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '전문휴양업.csv', encoding='euc-kr')
multi_hue = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '종합휴양업.csv', encoding='euc-kr')



# 숙박 시설 데이터
travel_suk = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '관광숙박업.csv', encoding='euc-kr')
travel_pen = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '관광펜션업.csv', encoding='euc-kr')
sigol_suk = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '농어촌민박업.csv', encoding='cp949')
foreign_suk = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '외국인관광도시민박업.csv', encoding='euc-kr')


# 음식점 데이터
gen_food = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '일반음식점.csv', encoding='cp949', low_memory=False)
rest_food = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '휴게음식점.csv', encoding='cp949')

