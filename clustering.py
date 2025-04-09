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
sports_data = sports_data[(sports_data['영업상태명'] != '폐업') & (sports_data['영업상태명'] != '휴업')]

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
culture_data = culture_data[(culture_data['영업상태명'] != '폐업') & (culture_data['영업상태명'] != '휴업')]


# 관광 휴양시설 데이터
circle = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '시내순환관광업.csv', encoding='euc-kr')
yayeong = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '일반야영장업.csv', encoding='cp949')
uone = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '일반유원시설업.csv', encoding='euc-kr')
car_ya = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '자동차야영장업.csv', encoding='euc-kr')
pro_hue = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '전문휴양업.csv', encoding='euc-kr')
multi_hue = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '종합휴양업.csv', encoding='euc-kr')

# 데이터 전처리
circle = circle.iloc[:, :28]
yayeong = yayeong.iloc[:, :28]
uone = uone.iloc[:, :28]
car_ya = car_ya.iloc[:, :28]
pro_hue = pro_hue.iloc[:, :28]
multi_hue = multi_hue.iloc[:, :28]

circle['시설종류'] = '시내순환관광'
yayeong['시설종류'] = '일반야영장'
car_ya['시설종류'] = '자동차야영장'
uone['시설종류'] = '일반유원'
pro_hue['시설종류'] = '전문휴양업'
multi_hue['시설종류'] = '종합휴양업'

dfs = [circle, yayeong, uone, car_ya, pro_hue, multi_hue]
travel_rest_data = pd.concat(dfs, ignore_index=True)
travel_rest_data = travel_rest_data[(travel_rest_data['영업상태명'] != '폐업') & (travel_rest_data['영업상태명'] != '휴업')]


# 숙박 시설 데이터
travel_suk = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '관광숙박업.csv', encoding='euc-kr')
travel_pen = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '관광펜션업.csv', encoding='euc-kr')
sigol_suk = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '농어촌민박업.csv', encoding='cp949')
foreign_suk = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '외국인관광도시민박업.csv', encoding='euc-kr')

# 데이터 전처리
travel_suk = travel_suk.iloc[:, :28]
travel_pen = travel_pen.iloc[:, :28]
sigol_suk = sigol_suk.iloc[:, :28]
foreign_suk = foreign_suk.iloc[:, :28]

travel_suk['시설종류'] = '관광숙박'
travel_pen['시설종류'] = '펜션'
sigol_suk['시설종류'] = '농어촌민박'
foreign_suk['시설종류'] = '외국인관광도시민박'

dfs = [travel_pen, travel_suk, foreign_suk, sigol_suk]
hotel_data = pd.concat(dfs, ignore_index=True)
hotel_data = hotel_data[(hotel_data['영업상태명'] != '폐업') & (hotel_data['영업상태명'] != '휴업')]


# 음식점 데이터
gen_food = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '일반음식점.csv', encoding='cp949', low_memory=False)
rest_food = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / '휴게음식점.csv', encoding='cp949')

# 전처리
gen_food = gen_food.iloc[:, :28]
rest_food = rest_food.iloc[:, :28]
gen_food['시설종류'] = '일반음식점'
rest_food['시설종류'] = '휴게음식점'

dfs = [gen_food, rest_food]
food_data = pd.concat(dfs, ignore_index=True)
food_data = food_data[(food_data['영업상태명'] != '폐업')]

# 지역명 컬럼 추가
# 지역명 딕셔너리
region = {
    5020000: '포항시',
    5050000: '경주시',
    5060000: '김천시',
    5070000: '안동시',
    5080000: '구미시',
    5090000: '영주시',
    5100000: '영천시',
    5110000: '상주시',
    5120000: '문경시',
    5130000: '경산시',
    5150000: '의성군',
    5160000: '청송군',
    5170000: '영양군',
    5180000: '영덕군',
    5190000: '청도군',
    5200000: '고령군',
    5210000: '성주군',
    5220000: '칠곡군',
    5230000: '예천군',
    5240000: '봉화군',
    5250000: '울진군',
    5260000: '울릉군'
}

food_data.loc[:, '지역명'] = food_data['개방자치단체코드'].map(region)
travel_rest_data.loc[:, '지역명'] = travel_rest_data['개방자치단체코드'].map(region)
sports_data.loc[:, '지역명'] = sports_data['개방자치단체코드'].map(region)
culture_data.loc[:, '지역명'] = culture_data['개방자치단체코드'].map(region)
hotel_data.loc[:, '지역명'] = hotel_data['개방자치단체코드'].map(region)


# 전처리 완료된 데이터
food_data
travel_rest_data
sports_data
culture_data
hotel_data

# 지역별로 개수 세보기
food_data.groupby('지역명').size()
travel_rest_data.groupby('지역명').size()
sports_data.groupby('지역명').size()
culture_data.groupby('지역명').size()
hotel_data.groupby('지역명').size()

