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


# 축제 데이터
festival_data = pd.read_csv(base_path / 'asset' / 'data' / 'travel' / 'festival.csv', encoding='cp949')
# 경북 축제 데이터로 변경
festival_data = festival_data[festival_data['제공기관명'].str.startswith('경상북도')]
festival_data.rename(columns={'제공기관명': '지역명'}, inplace=True)
festival_data['제공기관명'].unique()
festival_data['지역명'] = festival_data['지역명'].str.split().str[-1]


# 교통 접근성 점수를 위한 교통 데이터 불러오기
traffic_data = pd.read_csv(base_path / 'asset' / 'data' / 'traffic.csv', encoding='cp949', low_memory=False)
# 경북 데이터만 추출
traffic_data = traffic_data[traffic_data['행정구역시도명'] == '경상북도']
traffic_data.info()
traffic_data.columns
traffic_data['시설물대분류'].unique()

# 전처리 완료된 데이터
food_data           # 음식점
travel_rest_data    # 관광/휴양
sports_data         # 스포츠/레저 시설
culture_data        # 문화시설
hotel_data          # 숙박시설
festival_data       # 지역 축제


food_data.groupby('지역명').size()
hotel_data.groupby('지역명').size()
travel_rest_data.groupby('지역명').size()
culture_data.groupby('지역명').size()
festival_data.groupby('지역명').size()

festival_data[festival_data['지역명']=='영천시']




# 클러스터링
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce


# 1. 지역별 개수 집계 함수
def count_by_region(df, region_col='지역명'):
    return df[region_col].value_counts().rename_axis('지역명').reset_index(name='개수')

# 2. 각 데이터에서 지역별 개수 추출
food = count_by_region(food_data).rename(columns={'개수': '음식점'})
travel = count_by_region(travel_rest_data).rename(columns={'개수': '관광/휴양'})
culture = count_by_region(culture_data).rename(columns={'개수': '문화'})
hotel = count_by_region(hotel_data).rename(columns={'개수': '숙박'})
festival = count_by_region(festival_data).rename(columns={'개수': '축제'})

# 3. 하나의 테이블로 병합

dfs = [food, travel, culture, hotel, festival]
region_data = reduce(lambda left, right: pd.merge(left, right, on='지역명', how='outer'), dfs).fillna(0)

# ✅ 4. 변수 통합
region_data['문화/축제'] = region_data['문화'] + region_data['축제']
region_data['관광/음식'] = region_data['관광/휴양'] + region_data['음식점']

# ✅ 5. 기존 컬럼 제거 → '숙박', '문화/축제', '관광/음식'만 남김
region_data = region_data[['지역명', '숙박', '문화/축제', '관광/음식']]

# ✅ 6. 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(region_data.drop('지역명', axis=1))

# ✅ 7. K-Means 클러스터링
kmeans = KMeans(n_clusters=4, random_state=42)
region_data['cluster'] = kmeans.fit_predict(X_scaled)

# ✅ 8. 결과 확인
print(region_data.sort_values('cluster'))
print(region_data[region_data['지역명'] == '영천시'])
cluster_summary = region_data.groupby('cluster').mean(numeric_only=True)
print(cluster_summary)

# ✅ 9. PCA 시각화
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
region_data['PC1'] = pca_result[:, 0]
region_data['PC2'] = pca_result[:, 1]
print(pca.components_)


# 클러스터 번호 → 관광 유형 이름 매핑
cluster_names = {
    0: '전통 관광지',
    1: '소비 관광지',
    2: '문화 관광지',
    3: '체류형 관광지'
}


plt.figure(figsize=(10, 6))
# 영천시만 강조
yeongcheon = region_data[region_data['지역명'] == '영천시']
plt.scatter(
    yeongcheon['PC1'], yeongcheon['PC2'],
    s=300,  # 크기 키우기
    edgecolors='red', facecolors='none', linewidths=3,
    label='영천시'
)
sns.scatterplot(
    data=region_data,
    x='PC1', y='PC2',
    hue=region_data['cluster'].map(cluster_names),
    style=region_data['cluster'].map(cluster_names),
    s=100
)

# 각 지역명 텍스트로 표시
for i in range(len(region_data)):
    plt.text(
        region_data['PC1'][i] + 0.05,
        region_data['PC2'][i],
        region_data['지역명'][i],
        fontsize=9
    )

plt.title("경상북도 지역 관광 군집화 결과")
plt.axhline(1, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(4, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('관광지 규모')
plt.ylabel('전통형 vs 소비형')
plt.legend(title="관광 유형")
plt.tight_layout()
plt.show()


# 군집화 점수 판별
from sklearn.metrics import silhouette_score
score = silhouette_score(X_scaled, kmeans.labels_)
# 0. 5 ~0.7사이: 양호
print("Silhouette Score:", score)


##############################################################
# 엘보우 기법 사용해서 최적의 k찾기
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 클러스터링 대상 데이터 (지역명 제외)
X = region_data.drop(columns=['지역명', 'cluster', 'PC1', 'PC2'], errors='ignore')

# 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 엘보우 그래프
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)  # 군집 내 거리제곱합 (Within Cluster Sum of Squares)

# 시각화
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('엘보우 방법을 통한 최적 군집 수 찾기')
plt.xlabel('군집 수 (k)')
plt.ylabel('Inertia (군집 내 거리제곱합)')
plt.xticks(k_range)
plt.grid(True)
plt.show()
# 결과: 최적의 k는 4


'''
0	조용한 전통 관광지
1	소비 중심 도시 관광지
2	체류형, 문화 중심 특화지
3	축제/문화 중심 전통 관광지

'''


'''
결과 해석

1. 문화/축제 콘텐츠 강화
PC2도 낮은 수준이므로, 이벤트성 자극 요소 부족

→ 지역 축제 활성화 / 계절형 관광 요소 발굴이 필요

2. 체류형 관광 인프라 확충
숙박 자원 부족으로 인해 장기 체류 유도 어려움

→ 지역 특산물 기반 숙소, 체험형 민박, 테마형 숙소 개발 필요

3. 주변 도시와 연계형 관광지 구성
인근의 경산·청도·포항 등을 활용한 광역 관광 코스 연계 전략이 유효



✍️ 보고서용 문장 예시
영천시는 Cluster 0에 속하며, 
이는 경상북도 내에서도 음식점, 숙박시설, 문화·축제 자원이 전반적으로 부족한 지역군으로 분류됩니다. 
현재는 관광 목적지로서의 매력이 다소 약하며, 체류보다는 통과형 방문에 가까운 성격을 보입니다. 
이에 따라 지역 축제 및 체험형 콘텐츠 개발, 소규모 숙박 인프라 확충을 통해 군집 이동을 유도하고, 
경북 내 관광 거점 도시들과의 연계 전략을 강화해야 할 필요가 있습니다.
'''




import pandas as pd
import matplotlib.pyplot as plt

# 두 지역 자원 수를 비교하는 함수
def count_resources_by_city(city_name):
    return {
        '음식점': food_data[food_data['지역명'] == city_name].shape[0],
        '관광/휴양': travel_rest_data[travel_rest_data['지역명'] == city_name].shape[0],
        '문화시설': culture_data[culture_data['지역명'] == city_name].shape[0],
        '숙박': hotel_data[hotel_data['지역명'] == city_name].shape[0],
        '축제': festival_data[festival_data['지역명'] == city_name].shape[0]
    }

# 두 도시의 자원 데이터 수집
yeongcheon_resources = count_resources_by_city('영천시')
yeongju_resources = count_resources_by_city('영주시')

# 데이터프레임으로 변환
compare_df = pd.DataFrame([yeongcheon_resources, yeongju_resources], index=['영천시', '영주시'])

# 3. 자원별 비율 계산 (열 기준)
percent_df = compare_df.divide(compare_df.sum(axis=0), axis=1) * 100

# 4. 자원별 % 차이 계산
diff_df = abs(percent_df.loc['영천시'] - percent_df.loc['영주시'])

# 5. % 차이 출력
print("자원별 보유 비율 차이 (단위: %):")
print(diff_df.round(2))

# 6. 누적 비율 막대그래프 시각화
percent_df.T.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'lightcoral'])
plt.title("영천시 vs 영주시 관광 자원 비율 비교 (누적 비율)")
plt.ylabel("비율 (%)")
plt.xticks(rotation=45)
plt.legend(title="도시")
plt.tight_layout()
plt.show()

# 문화시설/숙박/축제 면에서 부족하다.