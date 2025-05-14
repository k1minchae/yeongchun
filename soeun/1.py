import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px


yc_fruit = gpd.read_file('../../data/yc.shp')
yc_b = gpd.read_file('../../data/yc_b.shp')
yc_fac = gpd.read_file('../../data/yc_fac.shp')
yc_garlic = gpd.read_file('../../data/yc_garlic.shp')
yc_rest = gpd.read_file('../../data/yc_rest.shp')
yc_rice = gpd.read_file('../../data/yc_rice.shp')


yc_fruit.info()
yc_b.info()
yc_fac.info()
yc_garlic.info()
yc_rest.info()
yc_rice.info()


yc_fruit = yc_fruit.to_crs(epsg=4326)
yc_b = yc_b.to_crs(epsg=4326)
yc_fac = yc_fac.to_crs(epsg=4326)
yc_garlic = yc_garlic.to_crs(epsg=4326)
yc_rest = yc_rest.to_crs(epsg=4326)
yc_rice = yc_rice.to_crs(epsg=4326)


fig, ax = plt.subplots(figsize=(10, 10))
yc_fruit.plot(ax=ax, edgecolor='red', facecolor='lightblue', alpha=0.5)
yc_b.plot(ax=ax, edgecolor='orange', facecolor='lightblue', alpha=0.5)
yc_fac.plot(ax=ax, edgecolor='blue', facecolor='lightblue', alpha=0.5)
yc_garlic.plot(ax=ax, edgecolor='green', facecolor='lightblue', alpha=0.5)
yc_rest.plot(ax=ax, edgecolor='purple', facecolor='lightblue', alpha=0.5)
yc_rice.plot(ax=ax, edgecolor='pink', facecolor='lightblue', alpha=0.5)
plt.show()

yc_fruit['crop_name'].value_counts()
yc_b['crop_name'].value_counts()
yc_fac['crop_name'].value_counts()
yc_garlic['crop_name'].value_counts()
yc_rest['crop_name'].value_counts()
yc_rice['crop_name'].value_counts()

len(yc_rice['jibun'].unique())
len(yc_rice['pnu'].unique())

us_fruit = yc_fruit[yc_fruit['emd_name'].str.contains('의성군')]
us_b = yc_b[yc_b['emd_name'].str.contains('의성군')]
us_fac = yc_fac[yc_fac['emd_name'].str.contains('의성군')]
us_garlic = yc_garlic[yc_garlic['emd_name'].str.contains('의성군')]
us_rest = yc_rest[yc_rest['emd_name'].str.contains('의성군')]
us_rice = yc_rice[yc_rice['emd_name'].str.contains('의성군')]

yc_fruit = yc_fruit[yc_fruit['emd_name'].str.contains('영천시')]
yc_b = yc_b[yc_b['emd_name'].str.contains('영천시')]
yc_fac = yc_fac[yc_fac['emd_name'].str.contains('영천시')]
yc_garlic = yc_garlic[yc_garlic['emd_name'].str.contains('영천시')]
yc_rest = yc_rest[yc_rest['emd_name'].str.contains('영천시')]
yc_rice = yc_rice[yc_rice['emd_name'].str.contains('영천시')]


us_df = pd.concat([us_fruit, us_b, us_fac, us_garlic, us_rest, us_rice], axis=0, ignore_index=True)
yc_df = pd.concat([yc_fruit, yc_b, yc_fac, yc_garlic, yc_rest, yc_rice], axis=0, ignore_index=True)

us_df.info()
yc_df.info()


us_df['crop_name'].value_counts()
yc_df['crop_name'].value_counts()

us_df.groupby('crop_name').sum('area')

us_df.info()

us_df.to_csv('../../data/us_df_250513.csv')
yc_df.to_csv('../../data/yc_df_250513.csv')



# ---
# (하)
# 작물별 재배 면적 Top 10 시각화
# ---


us_crop_name = yc_df['crop_name'].unique()
yc_crop_name = yc_df['crop_name'].unique()

us_filtered_crop_name = [name for name in us_crop_name if all(x not in name for x in ['기타', '경지', '시설'])]
yc_filtered_crop_name = [name for name in yc_crop_name if all(x not in name for x in ['기타', '경지', '시설'])]

us_filtered_df = us_df[us_df['crop_name'].isin(us_filtered_crop_name)]
yc_filtered_df = yc_df[yc_df['crop_name'].isin(yc_filtered_crop_name)]

fig = px.bar(
    us_filtered_df,
    x='crop_name',
    y='area',
    title='작물별 재배 면적',
    labels={'crop_name': '작물명', 'area': '총 재배 면적 (㎡)'},
    text='area'
)
fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
fig.update_layout(
    xaxis_tickangle=-45,
    yaxis_title='재배 면적 (㎡)',
    margin=dict(t=50, b=100),
    height=500,
    width=1000
)

fig.show()




# ---
import pandas as pd
import plotly.express as px

# 1. 작물별 재배 면적 합산
us_area = us_filtered_df.groupby('crop_name')['area'].sum().reset_index()
us_area.columns = ['crop_name', 'area_us']

yc_area = yc_filtered_df.groupby('crop_name')['area'].sum().reset_index()
yc_area.columns = ['crop_name', 'area_yc']

# 2. 병합 + 결측치 0 처리
merged = pd.merge(us_area, yc_area, on='crop_name', how='outer').fillna(0)

# 3. total 면적 기준 내림차순 정렬 후 Top 10
merged['total'] = merged['area_us'] + merged['area_yc']
top_crop = merged.sort_values(by='total', ascending=False).head(10)

# 4. plotly용 긴 형태로 변환
long_df = pd.melt(
    top_crop,
    id_vars='crop_name',
    value_vars=['area_us', 'area_yc'],
    var_name='지역',
    value_name='면적'
)

# 5. 컬럼 이름 보기 좋게 맵핑
long_df['지역'] = long_df['지역'].map({'area_us': '의성시', 'area_yc': '영천시'})

# 6. 그룹형 바 차트 그리기
fig = px.bar(
    long_df,
    x='crop_name',
    y='면적',
    color='지역',
    barmode='group',  # ← 요게 핵심!
    title='작물별 재배 면적 Top 10 비교 (영천시 vs 의성시)',
    labels={'crop_name': '작물명'}
)

fig.update_layout(
    xaxis_tickangle=-45,
    height=600
)

fig.show()





# ----
# 읍면동별 작물 재배 총면적 지도
# ---


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import numpy as np

# 1. 영천시 crop 데이터 불러오기 (예: 이미 필터링된 yc_gdf)
yc_gdf = yc_df.copy()
yc_gdf = yc_gdf.to_crs(epsg=5181)  # meter 단위로 좌표계 변환 (중요!)

# 2. hex grid 생성 함수
def generate_hex_grid(bounds, hex_size):
    xmin, ymin, xmax, ymax = bounds
    width = 3**0.5 * hex_size
    height = 2 * hex_size
    rows = int((ymax - ymin) / height) + 2
    cols = int((xmax - xmin) / width) + 2

    hexes = []
    for row in range(rows):
        for col in range(cols):
            x = xmin + col * width
            if row % 2 == 1:
                x += width / 2
            y = ymin + row * height * 0.75
            hex = Polygon([
                (x + hex_size * np.cos(np.radians(angle)),
                 y + hex_size * np.sin(np.radians(angle)))
                for angle in range(0, 360, 60)
            ])
            hexes.append(hex)
    return gpd.GeoDataFrame(geometry=hexes, crs=yc_gdf.crs)

# 3. hex grid 생성
hex_size = 500  # 반지름 단위 (500m)
hex_grid = generate_hex_grid(yc_gdf.total_bounds, hex_size)

# 4. Spatial join으로 각 hex 타일에 포함된 농지 필터링
joined = gpd.sjoin(yc_gdf[['geometry', 'area']], hex_grid, how='inner', predicate='intersects')

# 5. hex별 area 합산
hex_area = joined.groupby('index_right')['area'].sum()
hex_grid['area_sum'] = hex_grid.index.map(hex_area).fillna(0)

# 6. 시각화
fig, ax = plt.subplots(figsize=(12, 12))
hex_grid.plot(column='area_sum', cmap='YlOrRd', edgecolor='gray', linewidth=0.2, legend=True, ax=ax)
ax.set_title('영천시 육각 타일 기반 작물 재배 면적 분포', fontsize=16)
ax.axis('off')
plt.tight_layout()
plt.show()




# ---
# 비경지/논/과수 분포 지도
# ---

import json

def classify_crop(crop):
    if any(x in crop for x in ['비경지', '휴경지']):
        return '비/휴경지'
    elif any(x in crop for x in ['논_벼']):
        return '식량작물'
    elif any(x in crop for x in ['사과', '포도', '복숭아', '배', '과수_기타']):
        return '과수'
    elif any(x in crop for x in ['마늘', '양파', '배추']):
        return '채소'
    else:
        return '기타'


gdf = yc_df.copy()
gdf = gdf.to_crs(epsg=4326)

yc_df['crop_name'].value_counts()

# 2. 분류 컬럼 추가
gdf['crop_group'] = gdf['crop_name'].apply(classify_crop)

# 3. 표시할 그룹만 필터링
plot_gdf = gdf[gdf['crop_group'].isin(['비경지', '식량작물', '과수', '채소'])]


import plotly.express as px

# 색상 지정
color_dict = {
    '비경지': 'lightgray',
    '식량작물': 'gold',
    '과수': 'red',
    '채소': 'green'
}


# 5. 시각화
fig, ax = plt.subplots(figsize=(12, 12))

for group, data in plot_gdf.groupby('crop_group'):
    data.plot(ax=ax, color=color_dict[group], label=group, linewidth=0)

# 스타일 꾸미기
ax.set_title('영천시 작물 유형별 분포 지도', fontsize=16)
ax.axis('off')
ax.legend(title='작물 유형', loc='lower left')
plt.tight_layout()
plt.show()