import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from shapely.geometry import Point

a1214 = pd.read_csv('./data/a1214_2.csv')
a15 = pd.read_csv('./data/a15_2.csv')
a16 = pd.read_csv('./data/a16_2.csv')
a17 = pd.read_csv('./data/a17_2.csv')
a18 = pd.read_csv('./data/a18_2.csv')
a19 = pd.read_csv('./data/a19_2.csv')
a20 = pd.read_csv('./data/a20_2.csv')
a21 = pd.read_csv('./data/a21_2.csv')
a22 = pd.read_csv('./data/a22_2.csv')

df = pd.concat([a1214, a15, a16, a17, a18, a19, a20, a21, a22], axis=0, ignore_index=True)

ycdf = df[df['발생지시군구'] == '영천시']

ycdf


# 예: Ames, Iowa 경계 폴리곤 Shapefile 로드
gdf = gpd.read_file("../../kb/kb.shp")

# 좌표계 설정 (필요 시)
gdf = gdf.to_crs(epsg=4326)
gdf = gdf[gdf['SGG_NM'] == '경상북도 영천시']

# 폴리곤 맵에 그리기
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, edgecolor='black', facecolor='lightblue', alpha=0.5)
plt.show()


# ---


# 위도, 경도를 Point로 변환
ycdf['geometry'] = ycdf.apply(lambda row: Point(row['경도'], row['위도']), axis=1)

# GeoDataFrame으로 변환 (WGS84 좌표계로 설정)
accident_gdf = gpd.GeoDataFrame(ycdf, geometry='geometry', crs='EPSG:4326')

fig, ax = plt.subplots(figsize=(12, 12))

# 영천시 polygon 배경 지도
gdf.plot(ax=ax, color='lightgray', edgecolor='black')

# 사고 지점 찍기
accident_gdf.plot(ax=ax, color='red', markersize=5, alpha=0.3, label='교통사고')

ax.set_title('영천시 교통사고 위치', fontsize=16)
ax.axis('off')
ax.legend()
plt.tight_layout()
plt.show()