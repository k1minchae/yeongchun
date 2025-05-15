import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

cross_df = pd.read_csv('../data/crosswalk.csv')
cross_df[cross_df['시도명'] == '경상북도']['시군구명'].unique()


# 예: Ames, Iowa 경계 폴리곤 Shapefile 로드
gdf = gpd.read_file("../../kb/kb.shp")

# 좌표계 설정 (필요 시)
gdf = gdf.to_crs(epsg=4326)
gdf = gdf[gdf['SGG_NM'] == '경상북도 영천시']

# 폴리곤 맵에 그리기
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, edgecolor='black', facecolor='lightblue', alpha=0.5)
plt.show()