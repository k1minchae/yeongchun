import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# 시군구
sgg = gpd.read_file("../dashboard/asset/data/map/경상북도시군구.shp").to_crs(epsg=4326)

fig = px.choropleth(
    sgg,
    geojson=sgg.geometry,
    locations=sgg.index,
    color="ADZONE_NM",
    hover_name="ADZONE_NM",  # 마우스 올렸을 때 표시할 텍스트
    title="경상북도 시군구 지도"
)

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(
    margin={"r":0,"t":50,"l":0,"b":0},
    height=600
)

fig.show()



# 읍면동
# 1. SHP 파일 읽기 및 필터링
emd = gpd.read_file("C:/Users/USER/Desktop/TIL/프로젝트/yeongchun/dashboard/asset/data/map/영천시읍면동.shp")

# 2. 좌표계 WGS84로 변환 (plotly에서 필요)
emd = emd.to_crs(epsg=4326)

# 3. Plotly Express를 이용한 시각화
fig = px.choropleth(
    emd,
    geojson=emd.geometry,
    locations=emd.index,
    color="ADZONE_NM",  # 읍면동 이름에 따라 색상 다르게
    hover_name="ADZONE_NM",  # 마우스 올렸을 때 표시할 텍스트
    title="영천시 주요 읍면동 지도"
)

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(
    margin={"r":0,"t":50,"l":0,"b":0},
    height=600
)

fig.show()

# 유통센터 위도 경도
nongsan = pd.read_csv('C:/Users/USER/Desktop/TIL/프로젝트/yeongchun/dashboard/asset/data/유통센터위도경도.csv', encoding='cp949')


# 유통센터 시각화
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output

# 1. 데이터 불러오기
emd = gpd.read_file("C:/Users/USER/Desktop/TIL/프로젝트/yeongchun/dashboard/asset/data/map/영천시읍면동.shp").to_crs(epsg=4326)
nongsan = pd.read_csv("C:/Users/USER/Desktop/TIL/프로젝트/yeongchun/dashboard/asset/data/유통센터위도경도.csv", encoding='cp949')

# 2. 작물 목록 및 색상 팔레트
items = sorted(nongsan["주요품목"].dropna().unique().tolist())
color_palette = px.colors.qualitative.Plotly  # 색상 리스트
color_map = {item: color_palette[i % len(color_palette)] for i, item in enumerate(items)}

# 3. 체크박스 생성
checkboxes = [
    widgets.Checkbox(value=False, description=item, layout=widgets.Layout(width='200px'))
    for item in items
]
checkbox_group = widgets.VBox(checkboxes, layout=widgets.Layout(width='220px'))
output = widgets.Output()

# 4. 지도 업데이트 함수
def update_map(change=None):
    output.clear_output(wait=True)
    selected_items = [cb.description for cb in checkboxes if cb.value]
    filtered = nongsan[nongsan["주요품목"].isin(selected_items)]
    
    fig = go.Figure(go.Choroplethmapbox(
        geojson=emd.__geo_interface__,
        locations=emd.index,
        z=[1]*len(emd),
        colorscale="Greys",
        showscale=False,
        marker_opacity=0.3,
        marker_line_width=1
    ))

    # 5. 작물별로 마커 색상 다르게 추가
    for item in selected_items:
        sub_df = filtered[filtered["주요품목"] == item]
        fig.add_trace(go.Scattermapbox(
            lat=sub_df["위도"],
            lon=sub_df["경도"],
            mode='markers',
            name=item,  # 범례 이름
            marker=go.scattermapbox.Marker(size=10, color=color_map[item]),
            text=sub_df["주요품목"],
            hovertemplate="<b>센터명:</b> %{customdata[0]}<br><b>주요품목:</b> %{text}<extra></extra>",
            customdata=sub_df[["사업장명"]].values,
            showlegend=True
        ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=9.5,
        mapbox_center={"lat": nongsan["위도"].mean(), "lon": nongsan["경도"].mean()},
        margin={"r":0, "t":50, "l":0, "b":0},
        height=600,
        title="영천시 농산물 유통센터 - 선택된 주요품목",
        legend_title="주요품목"
    )

    with output:
        fig.show()

# 6. 체크박스에 이벤트 연결
for cb in checkboxes:
    cb.observe(update_map, 'value')

# 7. 화면 표시
layout = widgets.HBox([
    widgets.VBox([widgets.Label("표시할 주요품목을 선택하세요:"), checkbox_group]),
    output
])
display(layout)

# 8. 초기 지도 표시
update_map()