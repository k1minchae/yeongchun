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


import plotly.graph_objects as go
# 1. 읍면동 SHP 파일 읽기 및 좌표계 변환
emd = gpd.read_file("C:/Users/USER/Desktop/TIL/프로젝트/yeongchun/dashboard/asset/data/map/영천시읍면동.shp")
emd = emd.to_crs(epsg=4326)

# 2. 유통센터 데이터 불러오기
nongsan = pd.read_csv("C:/Users/USER/Desktop/TIL/프로젝트/yeongchun/dashboard/asset/data/유통센터위도경도.csv", encoding='cp949')

# 3. Mapbox 토큰 설정 (공용 토큰 또는 개인 토큰 필요 시 아래 주석 해제)
# px.set_mapbox_access_token('your_mapbox_token_here')

# 4. Choropleth (읍면동 경계)
fig = go.Figure(go.Choroplethmapbox(
    geojson=emd.__geo_interface__,  # GeoJSON 변환
    locations=emd.index,
    z=[1]*len(emd),  # 색상은 동일하게
    colorscale="Greys",
    showscale=False,
    marker_opacity=0.3,
    marker_line_width=1
))

# 5. 유통센터 점 찍기 (클릭 시 이름 표시)
fig.add_trace(go.Scattermapbox(
    lat=nongsan["위도"],
    lon=nongsan["경도"],
    mode='markers',
    marker=go.scattermapbox.Marker(size=10, color='red'),
    text=nongsan["주요품목"],            # hover 시 보이게 할 텍스트
    hovertemplate="<b>센터명:</b> %{customdata[0]}<br>" + 
                  "<b>주요품목:</b> %{text}<extra></extra>",
    customdata=nongsan[["사업장명"]].values  # %{customdata[0]} → 센터명
))

# 6. 레이아웃 설정
fig.update_layout(
    mapbox_style="carto-positron",  # 또는 "open-street-map", "carto-darkmatter"
    mapbox_zoom=9.5,
    mapbox_center={"lat": nongsan["위도"].mean(), "lon": nongsan["경도"].mean()},
    margin={"r":0, "t":50, "l":0, "b":0},
    title="영천시 농산물 유통센터 위치"
)

fig.show()




# dropbox 연동
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

# 1. 데이터 불러오기
emd = gpd.read_file("C:/Users/USER/Desktop/TIL/프로젝트/yeongchun/dashboard/asset/data/map/영천시읍면동.shp").to_crs(epsg=4326)
nongsan = pd.read_csv("C:/Users/USER/Desktop/TIL/프로젝트/yeongchun/dashboard/asset/data/유통센터위도경도.csv", encoding='cp949')

# 2. 주요품목 목록
items = sorted(nongsan["주요품목"].dropna().unique().tolist())

# 3. 체크박스 생성
checkboxes = [widgets.Checkbox(value=False, description=item) for item in items]
checkbox_group = widgets.VBox(checkboxes)

# 4. 버튼 추가
button = widgets.Button(description="지도 업데이트")
output = widgets.Output()

# 5. 버튼 클릭 시 지도 업데이트 함수
def on_button_click(b):
    output.clear_output()
    selected_items = [cb.description for cb in checkboxes if cb.value]
    filtered = nongsan[nongsan["주요품목"].isin(selected_items)]
    
    if filtered.empty:
        with output:
            print("선택된 품목에 해당하는 유통센터가 없습니다.")
        return

    fig = go.Figure(go.Choroplethmapbox(
        geojson=emd.__geo_interface__,
        locations=emd.index,
        z=[1]*len(emd),
        colorscale="Greys",
        showscale=False,
        marker_opacity=0.3,
        marker_line_width=1
    ))

    fig.add_trace(go.Scattermapbox(
        lat=filtered["위도"],
        lon=filtered["경도"],
        mode='markers',
        marker=go.scattermapbox.Marker(size=10, color='red'),
        text=filtered["주요품목"],
        hovertemplate="<b>센터명:</b> %{customdata[0]}<br><b>주요품목:</b> %{text}<extra></extra>",
        customdata=filtered[["사업장명"]].values
    ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=9.5,
        mapbox_center={"lat": nongsan["위도"].mean(), "lon": nongsan["경도"].mean()},
        margin={"r":0, "t":50, "l":0, "b":0},
        title="영천시 농산물 유통센터 - 선택된 주요품목"
    )

    with output:
        fig.show()

button.on_click(on_button_click)

# 6. 전체 위젯 출력
display(widgets.Label("표시할 주요품목을 선택하세요:"), checkbox_group, button, output)