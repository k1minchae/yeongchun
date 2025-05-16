import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False


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
nongsan = nongsan.loc[nongsan["주요품목"].isin(['사과', '포도', '복숭아'])]

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
    widgets.VBox([widgets.Label("주요품목을 선택하세요:"), checkbox_group]),
    output
])
display(layout)

# 8. 초기 지도 표시
update_map()

#####################################################################
# 과수 재배 현황
fruits = gpd.read_file("C:/Users/USER/Desktop/TIL/프로젝트/yeongchun/minchae/과수재배.shp")
fruits['crop_name'].unique()
fruits = fruits[~fruits['crop_name'].isin(['비경지', '과수_기타', '휴경지', '시설'])]

# 총 2175개중에
len(fruits.loc[fruits['crop_name'] == '사과']) / len(fruits) # 40%
len(fruits.loc[fruits['crop_name'] == '포도']) / len(fruits) # 35%
len(fruits.loc[fruits['crop_name'] == '복숭아']) / len(fruits)  # 23%


import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px

# 1. 과수 재배 데이터 불러오기 및 필터링
fruits = gpd.read_file("C:/Users/USER/Desktop/TIL/프로젝트/yeongchun/minchae/과수재배.shp")
fruits = fruits.loc[fruits['emd_name'].str.contains("영천")]  # 영천시 읍면동만 필터
fruits = fruits[fruits["crop_name"].isin(["사과", "포도", "복숭아"])]  # 세 작물만 필터

# 2. 좌표계 변환 (WGS84 위경도)
fruits = fruits.to_crs(epsg=4326)

# 3. 중심점 계산
fruits["lon"] = fruits.centroid.x
fruits["lat"] = fruits.centroid.y

# 4. 색상 지정
crop_list = ["사과", "포도", "복숭아"]
color_palette = px.colors.qualitative.Plotly
color_map = {crop: color_palette[i] for i, crop in enumerate(crop_list)}

# 5. 지도 시각화
fig = go.Figure()

# (1) 배경 폴리곤 (전체 경계)
fig.add_trace(go.Choroplethmapbox(
    geojson=fruits.__geo_interface__,
    locations=fruits.index,
    z=[1] * len(fruits),
    colorscale="Greys",
    showscale=False,
    marker_opacity=0.2,
    marker_line_width=0.5
))

# (2) 작물별 중심점 마커
for crop in crop_list:
    subset = fruits[fruits["crop_name"] == crop]
    fig.add_trace(go.Scattermapbox(
        lat=subset["lat"],
        lon=subset["lon"],
        mode="markers",
        name=crop,
        marker=go.scattermapbox.Marker(size=9, color=color_map[crop]),
        text=subset["area"],
        hovertemplate="<b>작물:</b> %{customdata[0]}<br><b>면적:</b> %{text}㎡<extra></extra>",
        customdata=subset[["crop_name"]].values
    ))

# 6. 레이아웃 설정
fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_zoom=9.3,
    mapbox_center={"lat": fruits["lat"].mean(), "lon": fruits["lon"].mean()},
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    height=650,
    title="영천시 과수 재배 현황 (사과, 포도, 복숭아)",
    legend_title="작물 종류"
)

fig.show()


# 각 과수별 경작지 KMeans 군집화
# 각 과일별 실루엣계수로 군집화 개수 결정

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. 과일 종류
crop_list = ["사과", "포도", "복숭아"]

# 2. 작물별 반복
for crop in crop_list:
    print(f"\n▶ 작물: {crop}")
    
    # (1) 해당 과일만 추출
    crop_df = fruits[fruits["crop_name"] == crop][["lat", "lon"]].dropna()
    coords = crop_df.to_numpy()
    
    if len(coords) < 3:
        print(" → 데이터가 부족하여 실루엣 계산 불가")
        continue
    
    # (2) 실루엣 계산
    silhouette_scores = []
    K = range(2, min(11, len(coords)))  # 군집 수 후보는 2~10 또는 샘플 수보다 작게

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(coords)
        score = silhouette_score(coords, labels)
        silhouette_scores.append(score)

    # (3) 최적 k 선택
    best_k = K[np.argmax(silhouette_scores)]
    print(f" → 최적 군집 수: {best_k} (실루엣 계수: {max(silhouette_scores):.4f})")

    # (4) 시각화
    plt.figure(figsize=(5, 3))
    plt.plot(K, silhouette_scores, marker='o')
    plt.title(f"{crop} - 실루엣 계수")
    plt.xlabel("군집 수 (k)")
    plt.ylabel("실루엣 계수")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


from sklearn.cluster import KMeans

apple = fruits[fruits["crop_name"] == "사과"][["lat", "lon"]]
grape = fruits[fruits["crop_name"] == "포도"][["lat", "lon"]]
peach = fruits[fruits["crop_name"] == "복숭아"][["lat", "lon"]]

# KMeans로 n개의 클러스터 중심 구하기 (후보지 후보)
kmeans_apple = KMeans(n_clusters=4, random_state=0).fit(apple[["lat", "lon"]])
kmeans_grape = KMeans(n_clusters=4, random_state=0).fit(grape[["lat", "lon"]])
kmeans_peach = KMeans(n_clusters=4, random_state=0).fit(peach[["lat", "lon"]])
후보지목록_사과 = kmeans_apple.cluster_centers_
후보지목록_포도 = kmeans_grape.cluster_centers_
후보지목록_복숭아 = kmeans_peach.cluster_centers_

# 클러스터 컬럼추가
# 1. 원본 유지 및 복사
fruits = fruits.copy()

# 2. 사과 클러스터 라벨 추가
fruits.loc[fruits["crop_name"] == "사과", "cluster"] = kmeans_apple.labels_

# 3. 포도 클러스터 라벨 추가
fruits.loc[fruits["crop_name"] == "포도", "cluster"] = kmeans_grape.labels_

# 4. 복숭아 클러스터 라벨 추가
fruits.loc[fruits["crop_name"] == "복숭아", "cluster"] = kmeans_peach.labels_

# 5. 정수형으로 변환
fruits["cluster"] = fruits["cluster"].astype(int)



# 계산
from haversine import haversine, Unit
import numpy as np
import pandas as pd

# 기존 유통센터 좌표
existing_centers = nongsan[["위도", "경도"]].to_numpy()

# 결과 저장
best_result = {
    "개선율": -np.inf,
    "작물": None,
    "후보지": None,
    "기존평균": None,
    "개선평균": None
}

# 작물별 기존 평균 거리 저장
average_distances = []

# 거리 계산 함수
def avg_dist_to_centers(df, centers):
    return df.apply(
        lambda row: min(haversine((row["lat"], row["lon"]), (c[0], c[1]), unit=Unit.KILOMETERS) for c in centers),
        axis=1
    ).mean()

# 작물별 반복
for crop_name, candidate_list in [
    ("사과", 후보지목록_사과),
    ("포도", 후보지목록_포도),
    ("복숭아", 후보지목록_복숭아),
]:
    crop_df = fruits[fruits["crop_name"] == crop_name][["lat", "lon"]].reset_index(drop=True)

    # 기존 평균 거리 계산
    기존평균 = avg_dist_to_centers(crop_df, existing_centers)
    average_distances.append({
        "작물": crop_name,
        "기존 평균거리 (km)": round(기존평균, 2)
    })

    # 후보지별 개선율 평가
    for candidate in candidate_list:
        개선센터 = np.vstack([existing_centers, candidate])
        개선평균 = avg_dist_to_centers(crop_df, 개선센터)
        개선율 = (기존평균 - 개선평균) / 기존평균 * 100

        if 개선율 > best_result["개선율"]:
            best_result.update({
                "개선율": 개선율,
                "작물": crop_name,
                "후보지": candidate,
                "기존평균": 기존평균,
                "개선평균": 개선평균
            })

# 출력: 작물별 기존 평균 거리
print("📌 작물별 기존 평균 거리:")
for row in average_distances:
    print(f"- {row['작물']}: {row['기존 평균거리 (km)']} km")

# 출력: 최적 후보지 정보
print("\n✅ 가장 개선 효과가 큰 후보지:")
print(f"- 작물: {best_result['작물']}")
print(f"- 후보지 위도, 경도: {best_result['후보지']}")
print(f"- 기존 평균 거리: {best_result['기존평균']:.2f} km")
print(f"- 후보지 추가 후 평균 거리: {best_result['개선평균']:.2f} km")
print(f"- 평균 거리 개선율: {best_result['개선율']:.2f} %")


# 각 작물별 후보지 시각화
from haversine import haversine, Unit
import numpy as np
import pandas as pd

# 기존 유통센터 좌표
existing_centers = nongsan[["위도", "경도"]].to_numpy()

# 거리 계산 함수
def avg_dist_to_centers(df, centers):
    return df.apply(
        lambda row: min(haversine((row["lat"], row["lon"]), (c[0], c[1]), unit=Unit.KILOMETERS) for c in centers),
        axis=1
    ).mean()

# 작물별 최적 후보지 결과 리스트
crop_results = []

for crop_name, candidate_list in [
    ("사과", 후보지목록_사과),
    ("포도", 후보지목록_포도),
    ("복숭아", 후보지목록_복숭아),
]:
    crop_df = fruits[fruits["crop_name"] == crop_name][["lat", "lon"]].reset_index(drop=True)

    # 기존 평균 거리
    기존평균 = avg_dist_to_centers(crop_df, existing_centers)

    # 후보지 중 가장 개선 효과 큰 것 찾기
    best_candidate = None
    best_개선평균 = None
    best_개선율 = -np.inf

    for candidate in candidate_list:
        개선센터 = np.vstack([existing_centers, candidate])
        개선평균 = avg_dist_to_centers(crop_df, 개선센터)
        개선율 = (기존평균 - 개선평균) / 기존평균 * 100

        if 개선율 > best_개선율:
            best_candidate = candidate
            best_개선평균 = 개선평균
            best_개선율 = 개선율

    crop_results.append({
        "작물": crop_name,
        "후보지 위도": round(best_candidate[0], 6),
        "후보지 경도": round(best_candidate[1], 6),
        "기존 평균거리 (km)": round(기존평균, 2),
        "개선 평균거리 (km)": round(best_개선평균, 2),
        "평균 거리 개선율 (%)": round(best_개선율, 2)
    })

# 결과 DataFrame 출력
results_df = pd.DataFrame(crop_results)
display(results_df)




# 📌 작물 목록 및 체크박스 생성
crops = sorted(fruits["crop_name"].dropna().unique().tolist())
color_palette = px.colors.qualitative.Plotly  # 색상 리스트
color_map = {item: color_palette[i % len(color_palette)] for i, item in enumerate(crops)}

checkboxes = [
    widgets.Checkbox(value=False, description=crop, layout=widgets.Layout(width="200px"))
    for crop in crops
]
checkbox_group = widgets.VBox(checkboxes, layout=widgets.Layout(width="220px"))
output = widgets.Output()

# 📌 지도 업데이트 함수
def update_plot(change=None):
    with output:
        clear_output(wait=True)
        selected_crops = [cb.description for cb in checkboxes if cb.value]
        if not selected_crops:
            print("하나 이상의 작물을 선택하세요.")
            return

        filtered_df = fruits[fruits["crop_name"].isin(selected_crops)]

        # 클러스터별 평균 좌표, 총 면적 계산
        grouped = filtered_df.groupby(["crop_name", "cluster"]).agg({
            "lat": "mean",
            "lon": "mean",
            "area": "sum"
        }).reset_index()

        # 면적 기준 마커 크기
        grouped["size"] = grouped["area"] / grouped["area"].max() * 40 + 10

        # 📌 지도 그리기 시작
        fig = go.Figure()

        # 📌 행정경계 (emd) 표시
        fig = fig.add_trace(go.Choroplethmapbox(
            geojson=emd.__geo_interface__,
            locations=emd.index,
            z=[1]*len(emd),
            showscale=False,
            marker_opacity=0.3,
            marker_line_width=1,
            colorscale="Greys"
        ))

        # 📌 작물별 군집 마커 표시
        for crop in selected_crops:
            sub = grouped[grouped["crop_name"] == crop]
            fig = fig.add_trace(go.Scattermapbox(
                lat=sub["lat"],
                lon=sub["lon"],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=sub["size"],
                    color=color_map[crop],
                    opacity=0.7
                ),
                text=[
                    f"{crop} - 군집 {row.cluster}<br>면적: {int(row.area)}㎡"
                    for _, row in sub.iterrows()
                ],
                hoverinfo='text',
                name=crop
            ))
        output.clear_output(wait=True)
        selected_items = [cb.description for cb in checkboxes if cb.value]
        filtered = nongsan[nongsan["주요품목"].isin(selected_items)]
    
        for item in selected_items:
            sub_df = filtered[filtered["주요품목"] == item]
            fig = fig.add_trace(go.Scattermapbox(
                lat=sub_df["위도"],
                lon=sub_df["경도"],
                mode='markers',
                name=item,  # 범례 이름
                marker=go.scattermapbox.Marker(size=10, color='red'),
                text=sub_df["주요품목"],
                hovertemplate="<b>센터명:</b> %{customdata[0]}<br><b>주요품목:</b> %{text}<extra></extra>",
                customdata=sub_df[["사업장명"]].values,
                showlegend=True
            ))

        fig = fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=10,
            mapbox_center={
                "lat": fruits["lat"].mean(),
                "lon": fruits["lon"].mean()
            },
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            height=700,
            title="작물별 클러스터 평균 위치 및 면적 (행정동 경계 포함)"
        )

        fig.show()

# 📌 체크박스에 이벤트 연결
for cb in checkboxes:
    cb.observe(update_plot, names="value")

# 📌 레이아웃 구성 및 출력
layout = widgets.HBox([
    widgets.VBox([widgets.Label("작물을 선택하세요:"), checkbox_group]),
    output
])
display(layout)
update_plot()