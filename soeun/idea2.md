
## ✅ 전체 분석 프레임 (3단계)

### 1. **수량 비교**

→ 다른 시군과 비교해서 *몇 개나 있는지?*

### 2. **적절성 평가**

→ 인구수, 면적, 유동인구, 시설 수 대비 *적정 수준인가?*

### 3. **공간 분석**

→ 지도에 찍어서 *어디가 부족한가?* → 위치 추천 가능

---

## ✅ 분석 지표 & 내용 아이디어 (최대한 많이)

### 📌 \[기본 통계 분석]

* **시군별 공공화장실 수**
* **화장실 수 / 인구 1만 명당**
* **화장실 수 / 면적 1km²당**
* **화장실 수 / 고령인구 비율**
* **화장실 수 / 유동인구 1만 명당**
* **화장실 수 / 관광지 수**
* **화장실 수 / 전통시장 수**
* **화장실 수 / 공공시설 수**

---

### 📌 \[공간적 시각화 분석]

* **화장실 위치 분포 지도 (folium / geopandas)**
* **읍면동별 화장실 수 Choropleth (색 진하게)**
* **주요 시설 반경 200m 이내 화장실 존재 여부 분석**
* **버스정류장, 공원, 시장 중심 Buffer 분석**
* **유동인구 대비 화장실 수 밀도 히트맵**
* **“여긴 없네?” → 사각지대 검출 지도**

---

### 📌 \[패턴/불균형 탐색]

* **화장실 수 대비 민원 건수 (불편함 지표)**
* **연도별 설치 수 증감 → 증가 중인지 정체 중인지**
* **남녀 화장실 비율 / 장애인 화장실 비율 비교**

---

## ✅ 활용 가능한 데이터 목록

| 데이터            | 설명             | 출처                           |
| -------------- | -------------- | ---------------------------- |
| 공공화장실 위치 정보    | 위치, 개방시간, 남녀 수 | 공공데이터포털                      |
| 시군별 인구/면적/고령자수 | 기준 인구지표        | KOSIS                        |
| 유동인구 (주야간/요일별) | 읍면동 단위 가능      | SKT, KT 유동인구 데이터 or Tmap API |
| 관광지/시장/공공시설 위치 | 밀집지 확인         | 공공데이터포털, OSM, 지자체            |
| 민원 정보 (화장실 관련) | 불편 제기 여부       | 영천시 홈페이지 or 정보공개요청           |
| 연도별 화장실 설치 데이터 | 시계열 분석         | 지자체 결산서, 예산서                 |
| 도로/버스정류장 위치    | 접근성 분석         | OSM + 국토부 대중교통 API           |
| 화장실 예산 지출 내역   | 정책 집중도 파악      | 영천시 예산서 / 공시자료               |

---

## ✅ 시각화 유형 모음

| 유형     | 시각화 예시                  |
| ------ | ----------------------- |
| 막대그래프  | 시군별 화장실 수 비교, 인구당 개수    |
| 선그래프   | 연도별 설치 수 추이             |
| 히트맵    | 유동인구 대비 화장실 밀도          |
| 점지도    | 화장실 위치 전체 지도 (folium)   |
| 버퍼분석   | 공원 반경 300m 이내 화장실 존재 여부 |
| 체크맵    | 사각지대(존재하지 않는 지역) 확인     |
| 레이더차트  | 인구·면적·시설수 대비 적절성 종합지표   |
| 워드클라우드 | 민원 키워드 추출 (있다면)         |

---

## ✅ 확장 가능 주제 (결합 분석)

1. **“유동인구는 많은데, 화장실은 부족한 지역은 어디?”**
2. **“고령인구 비중이 높은데 화장실이 없는 읍면동은?”**
3. **“공원/시장/관광지 중심 300m 이내에 화장실이 없는 곳은?”**
4. **“공공화장실 위치는 인구 밀도와 얼마나 일치하는가?”**
5. **“여성·장애인 화장실 비율이 낮은 읍면동은 어디?”**

---

## 👉 추천 워크플로우

1. **영천 + 경북 시군 전체 데이터 수집**
2. **인구, 면적, 유동인구 등 기준 정규화 지표 계산**
3. **folium/geopandas로 위치 시각화**
4. **버퍼 분석으로 사각지대 확인**
5. **적정성 점수 부여 → 낮은 지역 찾기**
6. **“신규 설치 우선 지역 제안”**

---
