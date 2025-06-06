
## 🧠 주제 요약

* **목표**: 고령 질환(특히 치매) 진료 인원과 장기요양보험 등급판정 현황을 활용해
  영천시의 **요양시설 수요를 간접적으로 추정**하고 **정책적 인사이트 도출**

---

## 📂 사용할 데이터 정리

* **① 노인장기요양보험 등급판정 현황**

  * 열 예상: 연도, 시도, 시군구, 등급구분(1\~5등급, 인지지원 등), 판정자 수
  * 필터: “경상북도” + “영천시”

* **② 시군구별 치매질환 진료 통계**

  * 열 예상: 연도, 시군구, 진료건수, 진료인원, 의원급/병원급 구분 가능성
  * 필터: “영천시” + “치매 관련 코드 (F00\~F03)” 포함

---

## 📊 분석 및 시각화 흐름

### 1. **시계열 비교 시각화: 등급판정자 수 vs 치매 진료인원**

* 목적: 치매 진료 증가 → 요양등급 신청 증가 → 요양시설 수요 증가 흐름 확인
* X축: 연도 / Y축: 진료인원 & 등급판정자 수 (이중축 가능)
* 시각화: 선그래프 (`plotly.express.line()` 또는 `plt.plot()`)

### 2. **등급별 분포 분석**

* 목적: 영천시 내 요양 수요의 등급별 구조 파악
* X축: 등급 (1\~5등급, 인지지원 등) / Y축: 인원 수
* 시각화: 막대그래프 (stacked 또는 grouped bar)

### 3. **영천 vs 인근 시군구 비교**

* 목적: 지역별 고령진료 부담 차이 분석 → 상대적 수요 해석
* 지표: 인구 1천 명당 등급판정자 수 or 치매 진료인원
* 지역: 영천, 경산, 청도, 대구 달성 등
* 시각화: 수평 막대그래프 또는 레이더 차트

### 4. **지도 시각화 (선택 사항)**

* 목적: 경북 내 요양 수요 분포의 공간적 시각화
* 도구: `GeoJSON` + `folium` or `geopandas`
* 색상: 등급판정자 수 또는 치매 진료율 기준

### 5. **요약 지표 카드형 시각화 (대시보드용)**

* 목적: 요양 수요 관련 핵심 지표를 카드형으로 표시
* 예시 지표:

  * 치매 진료인원: 1,258명
  * 장기요양 등급판정자: 632명
  * 1\~2등급 비율: 42%
  * 추정 요양시설 부족률: 18%

---

## 📈 예측 확장 (고급 옵션)

* 과거 5년 추세 기반으로 예측 모델 적용

  * 선형 회귀 or 시계열 모델 (예: ARIMA)
* 예측 지표:

  * 2025\~2027년 치매 진료인원 & 요양등급자 수
* 시각화:

  * 예측값은 선그래프에 점선으로 표시
* 인사이트 예:

  > “영천시는 2027년까지 약 750명 이상의 요양등급자가 필요할 것으로 보입니다.”

---

## 🧾 최종 결과물 아이디어

### 📊 Quarto 기반 대시보드 섹션 구성

1. 영천시 고령화 및 치매 개요
2. 치매 진료 추세 분석
3. 장기요양 등급판정자 추이 및 등급별 구조
4. 지역 간 비교 분석
5. 요양시설 수요 예측 및 정책 제언

---

## ✍ 인사이트 도출 예시

* 영천시는 **1\~2등급 고위험군 비중**이 높아, **시설형 요양기관 수요가 상대적으로 큼**
* 등급판정자 수에 비해 **치매 진료인원이 많지 않음** → **숨겨진 요양 수요** 존재 가능성
* 2027년까지 요양등급자 수는 **X% 증가**할 것으로 예측됨 → **시설·인력 대비 필요**

---
