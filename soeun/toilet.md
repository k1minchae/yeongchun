
## 🧠 주제

**영천시 고령 인구의 도보 이동 기반 공공화장실 접근성 분석**

## 🎯 목적

고령 인구 기준 도보 10\~15분 거리 내 공공화장실 밀집도를 분석하여
생활권 내 **화장실 접근성의 사각지대를 파악**하고
**우선 개선 지역을 도출**하는 데 목적이 있음

---

## 1. 데이터 수집 및 준비

* **인구통계 데이터**

  * 고령 인구 비율, 읍면동별 65세 이상 인구, 인구밀도
  * 출처: KOSIS, 통계청

* **공공화장실 위치 데이터**

  * 위치 좌표, 개방시간, 남녀/장애인 화장실 수
  * 출처: 공공데이터포털

* **도로망 데이터**

  * 도보 이동 가능한 경로 추정용 OSM 도로 네트워크
  * 출처: OpenStreetMap, osmnx

* **유동인구 데이터 (보조)**

  * 주야간 유동인구 대비 화장실 수 분석용
  * 출처: SKT 유동인구 API, T-map, KT 데이터

* **고령자 주요 동선 정보**

  * 전통시장, 복지관, 병원 등 고령자 주요 이용시설 위치
  * 출처: 지자체 공공시설 위치 데이터

---

## 2. 데이터 전처리

* **좌표계 일치**

  * 모든 데이터 WGS84 또는 EPSG:5181 등으로 통일

* **결측치 및 이상치 처리**

  * 위치 누락된 화장실 제거
  * 중복, 비정상 좌표 정제

* **시설군 정제**

  * 사용 대상별로 필터링 (공공화장실만 선택)
  * 시설 운영시간 기준 필터 (24시간 화장실 여부)

---

## 3. 기술적 분석

* **도보 10\~15분 거리 내 접근성 분석**

  * 도보 반경 500m~800m (경사 고려 시 400m~700m까지 축소 가능)
  * 주요 생활거점(시장, 복지관 등) 중심 버퍼 내 화장실 수 count
  * 전체 읍면동 기준 ‘도보 생활권 외곽지역’ 확인

* **공공화장실 적정성 평가**

  * 인구 1만 명당 / 면적 1km²당 / 고령인구 대비 / 시장·관광지 기준 대비 수치 분석
  * “인구 밀도는 높은데 화장실이 적은 곳” or “시장 근처인데 없는 곳” 식의 불균형 탐지

* **공공화장실 설치 사각지대 탐색**

  * 주요 보행 루트 상에서 buffer로 공공화장실 미존재 지역 확인
  * 쉼터, 정류장, 공원 주변 buffer (300m 이내 없음) 분석

---

## 4. 시각화

* **위치 시각화 (folium)**

  * 공공화장실 위치 점 지도
  * 쉼터, 정류장, 시장 등 다른 시설 오버레이

* **Choropleth 지도**

  * 읍면동별 고령자당 공공화장실 수 색상 시각화

* **접근성 히트맵**

  * 도보 10\~15분 내 커버되는 지역은 녹색, 사각지대는 회색/빨간색 등

* **버퍼 분석 시각화**

  * 시장, 병원 등 주요 지점 반경 300m 이내 화장실 유무 분석

* **종합지표 레이더차트**

  * 인구·면적·시설 대비 적정성 지표 시각화

---

## 5. 분석 및 결과 도출

* **접근성 사각지대 파악**

  * 보행 기준 생활권에 공공화장실이 부재한 지역 도출
  * 시장·복지관·공원 근처임에도 접근성이 낮은 지점 확인

* **정책 우선순위 설정**

  * 고령자 밀집 지역 중 시설 부족 지역 → 설치 우선순위 제안
  * 도보 이용률 높은 구간 중 화장실 미설치 지역 표시 → 개선 대상

---

## 6. 결과 요약 및 제언

* **주요 발견 사항 요약**

  * 고령자 활동 밀집 지역 중 다수는 공공화장실이 멀거나 접근 곤란
  * 일부 시장·복지시설 반경 300m 내 화장실 없음
  * 읍면동 간 시설 밀집도 격차 존재

* **정책 제안**

  * “시장 주변 쉼터와 병행된 공공화장실 추가 설치”
  * “고령자 주요 보행 루트 상 화장실 표지판, 유도체계 강화”
  * “점자블록·의자·경사로 보완 포함한 보행환경 종합 개선”

* **향후 연구 방향**

  * 민원 데이터를 활용한 체감 불편도 추가 분석
  * 실시간 화장실 개방 정보 API와 연동한 서비스 개발 가능성

---

