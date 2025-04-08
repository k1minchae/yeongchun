import pandas as pd


df = pd.read_csv(
    "/Users/parkhansl/Desktop/project_ych/data/경상북도 영천시_행사및축제_20221128.csv",
    encoding="euc-kr",
)

import pandas as pd
import requests  # ← 이 줄이 빠져 있었음

station = ["강남역", "서울역", "고속터미널역"]
place = pd.DataFrame(columns=["역", "x", "y"])

for i in station:
    url = "https://dapi.kakao.com/v2/local/search/keyword.json?query=" + i
    headers = {
        "Authorization": "ded2f220d018afef457c47c99c8dfe92"
    }  # ← 여기에 본인 키 입력!
    places = requests.get(url, headers=headers).json()["documents"][0]
    place_name = places["place_name"]
    x = float(places["x"])
    y = float(places["y"])
    place.loc[i] = [place_name, x, y]

print(place)


response = requests.get(url, headers=headers)
print(response.status_code)
print(response.json())


url = "https://dapi.kakao.com/v2/local/search/address.json"  # 요청할 url 주소
headers = {
    "Authorization": "KakaoAK a87248694cb79257a8289c04f92d5b35"
}  # REST API 키(유효한 키)

query = {"query": "종로구서린동70 알파빌딩"}  # 입력할 주소

result = requests.get(url, headers=headers, data=query).json()  # 카카오 API 요청

print(result)


"""
도로명 주소로 검색하는 api 코드

https://developers.kakao.com/docs/latest/ko/local/dev-guide#search-by-keyword
"주소를 좌표로 변환하기"


"""
# requests 라이브러리 설치
#!pip install requests

# 라이브러리 import
import json
import requests
import folium

# REST API 키 저장
rest_api_key = "a87248694cb79257a8289c04f92d5b35"


# 요청 코드
def lat_long(address):  # lat_long 함수 정의
    url = "https://dapi.kakao.com/v2/local/search/address.json?query=" + address
    headers = {"Authorization": "KakaoAK " + rest_api_key}

    try:
        response = requests.get(url, headers=headers)  # url 요청
        json_result = response.json()  # json 데이터(주소 결과) 파싱
        print(
            json.dumps(json_result, indent=4, ensure_ascii=False)
        )  # JSON 데이터 문자열 변환, 들여쓰기, 이스케이프 미처리 설정

        address_xy = json_result["documents"][0]["address"]  # json 파일 내 address 선택
        return float(address_xy["x"]), float(
            address_xy["y"]
        )  # address에서 경도, 위도 튜플 반환

    except Exception as e:  # 오류 발생 시 메시지 출력
        print(e)
        return None, None


"""

일단 할 것

단위 주소에 대해 위 경도 변환 / 지도 표시 / 마커 표시는 성공

df 주소 변환 및 저장
list에 각 좌표 ? set로 중복 제거?

하나하나 마커 표시

마커 표시 사이트 :
https://velog.io/@eodud0582/Folium

"""
df.head()

# 장소명 총 323개
df.장소명.nunique()

add = lat_long("부산 해운대구 센텀중앙로 48")
m = folium.Map(location=[add[1], add[0]], zoom_start=20)
folium.Marker([add[1], add[0]]).add_to(m)

m


"""
키워드로 검색하는 카카오 api
참조 : https://developers.kakao.com/docs/latest/ko/local/dev-guide#search-by-keyword

참조 사이트 : https://m.blog.naver.com/kiddwannabe/221812712712
키워드로 검색해서 xy 좌표만 dic 형태로 / map을 만들고 marker 모두 찍기
"""
import requests


searching = ["합정 스타벅스", "신촌 스타벅스"]

count = 0

for i in searching:
    print(i)
    url = "https://dapi.kakao.com/v2/local/search/keyword.json?query={}".format(i)
    headers = {"Authorization": "KakaoAK " + rest_api_key}
    places = requests.get(url, headers=headers).json()["documents"]

    xy_only = [{"x": place["x"], "y": place["y"]} for place in places]

    if count == 0 and len(xy_only) != 0:
        m = folium.Map(location=[xy_only[0]["y"], xy_only[1]["x"]], zoom_start=15)

    print(len(xy_only))

    count += 1

    for i in range(1, len(xy_only)):
        folium.Marker([xy_only[i]["y"], xy_only[i]["x"]]).add_to(m)

m


places

# xy_only = [{"x": place["x"], "y": place["y"]} for place in places]

# len(xy_only)

# xy_only[1]["x"]

m = folium.Map(location=[xy_only[0]["y"], xy_only[1]["x"]], zoom_start=15)
m


for i in range(1, len(xy_only)):
    folium.Marker([xy_only[i]["y"], xy_only[i]["x"]]).add_to(m)

m
