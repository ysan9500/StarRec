import streamlit as st
import json
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from dateutil import parser
import scrap
from datetime import datetime

# 파일 경로 설정
file_path = "database/filtered_embedding_result.json"
def extract_news_details(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # 뉴스 발행 날짜 추출
            publish_date_element = (
                    soup.find('meta', {'property': 'article:published_time'}) or
                    soup.find('time', {'class': 'entry-date published'}) or
                    soup.find('time', {'class': 'tnt-date asset-date text-muted'}) or
                    soup.find('time', {'datetime': True}) or
                    soup.find('p', {'class': 'mb-no'})
            )
            if publish_date_element:
                if publish_date_element.has_attr('content'):
                    publish_date_str = publish_date_element['content']
                elif publish_date_element.has_attr('datetime'):
                    publish_date_str = publish_date_element['datetime']
                else:
                    publish_date_str = publish_date_element.text.strip()
                try:
                    # 날짜만 파싱(시간 제외)
                    publish_date = parser.parse(publish_date_str, fuzzy=True).date().isoformat()
                except (ValueError, TypeError):
                    publish_date = "발행 날짜 없음"
            else:
                publish_date = "발행 날짜 없음"
            # 출판사 추출
            publisher = soup.find('meta', {'property': 'og:site_name'})
            if publisher and publisher.has_attr('content'):
                publisher = publisher['content']
            else:
                publisher = extract_publisher_from_url(url)
            return publish_date, publisher
        else:
            return "발행 날짜 없음", extract_publisher_from_url(url)
    except Exception as e:
        st.error(f"Error extracting details from {url}: {e}")
        return "발행 날짜 없음", extract_publisher_from_url(url)
def extract_publisher_from_url(url):
    try:
        domain = urlparse(url).netloc
        publisher = domain.split('.')[-2].capitalize()
        return publisher
    except Exception as e:
        st.error(f"Error extracting publisher from URL: {e}")
        return "출판사 없음"
if not os.path.exists(file_path):
    st.error(f"File not found: {file_path}")
else:
    with open(file_path, "r") as f:
        try:
            filtered_embedding_result_json = json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON: {e}")
            filtered_embedding_result_json = []
    if isinstance(filtered_embedding_result_json, list):
        title = st.title("⭐StarRec")
        day_name = datetime.today().strftime("%A")
        keywords = scrap.get_keywords("Monday")
        keyword_text = ""
        for keyword in keywords:
            keyword_text = keyword_text + keyword + ", "
        keyword_text = keyword_text[:-2]
        st.header(f"Today's keywords:")
        st.header(f"{keyword_text}")
        # CSS 스타일 추가
        st.markdown(
            """
            <style>
            .st-emotion-cache-1wmy9hl.e1f1d6gn1 {
                max-width: 1200px; /* 전체 페이지의 최대 너비 설정 */
                margin-left: auto;
                margin-right: auto;
                padding-left: 0; /* 좌측 여백 제거 */
                padding-right: 0; /* 우측 여백 제거 */
            }
            .news-container {
                display: flex;
                justify-content: space-between;
                margin: 20px auto; /* 상하 여백 20px, 중앙 정렬 */
                padding: 20px;
                border: 1px solid #E6E6E6;
                border-radius: 10px;
                width: 1000px; /* 원하는 너비 설정 */
                box-sizing: border-box; /* 패딩과 테두리를 너비에 포함 */
                margin-left: -15%; /* 왼쪽 여백 조정 */
            }
            .news-left {
                flex: 1; /* 너비 비율 조정 */
                padding-right: 20px; /* 뉴스 내용 영역의 우측 여백 */
                margin-left: 0; /* 왼쪽 여백 제거 */
            }
            .news-right {
                flex: 1.5; /* 뉴스 요약 부분의 너비를 좀 더 넓힘 */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        idx0 = 0
        for doc in filtered_embedding_result_json:
            # JSON 데이터의 metadata에서 title, source, summary 추출
            metadata = doc.get("kwargs", {}).get("metadata", {})
            title = metadata.get("title", "제목 없음")
            link = metadata.get("source", "#")
            if link != "#":
                publish_date, publisher = extract_news_details(link)
            else:
                publish_date, publisher = "발행 날짜 없음", "출판사 없음"
            
            with open("database/summary_list.json", "r") as fp:
                summary_list = json.load(fp).split("\n", 5)[1:]

            with open("database/scores.json", "r") as fp:
                scores = json.load(fp)
            score = scores[idx0]
            star = ""
            for i in range(int((score - 30)/2.5)):
                star += "⭐"

            ttt = summary_list[idx0].replace("\",","").replace("\"", "").replace("[","").replace("]","").strip("[]\\n")
            cols = st.columns(2)
            st.markdown(
                f"""
                <div class="news-container">
                    <div class="news-left">
                        <h3><a href="{link}" target="_blank">{title}</a></h3>
                        <p>Date: {publish_date}</p>
                        <p>Publisher: {publisher}</p>
                    </div>
                    <div class="news-right">
                        <p>{ttt}</p>
                        <p>{star}(Similarity: {score:.2f}%)</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            with st.form("my_form" + f"{idx0}"):
                st.write("Rate your interest!")
                slider_val = st.slider("Interest")

                # Every form must have a submit button.
                submitted = st.form_submit_button("Submit")
                if submitted:
                    st.write("Interest", slider_val)
            idx0 += 1
            
    else:
        st.error("Unexpected JSON data structure.")
