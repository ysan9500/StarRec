# # streamlit_app.py
#
# import streamlit as st
# import json
# import os
#
# # 파일 경로 설정
# file_path = "filtered_embedding_result_summary.json"
#
# if not os.path.exists(file_path):
#     st.error(f"File not found: {file_path}")
# else:
#     # JSON 파일에서 뉴스 데이터 로드
#     with open(file_path, "r") as f:
#         try:
#             filtered_embedding_result_json = json.load(f)
#         except json.JSONDecodeError as e:
#             st.error(f"Error decoding JSON: {e}")
#             filtered_embedding_result_json = []
#
#     # JSON 데이터 구조 확인 후 직접 표시
#     if isinstance(filtered_embedding_result_json, list):
#         st.title("추천 뉴스")
#
#         for doc in filtered_embedding_result_json:
#             metadata = doc.get("kwargs", {}).get("metadata", {})
#             title = metadata.get("title", "제목 없음")
#             link = metadata.get("source", "#")
#
#             if link != "#":
#                 st.markdown(f"[{title}]({link})")
#             else:
#                 st.write(title)
#     else:
#         st.error("Unexpected JSON data structure.")

import streamlit as st
import json
import os

# 파일 경로 설정
file_path = "filtered_embedding_result_summary.json"

# 파일이 존재하는지 확인
if not os.path.exists(file_path):
    st.error(f"File not found: {file_path}")
else:
    # JSON 파일에서 뉴스 데이터 로드
    with open(file_path, "r") as f:
        try:
            filtered_embedding_result_json = json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON: {e}")
            filtered_embedding_result_json = []

    # JSON 데이터 구조 확인 후 직접 표시
    if isinstance(filtered_embedding_result_json, list):
        st.title("추천 뉴스")

        for doc in filtered_embedding_result_json:
            # JSON 데이터의 metadata에서 title, source, summary 추출
            metadata = doc.get("kwargs", {}).get("metadata", {})
            title = metadata.get("title", "제목 없음")
            link = metadata.get("source", "#")
            summary = metadata.get("summary", "요약 없음")

            # 제목을 링크로 만들어 표시
            if link != "#":
                st.markdown(f"[{title}]({link})")
            else:
                st.write(title)

            # 요약본 표시
            st.write(f"**요약:** {summary}")
            st.write("---")
    else:
        st.error("Unexpected JSON data structure.")
