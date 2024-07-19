# TODO1: scrap 실행
# TODO2: 링크 글 스크랩
# TODO3: 문서 임베딩
# TODO4: 템플릿 프롬프트 + 유저 선호도 데이터배이스(선호, 비선호 조합 고려) + 임베딩된 문서
# TODO5: LLM에넣고 결과값 받기
# TODO6: 프론트엔드 streamlit

from dotenv import load_dotenv, dotenv_values
import scrap
import load_news
import embedding
import generate
from langchain_core.load import dumpd, dumps, load, loads
import json


def main():
    load_dotenv() 
    # scrap.scrap()
    # news = load.load_news()
    # preferred_news = load_news.load_preference(3)
    # unpreferred_news = load_news.load_preference(1)

    with open("database/news.json", "r") as fp1:
        news = loads(json.load(fp1))
    with open("database/preferred_news.json", "r") as fp2:
        preferred_news = loads(json.load(fp2))
    with open("database/unpreferred_news.json", "r") as fp3:
        unpreferred_news = loads(json.load(fp3))
    #
    # embedding_result = embedding.embedding(news, preferred_news)
    print(type(news[0]))
    print(type(preferred_news[0]))
    print(type(unpreferred_news[0]))

    # 임베딩 처리
    embedding_result_preferred = embedding.embedding(news, preferred_news)
    embedding_result_unpreferred = embedding.embedding(news, unpreferred_news)

    # # 선호 뉴스와 비선호 뉴스의 중복 제거
    # unpreferred_set = {doc for doc in embedding_result_unpreferred}
    # filtered_embedding_result = [
    #     doc for doc in embedding_result_preferred
    #     if doc not in unpreferred_set
    # ]
    
    # # 디버깅: 중복 제거된 결과 확인
    # print(f"Filtered Embedding Result Count: {len(filtered_embedding_result)}")
    # if filtered_embedding_result:
    #     print(f"First Element: {filtered_embedding_result[0]}")

    # 요약 생성
    cleaned = generate.cleanup(embedding_result_preferred)
    summaries = generate.summarize(cleaned)
    for summary in summaries:
        print(summary)

if __name__ == "__main__":
    main()