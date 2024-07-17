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
    scrap.scrap()
    #news = load.load_news()
    with open("database/news.json", "r") as fp:
        news = loads(json.load(fp))
    print('preferred news')
    preferred_news = load_news.load_preference(3)
    print('unpreferred news')
    unpreferred_news = load_news.load_preference(1)
    embedding_result = embedding.embedding(news, preferred_news)

    generate.summarize(embedding_result)

    # 결과 처리 (예시로 출력)
    # for idx, result in enumerate(embedding_result):
    #     print(f"[Top {idx + 1} Most Similar Document]")
    #     print(result)
    #     print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    main()