
from dotenv import load_dotenv
import embedding
import generate
from langchain_core.load import dumpd, dumps, load, loads
import json

def main():
    load_dotenv()

    with open("database/news.json", "r") as fp1:
        news = loads(json.load(fp1))
    with open("database/preferred_news.json", "r") as fp2:
        preferred_news = loads(json.load(fp2))
    with open("database/unpreferred_news.json", "r") as fp3:
        unpreferred_news = loads(json.load(fp3))

    # 임베딩 처리
    embedding_result_preferred = embedding.embedding(news, preferred_news)
    embedding_result_unpreferred = embedding.embedding(news, unpreferred_news)

    # 비선호 뉴스의 페이지 내용을 집합으로 만듦
    unpreferred_set = {doc.page_content for doc in embedding_result_unpreferred}

    # 중복된 뉴스 저장용 리스트
    duplicate_news = []

    # 선호 뉴스에서 비선호 뉴스와 중복되는 내용 확인
    filtered_embedding_result = []
    for doc in embedding_result_preferred:
        if doc.page_content in unpreferred_set:
            duplicate_news.append(doc)
        else:
            filtered_embedding_result.append(doc)

    # 중복된 뉴스 출력
    print(f"Number of Duplicate News: {len(duplicate_news)}")
    if duplicate_news:
        print("Duplicate News:")
        for idx, doc in enumerate(duplicate_news, start=1):
            print(f"{idx}. Title: {doc.metadata.get('title', 'No Title')}")
            print(f"   URL: {doc.metadata.get('source', 'No URL')}")
            print(f"   Description: {doc.metadata.get('description', 'No Description')}")
            print(f"   Original Content: {doc.page_content[:200]}...")  # 내용 일부만 출력
            print("\n" + "-"*80 + "\n")

    summary = generate.summarize(filtered_embedding_result)
    print(summary)
    print('summary type:', type(summary))

if __name__ == "__main__":
    main()
