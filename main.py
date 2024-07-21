# TODO1: scrap 실행
# TODO2: 링크 글 스크랩
# TODO3: 문서 임베딩
# TODO4: 템플릿 프롬프트 + 유저 선호도 데이터배이스(선호, 비선호 조합 고려) + 임베딩된 문서
# TODO5: LLM에넣고 결과값 받기
# TODO6: 프론트엔드 streamlit

from dotenv import load_dotenv, dotenv_values
import scrap
import load_news
import embedding2
import embedding3
import generate3
from langchain_core.load import dumpd, dumps, load, loads
import load_news
import json


def main():
    load_dotenv() 
    # scrap.scrap()
    # news = load_news.load_news()
    # preferred_news = load_news.load_preference(3)
    # unpreferred_news = load_news.load_preference(1)

    with open("database/news.json", "r") as fp1:
        news = loads(json.load(fp1))
    with open("database/preferred_news.json", "r") as fp2:
        preferred_news = loads(json.load(fp2))
    with open("database/unpreferred_news.json", "r") as fp3:
        unpreferred_news = loads(json.load(fp3))

    # print(type(news[0]))
    # print(type(preferred_news[0]))
    # print(type(unpreferred_news[0]))

    # # 임베딩 처리
    # embedding_result_preferred = embedding3.embedding(news, preferred_news)
    # # # embedding_result_unpreferred = embedding.embedding(news, unpreferred_news)
    
    # with open("database/filtered_embedding_result.json", "w") as f:
    #     json.dump(dumpd(embedding_result_preferred), f)

    with open("database/filtered_embedding_result.json", "r") as fp4:
        embedding_result_preferred = json.load(fp4)
        print(embedding_result_preferred[0]["kwargs"])
        generate3.summarize(embedding_result_preferred)
    with open("database/summaries.json", "r") as fp:
        summaries = json.load(fp)
        summary_list = summaries.split("SUMMARY:", 5)[1:]
        idx = 0
        for smry in summary_list:
            smry2 = smry.split("Write a summary of the following text delimited by triple backticks.")[0]
            smry3 = smry2.split("SOURCE")[0]
            smry4 = smry3.replace("\\n", "").replace("  ", "").replace("THE", "").replace("\"", "").replace(".,", ".").replace("\\t","").replace("\\", " ")
            summary_list[idx] = smry4
            idx += 1
        print(summary_list[0])
        string_representation = dumps(summary_list, pretty=True)
        with open("database/summary_list.json", "w") as fp:
            json.dump(string_representation, fp)

        with open("database/summary_list.json", "r") as fp:
                summary_list = json.load(fp).split("\n", 5)
                for a in summary_list:
                    print(a)
if __name__ == "__main__":
    main()