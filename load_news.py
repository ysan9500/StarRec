# Indexing: Load -> Split -> Store
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.load import dumpd, dumps, load, loads
import json
from cleantext import clean
import nest_asyncio
import bs4
from langchain_core.documents import Document

nest_asyncio.apply()
headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
           'Accept-Encoding': 'gzip, deflate',
           'Accept-Language': 'en-US,en;q=0.9',
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'}


def load_news():

    news_excel = 'news.xlsx'
    news_dict = pd.read_excel(news_excel, sheet_name=None)
    keywords = list(news_dict.keys())
    
    for keyword in keywords:
        links = news_dict[keyword]['link'].to_list()

        #TODO: Handling exceptions for requests taking too long.

        loader = WebBaseLoader(links, header_template=headers, verify_ssl=True, continue_on_failure=True,)
        loader.requests_per_second = 10
        docs = loader.aload()
        docs_text = [doc.page_content.replace("\n","") for doc in docs]
        docs = [Document(page_content=doc_text) for doc_text in docs_text]
        string_representation = dumps(docs, pretty=True)
        with open("database/news.json", "w") as fp:
            json.dump(string_representation, fp)
        #print(doc)
    
    return docs

def load_preference(num):
    preference_dict = pd.read_csv('preference.csv')
    preferences = preference_dict['O,X'].to_list()
    links = preference_dict['link'].to_list()
    selected_links = []
    idx = 0

    for preference in preferences:
        if preference == num:
            link = links[idx]
            selected_links.append(link)
            idx += 1

    loader = WebBaseLoader(selected_links, header_template=headers, verify_ssl=True, continue_on_failure=True,)
    loader.requests_per_second = 10
    docs = loader.aload()
    docs_text = [doc.page_content.replace("\n","") for doc in docs]
    docs = [Document(page_content=doc_text) for doc_text in docs_text]

    string_representation = dumps(docs, pretty=True)
    if num == 3:
        with open("database/preferred_news.json", "w") as fp:
            json.dump(string_representation, fp)
    else:
        with open("database/unpreferred_news.json", "w") as fp:
            json.dump(string_representation, fp)
    return docs


if __name__ == "__main__":
    #load_news()
    load_preference(3)
    load_preference(1)
