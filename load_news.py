# Indexing: Load -> Split -> Store
import pandas as pd
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.load import dumpd, dumps, load, loads
import json

import nest_asyncio

nest_asyncio.apply()
headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
           'Accept-Encoding': 'gzip, deflate',
           'Accept-Language': 'en-US,en;q=0.9',
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'}


def load_news():

    news_excel = 'news.xlsx'
    news_dict = pd.read_excel(news_excel, sheet_name=None)
    keywords = list(news_dict.keys())
    docs = []
    
    for keyword in keywords:
        links = news_dict[keyword]['link'].to_list()

#        for link in links:
#            if limit <= 0: break
#            loader = WebBaseLoader(link)
#            loader.requests_per_second = 1
#            doc = loader.load()
#            docs.append(doc)
#            print(link)
#            limit -= 1
#
#            #print(doc)

        #TODO: Handling exceptions for requests taking too long.

        loader = WebBaseLoader(links, header_template=headers, verify_ssl=True, continue_on_failure=True)
        loader.requests_per_second = 10
        doc = loader.aload()
        docs.append(doc)
        print(links)
        string_representation = dumps(docs, pretty=True)
        with open("database/news.json", "w") as fp:
            json.dump(string_representation, fp)
        #print(doc)
    
    return docs

def load_preference(num):
    preference_dict = pd.read_csv('preference.csv')
    preferences = preference_dict['O,X'].to_list()
    links = preference_dict['link'].to_list()
    docs = []
    idx = 0
    for preference in preferences:
        if preference == num:
            link = preference_dict['link'][idx]
            loader = WebBaseLoader(link)
            loader.requests_per_second = 10
            doc = loader.aload()
            docs.append(doc)
            print(link)
        idx += 1

    string_representation = dumps(docs, pretty=True)
    if num == 3:
        with open("database/preferred_news.json", "w") as fp:
            json.dump(string_representation, fp)
    else:
        with open("database/unpreferred_news.json", "w") as fp:
            json.dump(string_representation, fp)
    return docs


if __name__ == "__main__":
    load_news()
