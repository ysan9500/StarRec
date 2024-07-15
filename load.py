# Indexing: Load -> Split -> Store
import pandas as pd
import os
from langchain_community.document_loaders import WebBaseLoader

import nest_asyncio


def load_news():
    nest_asyncio.apply()
    news_excel = 'news.xlsx'
    news_dict = pd.read_excel(news_excel, sheet_name=None)
    keywords = list(news_dict.keys())
    docs = []
    
    for keyword in keywords:
        limit = 6
        links = news_dict[keyword]['link'].to_list()
        for link in links:
            if limit <= 0: break
            loader = WebBaseLoader(link)
            doc = loader.load()
            docs.append(doc)
            print(link)
            limit -= 1
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
            doc = loader.load()
            docs.append(doc)
            print(link)
        idx += 1
    return docs


if __name__ == "__main__":
    load_preference(3)
