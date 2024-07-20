#embedding2.py
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
import load_news
from langchain_core.load import dumpd, dumps, load, loads
import json
import os
os.environ['OPENAI_API_KEY']=''
def embedding(docs_news, docs_preference):
    # Initialize the splitter
    splitter = SentenceTransformersTokenTextSplitter()
    # Initialize the embedding model
    embeddings = OpenAIEmbeddings()
    # Create FAISS vector stores for preference and news documents
    vectorstore_preference = None
    vectorstore_news = None
    # Process preference documents
    if docs_preference:
        #docs_list_preference = [item for sublist in docs_preference for item in sublist]
        # Split the preference documents and add original content to metadata
        split_docs_preference = []
        for doc in docs_preference:
            parts = splitter.split_documents([doc])
            for part in parts:
                part.metadata['original_content'] = doc.page_content
                split_docs_preference.append(part)
        # Create the FAISS vector store for preference documents
        vectorstore_preference = FAISS.from_documents(
            documents=split_docs_preference,
            embedding=embeddings
        )
    # Process news documents
    if docs_news:
        #docs_list_news = [item for sublist in docs_news for item in sublist]
        # Split the news documents and add original content to metadata
        split_docs_news = []
        for doc in docs_news:
            parts = splitter.split_documents([doc])
            for part in parts:
                part.metadata['original_content'] = doc.page_content
                split_docs_news.append(part)
        # Create the FAISS vector store for news documents
        vectorstore_news = FAISS.from_documents(
            documents=split_docs_news,
            embedding=embeddings
        )
        # Create BM25 retriever for news documents
        bm25_retriever = BM25Retriever.from_documents(docs_news)
    # Create the ensemble retriever
    retrievers = []
    if vectorstore_preference and vectorstore_news:
        faiss_retriever = vectorstore_news.as_retriever()
        retrievers.append(faiss_retriever)
    if bm25_retriever:
        retrievers.append(bm25_retriever)
    ensemble_retriever = EnsembleRetriever(retrievers=retrievers)
    # Find the most similar documents in news documents to preference documents
    most_similar_contents = []
    if ensemble_retriever:
        for doc_preference in split_docs_preference:
            query_content = doc_preference.page_content
            similar_docs = ensemble_retriever.invoke(query_content, k=5)
            most_similar_contents.extend(similar_docs)
    return most_similar_contents[:5]
if __name__ == '__main__':
    with open("database/news.json", "r") as fp1:
        news = loads(json.load(fp1))
    with open("database/preferred_news.json", "r") as fp2:
        preferred_news = loads(json.load(fp2))
    with open("database/unpreferred_news.json", "r") as fp3:
        unpreferred_news = loads(json.load(fp3))
    embedding_result = embedding(news, preferred_news)
    print(type(embedding_result))
    print(type(embedding_result[0]))
    print(len(embedding_result))
    print(embedding_result)