from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import load_news

from langchain_core.load import dumpd, dumps, load, loads
import json

def embedding(docs_news, docs_preference):
    # Initialize the splitter
    splitter = SentenceTransformersTokenTextSplitter()

    # Initialize the embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vector store for news documents
    vectorstore_news = None

    # Process news documents
    if docs_news:
        docs_list_news = [item for sublist in docs_news for item in sublist]

        # Split the news documents and add original content to metadata
        split_docs_news = []
        for doc in docs_list_news:
            parts = splitter.split_documents([doc])
            for part in parts:
                part.metadata['original_content'] = doc.page_content
                split_docs_news.append(part)

        # Create the FAISS vector store for news documents
        vectorstore_news = FAISS.from_documents(
            documents=split_docs_news,
            embedding=embeddings
        )

    # Create BM25 retriever for preference documents
    bm25_retriever = None
    if docs_preference:
        docs_list_preference = [item for sublist in docs_preference for item in sublist]

        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_texts(
            [doc.page_content for doc in docs_list_preference],
            metadatas=[doc.metadata for doc in docs_list_preference]
        )
        bm25_retriever.k = 5

    # Create the FAISS retriever for news documents
    faiss_retriever = None
    if vectorstore_news:
        faiss_retriever = vectorstore_news.as_retriever(search_kwargs={"k": 5})

    # Initialize the ensemble retriever
    ensemble_retriever = None
    if bm25_retriever and faiss_retriever:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )

    # Find the most similar documents in news documents to preference documents
    most_similar_contents = []

    if ensemble_retriever:
        for doc_preference in docs_preference:
            query_content = doc_preference.page_content
            similar_docs = ensemble_retriever.invoke(query_content)
            most_similar_contents.extend(similar_docs)

    return most_similar_contents[:5]

if __name__=='__main__':
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

    # for idx, result in enumerate(embedding_result):
    #     print(f"[Top {idx + 1} Most Similar Document]")
    #     print(result)
    #     print("\n" + "-" * 80 + "\n")
