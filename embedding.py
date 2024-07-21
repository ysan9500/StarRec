#embedding.py
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import FAISS
import load_news

from langchain_core.load import dumpd, dumps, load, loads
import json

def embedding(docs_news, docs_preference):
    # Initialize the splitter
    splitter = SentenceTransformersTokenTextSplitter()

    # Initialize the embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

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

    # Find the most similar documents in news documents to preference documents
    most_similar_contents = []

    if vectorstore_preference and vectorstore_news:
        for doc_preference in split_docs_preference:
            query_content = doc_preference.page_content
            similar_docs = vectorstore_news.similarity_search(query_content, k=5)
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
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain_text_splitters import SentenceTransformersTokenTextSplitter
# from langchain_community.vectorstores import FAISS
# import load_news
# from langchain_core.load import dumpd, dumps, load, loads
# import json
# from langchain_core.load import dumpd, dumps, load, loads
# def embedding(docs_news, docs_preference):
#     # Initialize the splitter
#     splitter = SentenceTransformersTokenTextSplitter()
#     # Initialize the embedding model
#     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#     # Create FAISS vector stores for preference and news documents
#     vectorstore_preference = None
#     vectorstore_news = None
#     # Process preference documents
#     if docs_preference:
#         # Split the preference documents and add original content to metadata
#         split_docs_preference = []
#         for doc in docs_preference:
#             parts = splitter.split_documents([doc])
#             for part in parts:
#                 part.metadata['original_content'] = doc.page_content
#                 split_docs_preference.append(part)
#         # Create the FAISS vector store for preference documents
#         vectorstore_preference = FAISS.from_documents(
#             documents=split_docs_preference,
#             embedding=embeddings
#         )
#     # Process news documents
#     if docs_news:
#         # Split the news documents and add original content to metadata
#         split_docs_news = []
#         for doc in docs_news:
#             parts = splitter.split_documents([doc])
#             for part in parts:
#                 part.metadata['original_content'] = doc.page_content
#                 split_docs_news.append(part)
#         # Create the FAISS vector store for news documents
#         vectorstore_news = FAISS.from_documents(
#             documents=split_docs_news,
#             embedding=embeddings
#         )
#     # Find the most similar documents in news documents to preference documents
#     most_similar_contents = []
#     if vectorstore_preference and vectorstore_news:
#         for doc_preference in split_docs_preference:
#             query_content = doc_preference.page_content
#             similar_docs_with_scores = vectorstore_news.similarity_search_with_score(query_content, k=5)
#             # Extract documents from the results
#             for doc, score in similar_docs_with_scores:
#                 # Add similarity score to document metadata if needed
#                 doc.metadata['similarity_score'] = score
#                 most_similar_contents.append(doc)
#     # Ensure unique documents and limit to the top 5 results
#     unique_docs = {doc.metadata['original_content']: doc for doc in most_similar_contents}.values()
#     print(type(list(unique_docs)))
#     return list(unique_docs)[:5]
# if __name__ == '__main__':
#     with open("database/news.json", "r") as fp1:
#         news = loads(json.load(fp1))
#     with open("database/preferred_news.json", "r") as fp2:
#         preferred_news = loads(json.load(fp2))
#     with open("database/unpreferred_news.json", "r") as fp3:
#         unpreferred_news = loads(json.load(fp3))
#     embedding_result = embedding(news, preferred_news)
#     print(type(embedding_result))
#     print(type(embedding_result[0]))
#     print(len(embedding_result))
#     print(embedding_result)
#     # Print only the documents
#     for doc in embedding_result:
#         print(f"Document: {doc.page_content}\n")
#     # for idx, result in enumerate(embedding_result):
#     #     print(f"[Top {idx + 1} Most Similar Document]")
#     #     print(result)
#     #     print("\n" + "-" * 80 + "\n")