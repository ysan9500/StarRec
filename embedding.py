# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain_text_splitters import SentenceTransformersTokenTextSplitter
# from langchain_community.vectorstores import FAISS
# import load
#
#
# def embedding(docs_news, docs_preference):
#     # Initialize the splitter
#     splitter = SentenceTransformersTokenTextSplitter()
#
#     # Initialize the embedding model
#     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#
#     # Create FAISS vector stores for preference and news documents
#     vectorstore_preference = None
#     vectorstore_news = None
#
#     # Process preference documents
#     if docs_preference:
#         docs_list_preference = [item for sublist in docs_preference for item in sublist]
#
#         # Split the preference documents and add original content to metadata
#         split_docs_preference = []
#         for doc in docs_list_preference:
#             parts = splitter.split_documents([doc])
#             for part in parts:
#                 part.metadata['original_content'] = doc.page_content
#                 split_docs_preference.append(part)
#
#         # Create the FAISS vector store for preference documents
#         vectorstore_preference = FAISS.from_documents(
#             documents=split_docs_preference,
#             embedding=embeddings
#         )
#
#     # Process news documents
#     if docs_news:
#         docs_list_news = [item for sublist in docs_news for item in sublist]
#
#         # Split the news documents and add original content to metadata
#         split_docs_news = []
#         for doc in docs_list_news:
#             parts = splitter.split_documents([doc])
#             for part in parts:
#                 part.metadata['original_content'] = doc.page_content
#                 split_docs_news.append(part)
#
#         # Create the FAISS vector store for news documents
#         vectorstore_news = FAISS.from_documents(
#             documents=split_docs_news,
#             embedding=embeddings
#         )
#
#     # Find the most similar documents in news documents to preference documents
#     most_similar_contents = []
#
#     # if vectorstore_preference and vectorstore_news:
#     #     for doc_preference in split_docs_preference:
#     #         query_content = doc_preference.page_content
#     #         docs_and_scores = vectorstore_news.similarity_search_with_score(query_content)
#     #         most_similar_contents.extend(docs_and_scores[:5])  # Extend the list with the top 5 results
#     #
#     # # Sort by score to get the top 5 overall most similar documents
#     # most_similar_contents.sort(key=lambda x: x[1], reverse=True)
#     # top_5_similar_contents = most_similar_contents[:5]
#     #
#     # # Format the results for return
#     # results = []
#     # for content, score in top_5_similar_contents:
#     #     original_doc_content = content.metadata.get('original_content', 'Original content not found')
#     #     results.append({
#     #         "content": content.page_content,
#     #         "score": score,
#     #         "original_content": original_doc_content
#     #     })
#     #
#     # return results
#     if vectorstore_preference and vectorstore_news:
#         for doc_preference in split_docs_preference:
#             query_content = doc_preference.page_content
#             similar_docs = vectorstore_news.similarity_search(query_content, k=5)
#             most_similar_contents.extend(similar_docs)
#
#     # # Format the results for return
#     # results = []
#     # for content in most_similar_contents:
#     #     original_doc_content = content.metadata.get('original_content', 'Original content not found')
#     #     results.append({
#     #         "content": content.page_content,
#     #         "original_content": original_doc_content
#     #     })
#
#     print(type(most_similar_contents))
#     return most_similar_contents
#
# # if __name__ == "__main__":
# #     news = load.load_news()
# #     print('preferred news')
# #     preferred_news = load.load_preference(3)
# #
# #     embedding_result = embedding(news, preferred_news)


from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import FAISS
import load_news

def embedding(docs_news, docs_preference):
    # Initialize the splitter
    splitter = SentenceTransformersTokenTextSplitter()

    # Initialize the embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vector stores for preference and news documents
    vectorstore_preference = None
    vectorstore_news = None

    # Process preference documents
    if docs_preference:
        docs_list_preference = [item for sublist in docs_preference for item in sublist]

        # Split the preference documents and add original content to metadata
        split_docs_preference = []
        for doc in docs_list_preference:
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

    # Find the most similar documents in news documents to preference documents
    most_similar_contents = []

    if vectorstore_preference and vectorstore_news:
        for doc_preference in split_docs_preference:
            query_content = doc_preference.page_content
            similar_docs = vectorstore_news.similarity_search(query_content, k=5)

            # 디버깅: similar_docs의 구조 출력
            #print(f"similar_docs: {similar_docs}")

            # Extract the Document objects from similar_docs if necessary
            try:
                similar_docs = [doc for doc, score in similar_docs]
            except ValueError:
                similar_docs = similar_docs  # assume it's already a list of Document objects

            most_similar_contents.extend(similar_docs)

    return most_similar_contents

if __name__=='__main__':
    news = load_news.load_news()

    print('preferred news')
    preferred_news = load_news.load_preference(3)
    print('unpreferred news')

    embedding_result = embedding(news, preferred_news)
