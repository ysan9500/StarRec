# import random
# import numpy as np
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain_text_splitters import SentenceTransformersTokenTextSplitter
# import load
# from langchain_community.vectorstores import FAISS
#
# # def set_seed(seed):
# #     random.seed(seed)
# #     np.random.seed(seed)
#
# if __name__ == "__main__":
#     #set_seed(42)
#
#     docs = load.load_news()
#
#     # Flatten the loaded docs
#     docs_list = [item for sublist in docs for item in sublist]
#
#     # Initialize the splitter
#     splitter = SentenceTransformersTokenTextSplitter()
#
#     # Split the documents
#     split_docs = splitter.split_documents(docs_list)
#
#
#     # Initialize the embedding model
#     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#
#     vectorstore = FAISS.from_documents(
#         documents=split_docs,
#         embedding=embeddings
#     )
#
#     query = "What is MicroLED TV?"
#     docs_and_scores = vectorstore.similarity_search_with_score(query)
#     content, score = docs_and_scores[0]
#     print("[Content]")
#     print(content.page_content)
#     print("\n[Score]")
#     print(score)

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
import load
from langchain_community.vectorstores import FAISS

if __name__ == "__main__":
    docs = load.load_news()

    # Flatten the loaded docs
    docs_list = [item for sublist in docs for item in sublist]

    # Initialize the splitter
    splitter = SentenceTransformersTokenTextSplitter()

    # Split the documents and add original content to metadata
    split_docs = []
    for doc in docs_list:
        parts = splitter.split_documents([doc])
        for part in parts:
            part.metadata['original_content'] = doc.page_content
            split_docs.append(part)

    # Initialize the embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create the FAISS vector store
    vectorstore = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )

    # Perform similarity search with a query
    query = "What is MicroLED TV?"
    docs_and_scores = vectorstore.similarity_search_with_score(query)
    content, score = docs_and_scores[0]

    # Retrieve the original document content from metadata
    original_doc_content = content.metadata.get('original_content', 'Original content not found')

    print(content)
    # print("[Original Document Content]")
    # print(content)
    #print(original_doc_content)
    print("\n[Score]")
    print(score)
