from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
import load
from langchain_community.vectorstores import FAISS

if __name__ == "__main__":
    # Load news from both preference and general sources
    user_preference = 3  # 예시로 사용자 선호도를 설정

    print('preferred news')
    docs_preference = load.load_preference(user_preference)
    print('news')
    docs_news = load.load_news()

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

    # Find the most similar document in news documents to preference documents
    most_similar_content = None
    highest_score = -1.0

    if vectorstore_preference and vectorstore_news:
        for doc_preference in split_docs_preference:
            query_content = doc_preference.page_content
            docs_and_scores = vectorstore_news.similarity_search_with_score(query_content)
            content, score = docs_and_scores[0]  # Assume the most similar one is the first one

            if score > highest_score:
                highest_score = score
                most_similar_content = content

    # Retrieve the original document content from metadata
    if most_similar_content:
        original_doc_content = most_similar_content.metadata.get('original_content', 'Original content not found')

        print("[Most Similar Document Content]")
        print(most_similar_content)
        print("\n[Score]")
        print(highest_score)
    else:
        print("No similar document found.")