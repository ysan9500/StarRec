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
            docs_and_scores = vectorstore_news.similarity_search_with_score(query_content)
            most_similar_contents.extend(docs_and_scores[:5])  # Extend the list with the top 5 results

    # Sort by score to get the top 5 overall most similar documents
    most_similar_contents.sort(key=lambda x: x[1], reverse=True)
    top_5_similar_contents = most_similar_contents[:5]

    # Format the results for return
    results = []
    for content, score in top_5_similar_contents:
        original_doc_content = content.metadata.get('original_content', 'Original content not found')
        results.append({
            "content": content.page_content,
            "score": score,
            "original_content": original_doc_content
        })

    return results
