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

    # Split the documents
    split_docs = splitter.split_documents(docs_list)


    # Initialize the embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )

    query = "What is MicroLED TV?"
    docs_and_scores = vectorstore.similarity_search_with_score(query)
    content, score = docs_and_scores[0]
    print("[Content]")
    print(content.page_content)
    print("\n[Score]")
    print(score)


# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Sample documents
# doc1 = "The sky is blue."
# doc2 = "The sun is bright."
# 
# # Vectorize the documents
#
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
#
# # Calculate cosine similarity
#
# cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
# print(f"Cosine Similarity: {cosine_sim[0][1]}")