# from sentence_transformers import SentenceTransformer
#
# model = SentenceTransformer('all-MiniLM-L6-v2')
# documents = ["This is a document about AI.", "This is another document about machine learning."]
# document_embeddings = model.encode(documents)
#
# print(document_embeddings)

# split + embedding 
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
import load
from langchain.vectorstores import Chroma

if __name__ == "__main__":
    docs = load.load_news()

    # Flatten the loaded docs
    flattened_docs = [item.page_content for sublist in docs for item in sublist]

    # Initialize the splitter
    splitter = SentenceTransformersTokenTextSplitter()

    # Split the documents
    split_docs = [splitter.split_text(doc) for doc in flattened_docs]

    # Flatten the split documents
    flattened_split_docs = [item for sublist in split_docs for item in sublist]

    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for the split documents
    document_embeddings = model.encode(flattened_split_docs)

    # Print the generated embeddings
    print(document_embeddings)

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