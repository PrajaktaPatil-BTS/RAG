import os
import faiss
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


class PDFRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.chunks = []
        self.embeddings = None   # TF-IDF vectors

    def build_index(self, chunks):
        """
        chunks = list of dicts:
        [
            {"chunk": "text here", "page": 1},
            ...
        ]
        """
        self.chunks = chunks
        #extrct only text not page number
        texts = [c["chunk"] for c in chunks]

        # Create TF-IDF vectors
        self.embeddings = self.vectorizer.fit_transform(texts)

    def search(self, query, top_k=3):
        # convert query to TF-IDF
        q_vec = self.vectorizer.transform([query])

        # cosine similarity
        scores = cosine_similarity(q_vec, self.embeddings)[0]

        # top K indexes
        top_indices = scores.argsort()[::-1][:top_k]

        # return results
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx]["chunk"],
                "page": self.chunks[idx].get("page", None),
                "score": float(scores[idx])
            })

        return results

    #load data from file
    def load_index(self):
        if not all(os.path.exists(f) for f in ["faiss_tfidf.index", "tfidf_vectorizer.pkl", "chunks_with_pages.pkl"]):
            raise FileNotFoundError("FAISS index not found. Build it first.")
        self.index = faiss.read_index("faiss_tfidf.index")
        self.vectorizer = joblib.load("tfidf_vectorizer.pkl")
        self.chunks = joblib.load("chunks_with_pages.pkl")


