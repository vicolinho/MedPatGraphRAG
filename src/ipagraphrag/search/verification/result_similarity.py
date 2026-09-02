from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResultSimilarityModule:
    @staticmethod
    def tfidf_cosine_similarity(texts):
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)
        return sim_matrix

    # Optional: Add sentence-transformer similarity if needed
    # from sentence_transformers import SentenceTransformer, util
    @staticmethod
    def embedding_cosine_similarity(texts, model_name='all-MiniLM-L6-v2'):
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_tensor=True)
        sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        return sim_matrix