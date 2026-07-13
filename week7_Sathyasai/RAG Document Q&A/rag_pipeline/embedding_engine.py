"""
Handles Stage 3 of the pipeline: Embedding Creation.
Wraps a sentence-transformers model so both the document chunks
and the incoming user question can be converted into the same
vector space for similarity comparison.
"""

from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    def __init__(self, model_name):
        self.embedding_model = SentenceTransformer(model_name)

    def generate_embeddings(self, text_list):
        return self.embedding_model.encode(
            text_list, convert_to_numpy=True, show_progress_bar=False
        )

    def generate_single_embedding(self, text):
        return self.generate_embeddings([text])[0]
