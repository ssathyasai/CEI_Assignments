"""
Handles Stage 4 of the pipeline: Vector Database.
Uses FAISS to store chunk embeddings and quickly find the chunks
that are most similar to a given question embedding.
"""

import pickle
import faiss
import numpy as np


class VectorStore:
    def __init__(self, embedding_dimension):
        self.faiss_index = faiss.IndexFlatL2(embedding_dimension)
        self.stored_chunks = []

    def add_chunks(self, chunk_list, chunk_embeddings):
        self.faiss_index.add(np.array(chunk_embeddings).astype("float32"))
        self.stored_chunks.extend(chunk_list)

    def search_similar_chunks(self, question_embedding, top_k):
        question_vector = np.array([question_embedding]).astype("float32")
        distances, matched_positions = self.faiss_index.search(question_vector, top_k)

        matched_chunks = []
        for position in matched_positions[0]:
            if position != -1:
                matched_chunks.append(self.stored_chunks[position])

        return matched_chunks

    def save_to_disk(self, index_file_path, chunk_store_file_path):
        faiss.write_index(self.faiss_index, index_file_path)
        with open(chunk_store_file_path, "wb") as chunk_file:
            pickle.dump(self.stored_chunks, chunk_file)

    def load_from_disk(self, index_file_path, chunk_store_file_path):
        self.faiss_index = faiss.read_index(index_file_path)
        with open(chunk_store_file_path, "rb") as chunk_file:
            self.stored_chunks = pickle.load(chunk_file)
