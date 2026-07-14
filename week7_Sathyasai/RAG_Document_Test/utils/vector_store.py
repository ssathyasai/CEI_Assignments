"""Vector Store Management Module"""
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
import pickle
import os


class TFIDFEmbeddings(Embeddings):
    """Simple TF-IDF embeddings - no model download needed"""
    
    def __init__(self):
        self.vectorizer = None
        self.embedding_dim = 384
        self._fitted = False
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if self.vectorizer is None or not self._fitted:
            return [0.0] * self.embedding_dim
        
        try:
            tfidf = self.vectorizer.transform([text]).toarray()[0]
        except:
            return [0.0] * self.embedding_dim
        
        if len(tfidf) < self.embedding_dim:
            tfidf = np.pad(tfidf, (0, self.embedding_dim - len(tfidf)))
        else:
            tfidf = tfidf[:self.embedding_dim]
        return tfidf.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        self.vectorizer = TfidfVectorizer(
            max_features=self.embedding_dim,
            stop_words='english',
            lowercase=True,
            analyzer='word',
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=1,
            max_df=0.95
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()
        self._fitted = True
        
        embeddings = []
        for vec in tfidf_matrix:
            if len(vec) < self.embedding_dim:
                vec = np.pad(vec, (0, self.embedding_dim - len(vec)))
            else:
                vec = vec[:self.embedding_dim]
            embeddings.append(vec.tolist())
        
        return embeddings


@st.cache_resource
def get_embeddings():
    """Get cached embeddings"""
    return TFIDFEmbeddings()


class VectorStoreManager:
    """Manages vector embeddings and FAISS database with re-ranking"""
    
    def __init__(self):
        self.embeddings = get_embeddings()
        self.all_docs = []
    
    def create_vector_store(self, documents: List[Document]):
        """Create FAISS vector store from documents"""
        self.all_docs = documents
        vector_store = FAISS.from_documents(documents, self.embeddings)
        return vector_store
    
    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 4) -> List[Document]:
        """Re-rank documents using multiple signals"""
        if not documents:
            return documents
        
        # Get query embedding
        query_emb = np.array(self.embeddings.embed_query(query)).reshape(1, -1)
        
        # Get document embeddings
        doc_embs = np.array([self.embeddings.embed_query(doc.page_content) for doc in documents])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_emb, doc_embs)[0]
        
        # Calculate keyword overlap score
        query_words = set(query.lower().split())
        keyword_scores = []
        for doc in documents:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words & doc_words) / (len(query_words) + 1)
            keyword_scores.append(overlap)
        
        keyword_scores = np.array(keyword_scores)
        
        # Combined score (70% similarity, 30% keyword match)
        combined_scores = 0.7 * similarities + 0.3 * keyword_scores
        
        # Sort by combined score
        ranked_indices = np.argsort(combined_scores)[::-1]
        
        # Return top-k re-ranked documents
        return [documents[i] for i in ranked_indices[:top_k]]
    
    def save_vector_store(self, vector_store, path: str = "faiss_index"):
        """Save vector store to disk"""
        vector_store.save_local(path)
    
    def load_vector_store(self, path: str = "faiss_index"):
        """Load vector store from disk"""
        return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
