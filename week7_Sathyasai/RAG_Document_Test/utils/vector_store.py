"""Vector Store Management Module"""
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import warnings

# Suppress HuggingFace warnings
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")


class VectorStoreManager:
    """Manages vector embeddings and FAISS database"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{model_name}",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def create_vector_store(self, documents: List[Document]):
        """Create FAISS vector store from documents"""
        vector_store = FAISS.from_documents(documents, self.embeddings)
        return vector_store
    
    def save_vector_store(self, vector_store, path: str = "faiss_index"):
        """Save vector store to disk"""
        vector_store.save_local(path)
    
    def load_vector_store(self, path: str = "faiss_index"):
        """Load vector store from disk"""
        return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
