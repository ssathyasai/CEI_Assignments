"""Vector Store Management Module"""
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import warnings
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore")


@st.cache_resource
def get_embeddings():
    """Use HuggingFace Inference API (free, no model download needed)"""
    try:
        return HuggingFaceInferenceAPIEmbeddings(
            api_key="hf_default",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except:
        # Fallback to basic embeddings if HF API fails
        from langchain_community.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=384)


class VectorStoreManager:
    """Manages vector embeddings and FAISS database"""
    
    def __init__(self):
        self.embeddings = get_embeddings()
    
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
