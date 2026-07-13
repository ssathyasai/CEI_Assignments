"""Vector Store Module - Handles embeddings and FAISS"""
from typing import List, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    """Manages embeddings and FAISS vector store"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
    
    def create_vector_store(self, chunks: List[Any]) -> FAISS:
        """Create FAISS vector store from chunks"""
        if not chunks:
            raise ValueError("No chunks provided!")
        
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        return self.vector_store
    
    def get_vector_store(self) -> FAISS:
        """Get current vector store"""
        return self.vector_store
