"""
Vector Store Management Module
Handles embedding creation and FAISS vector storage
"""

from typing import List, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    """
    Manages embeddings and FAISS vector store operations.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: Name of the Hugging Face embedding model
        """
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
    
    def create_vector_store(self, chunks: List[Any]) -> FAISS:
        """
        Create FAISS vector store from text chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            FAISS vector store
        """
        if not chunks:
            raise ValueError("No chunks provided for embedding!")
        
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        return self.vector_store
    
    def get_vector_store(self) -> FAISS:
        """Get the current vector store."""
        return self.vector_store
    
    def save_vector_store(self, path: str = "./vector_store"):
        """Save vector store to disk."""
        if self.vector_store:
            self.vector_store.save_local(path)
    
    def load_vector_store(self, path: str = "./vector_store"):
        """Load vector store from disk."""
        try:
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except TypeError:
            # Fallback for older LangChain versions that don't support allow_dangerous_deserialization
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings
            )
        return self.vector_store

