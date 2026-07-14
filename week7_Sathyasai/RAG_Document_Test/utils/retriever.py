"""Retrieval System Module"""
from typing import List
from langchain_core.documents import Document


class RetrievalSystem:
    """Handles document retrieval from vector store with re-ranking"""
    
    def __init__(self, vector_store, vector_manager=None, top_k: int = 4):
        self.vector_store = vector_store
        self.vector_manager = vector_manager
        self.top_k = top_k
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k * 2}  # Fetch more for re-ranking
        )
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents for query"""
        # Get initial results
        docs = self.retriever.invoke(query)
        
        # Re-rank if manager available
        if self.vector_manager:
            docs = self.vector_manager.rerank_documents(query, docs, self.top_k)
        
        return docs[:self.top_k]
    
    def get_retriever(self):
        """Get the retriever object"""
        return self.retriever
