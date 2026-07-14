"""Retrieval System Module"""
from typing import List
from langchain_core.documents import Document


class RetrievalSystem:
    """Handles document retrieval from vector store"""
    
    def __init__(self, vector_store, top_k: int = 4):
        self.vector_store = vector_store
        self.top_k = top_k
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents for query"""
        return self.retriever.get_relevant_documents(query)
    
    def get_retriever(self):
        """Get the retriever object"""
        return self.retriever
