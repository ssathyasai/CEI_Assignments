"""Retrieval Module - Handles similarity search"""
from typing import List, Any, Tuple
from langchain_community.vectorstores import FAISS


class RetrievalSystem:
    """Handles similarity search and document retrieval"""
    
    def __init__(self, vector_store: FAISS, top_k: int = 4):
        self.vector_store = vector_store
        self.top_k = top_k
        self.retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    def retrieve(self, query: str) -> List[Any]:
        """Retrieve relevant documents"""
        return self.retriever.get_relevant_documents(query)
    
    def retrieve_with_scores(self, query: str) -> List[Tuple[Any, float]]:
        """Retrieve documents with similarity scores"""
        return self.vector_store.similarity_search_with_score(query, k=self.top_k)
