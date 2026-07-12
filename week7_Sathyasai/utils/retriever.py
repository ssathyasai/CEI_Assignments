"""
Retrieval System Module
Handles similarity search and document retrieval
"""

from typing import List, Any, Tuple
from langchain_community.vectorstores import FAISS


class RetrievalSystem:
    """
    Handles similarity search and retrieval of relevant document chunks.
    """
    
    def __init__(self, vector_store: FAISS, top_k: int = 4):
        """
        Initialize the retrieval system.
        
        Args:
            vector_store: FAISS vector store
            top_k: Number of chunks to retrieve
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )
    
    def retrieve(self, query: str) -> List[Any]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User's question
            
        Returns:
            List of relevant document chunks
        """
        retrieved_docs = self.retriever.get_relevant_documents(query)
        return retrieved_docs
    
    def retrieve_with_scores(self, query: str) -> List[Tuple[Any, float]]:
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: User's question
            
        Returns:
            List of tuples (document, score)
        """
        results = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        return results
