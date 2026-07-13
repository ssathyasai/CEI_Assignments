from typing import List, Dict, Any
import numpy as np

class RetrievalService:
    def __init__(self, vector_store, embedding_service):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.retrieval_log = []
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.embedding_service.encode_single(query)
            retrieved_chunks = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k
            )
            self._log_retrieval(query, retrieved_chunks)
            return retrieved_chunks
        except Exception as e:
            raise Exception(f"Error in retrieval: {str(e)}")
    
    def _log_retrieval(self, query: str, retrieved_chunks: List[Dict]):
        log_entry = {
            "query": query,
            "num_retrieved": len(retrieved_chunks),
            "top_scores": [chunk["similarity_score"] for chunk in retrieved_chunks[:3]]
        }
        self.retrieval_log.append(log_entry)
    
    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        if not retrieved_chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Context {i}] (Relevance: {chunk['similarity_score']:.3f})\n{chunk['text']}\n")
        
        return "\n".join(context_parts)