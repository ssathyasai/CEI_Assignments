"""
Retrieval Module
Implements hybrid retrieval (Vector similarity + BM25) and re-ranking (local Cross-Encoder or Cohere Rerank API).
"""

from typing import List, Dict, Any, Tuple, Union
import numpy as np
from utils.vector_store import VectorStoreManager, EmbeddingGenerator

class LocalReranker:
    """
    Local Cross-Encoder re-ranker using SentenceTransformers.
    Runs locally and is free.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device="cpu")

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        if not chunks:
            return []
        
        pairs = [[query, chunk["text"]] for chunk in chunks]
        scores = self.model.predict(pairs)
        
        # Sort chunks by score
        scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        
        # Softmax or sigmoid mapping for visual scores (optional, let's keep raw scores or mapped sigmoid)
        results = []
        for chunk, score in scored_chunks[:top_n]:
            # Convert logit score to a 0-1 confidence using sigmoid
            sigmoid_score = 1.0 / (1.0 + np.exp(-float(score)))
            results.append((chunk, sigmoid_score))
            
        return results


class CohereReranker:
    """
    Cloud-based Cohere Re-ranker.
    """
    def __init__(self, api_key: str, model_name: str = "rerank-english-v3.0"):
        import cohere
        self.co = cohere.Client(api_key)
        self.model_name = model_name

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        if not chunks:
            return []
            
        doc_texts = [chunk["text"] for chunk in chunks]
        response = self.co.rerank(
            query=query,
            documents=doc_texts,
            top_n=top_n,
            model=self.model_name
        )
        
        results = []
        for result in response.results:
            idx = result.index
            score = float(result.relevance_score)
            results.append((chunks[idx], score))
            
        return results


class HybridRetriever:
    """
    Retrieves the most contextually relevant document chunks using hybrid search & re-ranking.
    """
    def __init__(self, vector_manager: VectorStoreManager, embedding_generator: EmbeddingGenerator):
        self.vector_manager = vector_manager
        self.emb_gen = embedding_generator
        self.local_reranker = None

    def retrieve(
        self, 
        query: str, 
        search_strategy: str = "hybrid", 
        alpha: float = 0.7, 
        top_k_retrieve: int = 15,
        rerank_strategy: str = "none",
        cohere_api_key: str = None,
        top_n_final: int = 4
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieves context chunks.
        
        search_strategy: 'vector', 'keyword', 'hybrid'
        alpha: Weight for dense vector search (1.0 = vector only, 0.0 = keyword only)
        top_k_retrieve: Number of candidate chunks to fetch before re-ranking
        rerank_strategy: 'none', 'local', 'cohere'
        """
        # Step 1: Base Retrieval
        if search_strategy == "vector":
            query_emb = self.emb_gen.embed_query(query)
            candidates = self.vector_manager.vector_search(query_emb, top_k=top_k_retrieve)
        elif search_strategy == "keyword":
            candidates = self.vector_manager.keyword_search(query, top_k=top_k_retrieve)
        else: # hybrid
            query_emb = self.emb_gen.embed_query(query)
            dense_results = self.vector_manager.vector_search(query_emb, top_k=top_k_retrieve)
            sparse_results = self.vector_manager.keyword_search(query, top_k=top_k_retrieve)
            
            # Linear merge
            merged = {}
            for chunk, score in dense_results:
                text = chunk["text"]
                merged[text] = {
                    "chunk": chunk,
                    "dense_score": score,
                    "sparse_score": 0.0
                }
            for chunk, score in sparse_results:
                text = chunk["text"]
                if text in merged:
                    merged[text]["sparse_score"] = score
                else:
                    merged[text] = {
                        "chunk": chunk,
                        "dense_score": 0.0,
                        "sparse_score": score
                    }
                    
            candidate_list = []
            for text, info in merged.items():
                score = alpha * info["dense_score"] + (1.0 - alpha) * info["sparse_score"]
                candidate_list.append((info["chunk"], score))
            
            # Sort candidates by combined score
            candidate_list.sort(key=lambda x: x[1], reverse=True)
            candidates = candidate_list[:top_k_retrieve]

        # Step 2: Re-ranking
        if not candidates:
            return []
            
        chunks_only = [c[0] for c in candidates]
        
        if rerank_strategy == "local":
            if not self.local_reranker:
                self.local_reranker = LocalReranker()
            return self.local_reranker.rerank(query, chunks_only, top_n=top_n_final)
            
        elif rerank_strategy == "cohere":
            if not cohere_api_key:
                raise ValueError("Cohere API Key is required for Cohere Re-ranking")
            reranker = CohereReranker(api_key=cohere_api_key)
            return reranker.rerank(query, chunks_only, top_n=top_n_final)
            
        else: # 'none'
            # Return top_n_final from the base candidate retrieval list
            return candidates[:top_n_final]
