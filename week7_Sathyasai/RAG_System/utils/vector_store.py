"""
Vector Store Management Module
Handles dense embeddings generation, local FAISS and cloud Pinecone vector stores, 
and local BM25 keyword index creation.
"""

import os
import json
import pickle
import string
from typing import List, Dict, Any, Tuple, Union
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

class EmbeddingGenerator:
    """
    Generates text embeddings using Local SentenceTransformers, Google Gemini, or Cohere.
    """
    def __init__(self, provider: str = "local", api_key: str = None, model_name: str = None):
        """
        provider: 'local', 'gemini', or 'cohere'
        """
        self.provider = provider.lower()
        self.api_key = api_key
        
        if self.provider == "local":
            from sentence_transformers import SentenceTransformer
            self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name)
            self.dimension = 384
        elif self.provider == "gemini":
            import google.generativeai as genai
            if not api_key:
                raise ValueError("Google API key is required for Gemini Embeddings")
            genai.configure(api_key=api_key)
            self.model_name = model_name or "models/text-embedding-004"
            self.dimension = 768
        elif self.provider == "cohere":
            import cohere
            if not api_key:
                raise ValueError("Cohere API key is required for Cohere Embeddings")
            self.co = cohere.Client(api_key)
            self.model_name = model_name or "embed-english-v3.0"
            self.dimension = 1024
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    def embed_texts(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Embed a list of strings.
        """
        if not texts:
            return []
            
        if self.provider == "local":
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return [emb.tolist() for emb in embeddings]
            
        elif self.provider == "gemini":
            import google.generativeai as genai
            # Batch embedding query
            task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
            try:
                response = genai.embed_content(
                    model=self.model_name,
                    content=texts,
                    task_type=task_type
                )
                return response['embedding']
            except Exception as e:
                # Fallback if list fails, try single query embedding
                embeddings = []
                for t in texts:
                    res = genai.embed_content(model=self.model_name, content=t, task_type=task_type)
                    embeddings.append(res['embedding'])
                return embeddings
                
        elif self.provider == "cohere":
            input_type = "search_query" if is_query else "search_document"
            response = self.co.embed(
                texts=texts,
                model=self.model_name,
                input_type=input_type
            )
            return [emb for emb in response.embeddings]
            
        return []

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query.
        """
        return self.embed_texts([query], is_query=True)[0]


class VectorStoreManager:
    """
    Manages vector databases (FAISS, Pinecone) and keyword search indices (BM25).
    """
    def __init__(self, store_type: str = "faiss", pinecone_api_key: str = None, index_name: str = "rag-qa-bot"):
        self.store_type = store_type.lower()
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.chunks = []
        self.bm25 = None
        
        # FAISS state
        self.faiss_index = None
        
        # Pinecone state
        self.pc = None
        self.pc_index = None

    def initialize_store(self, dimension: int):
        """
        Setup database schemas.
        """
        if self.store_type == "faiss":
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.chunks = []
        elif self.store_type == "pinecone":
            from pinecone import Pinecone, ServerlessSpec
            if not self.pinecone_api_key:
                raise ValueError("Pinecone API Key is required for cloud vector store")
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Create serverless index if it doesn't exist
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            self.pc_index = self.pc.Index(self.index_name)
            self.chunks = [] # Keep local copies in memory for lookup if needed

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Insert text chunks and embedding vectors into indices.
        """
        self.chunks.extend(chunks)
        
        # 1. Update Dense Index
        if self.store_type == "faiss":
            embeddings_np = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_np)
            self.faiss_index.add(embeddings_np)
        elif self.store_type == "pinecone":
            vectors = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                # Pinecone vectors must have safe metadata strings/numbers
                chunk_id = f"c_{chunk['metadata'].get('chunk_id', i)}_{hash(chunk['text']) % 1000000}"
                vectors.append({
                    "id": chunk_id,
                    "values": emb,
                    "metadata": {
                        "text": chunk["text"],
                        "source": chunk["metadata"].get("source", "Unknown"),
                        "chunk_id": int(chunk["metadata"].get("chunk_id", i))
                    }
                })
            # Batch uploads of 100 vectors
            for j in range(0, len(vectors), 100):
                self.pc_index.upsert(vectors=vectors[j:j+100])

        # 2. Update BM25 Keyword Index
        self._build_bm25()

    def _tokenize(self, text: str) -> List[str]:
        """
        Helper tokenizer for BM25.
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def _build_bm25(self):
        """
        Rebuilds the BM25 index on all current document chunks.
        """
        if not self.chunks:
            return
        tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def vector_search(self, query_emb: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search using vector embeddings. Returns list of (chunk_dict, score).
        """
        if not self.chunks:
            return []
            
        if self.store_type == "faiss":
            query_np = np.array([query_emb]).astype('float32')
            faiss.normalize_L2(query_np)
            scores, indices = self.faiss_index.search(query_np, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(self.chunks):
                    # FAISS scores can be slightly larger than 1.0 due to float precision
                    res_score = min(max(float(score), 0.0), 1.0)
                    results.append((self.chunks[idx], res_score))
            return results
            
        elif self.store_type == "pinecone":
            res = self.pc_index.query(vector=query_emb, top_k=top_k, include_metadata=True)
            results = []
            for match in res.get("matches", []):
                chunk = {
                    "text": match["metadata"]["text"],
                    "metadata": {
                        "source": match["metadata"].get("source", "Unknown"),
                        "chunk_id": int(match["metadata"].get("chunk_id", 0))
                    }
                }
                # Normalize Pinecone cosine score
                score = (float(match["score"]) + 1.0) / 2.0 if match["score"] is not None else 0.0
                results.append((chunk, score))
            return results
            
        return []

    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search using BM25 keyword matching. Returns list of (chunk_dict, score).
        """
        if not self.bm25 or not self.chunks:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices by score desc
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        # Max score for normalization
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
        
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0: # Only return actual matches
                normalized_score = score / max_score
                results.append((self.chunks[idx], normalized_score))
        return results

    def save_local(self, directory: str = "faiss_index"):
        """
        Saves the local FAISS index and local chunks to a directory.
        """
        if self.store_type != "faiss" or not self.faiss_index:
            return
            
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, os.path.join(directory, "index.faiss"))
        
        # Save chunks metadata
        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load_local(self, directory: str = "faiss_index"):
        """
        Loads the local FAISS index and local chunks from a directory.
        """
        if self.store_type != "faiss":
            return
            
        faiss_path = os.path.join(directory, "index.faiss")
        chunks_path = os.path.join(directory, "chunks.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(chunks_path):
            self.faiss_index = faiss.read_index(faiss_path)
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            self._build_bm25()
        else:
            raise FileNotFoundError(f"FAISS index files not found in {directory}")
