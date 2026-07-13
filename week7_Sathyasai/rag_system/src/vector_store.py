import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import numpy as np

class VectorStoreService:
    def __init__(self, collection_name: str = "document_chunks"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        try:
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise Exception(f"Error initializing vector store: {str(e)}")
    
    def store_embeddings(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        try:
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            documents = [chunk["text"] for chunk in chunks]
            metadatas = [{"document": chunk["document"], "chunk_id": chunk["chunk_id"]} for chunk in chunks]
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            raise Exception(f"Error storing embeddings: {str(e)}")
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        try:
            results = self.collection.query(
                query_embeddings=query_embedding.reshape(1, -1).tolist(),
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved_chunks = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    retrieved_chunks.append({
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "similarity_score": 1 - results['distances'][0][i],
                        "id": results['ids'][0][i]
                    })
            
            return retrieved_chunks
        except Exception as e:
            raise Exception(f"Error searching vector store: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_documents": count
            }
        except Exception as e:
            return {"error": str(e)}