from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = None
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            raise Exception(f"Error loading embedding model: {str(e)}")
    
    def encode_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        if not self.model:
            self._load_model()
        
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            raise Exception(f"Error encoding texts: {str(e)}")
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]