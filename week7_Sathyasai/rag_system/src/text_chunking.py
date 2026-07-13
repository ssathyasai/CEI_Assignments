import re
from typing import List, Dict, Any

class TextChunking:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.log = []
    
    def chunk_by_size(self, text: str, document_name: str = "unknown") -> List[Dict[str, Any]]:
        chunks = []
        text = self._clean_text(text)
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_size = len(sentence_words)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "document": document_name,
                    "chunk_id": len(chunks),
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "sentence_count": len(current_chunk)
                })
                
                if self.overlap > 0:
                    overlap_words = current_chunk[-self.overlap:]
                    current_chunk = overlap_words.copy()
                    current_size = len(" ".join(overlap_words).split())
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "document": document_name,
                "chunk_id": len(chunks),
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "sentence_count": len(current_chunk)
            })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]', ' ', text)
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]