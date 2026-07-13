"""
Document Processing Module
Handles document ingestion (PDF, TXT, Hugging Face datasets) and chunking.
"""

import os
import tempfile
from typing import List, Dict, Any, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class DocumentProcessor:
    """
    Ingests documents from various sources and splits them into chunks.
    """
    
    @staticmethod
    def extract_text_from_pdf(file_path_or_bytes: Union[str, bytes]) -> str:
        """
        Extract raw text from a PDF file path or bytes.
        """
        try:
            if isinstance(file_path_or_bytes, bytes):
                # We need a file-like object for PdfReader if it is bytes, but tempfile is safer
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_path_or_bytes)
                    tmp_path = tmp.name
                try:
                    reader = PdfReader(tmp_path)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    return text
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
                reader = PdfReader(file_path_or_bytes)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    @staticmethod
    def extract_text_from_txt(file_path_or_bytes: Union[str, bytes]) -> str:
        """
        Extract raw text from a TXT file path or bytes.
        """
        try:
            if isinstance(file_path_or_bytes, bytes):
                return file_path_or_bytes.decode("utf-8", errors="ignore")
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        except Exception as e:
            raise Exception(f"Failed to extract text from TXT: {str(e)}")

    @staticmethod
    def load_hf_dataset(dataset_name: str, text_column: str, subset: str = None, split: str = "train", max_samples: int = 100) -> str:
        """
        Load a dataset from Hugging Face and extract text from the specified text column.
        """
        try:
            from datasets import load_dataset
            
            # Load dataset (with optional subset)
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            # Select column and extract text
            texts = []
            count = min(len(dataset), max_samples)
            for i in range(count):
                row = dataset[i]
                if text_column in row:
                    texts.append(str(row[text_column]))
            
            if not texts:
                raise ValueError(f"Column '{text_column}' not found or empty in dataset '{dataset_name}'")
            
            return "\n\n--- Document Separator ---\n\n".join(texts)
        except Exception as e:
            raise Exception(f"Failed to load Hugging Face dataset: {str(e)}")

    def split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100, source_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split a block of text into manageable chunks using RecursiveCharacterTextSplitter.
        
        Returns a list of dictionaries, where each dictionary contains:
        - 'text': the text content of the chunk
        - 'metadata': dict with source, chunk_id, etc.
        """
        if not text.strip():
            return []
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        raw_chunks = splitter.split_text(text)
        
        chunks = []
        for i, chunk in enumerate(raw_chunks):
            metadata = {
                "chunk_id": i,
                "chunk_length": len(chunk),
                "source": "Unknown"
            }
            if source_metadata:
                metadata.update(source_metadata)
            
            chunks.append({
                "text": chunk,
                "metadata": metadata
            })
            
        return chunks
