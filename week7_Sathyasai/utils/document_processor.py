"""
Document Processing Module
Handles PDF and TXT file loading and text chunking
"""

import tempfile
import os
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    Processes documents by loading and splitting them into chunks.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> List[Any]:
        """Load and extract text from a PDF file."""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")
    
    def load_text(self, file_path: str) -> List[Any]:
        """Load text from a TXT file."""
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            raise Exception(f"Error loading text file: {str(e)}")
    
    def process_uploaded_file(self, uploaded_file) -> List[Any]:
        """
        Process an uploaded file from Streamlit.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            List of document chunks
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Load based on file type
            if uploaded_file.name.endswith('.pdf'):
                documents = self.load_pdf(tmp_path)
            elif uploaded_file.name.endswith('.txt'):
                documents = self.load_text(tmp_path)
            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.name}")
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            return chunks
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)
