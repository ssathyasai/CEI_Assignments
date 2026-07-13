"""Document Processing Module - Handles PDF/TXT loading and chunking"""
import tempfile
import os
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Processes documents by loading and splitting into chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_uploaded_file(self, uploaded_file) -> List[Any]:
        """Process Streamlit uploaded file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_path)
            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.name}")
            
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            return chunks
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
