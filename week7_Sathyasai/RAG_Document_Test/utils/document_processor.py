"""Document Processing Module"""
import io
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles document loading and text chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """Process uploaded file and return chunks"""
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'pdf':
            return self._process_pdf(uploaded_file)
        elif file_type == 'txt':
            return self._process_text(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _process_pdf(self, uploaded_file) -> List[Document]:
        """Process PDF file"""
        # Save temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and split
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        # Clean up
        import os
        os.remove(temp_path)
        
        return chunks
    
    def _process_text(self, uploaded_file) -> List[Document]:
        """Process text file"""
        text_content = uploaded_file.read().decode('utf-8')
        documents = [Document(page_content=text_content, metadata={"source": uploaded_file.name})]
        chunks = self.text_splitter.split_documents(documents)
        return chunks
