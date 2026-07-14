"""Document Processing Module"""
import io
import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles document loading and text chunking with multiple strategies"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use recursive splitter with better separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",          # Triple newline (section break)
                "\n\n",            # Double newline (paragraph)
                "\n",              # Single newline (line)
                ". ",              # Sentence boundary
                "! ",              # Sentence boundary
                "? ",              # Sentence boundary
                " ",               # Word boundary
                ""                 # Character level (fallback)
            ]
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
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove extra newlines but keep paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def _process_pdf(self, uploaded_file) -> List[Document]:
        """Process PDF file with enhanced chunking"""
        # Save temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Load and split
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            # Combine all text first
            combined_text = "\n\n".join([doc.page_content for doc in documents])
            combined_text = self._clean_text(combined_text)
            
            # Create document with cleaned text
            clean_doc = Document(
                page_content=combined_text,
                metadata={"source": uploaded_file.name, "type": "pdf"}
            )
            
            # Split
            chunks = self.text_splitter.split_documents([clean_doc])
            
            # Ensure meaningful chunks
            chunks = self._filter_meaningful_chunks(chunks)
            
            return chunks
        finally:
            # Clean up
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _process_text(self, uploaded_file) -> List[Document]:
        """Process text file with enhanced chunking"""
        text_content = uploaded_file.read().decode('utf-8')
        text_content = self._clean_text(text_content)
        
        documents = [Document(
            page_content=text_content,
            metadata={"source": uploaded_file.name, "type": "txt"}
        )]
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Ensure meaningful chunks
        chunks = self._filter_meaningful_chunks(chunks)
        
        return chunks
    
    def _filter_meaningful_chunks(self, chunks: List[Document]) -> List[Document]:
        """Filter out very small or meaningless chunks"""
        meaningful_chunks = []
        
        for chunk in chunks:
            content = chunk.page_content.strip()
            
            # Skip if too short (less than 30 characters)
            if len(content) < 30:
                continue
            
            # Skip if only whitespace or special characters
            if not re.search(r'[a-zA-Z0-9]', content):
                continue
            
            meaningful_chunks.append(chunk)
        
        return meaningful_chunks
