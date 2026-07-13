import pypdf
from pathlib import Path

class DocumentIngestion:
    def __init__(self):
        self.supported_formats = ['.txt', '.pdf']
    
    def load_document(self, file_path: str) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        if extension == '.txt':
            return self._load_txt(file_path)
        elif extension == '.pdf':
            return self._load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _load_txt(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _load_pdf(self, file_path: Path) -> str:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text