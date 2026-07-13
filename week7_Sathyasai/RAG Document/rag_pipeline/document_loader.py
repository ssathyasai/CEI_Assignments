"""
Handles Stage 1 of the pipeline: Document Ingestion.
Reads PDFs and plain text files from a folder and returns their
raw text so the rest of the pipeline can chunk and embed it.
"""

import os
from pypdf import PdfReader


class DocumentLoader:
    def __init__(self, documents_folder):
        self.documents_folder = documents_folder

    def load_pdf_document(self, file_path):
        pdf_reader = PdfReader(file_path)
        extracted_pages = []

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text)

        return "\n".join(extracted_pages)

    def load_text_document(self, file_path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as text_file:
            return text_file.read()

    def load_all_documents(self):
        """
        Walks through the source folder and returns a list of
        dictionaries like {"file_name": ..., "raw_text": ...}
        """
        loaded_documents = []

        if not os.path.isdir(self.documents_folder):
            raise FileNotFoundError(
                f"Documents folder not found: {self.documents_folder}"
            )

        for file_name in sorted(os.listdir(self.documents_folder)):
            file_path = os.path.join(self.documents_folder, file_name)
            file_extension = os.path.splitext(file_name)[1].lower()

            if file_extension == ".pdf":
                document_text = self.load_pdf_document(file_path)
            elif file_extension in (".txt", ".md"):
                document_text = self.load_text_document(file_path)
            else:
                continue

            if document_text.strip():
                loaded_documents.append(
                    {"file_name": file_name, "raw_text": document_text}
                )

        return loaded_documents
