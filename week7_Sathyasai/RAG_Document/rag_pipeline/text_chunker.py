"""
Handles Stage 2 of the pipeline: Text Chunking.
Splitting long documents into smaller overlapping chunks improves
retrieval accuracy because embeddings work best on focused pieces
of text rather than entire documents.
"""


class TextChunker:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text_into_chunks(self, document_text, source_file_name):
        chunk_list = []
        start_position = 0
        text_length = len(document_text)

        while start_position < text_length:
            end_position = start_position + self.chunk_size
            chunk_text = document_text[start_position:end_position].strip()

            if chunk_text:
                chunk_list.append(
                    {"source_file": source_file_name, "text": chunk_text}
                )

            # Move the window forward, leaving some overlap so context
            # isn't lost right at the chunk boundary
            start_position += self.chunk_size - self.chunk_overlap

        return chunk_list

    def chunk_all_documents(self, loaded_documents):
        all_chunks = []

        for document in loaded_documents:
            document_chunks = self.split_text_into_chunks(
                document["raw_text"], document["file_name"]
            )
            all_chunks.extend(document_chunks)

        return all_chunks
