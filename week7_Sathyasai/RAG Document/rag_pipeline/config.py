"""
Central place for every setting the RAG pipeline needs.
Keeping these in one file makes it easy to tune chunk sizes,
swap models, or change the number of retrieved chunks without
touching the actual logic anywhere else.
"""

# Folder that holds the source documents (PDFs / text files)
SOURCE_DOCUMENTS_FOLDER = "sample_documents"

# Text chunking settings
CHUNK_SIZE_IN_CHARACTERS = 800
CHUNK_OVERLAP_IN_CHARACTERS = 100

# Embedding model used to turn text into vectors
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Language model used to generate the final answer
ANSWER_GENERATION_MODEL_NAME = "google/flan-t5-base"

# How many chunks to retrieve for every question
NUMBER_OF_CHUNKS_TO_RETRIEVE = 4

# Where the FAISS index gets saved so it isn't rebuilt every run
VECTOR_INDEX_FILE_PATH = "vector_index.faiss"
CHUNK_STORE_FILE_PATH = "chunk_store.pkl"
