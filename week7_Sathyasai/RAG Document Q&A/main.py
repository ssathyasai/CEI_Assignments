"""
Document Question Answering System (RAG)

Entry point that wires together every stage of the pipeline:
Document Ingestion -> Text Chunking -> Embedding Creation ->
Vector Database -> Query Processing -> Context Retrieval ->
Answer Generation.

Run:
    python main.py
"""

import os

from rag_pipeline import config
from rag_pipeline.document_loader import DocumentLoader
from rag_pipeline.text_chunker import TextChunker
from rag_pipeline.embedding_engine import EmbeddingEngine
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.answer_generator import AnswerGenerator


def build_knowledge_base():
    print("Loading documents from:", config.SOURCE_DOCUMENTS_FOLDER)
    document_loader = DocumentLoader(config.SOURCE_DOCUMENTS_FOLDER)
    loaded_documents = document_loader.load_all_documents()
    print(f"Loaded {len(loaded_documents)} document(s).")

    print("Splitting documents into chunks...")
    text_chunker = TextChunker(
        config.CHUNK_SIZE_IN_CHARACTERS, config.CHUNK_OVERLAP_IN_CHARACTERS
    )
    all_chunks = text_chunker.chunk_all_documents(loaded_documents)
    print(f"Created {len(all_chunks)} chunk(s).")

    print("Generating embeddings for each chunk...")
    embedding_engine = EmbeddingEngine(config.EMBEDDING_MODEL_NAME)
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    chunk_embeddings = embedding_engine.generate_embeddings(chunk_texts)

    print("Storing embeddings in the vector database...")
    embedding_dimension = chunk_embeddings.shape[1]
    vector_store = VectorStore(embedding_dimension)
    vector_store.add_chunks(all_chunks, chunk_embeddings)

    return embedding_engine, vector_store


def answer_user_questions(embedding_engine, vector_store):
    answer_generator = AnswerGenerator(config.ANSWER_GENERATION_MODEL_NAME)

    print("\nKnowledge base is ready. Type 'exit' to quit.\n")

    while True:
        user_question = input("Ask a question: ").strip()

        if user_question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not user_question:
            continue

        question_embedding = embedding_engine.generate_single_embedding(user_question)
        retrieved_chunks = vector_store.search_similar_chunks(
            question_embedding, config.NUMBER_OF_CHUNKS_TO_RETRIEVE
        )

        if not retrieved_chunks:
            print("Answer: I couldn't find anything relevant in the documents.\n")
            continue

        final_answer = answer_generator.generate_answer(user_question, retrieved_chunks)
        print("Answer:", final_answer)
        print("Sources:", ", ".join(sorted({c["source_file"] for c in retrieved_chunks})))
        print()


def main():
    if not os.path.isdir(config.SOURCE_DOCUMENTS_FOLDER):
        os.makedirs(config.SOURCE_DOCUMENTS_FOLDER, exist_ok=True)
        print(
            f"Created '{config.SOURCE_DOCUMENTS_FOLDER}' folder. "
            "Add your PDF or text files there and run this script again."
        )
        return

    embedding_engine, vector_store = build_knowledge_base()
    answer_user_questions(embedding_engine, vector_store)


if __name__ == "__main__":
    main()
