# Document Question Answering System (RAG)

A Retrieval-Augmented Generation (RAG) system that answers questions
based on your own documents (PDFs or text files), instead of relying
only on a language model's internal knowledge.

## How it works

1. **Document Ingestion** — PDFs/text files are loaded and converted to raw text.
2. **Text Chunking** — Text is split into smaller overlapping chunks.
3. **Embedding Creation** — Each chunk is converted into a vector using `sentence-transformers`.
4. **Vector Database** — Chunk embeddings are stored in a FAISS index.
5. **Query Processing** — The user's question is embedded the same way.
6. **Context Retrieval** — The most similar chunks are retrieved from FAISS.
7. **Answer Generation** — A language model (`flan-t5-base`) generates an answer grounded in the retrieved context.

## Project structure

```
week-7/
├── main.py                       # entry point, ties every stage together
├── requirements.txt
├── rag_pipeline/
│   ├── config.py                 # all tunable settings in one place
│   ├── document_loader.py        # Stage 1: Document Ingestion
│   ├── text_chunker.py           # Stage 2: Text Chunking
│   ├── embedding_engine.py       # Stage 3: Embedding Creation
│   ├── vector_store.py           # Stage 4: Vector Database
│   └── answer_generator.py       # Stage 7: Answer Generation
└── sample_documents/             # put your PDFs / text files here
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Put your PDFs or `.txt` files inside the `sample_documents/` folder
   (it gets created automatically on first run).
2. Run:

```bash
python main.py
```

3. Ask questions in the terminal. Type `exit` to quit.

## Example

```
Ask a question: What is the main idea of the document?
Answer: <generated answer grounded in the retrieved chunks>
Sources: my_notes.pdf
```

## Possible improvements

- Hybrid search (keyword + vector) for better retrieval
- Re-ranking retrieved chunks before generation
- Swapping in a stronger embedding or generation model
- A simple web UI on top of the same pipeline
