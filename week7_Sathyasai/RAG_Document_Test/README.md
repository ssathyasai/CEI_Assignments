# 📚 RAG Document Q&A System

A production-ready Retrieval-Augmented Generation (RAG) system that answers questions based on custom documents using Groq AI's ultra-fast inference.

## 🎯 Overview

This system implements a complete RAG pipeline that:
- Processes custom PDF and text documents
- Creates semantic embeddings using sentence-transformers
- Stores vectors in FAISS for fast similarity search
- Retrieves relevant context chunks for user queries
- Generates accurate, grounded answers using Groq's Llama 3.1 70B

## ✨ Features

- **Multi-format Support**: Upload PDF or TXT files
- **Configurable Chunking**: Adjust chunk size and overlap
- **Fast Retrieval**: FAISS vector database with semantic search
- **Ultra-fast Inference**: Groq AI (Llama 3.1 70B)
- **Interactive UI**: Clean Streamlit interface
- **Chat History**: Track questions and answers
- **Source Citations**: View retrieved document chunks
- **Real-time Metrics**: Monitor system performance

## 🏗️ Architecture

```
Document → Text Chunks → Embeddings → Vector Store → Retrieval → LLM → Answer
```

### Components

1. **Document Processor** (`document_processor.py`)
   - Loads PDF and text files
   - Splits text into manageable chunks
   - Preserves context with overlap

2. **Vector Store Manager** (`vector_store.py`)
   - Creates embeddings using all-MiniLM-L6-v2
   - Manages FAISS vector database
   - Enables fast similarity search

3. **Retrieval System** (`retriever.py`)
   - Converts queries to embeddings
   - Finds top-K most relevant chunks
   - Returns contextual documents

4. **Answer Generator** (`answer_generator.py`)
   - Uses Groq LLM (Llama 3.1 70B)
   - Generates grounded answers
   - Prevents hallucination

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Groq API Key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your repository
4. Add Groq API key in Streamlit Cloud secrets

## 📖 Usage

1. **Upload Document**
   - Navigate to "Upload" tab
   - Choose a PDF or TXT file
   - Enter your Groq API key
   - Click "Process Document"

2. **Ask Questions**
   - Go to "Ask Questions" tab
   - Type your question
   - Click "Ask"
   - View answer and source chunks

3. **Monitor Metrics**
   - Check "Metrics" tab for system stats
   - View chunking configuration
   - Track questions asked

## ⚙️ Configuration

### Chunking Parameters
- **Chunk Size**: 500-2000 chars (default: 1000)
- **Chunk Overlap**: 0-300 chars (default: 100)

### Retrieval Parameters
- **Top-K**: 1-10 chunks (default: 4)
- **Temperature**: 0.0-1.0 (default: 0.3)

### Model Information
- **LLM**: Llama 3.1 70B (Groq)
- **Embeddings**: all-MiniLM-L6-v2 (384 dims)
- **Vector DB**: FAISS (CPU)

## 📊 System Metrics

- **Embedding Dimension**: 384
- **Average Inference Time**: <1s with Groq
- **Supported File Types**: PDF, TXT
- **Max Upload Size**: 50MB

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain |
| LLM | Groq (Llama 3.1 70B) |
| Embeddings | Sentence Transformers |
| Vector Store | FAISS |
| UI | Streamlit |
| PDF Parser | PyPDF |

## 📁 Project Structure

```
week7_RAG_DOCUMENT_TEST/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Documentation
├── .python-version        # Python version
├── .gitignore            # Git ignore rules
├── .streamlit/
│   └── config.toml       # Streamlit theme
├── data/
│   └── sample.txt        # Sample document
└── utils/
    ├── __init__.py       # Package initialization
    ├── document_processor.py  # Document loading & chunking
    ├── vector_store.py        # Embeddings & FAISS
    ├── retriever.py           # Similarity search
    └── answer_generator.py    # LLM answer generation
```

## 🎓 Key Concepts

### Retrieval-Augmented Generation (RAG)
- Combines retrieval with generation
- Grounds answers in actual documents
- Reduces hallucination
- Works with private data

### Chunking Strategy
- RecursiveCharacterTextSplitter
- Maintains semantic coherence
- Preserves context with overlap

### Semantic Search
- Vector embeddings capture meaning
- Cosine similarity for relevance
- Fast approximate nearest neighbor search

## 🔍 Best Practices

1. **Chunk Size**: Balance between context and specificity
   - Too small: Loss of context
   - Too large: Diluted relevance

2. **Top-K Selection**: 
   - 3-5 chunks for most queries
   - More for complex questions

3. **Temperature**:
   - 0.0-0.3: Factual, deterministic
   - 0.5-0.7: Creative, varied

4. **Document Quality**:
   - Clean, well-formatted text
   - Avoid scanned images without OCR

## 🐛 Troubleshooting

### Import Errors
- Ensure all packages in `requirements.txt` are installed
- Use Python 3.10+

### Slow Performance
- Reduce chunk size
- Lower Top-K value
- Use Groq for faster inference

### Poor Answers
- Increase chunk overlap
- Adjust Top-K retrieval
- Check document quality

## 📝 Example Questions

For the sample document:
- "What is RAG?"
- "What are the key components of a RAG system?"
- "What are the benefits of using RAG?"
- "Which embedding model is used?"
- "What are the best practices for chunking?"

## 🎯 Validation Checklist

✅ Document ingestion (PDF/TXT)  
✅ Text chunking with overlap  
✅ Vector embeddings creation  
✅ FAISS vector store  
✅ Similarity-based retrieval  
✅ LLM answer generation  
✅ Source citation display  
✅ Interactive web interface  
✅ Configuration options  
✅ Performance metrics  

## 🚀 Future Enhancements

- Hybrid search (keyword + vector)
- Re-ranking for better relevance
- Multiple document support
- Conversation memory
- Answer evaluation metrics
- Export Q&A history

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Created as part of Week 7 assignment for Celebal Technologies Internship.

---

**Note**: Get your free Groq API key at [console.groq.com](https://console.groq.com) - No credit card required!
