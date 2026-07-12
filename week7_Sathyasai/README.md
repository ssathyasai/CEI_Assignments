# 📚 RAG Document Question Answering System

> A production-ready Retrieval-Augmented Generation (RAG) web application for intelligent document Q&A using Streamlit, LangChain, FAISS, and Google Gemini API.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green)](https://python.langchain.com/)

---

## 🎯 Overview

Upload your documents (PDF/TXT) and ask questions! This RAG system retrieves relevant information and generates accurate, context-grounded answers using AI.

**Key Features:**
- 📤 Drag-and-drop file upload
- 💬 Interactive chat interface  
- 🔍 Fast semantic search with FAISS
- 🤖 AI-powered answers with Google Gemini
- 📊 Real-time metrics dashboard
- ⚙️ Configurable parameters

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key ([Get free key](https://makersuite.google.com/app/apikey))

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd week8

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📖 Usage

1. **Enter API Key**: Paste your Google Gemini API key in the sidebar
2. **Upload Document**: Drop a PDF or TXT file in the upload area
3. **Process**: Click "Process Document" and wait ~10 seconds
4. **Ask Questions**: Type your question and get AI-generated answers!

---

## 📁 Project Structure

```
week8/
├── app.py                      # Main Streamlit application
├── utils/                      # Core modules
│   ├── document_processor.py  # Document loading & chunking
│   ├── vector_store.py         # FAISS vector storage
│   ├── retriever.py            # Similarity search
│   └── answer_generator.py     # LLM answer generation
├── data/                       # Sample documents
│   └── sample_document.txt
├── reference/                  # Reference materials
├── .streamlit/                 # Streamlit configuration
│   └── config.toml
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Web Framework** | Streamlit |
| **LLM** | Google Gemini Pro |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) |
| **Vector DB** | FAISS |
| **Orchestration** | LangChain |

---

## ⚙️ Configuration

Adjust these settings in the sidebar:
- **Chunk Size**: 500-2000 characters (default: 1000)
- **Chunk Overlap**: 0-300 characters (default: 100)  
- **Top-K Retrieval**: 1-10 chunks (default: 4)
- **Temperature**: 0.0-1.0 (default: 0.3)

---

## 🌐 Deployment

### Streamlit Cloud (Free)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo and deploy!

### Local
```bash
streamlit run app.py
```

---

## 🐛 Troubleshooting

**Issue**: DLL initialization failed  
**Fix**: Install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

**Issue**: Port already in use  
**Fix**: `streamlit run app.py --server.port 8502`

**Issue**: API key error  
**Fix**: Verify key at [Google AI Studio](https://makersuite.google.com/app/apikey)

---

## 📝 Example Questions

Try asking:
- "What is the main topic of this document?"
- "Summarize the key points"
- "What methodology is described?"
- "What are the conclusions?"

---

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

---

## 📄 License

This project is for educational purposes.

---

## 👤 Author

**Sathyasai**  
Celebal Internship - Week 7/8 Assignment

---

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) - RAG framework
- [Streamlit](https://streamlit.io/) - Web framework
- [FAISS](https://faiss.ai/) - Vector search
- [Google Gemini](https://ai.google.dev/) - LLM API

---

**⭐ Star this repo if you find it useful!**
