# 📚 RAG Document Q&A System

A production-ready RAG system that answers questions from your documents using Groq AI.

## 🎯 Live Demo

**Link**:https://ceiassignments-zp.streamlit.app/

**Try Locally**: `streamlit run app.py`

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```

### Get API Key
[Free Groq API key](https://console.groq.com) - No credit card needed

## ✨ Features

- **Upload Documents**: PDF or TXT files
- **Smart Chunking**: Intelligent text splitting with cleaning
- **Fast Retrieval**: TF-IDF embeddings + FAISS
- **Dual-Stage Search**: Retrieval + re-ranking
- **AI Answers**: Groq LLM (Llama 3.3 70B)
- **Source Citations**: View retrieved chunks
- **Configuration Presets**: Balanced & Precise modes
- **Chat History**: Track Q&A

## 📊 Recommended Settings

| Setting | Recommended | Range |
|---------|------------|-------|
| Chunk Size | 1000 | 50-2000 |
| Chunk Overlap | 100 | 0-300 |
| Top-K | 4 | 1-10 |
| Temperature | 0.3 | 0.0-1.0 |

**Presets**:
- **Balanced**: 1000/100/4/0.3 (default)
- **Precise**: 500/50/3/0.1 (accuracy)

## 📖 How to Use

1. **Upload** → Go to Upload tab → Select PDF/TXT
2. **Configure** → Set chunk size, overlap, top-k (optional)
3. **Ask** → Go to Ask Questions → Type your question
4. **View** → See answer + source chunks

## 🏗️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | TF-IDF (scikit-learn) |
| Vector DB | FAISS |
| Framework | LangChain |
| UI | Streamlit |

## 📁 Project Structure

```
RAG_Document_Test/
├── app.py                 # Main app
├── requirements.txt       # Dependencies
├── utils/
│   ├── document_processor.py    # Chunking
│   ├── vector_store.py          # Embeddings
│   ├── retriever.py             # Retrieval
│   └── answer_generator.py      # LLM
└── data/
    └── sample.txt         # Sample doc
```

## 🎯 System Flow

```
Document → Clean Text → Chunks → TF-IDF Embeddings → FAISS
                                          ↓
                              Dual-Stage Retrieval
                              (Fetch + Re-rank)
                                          ↓
                              Groq LLM (Strict RAG)
                                          ↓
                              Grounded Answer
```

## ⚙️ Configuration

- **Chunk Size**: 50-2000 chars. Larger = more context
- **Overlap**: 0-300 chars. Helps preserve relationships
- **Top-K**: 1-10. Optimal is 3-5
- **Temperature**: 0.0-1.0. Lower = more factual

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| API Key error | Add to Streamlit secrets: `GROQ_API_KEY` |
| Slow processing | Reduce chunk size or top-k |
| Poor answers | Check document is text-based (not image) |
| Model error | Auto-switches to fallback model |

## ✅ Features Implemented

- ✅ Document ingestion (PDF/TXT)
- ✅ Intelligent chunking
- ✅ TF-IDF embeddings
- ✅ FAISS storage
- ✅ Dual-stage retrieval
- ✅ Strict RAG prompt
- ✅ Model fallback
- ✅ Chat history
- ✅ Source citations
- ✅ Configuration presets

## 📚 Resources

- [Groq API](https://console.groq.com)
- [LangChain Docs](https://python.langchain.com)
- [Streamlit Docs](https://docs.streamlit.io)

## 📄 License

Educational purposes - Celebal Technologies Internship

---

**Status**: ✅ Production Ready | **Last Updated**: July 15, 2026
