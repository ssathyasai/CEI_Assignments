# 📚 RAG Document Q&A System

A production-ready Retrieval-Augmented Generation (RAG) system that answers questions based on custom documents using Groq AI's ultra-fast inference.

## 🎯 Quick Links

- **🔴 Live Demo**: [Coming Soon - Will be deployed on Streamlit Cloud](#live-demo)
- **📖 Documentation**: Complete guide below
- **🔑 API Key**: [Get free Groq API key](https://console.groq.com)

## 🎯 Overview

This system implements a complete RAG pipeline that:
- Processes custom PDF and text documents with intelligent chunking
- Creates semantic embeddings using TF-IDF with bigrams
- Stores vectors in FAISS for fast similarity search
- Retrieves and re-ranks relevant context chunks
- Generates accurate, grounded answers using Groq's Llama 3.3 70B
- Enforces strict RAG behavior (no hallucination)

## ✨ Features

- **Multi-format Support**: Upload PDF or TXT files
- **Intelligent Chunking**: 
  - Recursive splitting with multiple separators
  - Text cleaning and normalization
  - Meaningful chunk filtering
- **Advanced Retrieval**:
  - Dual-stage retrieval: fetch 2×top-k, then re-rank
  - Combined scoring: 70% semantic + 30% keyword match
  - TF-IDF embeddings with bigrams
- **Ultra-fast Inference**: Groq AI (Llama 3.3 70B + fallback to 3.1 8B)
- **Interactive UI**: Clean Streamlit interface with recommendations
- **Configuration Presets**: Balanced and Precise modes
- **Chat History**: Track questions and answers
- **Source Citations**: View retrieved document chunks
- **Real-time Metrics**: Monitor system performance

## 🏗️ Architecture

```
Document → Clean Text → Smart Chunks → TF-IDF Embeddings → FAISS Store
                                            ↓
                                    Dual-Stage Retrieval
                                    (Fetch + Re-rank)
                                            ↓
                                    Groq LLM (Strict RAG)
                                            ↓
                                    Grounded Answer
```

### Components

1. **Document Processor** (`document_processor.py`)
   - Loads PDF and text files with PyPDF
   - Recursive splitting with 8 separator levels
   - Text cleaning: normalize spaces, preserve structure
   - Filter meaningless chunks (<30 chars)

2. **Vector Store Manager** (`vector_store.py`)
   - TF-IDF embeddings with bigrams (scikit-learn)
   - 384-dimensional vectors (no model downloads needed)
   - FAISS vector database for fast similarity
   - Re-ranking: cosine similarity + keyword overlap

3. **Retrieval System** (`retriever.py`)
   - Converts queries to TF-IDF embeddings
   - Fetches top-k×2 candidates for re-ranking
   - Returns top-k best results
   - Zero external API calls

4. **Answer Generator** (`answer_generator.py`)
   - Groq LLM: Llama 3.3 70B (primary) → 3.1 8B (fallback)
   - Strict RAG prompt: only uses context
   - Model decommissioning detection & auto-fallback
   - Prevents hallucination with explicit instructions

## 📊 Recommendation Values

The app includes intelligent recommendations for best results:

| Parameter | Recommended | Range | Impact |
|-----------|------------|-------|---------|
| **Chunk Size** | 1000 | 50-2000 | Balance between context and specificity |
| **Chunk Overlap** | 100 | 0-300 | Context sharing between chunks (10-20%) |
| **Top-K Retrieval** | 4 | 1-10 | Number of relevant chunks (3-5 optimal) |
| **Temperature** | 0.3 | 0.0-1.0 | Factual answers (0.0-0.5 for RAG) |

### Quick Presets
- **Balanced Mode** (Default): 1000/100/4/0.3 - Good for most documents
- **Precise Mode**: 500/50/3/0.1 - Maximum accuracy, less context

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Groq API Key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# Clone repository
git clone https://github.com/ssathyasai/CEI_Assignments.git
cd week7_Sathyasai/RAG_Document_Test

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

### Deploy to Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy RAG system"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app" → Select your repository
   - Choose main branch and `app.py`

3. **Add Groq API Key**
   - Go to app settings → Secrets
   - Add: `GROQ_API_KEY = your_api_key_here`
   - Reboot app

## 📖 Usage Guide

### Step 1: Upload Document
- Navigate to **📤 Upload** tab
- Select a PDF or TXT file (up to 50MB)
- Click **🚀 Process Document**
- Monitor progress bar (typically 5-30 seconds)

### Step 2: Configure Settings (Optional)
- **Chunk Size**: Default 1000 is good for most documents
- **Chunk Overlap**: Default 100 captures context well
- **Top-K**: Default 4 balances relevance and context
- **Temperature**: Default 0.3 for factual answers
- Use presets for quick configuration

### Step 3: Ask Questions
- Go to **💬 Ask Questions** tab
- Type your question clearly
- Click **🔍 Ask**
- View answer and source chunks

### Step 4: Monitor Metrics
- Check **📊 Metrics** tab
- See chunk configuration used
- Track number of questions asked
- Review model and embedding info

## ⚙️ Configuration Details

### Chunking Strategy
```
Separators (in order):
1. \n\n\n (triple newline - section breaks)
2. \n\n (double newline - paragraphs)
3. \n (single newline - lines)
4. . (sentence period)
5. ! (sentence exclamation)
6. ? (sentence question)
7. space (word boundary)
8. "" (character level)
```

### Text Cleaning
- Normalize multiple spaces to single
- Preserve paragraph structure
- Remove leading/trailing whitespace
- Filter out non-meaningful content

### Re-ranking Algorithm
```
Score = 0.7 × cosine_similarity + 0.3 × keyword_overlap
```

### Strict RAG Prompt
- Answers ONLY from provided context
- No external knowledge injection
- Returns "document does not contain" if not found
- Prevents hallucination

## 📊 System Specifications

| Specification | Value |
|--------------|-------|
| **Embedding Dimension** | 384 |
| **TF-IDF Features** | Bigrams included |
| **Vector Store** | FAISS (CPU) |
| **Similarity Metric** | Cosine distance |
| **Re-ranking Strategy** | Dual-stage (fetch + rerank) |
| **Primary LLM** | Llama 3.3 70B |
| **Fallback LLM** | Llama 3.1 8B |
| **Max Tokens** | 1024 |
| **Response Time** | <2s per query (Groq API) |

## 🔧 Technical Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| **Framework** | LangChain | 1.4.9+ |
| **LLM** | Groq API | Ultra-fast inference |
| **Embeddings** | TF-IDF | Pure Python, no downloads |
| **Vector Store** | FAISS | CPU-optimized |
| **Text Splitter** | LangChain | RecursiveCharacterTextSplitter |
| **PDF Parser** | PyPDF | Handles PDF extraction |
| **UI** | Streamlit | Interactive web interface |
| **Python** | 3.10+ | Tested on 3.10 |

## 📁 Project Structure

```
week7_Sathyasai/RAG_Document_Test/
├── app.py                              # Main Streamlit app
├── requirements.txt                    # Dependencies
├── README.md                           # This file
├── .python-version                     # Python version
├── .gitignore                          # Git rules
├── .streamlit/
│   └── config.toml                    # Streamlit config
├── data/
│   └── sample.txt                     # Sample document
└── utils/
    ├── __init__.py                    # Package init
    ├── document_processor.py          # Document loading & chunking
    ├── vector_store.py                # Embeddings & FAISS
    ├── retriever.py                   # Dual-stage retrieval
    └── answer_generator.py            # LLM with model fallback
```

## 🎓 Key Concepts

### Retrieval-Augmented Generation (RAG)
- Combines retrieval with generation
- Grounds answers in actual documents
- Reduces hallucination
- Works with private/custom data
- Production-ready for enterprise use

### Intelligent Chunking
- Preserves semantic boundaries
- Maintains context with overlap
- Filters meaningless fragments
- Adapts to document structure

### Dual-Stage Retrieval
1. **Retrieval**: Fetch top-k×2 candidates (broad search)
2. **Re-ranking**: Score all candidates with combined metrics
3. **Output**: Return top-k best results

### Model Fallback
- Automatic detection of model decommissioning
- Seamless fallback to alternative model
- Transparent error reporting

## 🔍 Best Practices

### Document Preparation
- ✅ Use clean, well-formatted documents
- ✅ Ensure text is extractable (not just images)
- ✅ Provide context-rich content
- ❌ Avoid scanned PDFs without OCR
- ❌ Don't use binary/corrupted files

### Query Formulation
- ✅ Be specific and clear
- ✅ Use complete sentences
- ✅ Provide context when needed
- ❌ Don't ask multiple questions in one query
- ❌ Avoid ambiguous phrasing

### Parameter Tuning
- **More context needed**: Increase chunk size (1500+)
- **Too much noise**: Decrease chunk size (500-800)
- **Missing relationships**: Increase overlap (150-200)
- **Too slow**: Reduce top-k (2-3)
- **More precise**: Use Precise preset

## 🐛 Troubleshooting

### Issue: "GROQ_API_KEY not found"
**Solution**: Add secret to Streamlit Cloud
- App settings → Secrets → Add `GROQ_API_KEY`

### Issue: Slow processing
**Solution**: Optimize parameters
- Reduce chunk size (500-800)
- Lower top-k (2-3)
- Use Precise preset

### Issue: Poor answer quality
**Solution**: Improve document and query
- Check document is text-based (not image)
- Use clearer, more specific questions
- Increase chunk overlap
- Try Balanced preset

### Issue: "Model decommissioned" error
**Solution**: Automatic
- System auto-switches to fallback model
- No action needed

## 📝 Example Workflow

### Sample Document
```
RAG System Architecture:
The Retrieval-Augmented Generation (RAG) system consists of three main components:
1. Document Processor: Splits documents into chunks
2. Vector Store: Stores embeddings in FAISS
3. LLM Interface: Generates answers using Groq

Benefits of RAG:
- Grounded answers with citations
- Works with private documents
- Reduces hallucination
- Fast inference with Groq
```

### Sample Questions & Answers

**Q: What are the main components of RAG?**
A: The RAG system has three main components: (1) Document Processor - splits documents into chunks, (2) Vector Store - stores embeddings in FAISS, and (3) LLM Interface - generates answers using Groq.

**Q: What is RAG?**
A: RAG stands for Retrieval-Augmented Generation, a system that retrieves relevant documents and uses them to generate grounded answers.

## ✅ Validation Checklist

- ✅ Document ingestion (PDF/TXT)
- ✅ Intelligent text chunking with 8 separators
- ✅ Text cleaning and normalization
- ✅ Meaningful chunk filtering
- ✅ TF-IDF embeddings with bigrams
- ✅ FAISS vector storage
- ✅ Dual-stage retrieval + re-ranking
- ✅ Strict RAG prompt (no hallucination)
- ✅ Groq LLM with auto-fallback
- ✅ Interactive Streamlit UI
- ✅ Configuration recommendations
- ✅ Quick presets (Balanced/Precise)
- ✅ Source citation display
- ✅ Real-time metrics
- ✅ Chat history tracking

## 🚀 Future Enhancements

- [ ] Hybrid search (keyword + semantic)
- [ ] Query expansion for better retrieval
- [ ] Multi-document support
- [ ] Conversation memory (context from previous Q&A)
- [ ] Answer confidence scoring
- [ ] Export Q&A history to CSV/PDF
- [ ] Document metadata filtering
- [ ] Custom instruction templates

## 📄 License

This project is for educational purposes (Celebal Technologies Internship).

## 👨‍💻 Author

Created as part of **Week 7 Assignment** - Celebal Technologies Internship Program

---

## 🔴 Live Demo

**Status**: Coming soon on Streamlit Cloud

**Demo Link**: [Will be updated after deployment](https://share.streamlit.io)

**To Try Locally**:
```bash
git clone https://github.com/ssathyasai/CEI_Assignments.git
cd week7_Sathyasai/RAG_Document_Test
pip install -r requirements.txt
streamlit run app.py
```

---

## 🤝 Support

- 🔑 **Groq API Key**: [Get free at console.groq.com](https://console.groq.com)
- 📚 **LangChain Docs**: [python.langchain.com](https://python.langchain.com)
- 🎈 **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- 🐛 **Report Issues**: Create GitHub issue

---

**Last Updated**: July 15, 2026  
**Status**: ✅ Production Ready
