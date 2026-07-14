import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
import time

# Add the project root directory to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import custom modules
from src.document_ingestion import DocumentIngestion
from src.text_chunking import TextChunking
from src.embedding import EmbeddingService
from src.vector_store import VectorStoreService
from src.retrieval import RetrievalService
from src.generation import GenerationService

# Page configuration
st.set_page_config(
    page_title="RAG Document QA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #00ff88;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 10px rgba(0,255,136,0.3);
    }
    .answer-box {
        background: linear-gradient(135deg, #0d2818, #0d1b2a);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid #00ff88;
        margin: 1rem 0;
    }
    .chunk-card {
        background-color: #0d1b2a;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #00ff88;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.documents_processed = []
    st.session_state.total_chunks = 0
    st.session_state.rag_app = None

class RAGApp:
    def __init__(self):
        self.document_ingestion = DocumentIngestion()
        self.text_chunking = None
        self.embedding_service = None
        self.vector_store = None
        self.retrieval_service = None
        self.generation_service = None
        
    def initialize_services(self, api_key, chunk_size=500, overlap=50):
        """Initialize all services"""
        try:
            self.text_chunking = TextChunking(chunk_size=chunk_size, overlap=overlap)
            self.embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
            self.vector_store = VectorStoreService(collection_name="rag_documents")
            self.retrieval_service = RetrievalService(
                vector_store=self.vector_store,
                embedding_service=self.embedding_service
            )
            
            if api_key:
                self.generation_service = GenerationService(
                    api_key=api_key,
                    model="mixtral-8x7b-32768"
                )
            
            st.session_state.initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize services: {str(e)}")
            return False
    
    def process_document(self, file, doc_name):
        """Process a single document"""
        try:
            if file.type == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_path = tmp_file.name
                text = self.document_ingestion.load_document(tmp_path)
                os.unlink(tmp_path)
            else:
                text = file.getvalue().decode("utf-8")
            
            chunks = self.text_chunking.chunk_by_size(text, document_name=doc_name)
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.encode_texts(chunk_texts)
            self.vector_store.store_embeddings(chunks, embeddings)
            
            st.session_state.documents_processed.append(doc_name)
            st.session_state.total_chunks += len(chunks)
            
            return {"success": True, "chunks": len(chunks), "doc_name": doc_name}
        except Exception as e:
            return {"success": False, "error": str(e), "doc_name": doc_name}
    
    def query_system(self, question, top_k=3):
        """Query the RAG system"""
        try:
            retrieved_chunks = self.retrieval_service.retrieve_context(query=question, top_k=top_k)
            context = self.retrieval_service.format_context(retrieved_chunks)
            answer = self.generation_service.generate_answer(
                query=question,
                context=context,
                temperature=0.7,
                max_tokens=500
            )
            
            return {
                "success": True,
                "answer": answer,
                "context": context,
                "retrieved_chunks": retrieved_chunks
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_stats(self):
        """Get system statistics"""
        return {
            "documents_processed": len(st.session_state.documents_processed),
            "total_chunks": st.session_state.total_chunks,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "vector_store_count": self.vector_store.get_collection_stats().get("total_documents", 0)
        }

def main():
    st.markdown('<h1 class="main-header">🤖 Document QA System</h1>', unsafe_allow_html=True)
    st.markdown("### Retrieval-Augmented Generation (RAG) Pipeline")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Get API key
        api_key = None
        try:
            # Try Streamlit Cloud secrets
            api_key = st.secrets["GROQ_API_KEY"]
            st.success("✅ API Key loaded from secrets")
        except:
            # Manual input fallback
            api_key = st.text_input(
                "🔑 Groq API Key",
                type="password",
                help="Get your API key from console.groq.com",
                placeholder="Enter your API key here"
            )
        
        st.markdown("### 📝 Chunking Settings")
        chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
        
        if st.button("🚀 Initialize System", use_container_width=True):
            if not api_key:
                st.error("Please provide Groq API Key")
            else:
                rag_app = RAGApp()
                with st.spinner("Initializing services..."):
                    if rag_app.initialize_services(api_key, chunk_size, chunk_overlap):
                        st.session_state.rag_app = rag_app
                        st.success("✅ System initialized successfully!")
                    else:
                        st.error("Failed to initialize system")
        
        if st.session_state.initialized:
            st.markdown("---")
            st.markdown("### 📊 System Stats")
            rag_app = st.session_state.rag_app
            stats = rag_app.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats["documents_processed"])
                st.metric("Total Chunks", stats["total_chunks"])
            with col2:
                st.metric("Embedding Dim", stats["embedding_dimension"])
                st.metric("Vector Store", stats["vector_store_count"])
            
            st.markdown("---")
            st.markdown("### 📂 Processed Documents")
            for doc in st.session_state.documents_processed:
                st.text(f"• {doc}")
    
    # Main Content
    if not st.session_state.initialized:
        st.info("👈 Please configure and initialize the system from the sidebar")
        st.markdown("""
        ### 🚀 Quick Start Guide
        
        **For Streamlit Cloud Users:**
        1. Add `GROQ_API_KEY` to your Streamlit Cloud secrets
        2. Click "Initialize System"
        3. Upload documents and ask questions!
        
        **For Local Development:**
        1. Get Groq API key from [console.groq.com](https://console.groq.com)
        2. Enter it in the sidebar
        3. Click "Initialize System"
        4. Upload documents and ask questions!
        
        ### 📚 Features
        - 📄 Upload PDF and text documents
        - 🔍 Semantic search using embeddings
        - 🧠 AI-powered answers using Groq
        - 📊 Real-time system metrics
        - 🎯 Context-aware responses
        """)
        return
    
    rag_app = st.session_state.rag_app
    
    # Document Upload Section
    st.markdown("## 📤 Upload Documents")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Choose documents (PDF or TXT)",
            type=['pdf', 'txt'],
            accept_multiple_files=True
        )
    with col2:
        if uploaded_files and st.button("📂 Process All", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing: {file.name}")
                result = rag_app.process_document(file, file.name)
                if result["success"]:
                    st.success(f"✅ {file.name} - {result['chunks']} chunks created")
                else:
                    st.error(f"❌ {file.name} - {result.get('error', 'Unknown error')}")
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            progress_bar.empty()
            status_text.empty()
            st.success("✅ All documents processed!")
            st.rerun()
    
    # Query Section
    st.markdown("---")
    st.markdown("## 💬 Ask Questions")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "Enter your question",
            placeholder="e.g., What is the main topic of the document?"
        )
    with col2:
        top_k = st.number_input("Retrieve top K", min_value=1, max_value=10, value=3)
    
    if question and st.button("🔍 Get Answer", use_container_width=True, type="primary"):
        if not st.session_state.documents_processed:
            st.warning("⚠️ Please upload and process documents first!")
        else:
            with st.spinner("Searching and generating answer..."):
                result = rag_app.query_system(question, top_k)
                
                if result["success"]:
                    st.markdown("### 📝 Answer")
                    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                    
                    with st.expander("📚 View Retrieved Context", expanded=False):
                        for i, chunk in enumerate(result["retrieved_chunks"], 1):
                            score = chunk["similarity_score"]
                            st.markdown(
                                f'<div class="chunk-card">'
                                f'<strong>Chunk {i}</strong> '
                                f'<span style="color:#00ff88">(Relevance: {score:.3f})</span>'
                                f'<br>{chunk["text"][:500]}...'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()