"""
RAG Document Question Answering System - Streamlit Web Application
A Retrieval-Augmented Generation system with advanced optimizations and evaluation dashboards.
"""

import streamlit as st
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Import utilities
from utils.document_processor import DocumentProcessor
from utils.vector_store import EmbeddingGenerator, VectorStoreManager
from utils.retriever import HybridRetriever
from utils.answer_generator import AnswerGenerator
from utils.evaluator import RAGEvaluator

# Load environment variables if present
load_dotenv()

# App Configuration
st.set_page_config(
    page_title="HelixRAG - Cognitive Document Q&A",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
        color: #f8fafc;
    }
    
    /* Custom Title */
    .app-title-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .app-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        font-weight: 300;
    }
    
    /* Sidebar styling */
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f472b6 0%, #fb7185 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    
    /* Card design */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: border-color 0.3s ease, transform 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 242, 254, 0.4);
        transform: translateY(-2px);
    }
    
    /* Chat bubbles */
    .answer-box {
        background: rgba(15, 23, 42, 0.75);
        border-left: 4px solid #00f2fe;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
    }
    .chunk-box {
        background: rgba(30, 41, 59, 0.3);
        border-left: 3px solid #ec4899;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.8rem 0;
        font-size: 0.95rem;
    }
    
    /* Streamlit tabs styling override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(30, 41, 59, 0.25);
        border-radius: 8px;
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 0 24px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(30, 41, 59, 0.65) !important;
        border: 1px solid rgba(0, 242, 254, 0.4) !important;
        color: #00f2fe !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = None
if "emb_gen" not in st.session_state:
    st.session_state.emb_gen = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "answer_gen" not in st.session_state:
    st.session_state.answer_gen = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "stats" not in st.session_state:
    st.session_state.stats = {}
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "evaluation_report" not in st.session_state:
    st.session_state.evaluation_report = ""

def main():
    # Header
    st.markdown("""
    <div class="app-title-container">
        <div class="app-title">HelixRAG 🧬</div>
        <div class="app-subtitle">Optimized Cognitive Ingestion and Grounded QA Pipeline</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Setup
    with st.sidebar:
        st.markdown('<p class="sidebar-title">⚙️ Control Center</p>', unsafe_allow_html=True)
        
        # 1. API Keys Section
        st.subheader("🔑 Credentials")
        
        # Get defaults from environment variables
        env_gemini_key = os.getenv("GOOGLE_API_KEY", "")
        env_cohere_key = os.getenv("COHERE_API_KEY", "")
        env_pinecone_key = os.getenv("PINECONE_API_KEY", "")
        
        gemini_api_key = st.text_input(
            "Google Gemini API Key", 
            value=env_gemini_key, 
            type="password",
            placeholder="AIzaSy..."
        )
        cohere_api_key = st.text_input(
            "Cohere API Key", 
            value=env_cohere_key, 
            type="password",
            placeholder="For Embeddings/Reranking/LLM"
        )
        pinecone_api_key = st.text_input(
            "Pinecone API Key", 
            value=env_pinecone_key, 
            type="password",
            placeholder="For cloud-based indexing"
        )
        
        st.divider()
        
        # 2. Pipeline Settings
        st.subheader("🛠️ Engine Configurations")
        
        embedding_provider = st.selectbox(
            "Embedding Provider",
            options=["Local (SentenceTransformers)", "Gemini", "Cohere"],
            help="Model provider to generate vector representations"
        )
        
        vector_db_tool = st.selectbox(
            "Vector Database Tool",
            options=["FAISS (Local)", "Pinecone (Cloud)"],
            help="FAISS is fast and local. Pinecone connects to a serverless cloud instance."
        )
        
        llm_provider = st.selectbox(
            "Language Model Provider",
            options=["Gemini", "Cohere"],
            help="Generative LLM to output context-grounded responses"
        )
        
        st.divider()
        
        # 3. Dynamic Optimization Tuning
        st.subheader("🧪 Hyperparameters")
        
        chunk_size = st.slider(
            "Max Chunk Size",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of partitioned text blocks in characters"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=400,
            value=150,
            step=50,
            help="Overlap character count between adjacent chunks"
        )
        
        search_strategy = st.selectbox(
            "Retrieval Strategy",
            options=["Hybrid (Vector + Keyword)", "Vector Search", "Keyword (BM25) Search"],
            help="Choose the retrieval strategy. Hybrid combines dense semantic matching and keyword sparse scores."
        )
        
        alpha_weight = st.slider(
            "Hybrid Weight (Alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="1.0 = Dense Vector Search only. 0.0 = Sparse BM25 Search only."
        )
        
        rerank_strategy = st.selectbox(
            "Re-ranking Strategy",
            options=["None", "Local Cross-Encoder (Free)", "Cohere Rerank API"],
            help="Local Cross-Encoder runs a second-stage transformer locally to rank retrieved passages. Cohere uses cloud reranking."
        )
        
        top_k_candidates = st.number_input(
            "Candidates retrieved (top_k)",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
            help="Initial size of context pool retrieved before applying re-ranking"
        )
        
        top_n_final = st.number_input(
            "Final chunks for context (top_n)",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Number of final context blocks sent to the prompt"
        )
        
        generation_temp = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Higher values increase creativity; lower values ensure strict alignment with context."
        )
        
        if st.button("🗑️ Clear Cache", type="secondary"):
            st.session_state.vector_manager = None
            st.session_state.emb_gen = None
            st.session_state.retriever = None
            st.session_state.answer_gen = None
            st.session_state.document_processed = False
            st.session_state.chat_history = []
            st.session_state.stats = {}
            st.session_state.evaluation_results = None
            st.session_state.evaluation_report = ""
            st.rerun()

    # Main Area Tabs
    tabs = st.tabs([
        "📥 Ingest & Structure", 
        "💬 Conversational Q&A", 
        "📈 Pipeline Metrics & Evaluation"
    ])
    
    # ==================== TAB 1: INGESTION ====================
    with tabs[0]:
        st.subheader("Document Ingestion Module")
        st.write("Ingest domain custom text inputs: upload local files (PDF/TXT) or download Hugging Face dataset archives.")
        
        ingest_mode = st.radio("Select Ingestion Mode", ["File Upload (PDF/TXT)", "Hugging Face Archive"], horizontal=True)
        
        raw_text = ""
        doc_source = ""
        
        if ingest_mode == "File Upload (PDF/TXT)":
            uploaded_file = st.file_uploader("Upload local document", type=["pdf", "txt"])
            if uploaded_file:
                st.info(f"Loaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
                doc_source = uploaded_file.name
                
                # Load text
                with st.spinner("Extracting text from uploaded file..."):
                    try:
                        file_bytes = uploaded_file.read()
                        if uploaded_file.name.endswith(".pdf"):
                            raw_text = DocumentProcessor.extract_text_from_pdf(file_bytes)
                        else:
                            raw_text = DocumentProcessor.extract_text_from_txt(file_bytes)
                        st.success(f"Extracted {len(raw_text)} characters.")
                    except Exception as e:
                        st.error(f"Text extraction failed: {str(e)}")
        
        else:
            col1, col2 = st.columns(2)
            with col1:
                hf_name = st.text_input("Hugging Face Dataset Name", placeholder="e.g. wikitext", value="wikitext")
                hf_subset = st.text_input("Subset Config (Optional)", placeholder="e.g. wikitext-2-raw-v1", value="wikitext-2-raw-v1")
            with col2:
                hf_column = st.text_input("Text Column Name", placeholder="e.g. text", value="text")
                hf_max_samples = st.number_input("Max Sample Rows", min_value=10, max_value=1000, value=50, step=50)
            
            hf_split = st.selectbox("Split Name", ["train", "validation", "test"], index=0)
            
            if st.button("Download HF Dataset", type="secondary"):
                with st.spinner(f"Loading '{hf_name}' dataset split '{hf_split}'..."):
                    try:
                        raw_text = DocumentProcessor.load_hf_dataset(
                            dataset_name=hf_name,
                            text_column=hf_column,
                            subset=hf_subset if hf_subset else None,
                            split=hf_split,
                            max_samples=hf_max_samples
                        )
                        doc_source = f"Hugging Face Dataset: {hf_name}/{hf_subset or ''}"
                        st.success(f"Successfully downloaded Hugging Face archive. Extracted {len(raw_text)} characters.")
                        st.session_state["hf_downloaded_text"] = (raw_text, doc_source)
                    except Exception as e:
                        st.error(f"Failed to load HF dataset: {str(e)}")
            
            if "hf_downloaded_text" in st.session_state:
                raw_text, doc_source = st.session_state["hf_downloaded_text"]
                st.caption(f"Active Text Resource: {doc_source} ({len(raw_text)} characters)")
        
        if raw_text:
            st.divider()
            st.subheader("Process & Index Settings")
            st.write("Pre-process and vector-store the text into a similarity matching pipeline.")
            
            if st.button("🚀 Process & Index Document", type="primary"):
                # Validate keys
                keys_ok = True
                if embedding_provider == "Gemini" and not gemini_api_key:
                    st.error("Missing Google Gemini API Key in Control Center.")
                    keys_ok = False
                if embedding_provider == "Cohere" and not cohere_api_key:
                    st.error("Missing Cohere API Key in Control Center.")
                    keys_ok = False
                if vector_db_tool == "Pinecone (Cloud)" and not pinecone_api_key:
                    st.error("Missing Pinecone API Key in Control Center.")
                    keys_ok = False
                
                if keys_ok:
                    try:
                        status_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # 1. Chunking
                        status_text.text("✂️ Splitting document text into overlapping chunks...")
                        status_bar.progress(15)
                        
                        processor = DocumentProcessor()
                        chunks = processor.split_text(
                            text=raw_text,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            source_metadata={"source": doc_source}
                        )
                        
                        status_text.text(f"Created {len(chunks)} text chunks.")
                        status_bar.progress(35)
                        
                        # 2. Embedding creation
                        status_text.text("🔢 Connecting to embedding client & mapping text chunks...")
                        status_bar.progress(55)
                        
                        provider_key = ""
                        if embedding_provider == "Gemini":
                            provider_key = gemini_api_key
                            emb_prov_name = "gemini"
                            emb_model = "models/text-embedding-004"
                        elif embedding_provider == "Cohere":
                            provider_key = cohere_api_key
                            emb_prov_name = "cohere"
                            emb_model = "embed-english-v3.0"
                        else:
                            emb_prov_name = "local"
                            emb_model = "sentence-transformers/all-MiniLM-L6-v2"
                        
                        emb_generator = EmbeddingGenerator(
                            provider=emb_prov_name,
                            api_key=provider_key,
                            model_name=emb_model
                        )
                        
                        # Batch embed
                        chunk_texts = [c["text"] for c in chunks]
                        embeddings = []
                        
                        # Embed in batches to prevent API rate limit issues
                        batch_size = 32
                        for idx_b in range(0, len(chunk_texts), batch_size):
                            batch_texts = chunk_texts[idx_b:idx_b + batch_size]
                            batch_embs = emb_generator.embed_texts(batch_texts)
                            embeddings.extend(batch_embs)
                            
                        status_text.text("Storing vectors in database...")
                        status_bar.progress(80)
                        
                        # 3. Vector Database
                        db_type = "pinecone" if "Pinecone" in vector_db_tool else "faiss"
                        vector_manager = VectorStoreManager(
                            store_type=db_type,
                            pinecone_api_key=pinecone_api_key if db_type == "pinecone" else None
                        )
                        vector_manager.initialize_store(emb_generator.dimension)
                        vector_manager.add_documents(chunks, embeddings)
                        
                        # Setup retriever & generator
                        retriever = HybridRetriever(vector_manager, emb_generator)
                        
                        llm_prov_name = "gemini" if llm_provider == "Gemini" else "cohere"
                        llm_key = gemini_api_key if llm_provider == "Gemini" else cohere_api_key
                        answer_generator = AnswerGenerator(
                            provider=llm_prov_name,
                            api_key=llm_key,
                            temperature=generation_temp
                        )
                        
                        # Save state
                        st.session_state.vector_manager = vector_manager
                        st.session_state.emb_gen = emb_generator
                        st.session_state.retriever = retriever
                        st.session_state.answer_gen = answer_generator
                        st.session_state.document_processed = True
                        
                        # Set stats
                        st.session_state.stats = {
                            "source": doc_source,
                            "char_count": len(raw_text),
                            "chunk_count": len(chunks),
                            "avg_chunk_size": np.mean([len(c["text"]) for c in chunks]) if chunks else 0,
                            "dimension": emb_generator.dimension,
                            "embedding_model": emb_generator.model_name,
                            "vector_db": vector_db_tool,
                            "llm_model": answer_generator.model_name
                        }
                        
                        status_text.text("Pipeline processed successfully!")
                        status_bar.progress(100)
                        time.sleep(0.5)
                        
                        st.success("🎉 RAG Pipeline initialised and indexed!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Initialization Error: {str(e)}")
                        
        if st.session_state.document_processed:
            st.divider()
            st.subheader("📊 Ingested Stats & Chunk Visualizer")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Chunks", st.session_state.stats["chunk_count"])
            with col2:
                st.metric("Avg Chunk Size (chars)", f"{int(st.session_state.stats['avg_chunk_size'])}")
            with col3:
                st.metric("Vector Dimension", st.session_state.stats["dimension"])
            with col4:
                st.metric("Embedding Model", st.session_state.stats["embedding_model"].split('/')[-1])
            
            # Interactive Chunk Explorer
            st.write("#### Chunk Explorer")
            chunks_data = st.session_state.vector_manager.chunks
            selected_chunk_id = st.slider("Select chunk index to inspect", 0, len(chunks_data)-1, 0)
            
            c_info = chunks_data[selected_chunk_id]
            st.markdown(f"""
            <div class="glass-card">
                <h5>Chunk {selected_chunk_id} | Source: {c_info['metadata'].get('source', 'Unknown')}</h5>
                <p>Length: {len(c_info['text'])} chars</p>
                <div class="chunk-box" style="font-family: monospace;">
                    {c_info['text']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Text Chunk Distribution Chart
            st.write("#### Chunk Length Distribution")
            lens = [len(c["text"]) for c in chunks_data]
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(lens, bins=15, color="#00f2fe", edgecolor="#020617")
            ax.set_title("Distribution of Chunk Sizes (Characters)", color="white")
            ax.set_xlabel("Size (Characters)", color="white")
            ax.set_ylabel("Count", color="white")
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#475569')
            st.pyplot(fig)

    # ==================== TAB 2: CHATBOT ====================
    with tabs[1]:
        st.subheader("Grounded Question Answering Route")
        st.write("Submit natural language questions. The engine will retrieve context documents and generate grounded responses.")
        
        if not st.session_state.document_processed:
            st.warning("Please upload and index a document in the Ingest tab first.")
        else:
            # Custom styled input form
            with st.form("chat_form", clear_on_submit=False):
                query_str = st.text_input("Your Question", placeholder="Ask a question about the uploaded document...")
                submit_button = st.form_submit_button("🔍 Find Answer", type="primary")
            
            if submit_button and query_str:
                # Dynamic re-instantiation of LLM/Retriever to capture slider updates
                st.session_state.answer_gen.temperature = generation_temp
                
                # Check for needed API key if we dynamically change reranker
                valid_run = True
                if rerank_strategy == "Cohere Rerank API" and not cohere_api_key:
                    st.error("Cohere API Key is required for Cohere Reranking.")
                    valid_run = False
                
                # Double-check LLM API key
                if llm_provider == "Gemini" and not gemini_api_key:
                    st.error("Google Gemini API Key is required.")
                    valid_run = False
                elif llm_provider == "Cohere" and not cohere_api_key:
                    st.error("Cohere API Key is required.")
                    valid_run = False
                
                if valid_run:
                    with st.spinner("Executing retrieval and generation pipeline..."):
                        t_start = time.time()
                        
                        # 1. Retrieve
                        strat_key = "hybrid"
                        if "Vector" in search_strategy:
                            strat_key = "vector"
                        elif "Keyword" in search_strategy:
                            strat_key = "keyword"
                            
                        rerank_key = "none"
                        if "Local" in rerank_strategy:
                            rerank_key = "local"
                        elif "Cohere" in rerank_strategy:
                            rerank_key = "cohere"
                        
                        retrieved = st.session_state.retriever.retrieve(
                            query=query_str,
                            search_strategy=strat_key,
                            alpha=alpha_weight,
                            top_k_retrieve=top_k_candidates,
                            rerank_strategy=rerank_key,
                            cohere_api_key=cohere_api_key,
                            top_n_final=top_n_final
                        )
                        
                        t_retrieval = time.time() - t_start
                        
                        # 2. Generate
                        t_gen_start = time.time()
                        chunks_list = [c[0] for c in retrieved]
                        
                        answer = st.session_state.answer_gen.generate_answer(query_str, chunks_list)
                        t_generation = time.time() - t_gen_start
                        
                        t_total = time.time() - t_start
                        
                        # Append history
                        st.session_state.chat_history.append({
                            "query": query_str,
                            "answer": answer,
                            "retrieved": retrieved,
                            "latency": {
                                "retrieval": t_retrieval,
                                "generation": t_generation,
                                "total": t_total
                            }
                        })
            
            # Display Chat History
            if st.session_state.chat_history:
                st.divider()
                st.write("### Conversational Logs")
                
                for idx, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.container():
                        st.markdown(f"#### 👤 Query: {chat['query']}")
                        
                        # Grounded response container
                        st.markdown(f"""
                        <div class="answer-box">
                            <strong>💡 Generated Grounded Answer:</strong><br><br>
                            {chat['answer'].replace(chr(10), '<br>')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Latency stats
                        st.caption(f"⏱️ Pipeline Latency: Retrieval `{chat['latency']['retrieval']:.3f}s` | Generation `{chat['latency']['generation']:.3f}s` | Total `{chat['latency']['total']:.3f}s`")
                        
                        # Retrieved documents inspect list
                        with st.expander(f"📄 Retrieved Chunks details (Count: {len(chat['retrieved'])})"):
                            for i, (chunk, score) in enumerate(chat['retrieved']):
                                st.markdown(f"""
                                <div class="chunk-box">
                                    <strong>Document Chunk {i+1} | Source: {chunk['metadata'].get('source', 'Unknown')} | Match Score: {score:.4f}</strong><br><br>
                                    {chunk['text']}
                                </div>
                                """, unsafe_allow_html=True)
                        st.divider()

    # ==================== TAB 3: METRICS & EVALUATION ====================
    with tabs[2]:
        st.subheader("Dynamic Pipeline Optimizations & Evaluation Logs")
        st.write("Run pipeline validation logs on dynamic test queries and view aggregated metrics reports.")
        
        if not st.session_state.document_processed:
            st.warning("Please upload and index a document in the Ingest tab first.")
        else:
            st.write("### 🧪 Automated Validation Runner")
            st.write("Configure test questions to automatically run against the current pipeline configurations to verify correctness and measure performance.")
            
            default_test_queries = (
                "What is the main topic or objective discussed in the document?\n"
                "What are the core components used or referenced in the file?\n"
                "Give a summary of the conclusions or key findings described."
            )
            
            test_queries_input = st.text_area("Test Queries (One per line)", value=default_test_queries, height=120)
            
            if st.button("🚀 Run Pipeline Validation", type="primary"):
                questions = [q.strip() for q in test_queries_input.split("\n") if q.strip()]
                
                # Check for needed API key if we dynamically change reranker
                valid_run = True
                if rerank_strategy == "Cohere Rerank API" and not cohere_api_key:
                    st.error("Cohere API Key is required for Cohere Reranking.")
                    valid_run = False
                
                # Double-check LLM API key
                if llm_provider == "Gemini" and not gemini_api_key:
                    st.error("Google Gemini API Key is required.")
                    valid_run = False
                elif llm_provider == "Cohere" and not cohere_api_key:
                    st.error("Cohere API Key is required.")
                    valid_run = False
                
                if not questions:
                    st.error("Please enter at least one test query.")
                    valid_run = False
                    
                if valid_run:
                    with st.spinner(f"Evaluating {len(questions)} test queries..."):
                        evaluator = RAGEvaluator(st.session_state.retriever, st.session_state.answer_gen)
                        
                        strat_key = "hybrid"
                        if "Vector" in search_strategy:
                            strat_key = "vector"
                        elif "Keyword" in search_strategy:
                            strat_key = "keyword"
                            
                        rerank_key = "none"
                        if "Local" in rerank_strategy:
                            rerank_key = "local"
                        elif "Cohere" in rerank_strategy:
                            rerank_key = "cohere"
                            
                        results, report = evaluator.run_evaluation(
                            sample_questions=questions,
                            search_strategy=strat_key,
                            alpha=alpha_weight,
                            rerank_strategy=rerank_key,
                            cohere_api_key=cohere_api_key,
                            top_k_retrieve=top_k_candidates,
                            top_n_final=top_n_final,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        
                        st.session_state.evaluation_results = results
                        st.session_state.evaluation_report = report
                        
                        st.success("✅ Validation completed successfully! Log generated in `logs/validation_logs.json` and metrics report created in `logs/system_metrics_report.md`.")
            
            # Show results if present
            if st.session_state.evaluation_results:
                st.divider()
                st.write("### 📈 Aggregated Evaluation Metrics")
                
                avg_ret_lat = np.mean([r["retrieval_latency_sec"] for r in st.session_state.evaluation_results])
                avg_gen_lat = np.mean([r["generation_latency_sec"] for r in st.session_state.evaluation_results])
                avg_tot_lat = np.mean([r["total_latency_sec"] for r in st.session_state.evaluation_results])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Retrieval Latency", f"{avg_ret_lat:.3f}s")
                with col2:
                    st.metric("Avg Generation Latency", f"{avg_gen_lat:.3f}s")
                with col3:
                    st.metric("Avg Total Latency", f"{avg_tot_lat:.3f}s")
                
                st.divider()
                st.write("### 📄 System Metrics Report Preview")
                st.markdown(st.session_state.evaluation_report)
                
                # Download buttons
                st.divider()
                st.write("### 📥 Download Validation Logs & Report")
                
                col1, col2 = st.columns(2)
                with col1:
                    log_json = json.dumps(st.session_state.evaluation_results, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="Download Raw JSON Logs (`validation_logs.json`)",
                        data=log_json,
                        file_name="validation_logs.json",
                        mime="application/json"
                    )
                with col2:
                    st.download_button(
                        label="Download Markdown Report (`system_metrics_report.md`)",
                        data=st.session_state.evaluation_report,
                        file_name="system_metrics_report.md",
                        mime="text/markdown"
                    )

if __name__ == "__main__":
    main()
