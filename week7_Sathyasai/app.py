"""
RAG Document Question Answering System - Streamlit App
A Retrieval-Augmented Generation system for document-based Q&A
"""

import streamlit as st
import os
import time
from pathlib import Path

# Import utility modules
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.retriever import RetrievalSystem
from utils.answer_generator import AnswerGenerator

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #F63366;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chunk-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #F63366;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #0068c9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False


def main():
    """Main application function"""
    
    # Header
    st.markdown('<p class="main-header">📚 RAG Document Q&A System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your documents using AI-powered retrieval</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        st.divider()
        
        # System configuration
        st.subheader("📊 System Settings")
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of text chunks in characters"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=300,
            value=100,
            step=50,
            help="Overlap between chunks"
        )
        
        top_k = st.slider(
            "Top-K Retrieval",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of chunks to retrieve"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Generation randomness (lower = more focused)"
        )
        
        st.divider()
        
        # System info
        st.subheader("ℹ️ System Info")
        st.info("""
        **Embedding Model**: all-MiniLM-L6-v2
        
        **Vector DB**: FAISS
        
        **LLM**: Google Gemini Pro
        """)
        
        # Clear button
        if st.button("🗑️ Clear All", type="secondary"):
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.document_processed = False
            st.rerun()
    
    # Main content
    tabs = st.tabs(["📤 Upload & Process", "💬 Ask Questions", "📊 Metrics"])
    
    # Tab 1: Upload & Process
    with tabs[0]:
        st.header("Upload Your Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=['pdf', 'txt'],
            help="Upload a document to ask questions about"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                process_button = st.button("🚀 Process Document", type="primary")
            
            if process_button:
                if not api_key:
                    st.error("⚠️ Please enter your Google Gemini API key in the sidebar!")
                else:
                    process_document(uploaded_file, api_key, chunk_size, chunk_overlap, top_k, temperature)
        
        # Show processing status
        if st.session_state.document_processed:
            st.success("✅ Document processed successfully! Go to 'Ask Questions' tab.")
    
    # Tab 2: Ask Questions
    with tabs[1]:
        st.header("Ask Questions About Your Document")
        
        if not st.session_state.document_processed:
            st.warning("⚠️ Please upload and process a document first!")
        else:
            # Question input
            question = st.text_input(
                "Enter your question:",
                placeholder="What is this document about?"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                ask_button = st.button("🔍 Ask", type="primary")
            
            if ask_button and question:
                answer_question(question, api_key)
            
            # Display chat history
            if st.session_state.chat_history:
                st.divider()
                st.subheader("💬 Chat History")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.expander(f"Q: {chat['question'][:50]}...", expanded=(i==0)):
                        st.markdown(f"**Question:** {chat['question']}")
                        st.markdown(f'<div class="answer-box"><strong>Answer:</strong><br>{chat["answer"]}</div>', 
                                  unsafe_allow_html=True)
                        
                        if chat.get('sources'):
                            st.markdown("**📄 Source Chunks:**")
                            for j, source in enumerate(chat['sources'][:3]):
                                st.markdown(f'<div class="chunk-box"><strong>Chunk {j+1}:</strong><br>{source[:300]}...</div>',
                                          unsafe_allow_html=True)
    
    # Tab 3: Metrics
    with tabs[2]:
        st.header("📊 System Metrics")
        
        if st.session_state.document_processed:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Chunk Size", f"{chunk_size} chars")
                st.metric("Embedding Model", "all-MiniLM-L6-v2")
            
            with col2:
                st.metric("Chunk Overlap", f"{chunk_overlap} chars")
                st.metric("Vector DB", "FAISS")
            
            with col3:
                st.metric("Top-K Retrieval", top_k)
                st.metric("LLM", "Gemini Pro")
            
            st.divider()
            
            st.subheader("📈 Chat Statistics")
            st.metric("Total Questions Asked", len(st.session_state.chat_history))
        else:
            st.info("Process a document to see metrics")


def process_document(uploaded_file, api_key, chunk_size, chunk_overlap, top_k, temperature):
    """Process uploaded document and create vector store"""
    
    with st.spinner("🔄 Processing document..."):
        try:
            # Step 1: Process document
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📄 Loading document...")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = processor.process_uploaded_file(uploaded_file)
            
            status_text.text(f"✂️ Created {len(chunks)} chunks...")
            progress_bar.progress(40)
            time.sleep(0.5)
            
            # Step 2: Create embeddings and vector store
            status_text.text("🔢 Creating embeddings...")
            progress_bar.progress(60)
            
            vector_manager = VectorStoreManager()
            vector_store = vector_manager.create_vector_store(chunks)
            
            status_text.text("💾 Storing in vector database...")
            progress_bar.progress(80)
            time.sleep(0.5)
            
            # Step 3: Initialize retrieval system
            retrieval_system = RetrievalSystem(vector_store, top_k=top_k)
            
            # Step 4: Initialize answer generator
            answer_gen = AnswerGenerator(
                retriever=retrieval_system.retriever,
                api_key=api_key,
                temperature=temperature
            )
            
            # Store in session state
            st.session_state.vector_store = vector_store
            st.session_state.retrieval_system = retrieval_system
            st.session_state.answer_generator = answer_gen
            st.session_state.document_processed = True
            st.session_state.num_chunks = len(chunks)
            
            status_text.text("✅ Processing complete!")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            st.success(f"✅ Successfully processed document into {len(chunks)} chunks!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")


def answer_question(question, api_key):
    """Generate answer for user question"""
    
    with st.spinner("🤔 Thinking..."):
        try:
            # Retrieve and generate answer
            start_time = time.time()
            result = st.session_state.answer_generator.generate_answer(question)
            elapsed_time = time.time() - start_time
            
            answer = result['result']
            sources = result.get('source_documents', [])
            
            # Display answer
            st.markdown(f'<div class="answer-box"><strong>💡 Answer:</strong><br><br>{answer}</div>', 
                       unsafe_allow_html=True)
            
            # Display sources
            if sources:
                st.markdown("---")
                st.subheader("📄 Retrieved Source Chunks")
                
                for i, doc in enumerate(sources, 1):
                    with st.expander(f"📝 Chunk {i}"):
                        st.markdown(f'<div class="chunk-box">{doc.page_content}</div>', 
                                  unsafe_allow_html=True)
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.caption(f"Metadata: {doc.metadata}")
            
            # Show timing
            st.caption(f"⏱️ Response generated in {elapsed_time:.2f} seconds")
            
            # Add to chat history
            st.session_state.chat_history.append({
                'question': question,
                'answer': answer,
                'sources': [doc.page_content for doc in sources],
                'time': elapsed_time
            })
            
        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")


if __name__ == "__main__":
    main()
