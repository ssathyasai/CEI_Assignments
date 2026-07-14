"""RAG Document Q&A System - Streamlit App with Groq"""
import streamlit as st
import time
import os
import warnings
from utils import DocumentProcessor, VectorStoreManager, RetrievalSystem, AnswerGenerator

# Suppress warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="RAG Q&A System", page_icon="📚", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main-header {font-size: 2.5rem; font-weight: bold; color: #F63366; text-align: center;}
.chunk-box {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 3px solid #F63366;}
.answer-box {background-color: #e8f4f8; padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #0068c9;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'doc_processed' not in st.session_state:
    st.session_state.doc_processed = False
if 'model_name' not in st.session_state:
    st.session_state.model_name = "llama-3.3-70b-versatile"


def main():
    st.markdown('<p class="main-header">📚 RAG Document Q&A System</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#666;">Powered by Groq AI - Ultra Fast Inference</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Get API key from secrets (REQUIRED - no manual input)
        if "GROQ_API_KEY" not in st.secrets:
            st.error("❌ GROQ_API_KEY not found in Streamlit secrets")
            st.info("Please add GROQ_API_KEY to your Streamlit secrets")
            st.stop()
        
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("✅ Using Groq API key from secrets")
        
        st.divider()
        st.subheader("📊 Settings & Recommendations")
        
        # Chunk Size with recommendation
        st.markdown("**Chunk Size** (Recommended: 1000)")
        st.caption("Larger chunks = more context per retrieval. Good range: 500-1500")
        chunk_size = st.slider("Chunk Size", 50, 2000, 1000, 50, key="chunk_size")
        
        # Chunk Overlap with recommendation
        st.markdown("**Chunk Overlap** (Recommended: 100)")
        st.caption("Overlap = context sharing between chunks. 10-20% of chunk size")
        chunk_overlap = st.slider("Chunk Overlap", 0, 300, 100, 50, key="chunk_overlap")
        
        # Top-K with recommendation
        st.markdown("**Top-K Retrieval** (Recommended: 4)")
        st.caption("Number of most relevant chunks to retrieve. 3-5 is optimal")
        top_k = st.slider("Top-K Retrieval", 1, 10, 4, key="top_k")
        
        # Temperature with recommendation
        st.markdown("**Temperature** (Recommended: 0.3)")
        st.caption("Lower = factual answers. 0.0-0.5 for RAG systems")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1, key="temperature")
        
        st.divider()
        st.subheader("ℹ️ System Info")
        model_info = f"**Model**: {st.session_state.model_name}\n\n**Embeddings**: TF-IDF + Bigrams\n\n**Vector DB**: FAISS\n\n**Re-ranking**: Enabled"
        st.info(model_info)
        
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "💬 Ask Questions", "📊 Metrics"])
    
    # Tab 1: Upload
    with tab1:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose PDF or TXT", type=['pdf', 'txt'])
        
        if uploaded_file:
            st.success(f"✅ {uploaded_file.name}")
            if st.button("🚀 Process Document", type="primary"):
                process_document(uploaded_file, api_key, chunk_size, chunk_overlap, top_k, temperature)
        
        if st.session_state.doc_processed:
            st.success("✅ Document processed! Go to 'Ask Questions' tab.")
    
    # Tab 2: Questions
    with tab2:
        st.header("Ask Questions")
        if not st.session_state.doc_processed:
            st.warning("⚠️ Upload and process a document first!")
        else:
            question = st.text_input("Your question:", placeholder="What is this document about?")
            if st.button("🔍 Ask", type="primary") and question:
                answer_question(question, api_key)
            
            if st.session_state.chat_history:
                st.divider()
                st.subheader("💬 Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.expander(f"Q: {chat['question'][:50]}...", expanded=(i==0)):
                        st.markdown(f"**Question:** {chat['question']}")
                        st.markdown(f'<div class="answer-box"><strong>Answer:</strong><br>{chat["answer"]}</div>', unsafe_allow_html=True)
                        if chat.get('sources'):
                            st.markdown("**📄 Sources:**")
                            for j, src in enumerate(chat['sources'][:3]):
                                st.markdown(f'<div class="chunk-box"><strong>Chunk {j+1}:</strong><br>{src[:300]}...</div>', unsafe_allow_html=True)
    
    # Tab 3: Metrics
    with tab3:
        st.header("📊 System Metrics")
        if st.session_state.doc_processed:
            col1, col2, col3 = st.columns(3)
            col1.metric("Chunk Size", f"{chunk_size} chars")
            col1.metric("Model", st.session_state.model_name.replace("llama-", "Llama "))
            col2.metric("Chunk Overlap", f"{chunk_overlap} chars")
            col2.metric("Vector DB", "FAISS")
            col3.metric("Top-K", top_k)
            col3.metric("Embeddings", "TF-IDF")
            st.divider()
            st.metric("Questions Asked", len(st.session_state.chat_history))
        else:
            st.info("Process a document to see metrics")


def process_document(file, api_key, chunk_size, chunk_overlap, top_k, temp):
    with st.spinner("🔄 Processing..."):
        try:
            progress = st.progress(0)
            status = st.empty()
            
            status.text("📄 Loading document...")
            progress.progress(20)
            processor = DocumentProcessor(chunk_size, chunk_overlap)
            chunks = processor.process_uploaded_file(file)
            
            status.text(f"✂️ Created {len(chunks)} chunks...")
            progress.progress(40)
            
            status.text("🔢 Creating embeddings...")
            progress.progress(50)
            
            vector_mgr = VectorStoreManager()
            
            status.text(f"🔢 Generating embeddings for {len(chunks)} chunks...")
            progress.progress(60)
            vector_store = vector_mgr.create_vector_store(chunks)
            
            status.text("💾 Storing vectors...")
            progress.progress(80)
            retriever = RetrievalSystem(vector_store, vector_mgr, top_k)
            
            status.text("🤖 Initializing LLM...")
            progress.progress(90)
            answer_gen = AnswerGenerator(retriever.retriever, api_key, temperature=temp)
            
            st.session_state.vector_store = vector_store
            st.session_state.retriever = retriever
            st.session_state.answer_gen = answer_gen
            st.session_state.model_name = answer_gen.model_name
            st.session_state.doc_processed = True
            st.session_state.num_chunks = len(chunks)
            
            status.text("✅ Complete!")
            progress.progress(100)
            st.success(f"✅ Processed {len(chunks)} chunks! Using model: {answer_gen.model_name}")
            st.balloons()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Tips:\n- Check if GROQ_API_KEY is valid\n- Check internet connection\n- Try uploading a smaller document")


def answer_question(question, api_key):
    with st.spinner("🤔 Thinking..."):
        try:
            start = time.time()
            result = st.session_state.answer_gen.generate_answer(question)
            elapsed = time.time() - start
            
            answer = result['result']
            sources = result.get('source_documents', [])
            
            st.markdown(f'<div class="answer-box"><strong>💡 Answer:</strong><br><br>{answer}</div>', unsafe_allow_html=True)
            
            if sources:
                st.markdown("---")
                st.subheader("📄 Retrieved Chunks")
                for i, doc in enumerate(sources, 1):
                    with st.expander(f"📝 Chunk {i}"):
                        st.markdown(f'<div class="chunk-box">{doc.page_content}</div>', unsafe_allow_html=True)
            
            st.caption(f"⏱️ {elapsed:.2f}s | Model: {st.session_state.model_name}")
            
            st.session_state.chat_history.append({
                'question': question,
                'answer': answer,
                'sources': [d.page_content for d in sources]
            })
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info(f"💡 Model: {st.session_state.model_name}")


if __name__ == "__main__":
    main()
