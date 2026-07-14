"""Answer Generation Module - Uses Groq LLM"""
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class AnswerGenerator:
    """Generates answers using Groq LLM"""
    
    # List of models to try in order
    AVAILABLE_MODELS = [
        "llama-3.3-70b-versatile",    # Current primary model
        "llama-3.1-8b-instant",        # Lightweight fallback
    ]
    
    def __init__(self, retriever, api_key: str, model_name: str = None, temperature: float = 0.3):
        self.retriever = retriever
        self.temperature = temperature
        self.groq_api_key = api_key
        
        # Use provided model or try defaults
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.AVAILABLE_MODELS[0]
        
        # Initialize LLM - will be created lazily on first use
        self.llm = None
        self._init_llm()
        
        # Create prompt template - improved for better answers
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful AI assistant. Answer the question based on the provided context.
If the exact answer is not in the context, try to provide a helpful response using related information.

Context:
{context}

Question: {question}

Answer: Provide a comprehensive answer that addresses the question."""
        )
        
        # Create simple RAG chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _init_llm(self):
        """Initialize Groq LLM with model"""
        self.llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            groq_api_key=self.groq_api_key,
            max_tokens=1024,
            timeout=60
        )
    
    def _try_fallback_model(self):
        """Try next available model on failure"""
        current_idx = self.AVAILABLE_MODELS.index(self.model_name)
        if current_idx < len(self.AVAILABLE_MODELS) - 1:
            self.model_name = self.AVAILABLE_MODELS[current_idx + 1]
            self._init_llm()
            return True
        return False
    
    def generate_answer(self, query: str) -> Dict[str, Any]:
        """Generate answer for query"""
        try:
            # Get answer
            answer = self.rag_chain.invoke(query)
        except Exception as e:
            error_msg = str(e).lower()
            # Check if model is decommissioned
            if "decommissioned" in error_msg or "model_not_found" in error_msg.lower():
                if self._try_fallback_model():
                    # Recreate chain with new model
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    self.rag_chain = (
                        {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                        | self.prompt
                        | self.llm
                        | StrOutputParser()
                    )
                    # Try again with fallback
                    answer = self.rag_chain.invoke(query)
                else:
                    raise Exception(f"All models unavailable. Last error: {str(e)}")
            else:
                raise
        
        # Get source documents using invoke instead of get_relevant_documents
        docs = self.retriever.invoke(query)
        
        # Return in expected format
        return {
            "result": answer,
            "source_documents": docs
        }
