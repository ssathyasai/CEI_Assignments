"""Answer Generation Module - Uses Groq LLM"""
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


class AnswerGenerator:
    """Generates answers using Groq LLM"""
    
    def __init__(self, retriever, api_key: str, model_name: str = "llama-3.1-70b-versatile", temperature: float = 0.3):
        # Initialize Groq LLM
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=api_key,
            max_tokens=1024
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful AI assistant. Answer the question based ONLY on the context provided.
If the answer is not in the context, say "I cannot answer based on the provided documents."

Context:
{context}

Question: {input}

Answer:"""
        )
        
        # Create chains
        doc_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.qa_chain = create_retrieval_chain(retriever, doc_chain)
    
    def generate_answer(self, query: str) -> Dict[str, Any]:
        """Generate answer for query"""
        result = self.qa_chain.invoke({"input": query})
        # Convert to old format for compatibility
        return {
            "result": result["answer"],
            "source_documents": result.get("context", [])
        }
