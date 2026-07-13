"""
Answer Generation Module
Handles LLM-based answer generation using Google Gemini
"""

from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate


class AnswerGenerator:
    """
    Generates answers using retrieved context and Google Gemini LLM.
    """
    
    def __init__(self, retriever, api_key: str, model_name: str = "gemini-pro", temperature: float = 0.3):
        """
        Initialize the answer generator.
        
        Args:
            retriever: LangChain retriever object
            api_key: Google Gemini API key
            model_name: Name of the Gemini model
            temperature: Generation temperature
        """
        # Initialize Google Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
            max_output_tokens=1024
        )
        
        # Create custom prompt template
        prompt_template = """You are a helpful AI assistant that answers questions based on the provided context.
Use only the information from the context below to answer the question. If the answer cannot be found in the context, say "I cannot answer this based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
    
    def generate_answer(self, query: str) -> Dict[str, Any]:
        """
        Generate answer for a given query.
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with 'result' and 'source_documents'
        """
        result = self.qa_chain.invoke({"query": query})
        return result
