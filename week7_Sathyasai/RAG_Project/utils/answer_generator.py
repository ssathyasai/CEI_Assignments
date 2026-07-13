"""Answer Generation Module - Uses Groq LLM"""
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


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
        prompt_template = """You are a helpful AI assistant. Answer the question based ONLY on the context provided.
If the answer is not in the context, say "I cannot answer based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
    
    def generate_answer(self, query: str) -> Dict[str, Any]:
        """Generate answer for query"""
        return self.qa_chain.invoke({"query": query})
