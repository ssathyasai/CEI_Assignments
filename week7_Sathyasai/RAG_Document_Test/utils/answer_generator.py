"""Answer Generation Module - Uses Groq LLM"""
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class AnswerGenerator:
    """Generates answers using Groq LLM"""
    
    def __init__(self, retriever, api_key: str, model_name: str = "mixtral-8x7b-32768", temperature: float = 0.3):
        # Initialize Groq LLM with newer model
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=api_key,
            max_tokens=1024
        )
        
        self.retriever = retriever
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful AI assistant. Answer the question based ONLY on the context provided.
If the answer is not in the context, say "I cannot answer based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
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
    
    def generate_answer(self, query: str) -> Dict[str, Any]:
        """Generate answer for query"""
        # Get answer
        answer = self.rag_chain.invoke(query)
        
        # Get source documents
        docs = self.retriever.get_relevant_documents(query)
        
        # Return in expected format
        return {
            "result": answer,
            "source_documents": docs
        }
