"""
Answer Generation Module
Handles LLM-based answer generation using Google Gemini or Cohere, 
with strict context grounding to prevent hallucinations.
"""

from typing import List, Dict, Any

class AnswerGenerator:
    """
    Generates grounded answers based on retrieved context.
    """
    def __init__(self, provider: str = "gemini", api_key: str = None, model_name: str = None, temperature: float = 0.2):
        self.provider = provider.lower()
        self.api_key = api_key
        self.temperature = temperature
        
        if self.provider == "gemini":
            import google.generativeai as genai
            if not api_key:
                raise ValueError("Google API Key is required for Gemini LLM")
            genai.configure(api_key=api_key)
            self.model_name = model_name or "gemini-1.5-flash"
            self.model = genai.GenerativeModel(self.model_name)
        elif self.provider == "cohere":
            import cohere
            if not api_key:
                raise ValueError("Cohere API Key is required for Cohere LLM")
            self.co = cohere.Client(api_key)
            self.model_name = model_name or "command-r"
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Creates a grounded prompt with retrieved chunks and gets a response from the LLM.
        """
        if not retrieved_chunks:
            return "No relevant context was retrieved to answer this question."

        # Format context blocks
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk["metadata"].get("source", f"Chunk {i}")
            context_blocks.append(f"[Document Chunk {i+1} | Source: {source}]\n{chunk['text']}")
            
        context_str = "\n\n".join(context_blocks)
        
        # Build prompt template
        prompt = f"""You are a helpful AI assistant that answers questions based ONLY on the provided context.
Use only the information from the context below to answer the question. If the answer cannot be found in the context, say "I cannot answer this based on the provided documents."
Do not make up facts or extrapolate outside the provided documents. Ensure your answer is detailed and directly grounded in the provided snippets.

Context:
{context_str}

Question: {query}

Answer:"""

        # Call API
        try:
            if self.provider == "gemini":
                import google.generativeai as genai
                response = self.model.generate_content(
                    prompt,
                    generation_config={"temperature": self.temperature}
                )
                return response.text
                
            elif self.provider == "cohere":
                response = self.co.chat(
                    message=prompt,
                    model=self.model_name,
                    temperature=self.temperature
                )
                return response.text
        except Exception as e:
            return f"Error during generation: {str(e)}"
            
        return "Internal error during generation."
