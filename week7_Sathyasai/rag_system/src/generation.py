import groq
from typing import List, Dict, Any

class GenerationService:
    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768"):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        if not self.api_key:
            raise ValueError("Groq API key is required")
        
        try:
            self.client = groq.Groq(api_key=self.api_key)
        except Exception as e:
            raise Exception(f"Error initializing Groq client: {str(e)}")
    
    def generate_answer(self, query: str, context: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        try:
            prompt = self._create_prompt(query, context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Always ground your answers in the context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")
    
    def _create_prompt(self, query: str, context: str) -> str:
        return f"""
Based on the following context, please answer this question.

Context:
{context}

Question: {query}

Answer:"""