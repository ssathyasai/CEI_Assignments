"""
Handles Stage 7 of the pipeline: Answer Generation.
Takes the user's question plus the chunks retrieved from the
vector store and asks a language model to produce an answer that
is grounded in that retrieved context.
"""

from transformers import pipeline


class AnswerGenerator:
    def __init__(self, model_name):
        self.generation_pipeline = pipeline("text2text-generation", model=model_name)

    def build_prompt(self, user_question, retrieved_chunks):
        context_text = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)

        prompt = (
            "Answer the question using only the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {user_question}\n"
            "Answer:"
        )
        return prompt

    def generate_answer(self, user_question, retrieved_chunks):
        prompt = self.build_prompt(user_question, retrieved_chunks)
        model_output = self.generation_pipeline(
            prompt, max_new_tokens=200, do_sample=False
        )
        return model_output[0]["generated_text"].strip()
