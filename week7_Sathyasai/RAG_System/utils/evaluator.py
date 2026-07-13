"""
Evaluation Module
Runs validation checks on the RAG pipeline, logs retrieval performance, and generates system metrics reports.
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple
from utils.retriever import HybridRetriever
from utils.answer_generator import AnswerGenerator

class RAGEvaluator:
    """
    Validates RAG pipeline retrieval and generation performance using dynamic sample queries.
    """
    def __init__(self, retriever: HybridRetriever, answer_generator: AnswerGenerator):
        self.retriever = retriever
        self.answer_generator = answer_generator

    def run_evaluation(
        self, 
        sample_questions: List[str], 
        search_strategy: str = "hybrid",
        alpha: float = 0.7,
        rerank_strategy: str = "none",
        cohere_api_key: str = None,
        top_k_retrieve: int = 15,
        top_n_final: int = 4,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Runs evaluation on a list of questions.
        Saves logs to logs/validation_logs.json and generates logs/system_metrics_report.md
        """
        os.makedirs("logs", exist_ok=True)
        results = []
        
        for q in sample_questions:
            t0 = time.time()
            
            # Retrieve chunks
            try:
                retrieved = self.retriever.retrieve(
                    query=q,
                    search_strategy=search_strategy,
                    alpha=alpha,
                    top_k_retrieve=top_k_retrieve,
                    rerank_strategy=rerank_strategy,
                    cohere_api_key=cohere_api_key,
                    top_n_final=top_n_final
                )
                retrieval_time = time.time() - t0
            except Exception as e:
                retrieved = []
                retrieval_time = 0.0
                
            t1 = time.time()
            
            # Generate answer
            chunks_list = [item[0] for item in retrieved]
            scores_list = [item[1] for item in retrieved]
            
            try:
                answer = self.answer_generator.generate_answer(q, chunks_list)
                generation_time = time.time() - t1
            except Exception as e:
                answer = f"Failed to generate answer: {str(e)}"
                generation_time = 0.0
                
            total_time = time.time() - t0
            
            # Log results
            res_entry = {
                "question": q,
                "retrieval_latency_sec": round(retrieval_time, 3),
                "generation_latency_sec": round(generation_time, 3),
                "total_latency_sec": round(total_time, 3),
                "retrieved_chunks": [
                    {
                        "chunk_id": chunk["metadata"].get("chunk_id", i),
                        "text": chunk["text"][:200] + "...",
                        "source": chunk["metadata"].get("source", "Unknown"),
                        "score": round(score, 4)
                    }
                    for i, (chunk, score) in enumerate(retrieved)
                ],
                "answer": answer
            }
            results.append(res_entry)

        # Write detailed JSON validation log
        json_log_path = "logs/validation_logs.json"
        with open(json_log_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Compile System Metrics Report
        avg_retrieval_latency = sum(r["retrieval_latency_sec"] for r in results) / len(results) if results else 0.0
        avg_generation_latency = sum(r["generation_latency_sec"] for r in results) / len(results) if results else 0.0
        avg_total_latency = sum(r["total_latency_sec"] for r in results) / len(results) if results else 0.0
        
        report_content = f"""# System Metrics and Validation Report

This report documents the configuration details, performance evaluation, and validation logs of the RAG (Retrieval-Augmented Generation) QA pipeline.

## 📋 System Setup Configuration
- **Document Chunking Profile**:
  - Max Chunk Size: `{chunk_size}` characters
  - Chunk Overlap: `{chunk_overlap}` characters
  - Total Indexed Chunks: `{len(self.retriever.vector_manager.chunks)}`
- **Embedding Model Details**:
  - Model Name: `{self.retriever.emb_gen.model_name}`
  - Model Provider: `{self.retriever.emb_gen.provider}`
  - Vector Dimensionality: `{self.retriever.emb_gen.dimension}`
- **Vector Database Configuration**:
  - Database Tool: `{self.retriever.vector_manager.store_type.upper()}`
  - Distance Metric: `Cosine Similarity`
- **Retrieval & Reranking Settings**:
  - Search Strategy: `{search_strategy}` (alpha weight: `{alpha}`)
  - Candidate Retries (top_k_retrieve): `{top_k_retrieve}`
  - Re-ranking Strategy: `{rerank_strategy}`
  - Final Chunks Sent to LLM (top_n_final): `{top_n_final}`
- **Language Model Configuration**:
  - Model Name: `{self.answer_generator.model_name}`
  - Model Provider: `{self.answer_generator.provider}`
  - Temperature: `{self.answer_generator.temperature}`

---

## 📈 Aggregated Performance Metrics
- **Total Test Queries Run**: `{len(results)}`
- **Average Retrieval Latency**: `{avg_retrieval_latency:.3f} seconds`
- **Average Generation Latency**: `{avg_generation_latency:.3f} seconds`
- **Average Total End-to-End Latency**: `{avg_total_latency:.3f} seconds`

---

## 🧪 Detailed Validation Logs
Below is the evaluation log for the tested dynamic sample queries:

"""
        for i, res in enumerate(results):
            report_content += f"""### Query {i+1}: "{res['question']}"
- **Latency**: Retrieval: `{res['retrieval_latency_sec']:.3f}s` | Generation: `{res['generation_latency_sec']:.3f}s` | Total: `{res['total_latency_sec']:.3f}s`
- **Answer**:
  > {res['answer']}
  
- **Retrieved Chunks Used**:
"""
            if not res['retrieved_chunks']:
                report_content += "  *None retrieved.*\n\n"
            for chunk in res['retrieved_chunks']:
                report_content += f"  - Chunk {chunk['chunk_id']} (Source: `{chunk['source']}`) | Score: `{chunk['score']}`\n    *\"{chunk['text']}\"*\n"
            report_content += "\n---\n"

        # Write markdown report
        md_report_path = "logs/system_metrics_report.md"
        with open(md_report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return results, report_content
