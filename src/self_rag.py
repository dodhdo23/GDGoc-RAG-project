"""Self-RAG inference pipeline."""

from __future__ import annotations

from src.critique import AnswerCritique
from src.generator import LLMAnswerGenerator
from src.retriever import Retriever


class SelfRAG:
    def __init__(
        self,
        retriever: Retriever,
        generator: LLMAnswerGenerator,
        critique: AnswerCritique,
        max_iter: int = 2,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.critique = critique
        self.max_iter = max_iter

    def run(self, query: str, top_k: int = 3) -> dict:
        current_top_k = top_k
        retrieval_count = 0
        regen_count = 0
        previous_answer = None

        chunks = self.retriever.retrieve(query, top_k=current_top_k)
        retrieval_count += 1

        answer = self.generator.generate(query, chunks)["answer"]

        for _ in range(self.max_iter):
            result = self.critique.critique(query, answer, chunks)
            if not result["need_retrieval"]:
                break
            current_top_k *= 2
            previous_answer = answer
            chunks = self.retriever.retrieve(query, top_k=current_top_k)
            retrieval_count += 1
            answer = self.generator.generate(query, chunks, previous_answer)["answer"]
            regen_count += 1

        return {
            "query": query,
            "answer": answer,
            "retrieval_count": retrieval_count,
            "regeneration_count": regen_count,
            "chunks": chunks,
        }
