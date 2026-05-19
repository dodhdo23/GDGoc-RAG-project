"""Minimal baseline answer generation for end-to-end RAG demos."""

from __future__ import annotations

import re
from collections import Counter


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9가-힣]+", text.lower())


def _split_sentences(text: str) -> list[str]:
    text = _normalize(text)
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+|(?<=[다요함])\s+", text)
    return [part.strip() for part in parts if part.strip()]


class BaselineAnswerGenerator:
    """
    Simple extractive answer generator.

    This is intentionally lightweight and deterministic so the project can
    demonstrate an end-to-end baseline without depending on paid APIs.
    """

    def __init__(self, max_sentences: int = 3) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be > 0")
        self.max_sentences = max_sentences

    def generate(self, query: str, retrieved_chunks: list[dict]) -> dict:
        """Generate a grounded answer from retrieved chunks."""
        if not retrieved_chunks:
            return {
                "answer": "No relevant context was retrieved.",
                "used_contexts": [],
            }

        query_tokens = Counter(_tokenize(query))
        sentence_pool: list[dict] = []

        for item in retrieved_chunks:
            base_score = float(item["score"])
            source = item["metadata"].get("source", "unknown")
            for sentence in _split_sentences(item["text"]):
                sent_tokens = Counter(_tokenize(sentence))
                overlap = sum((query_tokens & sent_tokens).values())
                length_bonus = min(len(sent_tokens) / 20.0, 1.0)
                total_score = overlap * 2.0 + base_score + length_bonus
                sentence_pool.append(
                    {
                        "text": sentence,
                        "score": total_score,
                        "source": source,
                    }
                )

        if not sentence_pool:
            top_chunk = retrieved_chunks[0]
            return {
                "answer": _normalize(top_chunk["text"]),
                "used_contexts": [top_chunk],
            }

        ranked = sorted(sentence_pool, key=lambda item: item["score"], reverse=True)
        selected_sentences: list[str] = []
        used_sources: set[str] = set()

        for item in ranked:
            sentence = item["text"]
            if sentence in selected_sentences:
                continue
            selected_sentences.append(sentence)
            used_sources.add(item["source"])
            if len(selected_sentences) >= self.max_sentences:
                break

        answer = " ".join(selected_sentences)
        used_contexts = [
            item for item in retrieved_chunks if item["metadata"].get("source", "unknown") in used_sources
        ]

        return {
            "answer": answer,
            "used_contexts": used_contexts,
        }


class LLMAnswerGenerator:
    """OpenAI-based answer generator."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        query: str,
        retrieved_chunks: list[dict],
        previous_answer: str | None = None,
    ) -> dict:
        context = "\n\n".join(f"[{c['rank']}] {c['text']}" for c in retrieved_chunks)
        user_content = f"문서:\n{context}\n\n질문: {query}"
        if previous_answer:
            user_content += f"\n\n이전 답변: {previous_answer}"
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "주어진 문서를 근거로만 질문에 답하세요. 문서에 없는 내용은 말하지 마세요."},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
        )
        return {
            "answer": resp.choices[0].message.content,
            "used_contexts": retrieved_chunks,
        }
