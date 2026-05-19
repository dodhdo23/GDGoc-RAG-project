"""Objective evaluation helpers for the baseline RAG pipeline."""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "and",
    "or",
    "in",
    "on",
    "for",
    "with",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "도",
    "으로",
    "로",
    "와",
    "과",
    "하다",
    "있다",
    "되다",
}


def normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase text."""
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)



def tokenize(text: str) -> list[str]:
    """Tokenize English/Korean alphanumeric tokens."""
    tokens = re.findall(r"[A-Za-z0-9가-힣]+", normalize_text(text))
    return [token for token in tokens if token not in STOPWORDS]



def exact_match(prediction: str, reference: str) -> float:
    """Binary exact match after normalization."""
    return float(normalize_text(prediction) == normalize_text(reference))



def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)



def source_hit_at_k(retrieved_chunks: list[dict], gold_sources: list[str]) -> float:
    """Return 1.0 if any gold source appears in retrieved results."""
    if not gold_sources:
        return 0.0

    retrieved_sources = {item["metadata"].get("source", "") for item in retrieved_chunks}
    return float(any(source in retrieved_sources for source in gold_sources))



def grounded_token_recall(answer: str, retrieved_chunks: list[dict]) -> float:
    """
    Measure how much of the answer vocabulary is supported by retrieved context.

    This replaces subjective grounding checks with an objective lexical proxy.
    """
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0

    context_text = " ".join(item["text"] for item in retrieved_chunks)
    context_tokens = set(tokenize(context_text))
    supported = sum(1 for token in answer_tokens if token in context_tokens)
    return supported / len(answer_tokens)



def evaluate_example(
    question: str,
    predicted_answer: str,
    gold_answer: str,
    retrieved_chunks: list[dict],
    gold_sources: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate one question-answer example."""
    _ = question
    gold_sources = gold_sources or []
    return {
        "exact_match": exact_match(predicted_answer, gold_answer),
        "token_f1": token_f1(predicted_answer, gold_answer),
        "source_hit_at_k": source_hit_at_k(retrieved_chunks, gold_sources),
        "grounded_token_recall": grounded_token_recall(predicted_answer, retrieved_chunks),
    }



def average_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of metric dictionaries."""
    if not items:
        return {}

    keys = items[0].keys()
    return {key: mean(item[key] for item in items) for key in keys}



GROUNDEDNESS_PROMPT = """Question: {query}

Retrieved Context:
{context}

Generated Answer:
{answer}

위 답변이 Retrieved Context에 근거하고 있는지 판단해줘.
반드시 다음 중 하나로만 답해 (다른 말 없이):
GROUNDED
PARTIALLY_GROUNDED
UNGROUNDED""".strip()


def groundedness_score(query: str, answer: str, chunks: list[dict], api_key: str) -> str:
    """LLM으로 답변이 context에 근거하는지 판단. GROUNDED / PARTIALLY_GROUNDED / UNGROUNDED 반환."""
    from openai import OpenAI

    context = "\n\n".join(f"[{c['rank']}] {c['text']}" for c in chunks)
    prompt = GROUNDEDNESS_PROMPT.format(query=query, context=context, answer=answer)
    resp = OpenAI(api_key=api_key).chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    label = resp.choices[0].message.content.strip().upper()
    if label not in {"GROUNDED", "PARTIALLY_GROUNDED", "UNGROUNDED"}:
        return "PARTIALLY_GROUNDED"
    return label


def answer_contains_gold(predicted: str, gold: str) -> float:
    """generated answer에 gold answer가 포함되는지 (대소문자 무시)."""
    return float(normalize_text(gold) in normalize_text(predicted))


def context_contains_gold(retrieved_chunks: list[dict], gold: str) -> float:
    """retrieved context 중 하나라도 gold answer를 포함하는지."""
    gold_norm = normalize_text(gold)
    return float(any(gold_norm in normalize_text(c["text"]) for c in retrieved_chunks))


def compare_systems(
    baseline_result: dict,
    self_rag_result: dict,
    gold_answer: str,
    baseline_time: float = 0.0,
    self_rag_time: float = 0.0,
) -> dict:
    """두 시스템의 평가 지표를 나란히 반환합니다."""
    b_chunks = baseline_result.get("retrieved_chunks", [])
    s_chunks = self_rag_result.get("chunks", [])

    baseline_metrics = {
        "answer_contains_gold": answer_contains_gold(baseline_result["answer"], gold_answer),
        "context_contains_gold": context_contains_gold(b_chunks, gold_answer),
        "token_f1": token_f1(baseline_result["answer"], gold_answer),
        "grounded_token_recall": grounded_token_recall(baseline_result["answer"], b_chunks),
        "retrieval_count": 1.0,
        "regeneration_count": 0.0,
        "response_time": baseline_time,
    }
    self_rag_metrics = {
        "answer_contains_gold": answer_contains_gold(self_rag_result["answer"], gold_answer),
        "context_contains_gold": context_contains_gold(s_chunks, gold_answer),
        "token_f1": token_f1(self_rag_result["answer"], gold_answer),
        "grounded_token_recall": grounded_token_recall(self_rag_result["answer"], s_chunks),
        "retrieval_count": float(self_rag_result.get("retrieval_count", 1)),
        "regeneration_count": float(self_rag_result.get("regeneration_count", 0)),
        "response_time": self_rag_time,
    }

    return {
        "query": baseline_result["query"],
        "baseline": {"answer": baseline_result["answer"], "metrics": baseline_metrics},
        "self_rag": {"answer": self_rag_result["answer"], "metrics": self_rag_metrics},
    }


def load_eval_data(path: str) -> list[dict[str, Any]]:
    """Load evaluation data from a JSON file."""
    import json

    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Evaluation file must contain a list of examples.")
    return data
