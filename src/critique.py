"""LLM-based answer critique for Self-RAG."""

from __future__ import annotations

import json

from openai import OpenAI

CRITIQUE_PROMPT = """
질문: {query}

검색된 문서:
{context}

생성된 답변:
{answer}

위 답변을 다음 기준으로 평가하세요.

1. relevant: 답변이 질문과 관련 있는가?
2. supported: 답변이 검색된 문서에 근거하는가?
3. need_retrieval: 더 나은 답변을 위해 추가 검색이 필요한가?

JSON 형식으로만 출력:
{{"relevant": true/false, "supported": true/false, "need_retrieval": true/false}}
""".strip()


class AnswerCritique:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def critique(self, query: str, answer: str, chunks: list[dict]) -> dict:
        context = "\n\n".join(f"[{c['rank']}] {c['text']}" for c in chunks)
        prompt = CRITIQUE_PROMPT.format(query=query, context=context, answer=answer)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(resp.choices[0].message.content)
        return {
            "relevant": bool(result.get("relevant", True)),
            "supported": bool(result.get("supported", True)),
            "need_retrieval": bool(result.get("need_retrieval", False)),
        }
