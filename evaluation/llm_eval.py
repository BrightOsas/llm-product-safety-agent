"""
evaluation/llm_eval.py

Evaluates the final LLM output (not just retrieval): compares two prompt
variants using an LLM-as-judge, reports which one produces better answers.

Usage:
    python evaluation/llm_eval.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


sys.path.append(str(Path(__file__).resolve().parent.parent))
from ingest import build_index, build_vector_index, load_documents  # noqa: E402
from rag_helper import INSTRUCTIONS, INSTRUCTIONS_V2_STRUCTURED, RAGBase  # noqa: E402

load_dotenv()

RESULTS_PATH = Path(__file__).parent / "llm_eval_results.json"

EVAL_QUESTIONS = [
    "I have a Fisher-Price rock-n-play from around 2018, is it safe?",
    "Have there been any recalls for space heaters?",
    "Is my Instant Pot pressure cooker safe to still use?",
    "What's the hazard associated with recalled baby swings?",
    "I'm buying a used stroller secondhand, how do I check if it was recalled?",
    "Are Graco car seats generally reliable, safety-wise?",
]

JUDGE_PROMPT = """
You are judging two AI answers to the same question about product
recalls. Both answers were generated from the same retrieved CONTEXT.

QUESTION: {question}

CONTEXT PROVIDED TO BOTH:
{context}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Score each answer from 1-5 on:
- Groundedness (only uses facts from CONTEXT, no speculation)
- Clarity (easy to understand, well-organized)
- Actionability (tells the user what to do next, if relevant)

Return ONLY JSON in this exact format:
{{"answer_a": {{"groundedness": _, "clarity": _, "actionability": _}},
  "answer_b": {{"groundedness": _, "clarity": _, "actionability": _}},
  "winner": "A" or "B" or "tie"}}
""".strip()


def judge(client: OpenAI, model: str, question: str, context: str, answer_a: str, answer_b: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question, context=context, answer_a=answer_a, answer_b=answer_b
            ),
        }],
    )
    content = response.choices[0].message.content.strip()
    content = content.removeprefix("```json").removesuffix("```").strip()
    return json.loads(content)


def main():
    documents = load_documents()
    index = build_index(documents)
    vector_index = build_vector_index(documents)

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    rag_v1 = RAGBase(index=index, vector_index=vector_index, retrieval_mode="hybrid", use_reranking=True, instructions=INSTRUCTIONS)
    rag_v2 = RAGBase(index=index, vector_index=vector_index, retrieval_mode="hybrid", use_reranking=True, instructions=INSTRUCTIONS_V2_STRUCTURED)

    judgments = []
    wins = {"A": 0, "B": 0, "tie": 0}

    for question in EVAL_QUESTIONS:
        print(f"Evaluating: {question!r}")
        results = rag_v1.search(question)
        context = rag_v1.build_prompt(question, results)

        answer_a = rag_v1.llm(rag_v1.build_prompt(question, results), instructions=INSTRUCTIONS)
        answer_b = rag_v2.llm(rag_v2.build_prompt(question, results), instructions=INSTRUCTIONS_V2_STRUCTURED)

        verdict = judge(client, model, question, context, answer_a, answer_b)
        wins[verdict["winner"]] += 1

        judgments.append({
            "question": question,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "verdict": verdict,
        })

    overall_winner = max(wins, key=wins.get)

    output = {
        "prompt_a": "INSTRUCTIONS (default, conversational)",
        "prompt_b": "INSTRUCTIONS_V2_STRUCTURED (structured verdict format)",
        "win_counts": wins,
        "overall_winner": overall_winner,
        "judgments": judgments,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))

    print(f"\nWin counts: {wins}")
    print(f"Overall winner: Prompt {overall_winner}")
    print(f"Full results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()