"""
evaluation/ground_truth.py

Generates a retrieval evaluation set: for each recall document, ask an
LLM to produce a few natural questions a real person might ask that
should retrieve that specific document.

Usage:
    python evaluation/ground_truth.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


sys.path.append(str(Path(__file__).resolve().parent.parent))
from ingest import load_documents  # noqa: E402

load_dotenv()

OUTPUT_PATH = Path(__file__).parent / "ground_truth.json"

PROMPT = """
You are helping build a test set for a product-recall search system.

Given the recall record below, write exactly 3 short, natural questions a
regular person (not a lawyer) might type to find this exact recall. Vary
the phrasing — one specific (mentions the brand/product), one about the
hazard, one more casual/vague.

Recall record:
{text}

Return ONLY a JSON list of 3 strings, nothing else.
""".strip()



    
def generate_questions(client: OpenAI, doc_text: str, model: str) -> list[str]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(text=doc_text)}],
    )
    content = response.choices[0].message.content.strip()
    content = content.removeprefix("```json").removesuffix("```").strip()
    return json.loads(content)

def main(sample_size: int | None = 60):
    documents = load_documents()
    if sample_size:
        documents = documents[:sample_size]  # keep eval generation cheap

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    ground_truth = []
    for doc in documents:
        try:
            questions = generate_questions(client, doc["text"], model)
        except Exception as e:
            print(f"Skipping doc {doc['id']} due to error: {e}")
            continue

        for q in questions:
            ground_truth.append({"question": q, "doc_id": doc["id"]})

    OUTPUT_PATH.write_text(json.dumps(ground_truth, indent=2))
    print(f"Saved {len(ground_truth)} question/doc_id pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()