"""
evaluation/evaluate.py

1. Retrieval evaluation (Hit Rate, MRR): compares FOUR retrieval
   configurations against evaluation/ground_truth.json.
2. Tool-level evaluation of check_my_product: precision/recall against
   evaluation/product_test_cases.json.

Usage:
    python evaluation/evaluate.py
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
from ingest import build_index, build_vector_index, load_documents  # noqa: E402
from rag_helper import RAGBase  # noqa: E402
from tools import RecallTools  # noqa: E402

load_dotenv()

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"
TEST_CASES_PATH = Path(__file__).parent / "product_test_cases.json"
RESULTS_PATH = Path(__file__).parent / "retrieval_eval_results.json"


def hit_rate(relevance_lists: list[list[bool]]) -> float:
    return sum(any(rel) for rel in relevance_lists) / len(relevance_lists)


def mrr(relevance_lists: list[list[bool]]) -> float:
    total = 0.0
    for rel in relevance_lists:
        for rank, is_relevant in enumerate(rel, start=1):
            if is_relevant:
                total += 1 / rank
                break
    return total / len(relevance_lists)


def evaluate_one_config(rag: RAGBase, ground_truth: list[dict], num_results: int = 5) -> dict:
    relevance_lists = []
    for item in ground_truth:
        results = rag.search(item["question"], num_results=num_results)
        relevance = [r["id"] == item["doc_id"] for r in results]
        relevance_lists.append(relevance)

    return {
        "num_questions": len(ground_truth),
        "hit_rate": round(hit_rate(relevance_lists), 4),
        "mrr": round(mrr(relevance_lists), 4),
    }


def evaluate_retrieval_approaches(index, vector_index) -> dict:
    if not GROUND_TRUTH_PATH.exists():
        print(f"No ground truth found at {GROUND_TRUTH_PATH}. Run evaluation/ground_truth.py first.")
        return {}

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())

    configs = {
        "keyword_only": RAGBase(index=index, retrieval_mode="keyword"),
        "vector_only": RAGBase(index=index, vector_index=vector_index, retrieval_mode="vector"),
        "hybrid": RAGBase(index=index, vector_index=vector_index, retrieval_mode="hybrid"),
        "hybrid_plus_reranking": RAGBase(
            index=index, vector_index=vector_index, retrieval_mode="hybrid", use_reranking=True
        ),
    }

    results = {}
    for name, rag in configs.items():
        print(f"Evaluating retrieval config: {name} ...")
        results[name] = evaluate_one_config(rag, ground_truth)

    best_config = max(results, key=lambda name: results[name]["mrr"])
    results["_best_config"] = best_config

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nBest config by MRR: {best_config}")
    print(f"Results saved to {RESULTS_PATH}")
    return results


def evaluate_check_my_product(tools: RecallTools) -> dict:
    if not TEST_CASES_PATH.exists():
        print(f"No test cases found at {TEST_CASES_PATH}.")
        return {}

    data = json.loads(TEST_CASES_PATH.read_text())
    cases = data.get("cases", [])

    tp = fp = tn = fn = 0
    for case in cases:
        result = tools.check_my_product(case["description"])
        predicted = result["match_found"]
        expected = case["expected_match"]

        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and not expected:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    return {
        "num_cases": len(cases),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
    }


def main():
    documents = load_documents()
    index = build_index(documents)
    vector_index = build_vector_index(documents)

    print("=== Retrieval evaluation (comparing approaches) ===")
    retrieval_results = evaluate_retrieval_approaches(index, vector_index)
    print(json.dumps(retrieval_results, indent=2))

    print("\n=== check_my_product evaluation ===")
    rag = RAGBase(index=index, vector_index=vector_index, retrieval_mode="hybrid")
    tools = RecallTools(rag=rag, documents=documents)
    tool_results = evaluate_check_my_product(tools)
    print(json.dumps(tool_results, indent=2))


if __name__ == "__main__":
    main()