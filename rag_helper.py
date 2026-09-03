"""
rag_helper.py

The reusable RAG logic: search -> build prompt -> call the LLM.

Beyond plain keyword search, this module implements the course's
"best practices" techniques:

- Hybrid search: combines minsearch (keyword) and VectorIndex (semantic)
  results via Reciprocal Rank Fusion.
- Query rewriting: the LLM rewrites a casual user question into a more
  search-friendly query before retrieval.
- Document re-ranking: the LLM re-orders the top retrieved candidates by
  actual relevance to the original question.

Each technique can be toggled independently, which is also what lets
evaluation/evaluate.py compare configurations and justify which ones are
actually worth keeping (see README's "Best practices evaluation" section).
"""

import os

from openai import OpenAI


INSTRUCTIONS = """
You are a helpful, careful household product safety assistant.
Answer the QUESTION using only the facts in the CONTEXT below.
The CONTEXT is drawn from official CPSC (U.S. Consumer Product Safety
Commission) recall records.

Rules:
- If the CONTEXT contains a matching recall, clearly state the product,
  the specific model number(s) if available, the hazard, the date, and
  the recommended remedy.
- If the CONTEXT does NOT contain a clear match, say so plainly. Do not
  guess or imply a product is safe or unsafe without evidence.
- Keep the tone calm and practical, like a knowledgeable friend, not
  alarmist.
""".strip()

# A second prompt variant, evaluated against INSTRUCTIONS in
# evaluation/llm_eval.py so we can pick the better one with evidence
# rather than guessing (satisfies the "LLM evaluation" rubric criterion).
INSTRUCTIONS_V2_STRUCTURED = """
You are a household product safety assistant backed by official CPSC
recall data. Answer using ONLY the CONTEXT provided.

Respond in this structure:
1. Verdict: one line — "Recall found" or "No matching recall found"
2. Details (if found): product, manufacturer, date, hazard, remedy
3. What to do next: a short, practical recommendation

Never speculate beyond the CONTEXT. If nothing matches, say so and
suggest the user double-check the exact brand/model on cpsc.gov.
""".strip()

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()

QUERY_REWRITE_PROMPT = """
Rewrite the following user question into a short, keyword-rich search
query optimized for retrieving matching product recall records. Keep
brand names and product types. Remove filler words. Return ONLY the
rewritten query, nothing else.

User question: {question}
""".strip()

RERANK_PROMPT = """
Given the user QUESTION and a list of candidate recall records, return a
JSON list of the candidate indices (0-based) ordered from MOST to LEAST
relevant to the question. Only include indices that are at least
somewhat relevant; drop clearly irrelevant ones.

QUESTION: {question}

CANDIDATES:
{candidates}

Return ONLY a JSON list of integers, e.g. [2, 0, 3]
""".strip()


def format_context(results: list[dict]) -> str:
    entries = []
    for r in results:
        entries.append(
            f"- Title: {r['title']}\n"
            f"  Manufacturer: {r.get('manufacturer', 'unknown')}\n"
            f"  Model(s): {r.get('product_model', 'not specified')}\n"
            f"  Date: {r.get('recall_date', 'unknown')}\n"
            f"  Hazard: {r.get('hazard', 'not specified')}\n"
            f"  Remedy: {r.get('remedy', 'not specified')}\n"
            f"  Source: {r.get('url', 'n/a')}"
        )
    return "\n\n".join(entries) if entries else "(no matching recalls found)"


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Combine multiple ranked result lists (e.g. keyword search + vector
    search) into one fused ranking, using each document's rank position
    rather than raw scores (which aren't directly comparable across
    different retrieval methods).
    """
    scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_lookup[doc_id] = doc

    ranked_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [doc_lookup[i] for i in ranked_ids]


class RAGBase:
    """
    RAG helper supporting plain keyword search, vector search, hybrid
    (rank-fused) search, optional query rewriting, and optional LLM
    re-ranking of retrieved candidates.

    - `index`: anything with `.search(query, num_results, filter_dict)`
      (a minsearch Index, as built by ingest.build_index).
    - `vector_index` (optional): a VectorIndex (ingest.build_vector_index).
      When provided, `retrieval_mode="hybrid"` becomes available.
    """

    def __init__(
        self,
        index,
        vector_index=None,
        llm_client=None,
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
        model: str | None = None,
        retrieval_mode: str = "keyword",  # "keyword" | "vector" | "hybrid"
        use_query_rewriting: bool = False,
        use_reranking: bool = False,
    ):
        self.index = index
        self.vector_index = vector_index
        self.llm_client = llm_client or OpenAI()
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.retrieval_mode = retrieval_mode
        self.use_query_rewriting = use_query_rewriting
        self.use_reranking = use_reranking

    # ------------------------------------------------------------------
    # Query rewriting
    # ------------------------------------------------------------------
    def rewrite_query(self, question: str) -> str:
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": QUERY_REWRITE_PROMPT.format(question=question)}]
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten or question

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(self, query: str, num_results: int = 5, filter_dict: dict | None = None) -> list[dict]:
        search_query = self.rewrite_query(query) if self.use_query_rewriting else query
        filter_dict = filter_dict or {}

        if self.retrieval_mode == "vector":
            if self.vector_index is None:
                raise ValueError("retrieval_mode='vector' requires a vector_index")
            results = self.vector_index.search(search_query, num_results=num_results, filter_dict=filter_dict)

        elif self.retrieval_mode == "hybrid":
            if self.vector_index is None:
                raise ValueError("retrieval_mode='hybrid' requires a vector_index")
            keyword_results = self.index.search(search_query, filter_dict=filter_dict, num_results=num_results * 2)
            vector_results = self.vector_index.search(search_query, num_results=num_results * 2, filter_dict=filter_dict)
            fused = reciprocal_rank_fusion([keyword_results, vector_results])
            results = fused[:num_results]

        else:  # "keyword" (default)
            results = self.index.search(search_query, filter_dict=filter_dict, num_results=num_results)

        if self.use_reranking and results:
            results = self._rerank(query, results)

        return results

    def _rerank(self, question: str, results: list[dict]) -> list[dict]:
        candidates_str = "\n".join(
            f"[{i}] {r['title']} - {r.get('hazard', '')}" for i, r in enumerate(results)
        )
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": RERANK_PROMPT.format(question=question, candidates=candidates_str)}],
            )
            content = response.choices[0].message.content.strip()
            content = content.removeprefix("```json").removesuffix("```").strip()
            import json

            order = json.loads(content)
            reranked = [results[i] for i in order if 0 <= i < len(results)]
            return reranked or results
        except Exception:
            # if reranking fails for any reason, fall back to original order
            return results

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def build_prompt(self, question: str, results: list[dict]) -> str:
        context = format_context(results)
        return self.prompt_template.format(question=question, context=context)

    def llm(self, prompt: str, instructions: str | None = None) -> str:
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": instructions or self.instructions},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def answer(self, question: str, num_results: int = 5, filter_dict: dict | None = None) -> dict:
        """Full RAG pipeline: retrieve -> prompt -> generate."""
        results = self.search(question, num_results=num_results, filter_dict=filter_dict)
        prompt = self.build_prompt(question, results)
        answer_text = self.llm(prompt)
        return {
            "answer": answer_text,
            "retrieved": results,
        }
