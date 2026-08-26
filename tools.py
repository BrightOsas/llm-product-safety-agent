"""
tools.py

The three tools the agent can call. Each one is a plain Python function
first (so it can be tested standalone), then wrapped with an OpenAI
function-calling schema in `TOOL_SCHEMAS` for use in agent.py.
"""

from collections import Counter

from rag_helper import RAGBase


class RecallTools:
    """
    Wraps a RAGBase (search + LLM) and the raw document list so the three
    agent tools can both search semantically and aggregate over metadata.
    """

    def __init__(self, rag: RAGBase, documents: list[dict]):
        self.rag = rag
        self.documents = documents

    # ------------------------------------------------------------------
    # Tool 1: search_recalls
    # ------------------------------------------------------------------
    def search_recalls(self, query: str, category: str | None = None, num_results: int = 10, list_all: bool = False) -> dict:
        """
        Free-text search over the recall knowledge base, optionally
        filtered by category (e.g. "stroller", "space heater").

        If list_all=True and a category is given, returns ALL recalls in
        that category directly from the document list (up to 50), instead
        of a top-N semantic search — use this when the user wants a
        complete list ("what models of cribs have been recalled", "list
        all pressure cooker recalls") rather than the most relevant few.
        """
        if list_all and category:
            matches = [d for d in self.documents if d.get("category") == category]
            matches = matches[:50]  # hard cap so the answer stays usable
            return {
                "query": query,
                "category": category,
                "mode": "list_all",
                "count": len(matches),
                "results": [
                    {
                        "title": r["title"],
                        "manufacturer": r["manufacturer"],
                        "date": r["recall_date"],
                        "hazard": r["hazard"],
                        "remedy": r["remedy"],
                        "url": r["url"],
                    }
                    for r in matches
                ],
            }

        filter_dict = {"category": category} if category else {}
        results = self.rag.search(query, num_results=num_results, filter_dict=filter_dict)
        return {
            "query": query,
            "category": category,
            "mode": "top_n",
            "count": len(results),
            "results": [
                {
                    "title": r["title"],
                    "manufacturer": r["manufacturer"],
                    "date": r["recall_date"],
                    "hazard": r["hazard"],
                    "remedy": r["remedy"],
                    "url": r["url"],
                }
                for r in results
            ],
        }
    # ------------------------------------------------------------------
    # Tool 2: check_my_product
    # ------------------------------------------------------------------

    def check_my_product(self, description: str, num_results: int = 5) -> dict:
        """
        Given a loose, free-text description of a product someone owns
        (e.g. "Fisher-Price rock-n-play from around 2018"), determine
        whether it matches any known recall.
        """
        results = self.rag.search(description, num_results=num_results)

        if not results:
            return {
                "description": description,
                "match_found": False,
                "confidence": "low",
                "matches": [],
                "note": "No similar recalls found in the knowledge base.",
            }

        top_candidates = results[:3]
        candidates_str = "\n".join(
            f"[{i}] {r['title']} | Manufacturer: {r.get('manufacturer', 'unknown')}"
            for i, r in enumerate(top_candidates)
        )

        verdict_prompt = (
            f"A user described a product they own: \"{description}\"\n\n"
            f"Here are the closest candidate recall records found by search:\n{candidates_str}\n\n"
            f"Does the user's description genuinely match ANY of these candidates "
            f"(same product type AND same or very similar brand)? Respond with ONLY "
            f"JSON: {{\"match_index\": <index or -1 if none genuinely match>, "
            f"\"confidence\": \"high\"|\"medium\"|\"low\"}}"
        )

        try:
            response = self.rag.llm_client.chat.completions.create(
                model=self.rag.model,
                messages=[{"role": "user", "content": verdict_prompt}],
            )
            content = response.choices[0].message.content.strip()
            content = content.removeprefix("```json").removesuffix("```").strip()
            import json
            verdict = json.loads(content)
        except Exception:
            verdict = {"match_index": -1, "confidence": "low"}

        match_index = verdict.get("match_index", -1)

        if match_index is None or match_index < 0 or match_index >= len(top_candidates):
            return {
                "description": description,
                "match_found": False,
                "confidence": verdict.get("confidence", "low"),
                "matches": [],
                "note": "No genuine match confirmed, even though similar-looking records were found.",
            }

        matched = top_candidates[match_index]
        return {
            "description": description,
            "match_found": True,
            "confidence": verdict.get("confidence", "medium"),
            "matches": [{
                "title": matched["title"],
                "manufacturer": matched["manufacturer"],
                "date": matched["recall_date"],
                "hazard": matched["hazard"],
                "remedy": matched["remedy"],
                "url": matched["url"],
            }],
        }

    # ------------------------------------------------------------------
    # Tool 3: compare_brand_safety
    # ------------------------------------------------------------------
    def compare_brand_safety(self, brand_a: str, brand_b: str) -> dict:
        """
        Compare recall counts and hazard types between two manufacturers,
        computed directly from the ingested documents (not just search).
        """

        def brand_stats(brand: str) -> dict:
            matches = [
                d for d in self.documents
                if brand.lower() in (d.get("manufacturer") or "").lower()
            ]
            hazard_counts = Counter()
            for d in matches:
                for h in (d.get("hazard") or "").split(", "):
                    if h:
                        hazard_counts[h] += 1
            total_injuries = sum(d.get("injury_count", 0) for d in matches)
            return {
                "brand": brand,
                "recall_count": len(matches),
                "top_hazards": hazard_counts.most_common(5),
                "total_reported_injuries": total_injuries,
            }

        return {
            "brand_a": brand_stats(brand_a),
            "brand_b": brand_stats(brand_b),
        }


# ----------------------------------------------------------------------
# OpenAI function-calling schemas
# ----------------------------------------------------------------------
TOOL_SCHEMAS = [
        {
        "type": "function",
        "function": {
            "name": "search_recalls",
            "description": "Search the CPSC recall knowledge base by free text, optionally filtered by product category. Set list_all=true when the user wants a complete/full list of recalls in a category (e.g. 'what cribs have been recalled', 'list all pressure cooker recalls') rather than just the most relevant few.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, e.g. 'stroller wheel'"},
                    "category": {"type": "string", "description": "Category filter, e.g. 'stroller', 'space heater'. Required if list_all is true."},
                    "list_all": {"type": "boolean", "description": "If true, return ALL recalls in the category instead of just the top matches. Use for 'list all', 'every', 'complete list' type requests."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_my_product",
            "description": "Check whether a product the user owns (described in free text) matches any known CPSC recall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Free-text description of the product, e.g. 'Fisher-Price rock-n-play from around 2018'",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_brand_safety",
            "description": "Compare recall history (count, hazard types, injuries) between two manufacturers/brands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "brand_a": {"type": "string"},
                    "brand_b": {"type": "string"},
                },
                "required": ["brand_a", "brand_b"],
            },
        },
    },
]
