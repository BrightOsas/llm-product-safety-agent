"""
vector_search.py

A minimal embedding-based (semantic/vector) search index, used alongside
minsearch's keyword search in ingest.py. Having two independent retrieval
approaches lets us:

1. Evaluate them separately (evaluation/evaluate.py) and pick the best,
   or combine them.
2. Build a hybrid search (rag_helper.py) that blends keyword + vector
   scores, per the course's "best practices" section.

Kept dependency-light: uses OpenAI's embeddings API + numpy cosine
similarity rather than a full vector database, since our corpus is only
a few hundred documents (a real vector DB like Qdrant would be overkill
here, though the same interface could be swapped in later).
"""

from __future__ import annotations

import os

import numpy as np
from openai import OpenAI

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


class VectorIndex:
    def __init__(self, client: OpenAI | None = None, model: str = EMBEDDING_MODEL):
        self.client = client or OpenAI()
        self.model = model
        self.documents: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        # batch in chunks of 100 to stay well under API limits
        vectors = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend([d.embedding for d in resp.data])
        return np.array(vectors, dtype=np.float32)

    def fit(self, documents: list[dict]):
        self.documents = documents
        texts = [d["text"] for d in documents]
        self.embeddings = self._embed(texts)
        # normalize once so search is a plain dot product (cosine similarity)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self.embeddings = self.embeddings / norms
        return self

    def search(self, query: str, num_results: int = 5, filter_dict: dict | None = None) -> list[dict]:
        if self.embeddings is None:
            raise RuntimeError("VectorIndex.fit() must be called before search()")

        query_vec = self._embed([query])[0]
        query_vec = query_vec / (np.linalg.norm(query_vec) or 1e-9)

        scores = self.embeddings @ query_vec  # cosine similarity per doc

        candidates = list(enumerate(scores))
        if filter_dict:
            candidates = [
                (i, s) for i, s in candidates
                if all(self.documents[i].get(k) == v for k, v in filter_dict.items())
            ]

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:num_results]
        return [self.documents[i] for i, _ in top]
