"""
ingest.py

Loads the raw CPSC recall records (data/recalls_raw.json), converts each
one into a natural-language document plus structured metadata, and builds
a minsearch index that can be reused across the app, the agent tools, and
the evaluation scripts.

This mirrors the course's ingest.py pattern: one file responsible for
"raw data -> searchable index", imported everywhere else.
"""

import json
from pathlib import Path

from minsearch import Index

RAW_DATA_PATH = Path(__file__).parent / "data" / "recalls_raw.json"


def _first_or_none(value):
    """CPSC fields are sometimes lists (e.g. multiple manufacturers)."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _join_names(value, field="Name"):
    """Flatten CPSC's nested manufacturer/hazard/injury structures."""
    if not value:
        return ""
    if isinstance(value, list):
        names = []
        for item in value:
            if isinstance(item, dict):
                names.append(str(item.get(field, item)))
            else:
                names.append(str(item))
        return ", ".join(n for n in names if n)
    return str(value)


def record_to_document(record: dict) -> dict:
    """Convert one raw CPSC recall record into a retrievable document."""
    recall_id = record.get("RecallID")
    title = record.get("Title") or record.get("RecallTitle") or ""
    date = record.get("RecallDate", "")
    manufacturers = _join_names(record.get("Manufacturers"), field="Name")
    hazards = _join_names(record.get("Hazards"), field="Name")
    remedies = _join_names(record.get("Remedies"), field="Name")
    description = record.get("Description", "") or ""
    category = record.get("_query_category", "")
    url = record.get("URL", "")

    products = record.get("Products", [])
    product_names = _join_names(products, field="Name")
    product_models = ""
    if isinstance(products, list):
        models = [p.get("Model", "") for p in products if isinstance(p, dict)]
        product_models = ", ".join(m for m in models if m)

    injuries = record.get("Injuries", [])
    injury_count = len(injuries) if isinstance(injuries, list) else 0

    text = (
        f"{title}. "
        f"Product: {product_names or category}. "
        f"Manufacturer: {manufacturers or 'unknown'}. "
        f"Recalled on {date}. "
        f"Hazard: {hazards or 'not specified'}. "
        f"Description: {description}. "
        f"Remedy: {remedies or 'not specified'}. "
        f"Reported injuries: {injury_count}."
    )

    return {
        "id": str(recall_id),
        "text": text,
        "title": title,
        "category": category,
        "manufacturer": manufacturers,
        "product_name": product_names,
        "product_model": product_models,
        "hazard": hazards,
        "remedy": remedies,
        "recall_date": date,
        "injury_count": injury_count,
        "url": url,
    }


def load_documents(raw_path: Path = RAW_DATA_PATH) -> list[dict]:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Run `python data/fetch_data.py` first."
        )
    raw_records = json.loads(raw_path.read_text())
    return [record_to_document(r) for r in raw_records]



def build_index(documents: list[dict] | None = None) -> Index:
    """Build (and fit) a minsearch Index (keyword/text search) over the recall documents."""
    if documents is None:
        documents = load_documents()

    index = Index(
        text_fields=["text", "title", "product_name", "hazard", "remedy"],
        keyword_fields=["category", "manufacturer"],
    )
    index.fit(documents)
    return index


def build_vector_index(documents: list[dict] | None = None):
    """Build (and fit) a VectorIndex (embedding/semantic search) over the recall documents."""
    from vector_search import VectorIndex

    if documents is None:
        documents = load_documents()

    vector_index = VectorIndex()
    vector_index.fit(documents)
    return vector_index


def build_vector_index(documents: list[dict] | None = None):
    """Build (and fit) a VectorIndex (embedding/semantic search) over the recall documents."""
    from vector_search import VectorIndex

    if documents is None:
        documents = load_documents()

    vector_index = VectorIndex()
    vector_index.fit(documents)
    return vector_index


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    idx = build_index(docs)
    print("Index built successfully.")

    # quick smoke test
    results = idx.search("stroller wheel falling off", num_results=3)
    for r in results:
        print("-", r["title"][:80])
