"""
evaluation/generate_test_cases.py

Automatically builds evaluation/product_test_cases.json from your real
fetched data: picks 12 real recalled products (as positive test cases)
and pairs them with 12 made-up, clearly-different everyday products (as
negative test cases).

Usage:
    python evaluation/generate_test_cases.py
"""

import json
import random
from pathlib import Path

RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "recalls_raw.json"
OUTPUT_PATH = Path(__file__).parent / "product_test_cases.json"

# Generic, everyday product descriptions unlikely to match anything in
# our recall categories — used as negative (not recalled) test cases.
NEGATIVE_EXAMPLES = [
    "a plain wooden stepstool I bought at a hardware store",
    "a basic ceramic coffee mug set from a home goods store",
    "a manual can opener, no brand name I recognize",
    "a plastic laundry basket from a discount store",
    "a set of stainless steel mixing bowls",
    "a simple corkboard for pinning notes",
    "a canvas tote bag from a farmers market",
    "a basic desk lamp with no smart features",
    "a wooden cutting board",
    "a pack of plain cotton dish towels",
    "a small houseplant pot with drainage holes",
    "a non-electric hand whisk",
]


def main():
    records = json.loads(RAW_DATA_PATH.read_text())
    sample = random.sample(records, min(12, len(records)))

    cases = []
    for r in sample:
        title = r.get("Title", "")
        manufacturers = r.get("Manufacturers", [])
        mfr_name = manufacturers[0].get("Name", "") if manufacturers else ""
        products = r.get("Products", [])
        product_name = products[0].get("Name", "") if products else ""
        description = f"{mfr_name} {product_name}".strip() or title[:60]

        cases.append({
            "description": description,
            "expected_match": True,
            "expected_doc_id": str(r.get("RecallID")),
        })

    for neg in NEGATIVE_EXAMPLES:
        cases.append({
            "description": neg,
            "expected_match": False,
            "expected_doc_id": None,
        })

    OUTPUT_PATH.write_text(json.dumps({"cases": cases}, indent=2))
    print(f"Generated {len(cases)} test cases ({len(sample)} positive, {len(NEGATIVE_EXAMPLES)} negative)")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()