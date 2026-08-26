"""
fetch_data.py

Pulls recall records from the CPSC SaferProducts.gov REST API for a fixed
set of everyday product categories, and saves the combined raw results to
data/recalls_raw.json.

The API is free and requires no authentication:
https://www.saferproducts.gov/RestWebServices/Recall

Usage:
    python data/fetch_data.py
"""

import json
import time
from pathlib import Path

import requests

BASE_URL = "https://www.saferproducts.gov/RestWebServices/Recall"

# Everyday categories chosen to make the demo relatable: baby/kids gear,
# kitchen appliances, and home electronics/heating.
CATEGORIES = [
    "stroller",
    "car seat",
    "crib",
    "baby swing",
    "high chair",
    "space heater",
    "pressure cooker",
    "air fryer",
    "coffee maker",
    "battery charger",
]

OUTPUT_PATH = Path(__file__).parent / "recalls_raw.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def fetch_category(product_name: str) -> list[dict]:
    """Fetch all recall records whose ProductName matches the given term."""
    resp = requests.get(
        BASE_URL,
        params={"format": "json", "ProductName": product_name},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    for record in data:
        record["_query_category"] = product_name
    return data


def main():
    all_records = []
    seen_ids = set()

    for category in CATEGORIES:
        print(f"Fetching recalls for: {category!r} ...")
        try:
            records = fetch_category(category)
        except requests.RequestException as e:
            print(f"  Failed to fetch {category!r}: {e}")
            continue

        new_count = 0
        for r in records:
            rid = r.get("RecallID")
            if rid is not None and rid in seen_ids:
                continue
            if rid is not None:
                seen_ids.add(rid)
            all_records.append(r)
            new_count += 1

        print(f"  Got {len(records)} records ({new_count} new)")
        time.sleep(0.5)  # be polite to the API

    OUTPUT_PATH.write_text(json.dumps(all_records, indent=2))
    print(f"\nSaved {len(all_records)} unique recall records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
