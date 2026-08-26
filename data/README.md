# Data ingestion

## How it works

1. `fetch_data.py` pulls recall records from the CPSC SaferProducts.gov
   API for 10 fixed categories (see root README) and overwrites
   `recalls_raw.json` with a full fresh snapshot.
2. `../ingest.py` reads `recalls_raw.json`, converts each record into a
   document (text + metadata, including product model numbers), and
   builds the keyword (`minsearch`) and vector (embeddings) indices used
   by the app.

## This is a manual, on-demand pipeline — not automatic

- Running the app (`streamlit run app.py`) does **not** re-fetch from
  CPSC. It only rebuilds the search indices from whatever is currently
  sitting in `recalls_raw.json` on disk.
- If CPSC publishes a new recall, the app has no way of knowing until
  someone manually re-runs:
```bash
  python data/fetch_data.py
  python ingest.py   # optional sanity check; app.py rebuilds indices itself on start
```
- Restarting Streamlit re-reads the same local file — it does not
  trigger a new fetch.

## Why this is fine for this project

Batch/on-demand ingestion (rerun manually when you want fresh data) is a
standard, expected pattern for this kind of project — it's exactly what
`ingest.py` is designed to do. Continuous/scheduled ingestion (e.g. a
daily cron job that re-fetches and rebuilds automatically) would be a
reasonable future enhancement, not a requirement.