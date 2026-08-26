# Is My Stuff Safe? — A Household Product Recall Checker Agent

**LLM Zoomcamp Capstone Project**

## Problem description

Most people have no idea whether the products sitting in their home right
now — a stroller, a car seat, a space heater, a pressure cooker, an air
fryer — have ever been subject to an official safety recall. Checking
manually means digging through the CPSC website by hand, knowing the
exact product name/model, and reading dense legal language.

**"Is My Stuff Safe?"** is a conversational agent you describe a product
to in plain language — *"I have a Cosco Rock 'N Roller baby stroller
from around 2005, is it safe?"* or *"Is my COSORI air fryer model
CP158-AF safe to use?"* — and it tells you if it matches a known recall,
what the hazard was, and what to do about it, grounded in real CPSC data
(including specific model numbers).

## Dataset

**CPSC SaferProducts.gov REST API** (free, no key required — see
`data/fetch_data.py`), fetched across **10 everyday categories**:
`stroller, car seat, crib, baby swing, high chair, space heater,
pressure cooker, air fryer, coffee maker, battery charger` — **428
recall documents** total.

> **Scope limitation**: only these 10 categories were fetched (not
> CPSC's full ~6,700-record database). Questions outside these
> categories correctly return "no confirmed match" rather than guessing.

> Per course rules, this project uses an independent dataset, not the
> DataTalks.Club FAQ documents.

## Project structure

product-recall-checker/
├── README.md

├── requirements.txt / .env.example / Dockerfile / docker-compose.yml

├── data/fetch_data.py + README.md  # pulls raw recall data — see data/README.

├── ingest.py                        # raw data -> documents -> indices

├── vector_search.py                  # semantic (embedding) search index

├── rag_helper.py                      # hybrid search + prompt + LLM

├── tools.py                            # the 3 agent tools

├── agent.py                             # function-calling agent loop

├── app.py                                # Streamlit chat interface

├── pages/1_Monitoring.py                  # monitoring dashboard

├── monitoring/db.py + README.md            # SQLite logging — see monitoring/README.md

└── evaluation/ (scripts + results) + README.md  # see evaluation/README.md


## Agent tools

| Tool | Purpose |
|---|---|
| `search_recalls(query, category=None)` | Free-text + category-filtered search |
| `check_my_product(description)` | Checks a described product against known recalls |
| `compare_brand_safety(brand_a, brand_b)` | Compares recall history between two brands |

## Retrieval flow

`rag_helper.RAGBase` supports keyword, vector, hybrid, and
hybrid+reranking retrieval, plus optional query rewriting. See
**[`evaluation/README.md`](evaluation/README.md)** for which one wins
and why.

## How to run

1. Copy `.env.example` to `.env`, add your `OPENAI_API_KEY`.
2. `pip install -r requirements.txt`
3. `python data/fetch_data.py`
4. `python ingest.py` (sanity check)
5. `streamlit run app.py`

Or: `docker compose up --build`

### Example questions

- *I have a Cosco Rock 'N Roller baby stroller from around 2005, is it safe?*
- *Is my COSORI air fryer model CP158-AF safe to use?*
- *Compare recall history between Graco and Chicco for car seats.*

**Note**: `check_my_product` works best with a **brand** included —
"my air fryer" alone can't be matched to one specific recall among several.

<img width="656" height="402" alt="image" src="https://github.com/user-attachments/assets/877e2deb-7c34-45ca-b38a-30ebb6a9b3bd" />

<img width="649" height="350" alt="image" src="https://github.com/user-attachments/assets/92439f2b-8e54-4a63-b874-6fea316141f3" />




<img width="653" height="415" alt="image" src="https://github.com/user-attachments/assets/1f61238f-b4cc-435e-a483-c1085530c159" />


## Tech stack

CPSC API · `minsearch` (keyword) + OpenAI embeddings (vector) + RRF
(hybrid) · OpenAI `gpt-4o-mini` (agent + function calling) · SQLite +
Streamlit (monitoring) · Docker

## Further reading

- **[`evaluation/README.md`](evaluation/README.md)** — retrieval
  comparison, a real bug found and fixed in `check_my_product`, and LLM
  prompt comparison, all with real numbers.
- **[`monitoring/README.md`](monitoring/README.md)** — what's logged
  and how the dashboard works.
- **[`data/README.md`](data/README.md)** — how ingestion works, and why
  it's a manual/on-demand pipeline rather than automatic.
