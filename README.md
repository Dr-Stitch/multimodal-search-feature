# Multimodal Product Search Agent - Implementation Plan

## Overview

Build a new Jupyter Notebook agent (`product-search-agent/`) that lets users search a product catalog by providing a query image + text description. The system uses GPT-4o vision to describe images and `text-embedding-3-small` to embed combined text, stored and searched via Qdrant in-memory vector DB.

## Tech Stack

- **Embedding:** OpenAI `text-embedding-3-small` (1536-dim vectors)
- **Vision:** OpenAI `gpt-4o` (image → rich text description)
- **Vector DB:** Qdrant in-memory (`QdrantClient(":memory:")`), no Docker required
- **Delivery:** Jupyter Notebook, consistent with existing agent patterns

## Directory Structure

```
AI-Agents/
└── product-search-agent/
    ├── product-search-agent.ipynb
    ├── .env                   ← OPENAI_API_KEY
    ├── README.md
    └── data/
        ├── products.csv
        └── images/
            ├── SKU001.jpg
            └── ...
```

## Product Dataset Format (products.csv)

| Column             | Required | Notes                                       |
| ------------------ | -------- | ------------------------------------------- |
| `product_id`       | Yes      | Unique identifier                           |
| `image_path`       | Yes      | Relative path e.g. `data/images/SKU001.jpg` |
| `name`             | Yes      | Display name                                |
| `text_description` | Yes      | Human-written description                   |
| `category`         | No       | Defaults to `""`                            |
| `price`            | No       | Defaults to `0.0`                           |
| `brand`            | No       | Defaults to `""`                            |

## Required Packages

```
openai  qdrant-client  python-dotenv  pandas  Pillow
```

## .env Variables

```
OPENAI_API_KEY=sk-...
```

## Notebook Structure (Cell-by-Cell)

### Section 1 — Setup

- **Cell 0 (Markdown):** Title + workflow overview
- **Cell 1 (Code):** `# install openai qdrant-client python-dotenv pandas Pillow` + `!pip install ...`
- **Cell 2 (Code):** All imports
- **Cell 3 (Code):** `load_dotenv()`, load `OPENAI_API_KEY`, raise `ValueError` if missing
- **Cell 4 (Code):** Config constants (`EMBEDDING_MODEL`, `VECTOR_DIM=1536`, `VISION_MODEL`, `COLLECTION_NAME`, `TOP_K=5`, `DATA_CSV_PATH`, `RATE_LIMIT_DELAY=0.5`)

### Section 2 — Class Definitions (one per cell)

**Cell 5 — `DataIngestionAgent`**

- Loads and validates `products.csv`
- Checks required columns, fills optional columns with defaults
- Drops rows with missing required fields (with warning)
- Returns `list[dict]`

**Cell 6 — `VisionDescriptionAgent`**

- Accepts image file path → base64-encodes it → calls GPT-4o vision API
- Validates image exists and is readable with Pillow before sending
- System prompt instructs GPT-4o to describe colors, materials, shape, style, use case
- Uses `"detail": "low"` (cheaper, suitable for product thumbnails)
- Returns descriptive text string

**Cell 7 — `EmbeddingAgent`**

- Calls `text-embedding-3-small` API on any text string
- `combine_descriptions(vision_desc, text_desc)` merges both with `"Visual: ... | Description: ..."` format
- Raises `ValueError` on empty input before any API call

**Cell 8 — `VectorStoreAgent`**

- Initializes `QdrantClient(":memory:")` on construction
- Creates collection with `COSINE` distance, configurable vector dim
- `upsert(point_id, vector, payload)` — inserts a single product
- `search(query_vector, top_k)` — returns `list[ScoredPoint]`
- `count()` — returns number of indexed points

**Cell 9 — `ProductSearchAgent`**

- Orchestrates the full query pipeline:
    1. Describe query image with `VisionDescriptionAgent` (if image provided)
    2. Combine with query text via `EmbeddingAgent.combine_descriptions`
    3. Embed with `EmbeddingAgent.embed`
    4. Search via `VectorStoreAgent.search`
- Supports image-only, text-only, or image+text queries
- Returns `list[dict]` with rank, score, and all product metadata

**Cell 10 — `ResponseAgent`**

- `format_results(results, query_text)` → formatted string
- `display(results, query_text)` → prints to stdout
- Follows `ResponseAgent` pattern from weather-agent exactly

### Section 3 — Initialization

**Cell 11:** Instantiate all agents, print confirmation

### Section 4 — Ingest Phase

**Cell 12 (Markdown):** Ingest phase description + cost warning (one GPT-4o + one embedding call per product)

**Cell 13 (Code):** Ingest loop

- For each product: vision describe → combine → embed → upsert
- Per-product `try/except`: vision failure falls back to text-only; embedding failure skips product
- `time.sleep(RATE_LIMIT_DELAY)` between calls
- Prints progress `(idx/total)` per product
- Reports `failed_products` list at end

### Section 5 — Query Phase

**Cell 14 (Markdown):** Query phase description

**Cell 15:** Example query with image + text

```python
QUERY_IMAGE_PATH = "data/images/query_sample.jpg"
QUERY_TEXT = "lightweight hiking backpack in a neutral color"
results = search_agent.search(QUERY_IMAGE_PATH, QUERY_TEXT)
response_agent.display(results, QUERY_TEXT)
```

**Cell 16:** Text-only query example (`query_image_path=None`)

### Section 6 — Verification

**Cell 17 (Markdown):** Verification section header

**Cell 18:** Assert `vector_store.count()` == products loaded - failed count

**Cell 19:** Retrieve point `id=0`, print payload keys + vector length

**Cell 20:** Self-consistency test — search product[0] with its own image + description, verify it ranks #1

## Architecture Data Flow

```
[products.csv + images/]
        │
DataIngestionAgent          validates + yields product dicts
        │
VisionDescriptionAgent      GPT-4o: image → rich text
        │
EmbeddingAgent.combine      merges vision text + human text
        │
EmbeddingAgent.embed        → 1536-dim vector
        │
VectorStoreAgent.upsert     stores vector + payload in Qdrant

─── QUERY TIME ──────────────────────────────────────
[query image + text]
        │
ProductSearchAgent.search
  ├─ VisionDescriptionAgent.describe  (query image → text)
  ├─ EmbeddingAgent.combine_descriptions
  ├─ EmbeddingAgent.embed             (→ query vector)
  └─ VectorStoreAgent.search          (cosine top-K)
        │
ResponseAgent.display       formatted ranked output
```

## Error Handling

| Scenario                     | Where                    | Handling                                                         |
| ---------------------------- | ------------------------ | ---------------------------------------------------------------- |
| Missing `.env` / API key     | Cell 3                   | `ValueError` with setup instructions                             |
| CSV not found                | `DataIngestionAgent`     | `FileNotFoundError`                                              |
| CSV missing required columns | `DataIngestionAgent`     | `ValueError` listing missing columns                             |
| Image file not found         | `VisionDescriptionAgent` | `FileNotFoundError`; ingest loop catches, skips with warning     |
| Corrupt image                | `VisionDescriptionAgent` | Pillow `verify()` raises; caught in ingest loop                  |
| Unsupported image format     | `VisionDescriptionAgent` | `ValueError` listing supported formats (.jpg, .png, .webp, .gif) |
| GPT-4o API failure           | Ingest loop              | Falls back to text-only embedding (vision_desc = "")             |
| Embedding API failure        | Ingest loop              | Product skipped, added to `failed_products`                      |
| Empty text to embed          | `EmbeddingAgent`         | `ValueError` before API call                                     |
| Empty search query           | `ProductSearchAgent`     | `ValueError` before any API calls                                |

## Key Notes

- **Qdrant is in-memory only** — re-run ingest after every kernel restart. For persistence, change `":memory:"` to `QdrantClient(path="./qdrant_data")` (noted as comment in Cell 11)
- **Switching to `text-embedding-3-large`**: change both `EMBEDDING_MODEL` and `VECTOR_DIM=3072` in Cell 4 only
- **`"detail": "low"`** on GPT-4o vision — faster and cheaper; change to `"high"` if product images contain fine text or logos
- **Rate limiting** — `RATE_LIMIT_DELAY=0.5s` suits OpenAI Tier 1; reduce to `0.1s` for Tier 3+

## Critical Reference Files

- `weather-agent/weather-agent.ipynb` — multi-class pattern, env var loading, initialization cell
- `restaurent-query-agent/bucket.fud.ipynb` — install comment convention, pipeline structure
- `weather-agent/README.md` — README template to follow

## Verification Steps

1. Run all cells top-to-bottom with a small sample CSV (3-5 products + images)
2. **Cell 18:** Assert indexed count matches CSV row count
3. **Cell 19:** Confirm payload keys and vector dim are correct
4. **Cell 20:** Self-consistency test — product[0] searched by its own image should rank #1
5. **Cell 15:** Run a real cross-product query — verify results are semantically relevant
6. **Cell 16:** Run text-only query (`query_image_path=None`) — confirm it still works
