# Multimodal Product Search Agent - Implementation Plan

## Overview

Build a new Jupyter Notebook agent (`product-search-agent/`) that lets users search a product catalog by providing a query image, text description, or both. The system uses `google/siglip-base-patch16-224` (CLIP-based, local, free) to encode both images and text directly into the same 768-dim vector space — no paid APIs needed at all. Vectors are stored and searched via Qdrant in-memory vector DB.

## Tech Stack

- **CLIP Encoder:** `google/siglip-base-patch16-224` via HuggingFace `transformers` (local, free)
    - Encodes both images AND text into the same 768-dim vector space
    - Replaces both the vision model and the embedding model entirely
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
transformers  torch  torchvision  qdrant-client  python-dotenv  pandas  Pillow
```

> `torch` + `transformers` load SigLIP locally. Everything runs fully offline — no API keys needed.

## .env Variables

None required for the current implementation. `.env` file and `python-dotenv` are kept in the setup so credentials can be added later (e.g. for future Gemini query expansion) without restructuring.

## Notebook Structure (Cell-by-Cell)

### Section 1 — Setup

- **Cell 0 (Markdown):** Title + workflow overview
- **Cell 1 (Code):** `# install transformers torch torchvision qdrant-client python-dotenv pandas Pillow` + `!pip install ...`
- **Cell 2 (Code):** All imports
- **Cell 3 (Code):** `load_dotenv()` — no required keys for now; placeholder for future credentials
- **Cell 4 (Code):** Config constants (`CLIP_MODEL_NAME="google/siglip-base-patch16-224"`, `VECTOR_DIM=768`, `COLLECTION_NAME`, `TOP_K=5`, `DATA_CSV_PATH`)

### Section 2 — Class Definitions (one per cell)

**Cell 5 — `DataIngestionAgent`**

- Loads and validates `products.csv`
- Checks required columns, fills optional columns with defaults
- Drops rows with missing required fields (with warning)
- Returns `list[dict]`

**Cell 6 — `CLIPEncoderAgent`**

- Loads `google/siglip-base-patch16-224` processor and model on construction (cached after first load)
- `encode_image(image_path)` → validates file exists + Pillow-readable → returns 768-dim numpy vector
- `encode_text(text)` → raises `ValueError` on empty string → returns 768-dim numpy vector
- Both methods L2-normalize the output vector before returning (required for correct cosine similarity in Qdrant)
- Supports `.jpg`, `.png`, `.webp`, `.gif`

**Cell 7 — `VectorStoreAgent`**

- Initializes `QdrantClient(":memory:")` on construction
- Creates collection with `COSINE` distance, configurable vector dim
- `upsert(point_id, vector, payload)` — inserts a single product
- `search(query_vector, top_k)` — returns `list[ScoredPoint]`
- `count()` — returns number of indexed points

**Cell 8 — `ProductSearchAgent`**

- Orchestrates the full query pipeline:
    1. If image provided → `CLIPEncoderAgent.encode_image` → image vector
    2. If text provided → `CLIPEncoderAgent.encode_text` → text vector
    3. If both provided → average the two vectors, re-normalize → combined query vector
    4. Search via `VectorStoreAgent.search`
- Supports image-only, text-only, or image+text queries
- Returns `list[dict]` with rank, score, and all product metadata

**Cell 9 — `ResponseAgent`**

- `format_results(results, query_text)` → formatted string
- `display(results, query_text)` → prints to stdout
- Follows `ResponseAgent` pattern from weather-agent exactly

### Section 3 — Initialization

**Cell 10:** Instantiate all agents, print confirmation

### Section 4 — Ingest Phase

**Cell 11 (Markdown):** Ingest phase description — entirely free and local, no API calls

**Cell 12 (Code):** Ingest loop

- For each product:
    1. `CLIPEncoderAgent.encode_image(product["image_path"])` → image vector
    2. `CLIPEncoderAgent.encode_text(product["text_description"])` → text vector
    3. Average both vectors, re-normalize → combined product vector
    4. `VectorStoreAgent.upsert(point_id, combined_vector, payload)`
- Per-product `try/except`: image encode failure falls back to text-only vector; text encode failure falls back to image-only vector; both fail → product skipped, added to `failed_products`
- No `time.sleep` needed (all local, no rate limits)
- Prints progress `(idx/total)` per product
- Reports `failed_products` list at end

### Section 5 — Query Phase

**Cell 13 (Markdown):** Query phase description

**Cell 14:** Example query with image + text

```python
QUERY_IMAGE_PATH = "data/images/query_sample.jpg"
QUERY_TEXT = "lightweight hiking backpack in a neutral color"
results = search_agent.search(QUERY_IMAGE_PATH, QUERY_TEXT)
response_agent.display(results, QUERY_TEXT)
```

**Cell 15:** Text-only query example (`query_image_path=None`)

### Section 6 — Verification

**Cell 16 (Markdown):** Verification section header

**Cell 17:** Assert `vector_store.count()` == products loaded - failed count

**Cell 18:** Retrieve point `id=0`, print payload keys + vector length

**Cell 19:** Self-consistency test — search product[0] with its own image + description, verify it ranks #1

## Architecture Data Flow

```
INGEST TIME (fully local, no API calls)
─────────────────────────────────────────────────────
[products.csv + images/]
        │
DataIngestionAgent              validates + yields product dicts
        │
        ├─ CLIPEncoderAgent.encode_image(image_path)   → 768-dim image vector
        ├─ CLIPEncoderAgent.encode_text(text_desc)     → 768-dim text vector
        └─ average + L2-normalize                      → 768-dim combined vector
                │
VectorStoreAgent.upsert         stores vector + payload in Qdrant


QUERY TIME
─────────────────────────────────────────────────────
[query image]  ──► CLIPEncoderAgent.encode_image  ──► image vector
                                                          │
[query text]   ──► CLIPEncoderAgent.encode_text   ──► text vector
                                                          │
                        average + L2-normalize    ──► query vector
                                                          │
                   VectorStoreAgent.search         cosine top-K
                                                          │
ResponseAgent.display           formatted ranked output
```

## Error Handling

| Scenario                      | Where                | Handling                                                              |
| ----------------------------- | -------------------- | --------------------------------------------------------------------- |
| CSV not found                 | `DataIngestionAgent` | `FileNotFoundError`                                                   |
| CSV missing required columns  | `DataIngestionAgent` | `ValueError` listing missing columns                                  |
| Image file not found          | `CLIPEncoderAgent`   | `FileNotFoundError`; ingest loop catches, falls back to text-only     |
| Corrupt image                 | `CLIPEncoderAgent`   | Pillow `verify()` raises; ingest loop falls back to text-only         |
| Unsupported image format      | `CLIPEncoderAgent`   | `ValueError` listing supported formats (.jpg, .png, .webp, .gif)      |
| Image encode failure (ingest) | Ingest loop          | Falls back to text-only vector                                        |
| Text encode failure (ingest)  | Ingest loop          | Falls back to image-only vector; both fail → skip + `failed_products` |
| Empty text to encode          | `CLIPEncoderAgent`   | `ValueError` before model call                                        |
| Empty search query            | `ProductSearchAgent` | `ValueError` before any encoding                                      |

## Key Notes

- **Qdrant is in-memory only** — re-run ingest after every kernel restart. For persistence, change `":memory:"` to `QdrantClient(path="./qdrant_data")` (noted as comment in Cell 10)
- **SigLIP model is downloaded once** and cached by HuggingFace in `~/.cache/huggingface/` (~400 MB). Subsequent runs load from cache instantly
- **No rate limiting needed** — all encoding is local
- **Upgrading to a heavier CLIP model**: swap `CLIP_MODEL_NAME` to `laion/CLIP-ViT-H-14-laion2B` and set `VECTOR_DIM=1024` in Cell 4; requires GPU for reasonable speed
- **Vector averaging for image+text queries** — averaging the image and text vectors then re-normalizing is the standard CLIP fusion approach; gives balanced image+text relevance

## Future Enhancements

- **Vague query enrichment** — add a `QueryExpansionAgent` that calls Gemini (free tier) when the user's text query is short or vague. It would expand e.g. `"camping stuff"` into a detailed description before SigLIP encodes it. Requires adding `google-generativeai` package and `GEMINI_API_KEY` to `.env`. The `ProductSearchAgent.search()` signature does not need to change — expansion would be an internal step before `CLIPEncoderAgent.encode_text`.

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
