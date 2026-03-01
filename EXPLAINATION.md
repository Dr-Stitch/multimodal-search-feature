## Full Code Explanation

### Overview

This notebook implements a **multimodal product search system** that lets you find products using images, text, or both. It uses **SigLIP** (a CLIP variant from Google) to encode both images and text into the same 768-dimensional vector space, then stores them in an in-memory **Qdrant** vector database for similarity search.

---

## Architecture

┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ products.csv │────▶│ DataIngestionAgent│────▶│ Product List │
└─────────────────┘ └──────────────────┘ └────────┬────────┘
│
▼
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Product Image │────▶│ CLIPEncoderAgent │────▶│ 768-dim Vector │
│ Product Text │────▶│ │────▶│ (averaged) │
└─────────────────┘ └──────────────────┘ └────────┬────────┘
│
▼
┌──────────────────┐ ┌─────────────────┐
│ VectorStoreAgent │◀───│ Qdrant Upsert │
└────────┬─────────┘ └─────────────────┘
│
Query ──────────────────▶│
▼
┌──────────────────┐ ┌─────────────────┐
│ProductSearchAgent│────▶│ ResponseAgent │
└──────────────────┘ └─────────────────┘

---

## Cell-by-Cell Breakdown

### Cell 2 — Imports

| Import                       | Purpose                                |
| ---------------------------- | -------------------------------------- |
| `os`                         | File path checks                       |
| `time`                       | (unused placeholder)                   |
| `warnings`                   | Emit non-fatal warnings                |
| `numpy`                      | Vector math (averaging, L2 norm)       |
| `pandas`                     | Read CSV product catalog               |
| `PIL.Image`                  | Load and validate images               |
| `dotenv.load_dotenv`         | Load `.env` file (future API keys)     |
| `transformers.AutoProcessor` | Preprocesses images/text for the model |
| `transformers.AutoModel`     | Loads SigLIP model                     |
| `torch`                      | PyTorch runtime for inference          |
| `qdrant_client.*`            | Vector database client and models      |

---

### Cell 3 — Environment

load_dotenv()
Loads environment variables from `.env`. Currently a placeholder since SigLIP runs locally.

---

### Cell 4 — Configuration

| Constant                  | Value                              | Purpose                   |
| ------------------------- | ---------------------------------- | ------------------------- |
| `CLIP_MODEL_NAME`         | `"google/siglip-base-patch16-224"` | HuggingFace model ID      |
| `VECTOR_DIM`              | `768`                              | Embedding dimension       |
| `COLLECTION_NAME`         | `"product_catalog"`                | Qdrant collection name    |
| `TOP_K`                   | `5`                                | Default number of results |
| `DATA_CSV_PATH`           | `"data/products.csv"`              | Product data file         |
| `SUPPORTED_IMAGE_FORMATS` | `{".jpg", ".jpeg", ...}`           | Allowed image extensions  |

---

## Class Explanations

### `DataIngestionAgent`

**Purpose**: Loads and validates the product CSV.

| Method     | Parameters      | Returns      | Description                                                      |
| ---------- | --------------- | ------------ | ---------------------------------------------------------------- |
| `__init__` | `csv_path: str` | —            | Stores the CSV path                                              |
| `load`     | —               | `list[dict]` | Reads CSV, validates columns, fills defaults, drops invalid rows |

**Required CSV columns**: `product_id`, `image_path`, `name`, `text_description`  
**Optional columns** (auto-filled): `category`, `price`, `brand`

---

### `CLIPEncoderAgent`

**Purpose**: Encodes images and text into 768-dim vectors using SigLIP.

| Method            | Parameters                          | Returns      | Description                                                                        |
| ----------------- | ----------------------------------- | ------------ | ---------------------------------------------------------------------------------- |
| `__init__`        | `model_name: str = CLIP_MODEL_NAME` | —            | Downloads and loads the SigLIP model (~400MB, cached)                              |
| `_validate_image` | `image_path: str`                   | `PIL.Image`  | Checks file exists, format is supported, image isn't corrupt                       |
| `_l2_normalize`   | `vector: np.ndarray`                | `np.ndarray` | Normalizes vector to unit length (required for cosine similarity)                  |
| `encode_image`    | `image_path: str`                   | `np.ndarray` | Opens image → preprocesses → runs through model → returns L2-normed 768-dim vector |
| `encode_text`     | `text: str`                         | `np.ndarray` | Tokenizes text → runs through model → returns L2-normed 768-dim vector             |

**How encoding works**:

with torch.no_grad(): # Disable gradient tracking
inputs = self.processor(images=img, ...) # Convert to tensor
features = self.model.get_image_features(...) # Forward pass → 768 floats
vector = features.squeeze().numpy() # Remove batch dim, convert to numpy

---

### `VectorStoreAgent`

**Purpose**: Manages the Qdrant in-memory vector database.

| Method     | Parameters                                             | Returns             | Description                                                         |
| ---------- | ------------------------------------------------------ | ------------------- | ------------------------------------------------------------------- |
| `__init__` | `collection_name: str`, `vector_dim: int`              | —                   | Creates in-memory Qdrant client and collection with cosine distance |
| `upsert`   | `point_id: int`, `vector: np.ndarray`, `payload: dict` | —                   | Insert/update a single vector with metadata                         |
| `search`   | `query_vector: np.ndarray`, `top_k: int = 5`           | `list[ScoredPoint]` | Find the `top_k` most similar vectors                               |
| `count`    | —                                                      | `int`               | Returns total number of indexed vectors                             |

**Qdrant concepts**:

- **Collection**: A named container for vectors (like a database table)
- **Point**: A vector + metadata payload + unique ID
- **Distance.COSINE**: Similarity metric (1.0 = identical, 0.0 = orthogonal)

---

### `ProductSearchAgent`

**Purpose**: Orchestrates the full search pipeline.

| Method             | Parameters                                                                 | Returns      | Description                                            |
| ------------------ | -------------------------------------------------------------------------- | ------------ | ------------------------------------------------------ |
| `__init__`         | `clip_encoder: CLIPEncoderAgent`, `vector_store: VectorStoreAgent`         | —            | Stores references to encoder and vector store          |
| `_combine_vectors` | `*vectors: np.ndarray`                                                     | `np.ndarray` | Averages multiple vectors and L2-normalizes the result |
| `search`           | `query_image_path: str = None`, `query_text: str = None`, `top_k: int = 5` | `list[dict]` | Main search method (see below)                         |

**`search()` flow**:

1. Validate at least one input is provided
2. Encode image (if provided) → 768-dim vector
3. Encode text (if provided) → 768-dim vector
4. If both: average + re-normalize
5. Query Qdrant for `top_k` nearest neighbors
6. Format results with rank, score, and metadata

---

### `ResponseAgent`

**Purpose**: Formats and displays search results.

| Method           | Parameters                                      | Returns | Description                                       |
| ---------------- | ----------------------------------------------- | ------- | ------------------------------------------------- |
| `format_results` | `results: list[dict]`, `query_text: str = None` | `str`   | Builds a formatted string with all result details |
| `display`        | `results: list[dict]`, `query_text: str = None` | —       | Prints formatted results to stdout                |

---

## Ingest Phase (Cell 12)

For each product in the CSV:

1. Load image → encode_image() → 768-dim image_vector
2. Load text → encode_text() → 768-dim text_vector
3. combined = average([image_vector, text_vector])
4. combined = L2_normalize(combined)
5. upsert(id=idx, vector=combined, payload={product metadata})

**Fallback logic**: If image encoding fails, use text-only; if text fails, use image-only; if both fail, skip the product.

---

## Query Phase (Cells 14-15)

**Three query modes**:

| Mode         | Code                                               | What happens                          |
| ------------ | -------------------------------------------------- | ------------------------------------- |
| Image + Text | `search(query_image_path="...", query_text="...")` | Both encoded, averaged, then searched |
| Image only   | `search(query_image_path="...")`                   | Image vector used directly            |
| Text only    | `search(query_text="...")`                         | Text vector used directly             |

---

## Verification Phase (Cells 17-19)

| Test                 | Purpose                                                                               |
| -------------------- | ------------------------------------------------------------------------------------- |
| **Count check**      | Verifies indexed count matches expected (products - failures)                         |
| **Point inspection** | Retrieves point #0 and verifies vector dimension = 768                                |
| **Self-consistency** | Searches for product[0] using its own image+text — it should rank #1 with score ≈ 1.0 |

---

## Key Concepts

### Why average image + text vectors?

CLIP models map images and text to the **same vector space**. Averaging them creates a "combined" representation that captures both visual and semantic information, improving search quality for multimodal queries.

### Why L2-normalize?

Cosine similarity is computed as dot product when vectors are unit-length. Pre-normalizing ensures consistent similarity scores between 0 and 1.

### Why in-memory Qdrant?

Fast prototyping. For persistence, change `QdrantClient(":memory:")` to `QdrantClient(path="./qdrant_data")`.
