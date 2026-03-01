# Modular Product Search Agent

This project provides a scalable, production-ready multimodal product search system using CLIP (SigLIP) and Qdrant. The codebase is modular, separating offline ingestion from online search and API serving.

---

## Directory Structure

```
modular-product-search-agent/
├── train_and_ingest.py   # Offline: data ingestion, encoding, and vector DB population
├── search_feature.py     # Online: search logic for API integration
├── search_api_fastapi.py # FastAPI server for production API
├── USER_MANUAL.md        # Detailed integration and usage guide
├── README.md             # This file
```

---

## Setup & Environment

1. **Clone the repository and install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2. **Configure environment variables:**
    - Create a `.env` file in the project root with:
      ```env
      QDRANT_URL=https://your-qdrant-url:6333
      QDRANT_API_KEY=your_api_key_here
      CLIP_MODEL_NAME=google/siglip-base-patch16-224
      QDRANT_COLLECTION=product_catalog
      DATA_CSV_PATH=../data/products.csv
      ```
    - Never hardcode secrets in code. Always use `.env` or environment variables.

---

## 1. train_and_ingest.py

**Purpose:**
- Loads product data, encodes images/text, and populates the Qdrant vector database.
- Run this script whenever you want to (re)build your product search index.

**Usage:**
```bash
python train_and_ingest.py
```

---

## 2. search_feature.py

**Purpose:**
- Loads the trained Qdrant vector DB and runs search queries (image, text, or both).
- Integrate the `ProductSearchFeature` class into your backend (Flask, FastAPI, etc.).

**Usage:**
```bash
python search_feature.py
```
Or import and use in your API server.

---

## 3. search_api_fastapi.py

**Purpose:**
- FastAPI-based API for multimodal product search.
- Accepts text, image, or both as input and returns ranked product results.

**Usage:**
```bash
uvicorn search_api_fastapi:app --host 0.0.0.0 --port 8000 --reload
```
Then POST to `http://localhost:8000/search` with form fields `query_text`, `image`, and `top_k`.

---

## Security & Best Practices
- Store all secrets (API keys, URLs) in `.env`.
- Validate and sanitize all file uploads.
- For production, secure your API endpoints and handle file uploads safely.

---

## References
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/)

---

For advanced integration, see `USER_MANUAL.md` or contact the project maintainer.
