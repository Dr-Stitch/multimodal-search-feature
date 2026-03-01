# Modular Product Search Agent Manual

This manual explains the structure and usage of the modular-product-search-agent directory, which separates the training (offline) and feature (online) phases for scalable, maintainable deployment.

---

## Directory Structure

```
modular-product-search-agent/
├── train_and_ingest.py   # Offline: data ingestion, encoding, and vector DB population
├── search_feature.py     # Online: search API for website integration
```

---

## 1. train_and_ingest.py

**Purpose:**

- Handles the offline phase: loads product data, encodes images/text, and populates the Qdrant vector database.
- Run this script whenever you want to (re)build your product search index.

**Key Steps:**

1. Load product data from CSV
2. Encode images and text using CLIP (SigLIP)
3. Store vectors and metadata in Qdrant (vector DB)

**How to Use:**

- Place your product CSV and images in the appropriate locations.
- Adjust `DATA_CSV_PATH` and image paths as needed.
- Run:
    ```bash
    python train_and_ingest.py
    ```
- The script will process all products and populate the Qdrant DB for later search.

---

## 2. search_feature.py

**Purpose:**

- Provides the online feature for your website: loads the trained Qdrant vector DB and runs search queries (image, text, or both).
- Integrate this as a backend API or service for your web app.

**Key Steps:**

1. Load the trained Qdrant vector DB
2. Encode user queries (image, text, or both)
3. Search for similar products and return results

**How to Use:**

- Ensure the Qdrant DB is populated (run `train_and_ingest.py` first).
- Integrate the `ProductSearchFeature` class into your backend (Flask, FastAPI, etc.).
- Example usage (for testing):
    ```bash
    python search_feature.py
    ```
- For production, expose the `search` method as an API endpoint.

---

## 3. Customization & Extension

- Both scripts are heavily commented for readability and easy modification.
- You can switch to image URL support, add new metadata fields, or change the model as needed.
- For large-scale deployments, consider:
    - Running Qdrant as a persistent service
    - Serving the CLIP model via a dedicated inference server
    - Adding caching, batching, or async processing

---

## 4. Troubleshooting

- Ensure all dependencies are installed (see requirements in the main README or notebook).
- Check file paths and permissions for images and CSVs.
- Review script output for warnings about missing or failed data.

---

## 5. References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/)

---

For further questions or advanced integration, see the main project README or contact the project maintainer.
