
"""
train_and_ingest.py
-------------------
This script handles the offline phase: loading product data, encoding images/text, and populating the Qdrant vector database.
Run this script whenever you want to (re)build your product search index.

Steps:
1. Load product data from CSV
2. Encode images and text using CLIP (SigLIP)
3. Store vectors and metadata in Qdrant (vector DB)
4. Designed for batch/offline use (not for live queries)
"""


# --- Imports ---
import os
import io
import warnings
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# --- Configuration ---
CLIP_MODEL_NAME = "google/siglip-base-patch16-224"  # Model name for CLIP/SigLIP
VECTOR_DIM = 768  # Embedding dimension
COLLECTION_NAME = "product_catalog"  # Qdrant collection name
DATA_CSV_PATH = "../data/products.csv"  # Path to product CSV (edit as needed)
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}  # Allowed image types


# --- Data Ingestion Agent ---
class DataIngestionAgent:
    """
    Loads, validates, and returns product data from a CSV file.
    Ensures required columns are present and fills missing optional columns.
    """
    REQUIRED_COLUMNS = {"product_id", "image_path", "name", "text_description"}
    OPTIONAL_DEFAULTS = {"category": "", "price": 0.0, "brand": ""}

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self) -> list[dict]:
        """
        Load and validate products from CSV.
        Returns a list of product dictionaries.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        # Check for required columns
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        # Fill missing optional columns with defaults
        for col, default in self.OPTIONAL_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
        # Drop rows with missing required fields
        before_count = len(df)
        df = df.dropna(subset=list(self.REQUIRED_COLUMNS))
        dropped = before_count - len(df)
        if dropped > 0:
            warnings.warn(f"Dropped {dropped} rows with missing required fields")
        products = df.to_dict(orient="records")
        print(f"Loaded {len(products)} products from {self.csv_path}")
        return products


# --- CLIP Encoder Agent (Local file mode) ---
class CLIPEncoderAgent:
    """
    Encodes images and text into 768-dim vectors using SigLIP (CLIP).
    Only supports local image file paths.
    """
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        print(f"Loading model '{model_name}'...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        print("Model loaded ✓")

    def _validate_image(self, image_path: str) -> Image.Image:
        """
        Validate and open an image from a local file path.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format '{ext}'. Supported: {SUPPORTED_IMAGE_FORMATS}")
        try:
            img = Image.open(image_path).convert("RGB")
            img.verify()  # Check for corruption
            img = Image.open(image_path).convert("RGB")  # Re-open after verify
            return img
        except Exception as e:
            raise ValueError(f"Cannot read image '{image_path}': {e}")

    def _l2_normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        L2-normalize a vector for cosine similarity.
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Encode an image (from local path) into a 768-dim L2-normalized vector.
        """
        img = self._validate_image(image_path)
        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt")
            image_features = self.model.get_image_features(**inputs)
            vector = image_features.squeeze().numpy()
        return self._l2_normalize(vector)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into a 768-dim L2-normalized vector.
        """
        if not text or not text.strip():
            raise ValueError("Cannot encode empty text")
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            text_features = self.model.get_text_features(**inputs)
            vector = text_features.squeeze().numpy()
        return self._l2_normalize(vector)


# --- Vector Store Agent ---
class VectorStoreAgent:
    """
    Manages Qdrant vector database for storing product vectors and metadata.
    """
    def __init__(self, collection_name: str = COLLECTION_NAME, vector_dim: int = VECTOR_DIM):
        # Use persistent storage for Qdrant (not in-memory)
        self.client = QdrantClient(path="../qdrant_data")
        self.collection_name = collection_name
        # Recreate collection (drops if exists)
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        print(f"Collection '{self.collection_name}' created (dim={vector_dim}, COSINE)")

    def upsert(self, point_id: int, vector: np.ndarray, payload: dict):
        """
        Insert or update a single product vector with metadata.
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector.tolist(), payload=payload)],
        )


# --- Main Ingestion Script ---
if __name__ == "__main__":
    # Initialize agents
    ingestion_agent = DataIngestionAgent(csv_path=DATA_CSV_PATH)
    clip_encoder = CLIPEncoderAgent(model_name=CLIP_MODEL_NAME)
    vector_store = VectorStoreAgent(collection_name=COLLECTION_NAME, vector_dim=VECTOR_DIM)

    # Load products from CSV
    products = ingestion_agent.load()
    for idx, product in enumerate(products):
        product_id = product["product_id"]
        image_path = product["image_path"]
        text_desc = product["text_description"]
        try:
            image_vector = None
            text_vector = None
            # Encode image (if possible)
            try:
                image_vector = clip_encoder.encode_image(image_path)
            except Exception as e:
                warnings.warn(f"[{product_id}] Image encode failed ({e}), using text-only")
            # Encode text (if possible)
            try:
                text_vector = clip_encoder.encode_text(text_desc)
            except Exception as e:
                warnings.warn(f"[{product_id}] Text encode failed ({e}), using image-only")
            # Combine vectors (average if both, else use available)
            if image_vector is not None and text_vector is not None:
                combined = np.mean([image_vector, text_vector], axis=0)
                norm = np.linalg.norm(combined)
                if norm > 0:
                    combined = combined / norm
            elif image_vector is not None:
                combined = image_vector
            elif text_vector is not None:
                combined = text_vector
            else:
                raise ValueError("Both image and text encoding failed")
            # Build payload (all product metadata)
            payload = {
                "product_id": product_id,
                "name": product["name"],
                "text_description": text_desc,
                "image_path": image_path,
                "category": product.get("category", ""),
                "price": float(product.get("price", 0.0)),
                "brand": product.get("brand", ""),
            }
            # Upsert to Qdrant
            vector_store.upsert(point_id=idx, vector=combined, payload=payload)
            print(f"  ({idx + 1}/{len(products)}) ✓ {product_id} — {product['name']}")
        except Exception as e:
            print(f"  ({idx + 1}/{len(products)}) ✗ {product_id} — FAILED: {e}")
    print(f"Ingest complete: {len(products)} products processed.")
