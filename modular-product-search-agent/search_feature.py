"""
search_feature.py
-----------------
This script provides the online feature for your website: loading the trained Qdrant vector DB and running search queries (image, text, or both).
Integrate this as a backend API or service for your web app.
"""

import os
import io
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient

# --- Config ---
CLIP_MODEL_NAME = "google/siglip-base-patch16-224"
VECTOR_DIM = 768
COLLECTION_NAME = "product_catalog"
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# --- CLIP Encoder (same as training) ---
class CLIPEncoderAgent:
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _validate_image(self, image_path: str) -> Image.Image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format '{ext}'. Supported: {SUPPORTED_IMAGE_FORMATS}")
        img = Image.open(image_path).convert("RGB")
        return img

    def _l2_normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def encode_image(self, image_path: str) -> np.ndarray:
        img = self._validate_image(image_path)
        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt")
            image_features = self.model.get_image_features(**inputs)
            vector = image_features.squeeze().numpy()
        return self._l2_normalize(vector)

    def encode_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Cannot encode empty text")
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            text_features = self.model.get_text_features(**inputs)
            vector = text_features.squeeze().numpy()
        return self._l2_normalize(vector)

# --- Search Feature ---
class ProductSearchFeature:
    def __init__(self, collection_name: str = COLLECTION_NAME, vector_dim: int = VECTOR_DIM):
        self.client = QdrantClient(path="../qdrant_data")  # Persistent storage
        self.collection_name = collection_name
        self.encoder = CLIPEncoderAgent()

    def _combine_vectors(self, *vectors: np.ndarray) -> np.ndarray:
        combined = np.mean(vectors, axis=0)
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined

    def search(self, query_image_path: str = None, query_text: str = None, top_k: int = 5):
        vectors = []
        if query_image_path is not None:
            image_vector = self.encoder.encode_image(query_image_path)
            vectors.append(image_vector)
        if query_text is not None and query_text.strip():
            text_vector = self.encoder.encode_text(query_text)
            vectors.append(text_vector)
        if not vectors:
            raise ValueError("Must provide at least one of: query_image_path, query_text")
        if len(vectors) == 1:
            query_vector = vectors[0]
        else:
            query_vector = self._combine_vectors(*vectors)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
        )
        return results

# Example usage (for testing only)
if __name__ == "__main__":
    searcher = ProductSearchFeature()
    # Example: search by text
    results = searcher.search(query_text="lightweight hiking backpack")
    for r in results:
        print(r)
