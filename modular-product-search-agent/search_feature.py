
"""
search_feature.py
-----------------
This script provides the online feature for your website: loading the trained Qdrant vector DB and running search queries (image, text, or both).
Integrate this as a backend API or service for your web app.

Steps:
1. Load the trained Qdrant vector DB
2. Encode user queries (image, text, or both)
3. Search for similar products and return results
4. Designed for live/online use (website backend)
"""


# --- Imports ---
import os
import io
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient


# --- Configuration ---
CLIP_MODEL_NAME = "google/siglip-base-patch16-224"  # Model name for CLIP/SigLIP
VECTOR_DIM = 768  # Embedding dimension
COLLECTION_NAME = "product_catalog"  # Qdrant collection name
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}  # Allowed image types


# --- CLIP Encoder Agent (same as training) ---
class CLIPEncoderAgent:
    """
    Encodes images and text into 768-dim vectors using SigLIP (CLIP).
    Only supports local image file paths.
    """
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _validate_image(self, image_path: str) -> Image.Image:
        """
        Validate and open an image from a local file path.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format '{ext}'. Supported: {SUPPORTED_IMAGE_FORMATS}")
        img = Image.open(image_path).convert("RGB")
        return img

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


# --- Product Search Feature ---
class ProductSearchFeature:
    """
    Loads the trained Qdrant vector DB and runs search queries (image, text, or both).
    Intended for use as a backend API/service for your website.
    """
    def __init__(self, collection_name: str = COLLECTION_NAME, vector_dim: int = VECTOR_DIM):
        # Connect to persistent Qdrant DB
        self.client = QdrantClient(path="../qdrant_data")
        self.collection_name = collection_name
        self.encoder = CLIPEncoderAgent()

    def _combine_vectors(self, *vectors: np.ndarray) -> np.ndarray:
        """
        Average multiple vectors and L2-normalize the result.
        """
        combined = np.mean(vectors, axis=0)
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined

    def search(self, query_image_path: str = None, query_text: str = None, top_k: int = 5):
        """
        Search for similar products using image, text, or both.
        Returns a list of Qdrant search results.
        """
        vectors = []
        # Encode image if provided
        if query_image_path is not None:
            image_vector = self.encoder.encode_image(query_image_path)
            vectors.append(image_vector)
        # Encode text if provided
        if query_text is not None and query_text.strip():
            text_vector = self.encoder.encode_text(query_text)
            vectors.append(text_vector)
        if not vectors:
            raise ValueError("Must provide at least one of: query_image_path, query_text")
        # Combine vectors (average if both, else use single)
        if len(vectors) == 1:
            query_vector = vectors[0]
        else:
            query_vector = self._combine_vectors(*vectors)
        # Search Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
        )
        return results


# Example usage (for testing only)
if __name__ == "__main__":
    # Initialize search feature
    searcher = ProductSearchFeature()
    # Example: search by text
    results = searcher.search(query_text="lightweight hiking backpack")
    for r in results:
        print(r)
