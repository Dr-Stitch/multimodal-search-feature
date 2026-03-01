import os
import io
import logging
import numpy as np
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# --- Load environment variables from .env file (for config and secrets) ---
load_dotenv()

# --- Configuration ---
# Model name for CLIP/SigLIP
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "google/siglip-base-patch16-224")
# Embedding vector dimension
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 768))
# Qdrant collection name
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "product_catalog")
# Supported image file extensions
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# Qdrant remote config (URL and API key loaded from .env)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("search_feature")

# --- CLIP Encoder Agent ---
class CLIPEncoderAgent:
    """
    Encodes images and text into vectors using SigLIP (CLIP) model.
    Handles both local file paths and image URLs.
    """
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        # Load processor and model from HuggingFace
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _validate_image(self, image_path: str) -> Image.Image:
        # If image_path is a URL, download and open it
        if image_path.startswith("http"):
            response = requests.get(image_path, stream=True)
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        # Otherwise, open local file
        return Image.open(image_path).convert("RGB")

    def _l2_normalize(self, vector: np.ndarray) -> np.ndarray:
        # Normalize vector to unit length (for cosine similarity)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def encode_image(self, image_path: str) -> np.ndarray:
        # Encode an image file or URL to a vector
        img = self._validate_image(image_path)
        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt")
            outputs = self.model.get_image_features(**inputs)
            # Use pooler_output for SigLIP/CLIP models
            vector = outputs.pooler_output.squeeze().numpy()
        return self._l2_normalize(vector)

    def encode_text(self, text: str) -> np.ndarray:
        # Encode a text string to a vector
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            outputs = self.model.get_text_features(**inputs)
            vector = outputs.pooler_output.squeeze().numpy()
        return self._l2_normalize(vector)

# --- Product Search Feature ---
class ProductSearchFeature:
    """
    Provides search functionality over a Qdrant vector database.
    Encodes queries and retrieves similar products.
    """
    def __init__(self, collection_name: str = COLLECTION_NAME, vector_dim: int = VECTOR_DIM):
        try:
            # Connect to Qdrant Cloud using URL and API key
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            self.collection_name = collection_name
            self.encoder = CLIPEncoderAgent()
            logger.info(f"Connected to Qdrant Cloud collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def search(self, query_image_path: str = None, query_text: str = None, top_k: int = 5):
        """
        Search for similar products using an image, text, or both.
        Returns a list of Qdrant search results (points).
        """
        vectors = []
        # Encode image if provided
        if query_image_path:
            vectors.append(self.encoder.encode_image(query_image_path))
        # Encode text if provided
        if query_text:
            vectors.append(self.encoder.encode_text(query_text))

        if not vectors:
            raise ValueError("No query provided")

        # Average vectors if both image and text are provided
        query_vector = np.mean(vectors, axis=0)
        # Normalize the query vector
        query_vector = query_vector / np.linalg.norm(query_vector)

        # Query Qdrant for similar points
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k
        ).points

if __name__ == "__main__":
    try:
        # Example usage: search by text
        searcher = ProductSearchFeature()
        results = searcher.search(query_text="dining table")
        for r in results:
            print(f"Score: {r.score:.4f} | Name: {r.payload.get('name')}")
    except Exception as e:
        print(f"Error: {e}")
