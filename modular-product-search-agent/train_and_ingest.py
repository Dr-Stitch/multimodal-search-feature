import os
import io
import warnings
import logging
import pandas as pd
import numpy as np
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Configuration ---
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "google/siglip-base-patch16-224")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 768))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "product_catalog")
DATA_CSV_PATH = os.getenv("DATA_CSV_PATH", "/content/products.csv")
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# Qdrant remote config (all loaded from .env or environment)
# QDRANT_URL: The full URL to your Qdrant Cloud/cluster endpoint (e.g. https://xxxx.aws.cloud.qdrant.io:6333)
# QDRANT_API_KEY: Your Qdrant API key (never hardcode in code, always use .env or environment)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")  # Optional, for self-hosted Qdrant
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))  # Optional, for self-hosted Qdrant

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_and_ingest")

# --- Data Ingestion Agent ---
class DataIngestionAgent:
    REQUIRED_COLUMNS = {"product_id", "image_path", "name", "text_description"}
    OPTIONAL_DEFAULTS = {"category": "", "price": 0.0, "brand": ""}

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self) -> list[dict]:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        for col, default in self.OPTIONAL_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
        before_count = len(df)
        df = df.dropna(subset=list(self.REQUIRED_COLUMNS))
        dropped = before_count - len(df)
        if dropped > 0:
            warnings.warn(f"Dropped {dropped} rows with missing required fields")
        products = df.to_dict(orient="records")
        print(f"Loaded {len(products)} products from {self.csv_path}")
        return products

# --- CLIP Encoder Agent ---
class CLIPEncoderAgent:
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        print(f"Loading model '{model_name}'...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        print("Model loaded ✓")

    def _validate_image(self, image_path: str) -> Image.Image:
        if image_path.startswith("http://") or image_path.startswith("https://"):
            try:
                response = requests.get(image_path, stream=True)
                response.raise_for_status()
                img_data = io.BytesIO(response.content)
                img = Image.open(img_data).convert("RGB")
                return img
            except Exception as e:
                raise ValueError(f"Cannot download/read image from URL '{image_path}': {e}")
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            try:
                img = Image.open(image_path).convert("RGB")
                return img
            except Exception as e:
                raise ValueError(f"Cannot read image '{image_path}': {e}")

    def _l2_normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def encode_image(self, image_path: str) -> np.ndarray:
        img = self._validate_image(image_path)
        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt")
            outputs = self.model.get_image_features(**inputs)
            vector = outputs.pooler_output.squeeze().numpy()
        return self._l2_normalize(vector)

    def encode_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Cannot encode empty text")
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            outputs = self.model.get_text_features(**inputs)
            vector = outputs.pooler_output.squeeze().numpy()
        return self._l2_normalize(vector)

# --- Vector Store Agent ---
class VectorStoreAgent:
    def __init__(self, collection_name: str = COLLECTION_NAME, vector_dim: int = VECTOR_DIM):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection_name = collection_name
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{self.collection_name}' created")

    def upsert(self, point_id: int, vector: np.ndarray, payload: dict):
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector.tolist(), payload=payload)],
        )

if __name__ == "__main__":
    try:
        ingestion_agent = DataIngestionAgent(csv_path=DATA_CSV_PATH)
        clip_encoder = CLIPEncoderAgent(model_name=CLIP_MODEL_NAME)
        vector_store = VectorStoreAgent(collection_name=COLLECTION_NAME, vector_dim=VECTOR_DIM)
        products = ingestion_agent.load()
        for idx, product in enumerate(products):
            try:
                img_vec = clip_encoder.encode_image(product["image_path"])
                txt_vec = clip_encoder.encode_text(product["text_description"])
                # Fix: Use clip_encoder method instead of undefined 'self'
                combined = clip_encoder._l2_normalize(np.mean([img_vec, txt_vec], axis=0))
                payload = {**product, "price": float(product.get("price", 0.0))}
                vector_store.upsert(point_id=idx, vector=combined, payload=payload)
                logger.info(f"Processed {product['product_id']}")
            except Exception as e:
                logger.error(f"Failed {product['product_id']}: {e}")
    except Exception as e:
        logger.critical(f"Fatal: {e}")