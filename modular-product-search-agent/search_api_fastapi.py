"""
search_api_fastapi.py
--------------------
FastAPI-based API for multimodal product search using the ProductSearchFeature class.
Accepts text, image, or both as input and returns ranked product results.
"""

import os
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from search_feature import ProductSearchFeature
import uvicorn

app = FastAPI(title="Multimodal Product Search API")

# Allow CORS for local development (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

searcher = ProductSearchFeature()

@app.post("/search")
async def search(
    query_text: str = Form(None),
    image: UploadFile = File(None),
    top_k: int = Form(5)
):
    """
    Accepts a text query, an image upload, or both. Returns ranked product results.
    """
    image_path = None
    try:
        # Save uploaded image to a temp file if provided
        if image is not None:
            ext = os.path.splitext(image.filename)[1].lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                raise HTTPException(status_code=400, detail="Unsupported image format.")
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            image_path = os.path.join(temp_dir, image.filename)
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image.file, f)

        # Run search
        results = searcher.search(query_image_path=image_path, query_text=query_text, top_k=top_k)
        formatted = [
            {
                "rank": i+1,
                "score": round(r.score, 4),
                **r.payload
            } for i, r in enumerate(results)
        ]
        return JSONResponse({"results": formatted})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

if __name__ == "__main__":
    uvicorn.run("search_api_fastapi:app", host="0.0.0.0", port=8000, reload=True)
