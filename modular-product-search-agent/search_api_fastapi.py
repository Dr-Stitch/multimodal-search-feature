import nest_asyncio
import uvicorn
import threading
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import logging

# 1. Allow uvicorn to run inside the notebook's event loop (for Colab/Jupyter compatibility)
nest_asyncio.apply()

# 2. Import the ProductSearchFeature from search_feature.py (handles all search logic)
from search_feature import ProductSearchFeature

# Logging setup for server events and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('fastapi_colab')

# Create FastAPI app instance
app = FastAPI(title='Multimodal Product Search API')

# Enable CORS for all origins (for easy testing and frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Global searcher instance (will be initialized on server start)
searcher = None

def load_searcher():
    """
    Initialize the ProductSearchFeature (loads model and connects to Qdrant).
    Sets the global searcher variable.
    """
    global searcher
    try:
        searcher = ProductSearchFeature()
        logger.info("ProductSearchFeature successfully initialized.")
    except Exception as e:
        logger.error(f'Failed to initialize searcher: {e}')

@app.post('/search')
async def search(
    query_text: str = Form(None),
    image: UploadFile = File(None),
    top_k: int = Form(5)
):
    """
    Accepts a text query, an image upload, or both. Returns ranked product results.
    """
    if searcher is None:
        raise HTTPException(status_code=503, detail='Search system is still loading or failed to initialize.')

    image_path = None
    try:
        # Save uploaded image to a temp file if provided
        if image:
            temp_dir = 'temp_uploads'
            os.makedirs(temp_dir, exist_ok=True)
            image_path = os.path.join(temp_dir, image.filename)
            with open(image_path, 'wb') as f:
                shutil.copyfileobj(image.file, f)

        # Run the multimodal search (image, text, or both)
        results = searcher.search(query_image_path=image_path, query_text=query_text, top_k=top_k)

        # Format results for API response
        formatted = [
            {
                'rank': i+1,
                'score': round(r.score, 4),
                **r.payload
            } for i, r in enumerate(results)
        ]
        return JSONResponse({'results': formatted})
    except Exception as e:
        logger.error(f'Search failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file after search
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

def run_server():
    """
    Loads the searcher and starts the FastAPI server (for notebook/Colab use).
    """
    load_searcher()
    config = uvicorn.Config(app=app, host='0.0.0.0', port=8000, log_level='info')
    server = uvicorn.Server(config)
    server.run()

if __name__ == '__main__':
    # Run server in a background thread to keep notebook cell interactive
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print('FastAPI server is starting in the background on http://127.0.0.1:8000')