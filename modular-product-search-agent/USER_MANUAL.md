# Modular Product Search Agent — User & Integration Manual

This manual explains how to use and integrate the modular product search feature into your website, including how users interact with it, how to handle text/image/both queries, and how results are returned.

---

## 1. Overview

The modular product search agent is designed for seamless integration into any web application. It enables users to search your product catalog using text, images, or both, and returns the most relevant products using advanced multimodal search.

---

## 2. User Experience

### How Users Search

- **Text Query:** Users type a description (e.g., "red running shoes size 10").
- **Image Query:** Users upload a product photo (e.g., a picture of a backpack).
- **Combined Query:** Users provide both an image and text for the most accurate results.

### What Happens

- The website frontend collects the user's query (text, image, or both).
- The query is sent to the backend API powered by `search_feature.py`.
- The backend encodes the query, searches the vector database, and returns the top matches.
- The frontend displays the results (product name, image, description, etc.) to the user.

---

## 3. Integration Guide

### Backend (Python API)

```json
{
	"query_text": "string (optional)",
	"query_image_path": "string (optional, path to uploaded image)",
	"top_k": 5
}
```

1. Save the uploaded image (if any) to a temporary path.
2. Pass the text and/or image path to `ProductSearchFeature.search()`.
3. Return the search results as JSON.

#### Example (Flask-style pseudocode):

```python
from flask import Flask, request, jsonify
from search_feature import ProductSearchFeature

app = Flask(__name__)
searcher = ProductSearchFeature()

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query_text = data.get('query_text')
    query_image_path = data.get('query_image_path')  # Handle file uploads separately
    top_k = data.get('top_k', 5)
    results = searcher.search(query_image_path=query_image_path, query_text=query_text, top_k=top_k)
    # Format results for frontend
    formatted = [
        {
            'rank': i+1,
            'score': round(r.score, 4),
            **r.payload
        } for i, r in enumerate(results)
    ]
    return jsonify({'results': formatted})
```

### Backend Integration Examples

#### Python (Flask-style pseudocode)

Import and initialize the `ProductSearchFeature` class from `search_feature.py`. Expose an API endpoint (e.g., `/search`) that accepts POST requests with the following JSON payload:

```json
{
	"query_text": "string (optional)",
	"query_image_path": "string (optional, path to uploaded image)",
	"top_k": 5
}
```

Example handler:

```python
from flask import Flask, request, jsonify
from search_feature import ProductSearchFeature

app = Flask(__name__)
searcher = ProductSearchFeature()

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query_text = data.get('query_text')
    query_image_path = data.get('query_image_path')  # Handle file uploads separately
    top_k = data.get('top_k', 5)
    results = searcher.search(query_image_path=query_image_path, query_text=query_text, top_k=top_k)
    formatted = [
        {
            'rank': i+1,
            'score': round(r.score, 4),
            **r.payload
        } for i, r in enumerate(results)
    ]
    return jsonify({'results': formatted})
```

#### Node.js (Express + Python Child Process)

You can use Node.js (Express) as your backend and call the Python search script as a child process. For image uploads, use `multer` to handle file uploads, then pass the file path and text to the Python script.

**Install dependencies:**

```bash
npm install express multer
```

**Example Express route:**

```js
const express = require("express");
const multer = require("multer");
const { spawn } = require("child_process");
const app = express();
const upload = multer({ dest: "uploads/" });

app.use(express.json());

app.post("/search", upload.single("image"), (req, res) => {
	const queryText = req.body.query_text;
	const imagePath = req.file ? req.file.path : null;
	const topK = req.body.top_k || 5;

	// Call the Python script as a child process
	const py = spawn("python", [
		"modular-product-search-agent/search_feature.py",
	]);
	let output = "";

	py.stdin.write(
		JSON.stringify({
			query_text: queryText,
			query_image_path: imagePath,
			top_k: topK,
		}),
	);
	py.stdin.end();

	py.stdout.on("data", (data) => {
		output += data.toString();
	});

	py.stdout.on("end", () => {
		try {
			const results = JSON.parse(output);
			res.json(results);
		} catch (e) {
			res.status(500).json({ error: "Failed to parse Python output" });
		}
	});
});

app.listen(3000, () => console.log("Server running on port 3000"));
```

**Note:** For production, use a more robust inter-process communication (e.g., REST API between Node.js and Python via FastAPI or Flask) and handle file cleanup.

### Frontend

### Frontend

- Provide a search form with:
    - Text input for queries
    - File upload for images
    - Optionally, both fields together
- On submit, send the data to the backend `/search` endpoint.
- Display the returned product list (name, image, description, etc.) to the user.

#### Image Upload Technology & Recommendations

- **Best Practice:** Use a modern file upload library (e.g., `multer` for Node.js, `Flask-Uploads` or `Werkzeug` for Flask, or built-in file upload for FastAPI).
- **Frontend:** Use an `<input type="file">` element for image uploads. For a better user experience, support drag-and-drop and preview.
- **Backend:** Save the uploaded image to a temporary directory and pass the file path to the search backend.
- **Security:** Always validate and sanitize uploaded files. Limit file size and type.
- **Preferred Method:**
    - For most web apps, uploading the image file to the backend (not just a URL) is more reliable and secure, as it avoids broken links and ensures the backend can access the image for encoding.
    - If you must use URLs, validate that the backend can access and download the image.

**Summary:**

- Use file upload for best reliability and user experience.

---

## 4. Query Types & Handling

| Query Type   | User Action        | Backend Handling                     |
| ------------ | ------------------ | ------------------------------------ |
| Text only    | Enter text         | Encode text, search                  |
| Image only   | Upload image       | Encode image, search                 |
| Text + Image | Both fields filled | Encode both, average vectors, search |

- If both fields are empty, return an error.
- If only one is provided, use that modality.
- If both, combine for best results.

---

## 5. Result Format

The backend returns a list of results, each with:

- `rank`: Position in the result list
- `score`: Similarity score (higher = more similar)
- `product_id`, `name`, `text_description`, `image_path`, `category`, `price`, `brand` (from your catalog)

Example response:

```json
{
  "results": [
    {
      "rank": 1,
      "score": 0.9876,
      "product_id": "123",
      "name": "Red Running Shoes",
      "text_description": "Lightweight shoes for running...",
      "image_path": "images/123.jpg",
      "category": "Shoes",
      "price": 59.99,
      "brand": "FastFeet"
    },
    ...
  ]
}
```

---

## 6. Deployment Notes

- Ensure the Qdrant DB is populated (run `train_and_ingest.py` first).
- The backend must have access to the trained Qdrant DB and the CLIP model.
- For production, secure the API and handle file uploads safely.
- For large catalogs or high traffic, consider scaling the backend and vector DB.

---

## 7. Advanced Tips

- You can extend the API to accept image files directly (not just paths).
- Add filters (price, category, etc.) to the search endpoint for hybrid search.
- Log queries and results for analytics and improvement.

---

For further integration help, see the main README or contact the project maintainer.
