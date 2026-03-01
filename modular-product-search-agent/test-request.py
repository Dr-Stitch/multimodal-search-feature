import requests
import json

# This works because both the server and this request are in the same Colab runtime
url = "http://127.0.0.1:8000/search"
data = {"query_text": "dining table", "top_k": 3}

try:
    response = requests.post(url, data=data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Internal test failed: {e}\nMake sure the FastAPI server cell is currently running.")