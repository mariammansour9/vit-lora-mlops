"""Call the Dockerized API:  python client.py path/to/image.jpg [--url http://localhost:8000]"""
import argparse
import json

import requests

p = argparse.ArgumentParser()
p.add_argument("image")
p.add_argument("--url", default="http://localhost:8000")
p.add_argument("--top-k", type=int, default=3)
a = p.parse_args()

with open(a.image, "rb") as f:
    r = requests.post(f"{a.url}/predict", params={"top_k": a.top_k}, files={"file": f}, timeout=60)
print(r.status_code)
print(json.dumps(r.json(), indent=2))
