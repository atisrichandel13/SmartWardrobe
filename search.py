import os
import json
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor

# ── Load Model ────────────────────────────────────────────────────────────────

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
tokenizer = CLIPTokenizer.from_pretrained("patrickjohncyh/fashion-clip")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

model = model.to(device)
model.eval()

# ── Encode ────────────────────────────────────────────────────────────────────

def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype("float32")

def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype("float32")

# ── Index ─────────────────────────────────────────────────────────────────────

INDEX_PATH = "wardrobe.index"
MAPPING_PATH = "wardrobe_mapping.json"

def build_index(wardrobe_dir):
    image_paths = [
        os.path.join(wardrobe_dir, f)
        for f in os.listdir(wardrobe_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Encoding {len(image_paths)} images...")
    embeddings = []
    mapping = []

    for path in image_paths:
        try:
            emb = encode_image(path)
            embeddings.append(emb)
            mapping.append(path)
            print(f"  Encoded: {os.path.basename(path)}")
        except Exception as e:
            print(f"  Failed: {path} — {e}")

    embeddings = np.vstack(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity on normalized vectors
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(MAPPING_PATH, "w") as f:
        json.dump(mapping, f)

    print(f"\nIndex built with {index.ntotal} images and saved to disk.")
    return index, mapping

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    return index, mapping

# ── Search ────────────────────────────────────────────────────────────────────

def search_by_text(query, top_k=5):
    index, mapping = load_index()
    query_embedding = encode_text(query)
    scores, indices = index.search(query_embedding, top_k)
    results = [
        {"image_path": mapping[i], "score": float(scores[0][rank])}
        for rank, i in enumerate(indices[0])
    ]
    return results

def search_by_image(image_path, top_k=5):
    index, mapping = load_index()
    query_embedding = encode_image(image_path)
    scores, indices = index.search(query_embedding, top_k)
    results = [
        {"image_path": mapping[i], "score": float(scores[0][rank])}
        for rank, i in enumerate(indices[0])
    ]
    return results

import requests
from io import BytesIO

def build_index_from_urls(urls: list, filenames: list):
    """Build FAISS index from a list of Cloudinary URLs."""
    print(f"Encoding {len(urls)} images from Cloudinary...")
    embeddings = []
    mapping = []

    for url, filename in zip(urls, filenames):
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            inputs = image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                embedding = model.get_image_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            emb = embedding.cpu().numpy().astype("float32")
            embeddings.append(emb)
            mapping.append(filename)
            print(f"  Encoded: {filename}")
        except Exception as e:
            print(f"  Failed: {filename} — {e}")

    embeddings = np.vstack(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(MAPPING_PATH, "w") as f:
        json.dump(mapping, f)

    print(f"\nIndex built with {index.ntotal} images.")
    return index, mapping

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build index from mock wardrobe
    build_index("mock_wardrobe")

    # Test text search
    print("\nText search: 'blue denim jacket'")
    results = search_by_text("blue denim jacket", top_k=3)
    for r in results:
        print(f"  {os.path.basename(r['image_path'])}  score: {r['score']:.4f}")

    # Test image search
    print("\nImage search using white_tshirt.jpg as query")
    results = search_by_image("mock_wardrobe/white_tshirt.jpg", top_k=3)
    for r in results:
        print(f"  {os.path.basename(r['image_path'])}  score: {r['score']:.4f}")
