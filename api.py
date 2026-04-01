import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from search import build_index_from_urls, search_by_text, search_by_image
from cloudinary_helper import upload_image, get_all_images, delete_image, download_image_to_temp

load_dotenv()

app = FastAPI(title="SmartWardrobe API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Wardrobe ──────────────────────────────────────────────────────────────────

@app.get("/wardrobe")
def get_wardrobe():
    """Return all images in the wardrobe from Cloudinary."""
    images = get_all_images()
    return {"total": len(images), "items": images}


@app.post("/wardrobe/add")
async def add_to_wardrobe(file: UploadFile = File(...)):
    """Upload a new clothing image to Cloudinary and rebuild the FAISS index."""
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only jpg and png files are supported.")

    # Save temporarily
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Upload to Cloudinary
    public_id = os.path.splitext(file.filename)[0]
    result = upload_image(temp_path, public_id=public_id)
    os.remove(temp_path)

    # Rebuild FAISS index
    all_images = get_all_images()
    urls = [img["url"] for img in all_images]
    filenames = [img["filename"] for img in all_images]
    build_index_from_urls(urls, filenames)

    return {
        "message": f"{file.filename} uploaded to Cloudinary.",
        "url": result["url"],
        "public_id": result["public_id"]
    }


# ── Search ────────────────────────────────────────────────────────────────────

@app.post("/search/text")
def text_search(query: str, top_k: int = 5):
    """Search wardrobe by text query."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    results = search_by_text(query, top_k=top_k)
    for r in results:
        filename = os.path.basename(r["image_path"])
        r["filename"] = filename
        r["garment_id"] = os.path.splitext(filename)[0]
        del r["image_path"]
    return {"query": query, "results": results}


@app.post("/search/image")
async def image_search(file: UploadFile = File(...), top_k: int = 5):
    """Search wardrobe by uploading a reference image."""
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only jpg and png files are supported.")

    temp_filename = f"{uuid.uuid4()}.jpg"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = search_by_image(temp_path, top_k=top_k)
    os.remove(temp_path)

    for r in results:
        filename = os.path.basename(r["image_path"])
        r["filename"] = filename
        r["garment_id"] = os.path.splitext(filename)[0]
        del r["image_path"]
    return {"results": results}


# ── Stubbed Endpoints ─────────────────────────────────────────────────────────

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """STUB — Gagandeep's SAM segmentation module."""
    return {
        "message": "Segmentation stub — Gagandeep's module will replace this.",
        "garment_id": str(uuid.uuid4()),
        "crop_path": "mock_wardrobe/white_tshirt.jpg"
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """STUB — Krutika's ViT classifier module."""
    return {
        "message": "Classification stub — Krutika's module will replace this.",
        "garment_id": str(uuid.uuid4()),
        "category": "t-shirt",
        "color": "white",
        "pattern": "solid",
        "style": "casual",
        "embedding_vector": [0.1, 0.2, 0.3]
    }


@app.get("/recommend/{garment_id}")
def recommend(garment_id: str):
    """STUB — Krutika's compatibility model."""
    return {
        "message": "Compatibility stub — Krutika's module will replace this.",
        "garment_id": garment_id,
        "recommendations": [
            {"filename": "black_jeans.jpg", "score": 0.91},
            {"filename": "white_sneakers.jpg", "score": 0.87},
            {"filename": "grey_hoodie.jpg", "score": 0.82},
        ]
    }


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)