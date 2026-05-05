import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv
import os
import requests
from PIL import Image
from io import BytesIO

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

FOLDER = "smartwardrobe"

def upload_image(image_path: str, public_id: str = None):
    """Upload a local image to Cloudinary and return its URL."""
    result = cloudinary.uploader.upload(
        image_path,
        folder=FOLDER,
        public_id=public_id,
        overwrite=True,
        resource_type="image"
    )
    return {
        "url": result["secure_url"],
        "public_id": result["public_id"]
    }

def delete_image(public_id: str):
    """Delete an image from Cloudinary."""
    result = cloudinary.api.delete_resources([public_id])
    return result

def get_all_images():
    """Get all images in the smartwardrobe folder."""
    result = cloudinary.api.resources(
        type="upload",
        prefix=FOLDER,
        max_results=200
    )
    images = [
        {
            "url": r["secure_url"],
            "public_id": r["public_id"],
            "filename": r["public_id"].split("/")[-1]
        }
        for r in result["resources"]
    ]
    return images

def download_image_to_temp(url: str, save_path: str):
    """Download a Cloudinary image to a local temp path for inference."""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((224, 224))
    img.save(save_path)
    return save_path