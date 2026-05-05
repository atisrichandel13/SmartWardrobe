from cloudinary_helper import get_all_images
from search import build_index_from_urls

print("Fetching images from Cloudinary...")
images = get_all_images()

urls = [img["url"] for img in images]
filenames = [img["filename"] for img in images]

print(f"Found {len(images)} images. Building index...")
build_index_from_urls(urls, filenames)

print("Done! FAISS index built from Cloudinary images.")