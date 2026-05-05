from cloudinary_helper import upload_image
import os

wardrobe_dir = "mock_wardrobe"
images = [f for f in os.listdir(wardrobe_dir) if f.lower().endswith((".jpg", ".png"))]

print(f"Uploading {len(images)} images to Cloudinary...")
for filename in images:
    path = os.path.join(wardrobe_dir, filename)
    public_id = os.path.splitext(filename)[0]
    result = upload_image(path, public_id=public_id)
    print(f"  Uploaded: {filename} → {result['url']}")

print("\nDone! All images are on Cloudinary.")