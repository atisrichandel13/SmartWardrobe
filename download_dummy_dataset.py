import os
import requests
from PIL import Image
from io import BytesIO

# Small set of clothing images from the web
mock_images = {
    "blue_denim_jacket.jpg": "https://images.unsplash.com/photo-1551537482-f2075a1d41f2?w=400",
    "white_tshirt.jpg": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400",
    "black_jeans.jpg": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400",
    "red_dress.jpg": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=400",
    "white_sneakers.jpg": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400",
    "black_leather_jacket.jpg": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400",
    "floral_dress.jpg": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400",
    "grey_hoodie.jpg": "https://images.unsplash.com/photo-1556821840-3a63f15732ce?w=400",
    "beige_trousers.jpg": "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=400",
    "blue_formal_shirt.jpg": "https://images.unsplash.com/photo-1602810318383-e386cc2a3ccf?w=400",
    "black_skirt.jpg": "https://images.unsplash.com/photo-1583496661160-fb5886a0aaaa?w=400",
    "brown_boots.jpg": "https://images.unsplash.com/photo-1542838132-92c53300491e?w=400",
    "yellow_summer_dress.jpg": "https://images.unsplash.com/photo-1496747611176-843222e1e57c?w=400",
    "navy_blazer.jpg": "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=400",
    "striped_tshirt.jpg": "https://images.unsplash.com/photo-1523381294911-8d3cead13475?w=400",
}

def download_mock_wardrobe():
    save_dir = "mock_wardrobe"
    os.makedirs(save_dir, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0"}

    print(f"Downloading {len(mock_images)} clothing images...")
    for filename, url in mock_images.items():
        save_path = os.path.join(save_dir, filename)
        if os.path.exists(save_path):
            print(f"  Skipping {filename} (already exists)")
            continue
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize((224, 224))
            img.save(save_path)
            print(f"  Downloaded {filename}")
        except Exception as e:
            print(f"  Failed {filename}: {e}")

    print(f"\nDone! Images saved to ./{save_dir}/")
    print(f"Total: {len(os.listdir(save_dir))} images")

if __name__ == "__main__":
    download_mock_wardrobe()