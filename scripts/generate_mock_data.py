"""
scripts/generate_mock_data.py

Generates realistic mock datasets for development and unit testing.
Run this to start working BEFORE the real datasets are downloaded.

Creates:
  data/mock_deepfashion/     – synthetic DeepFashion-format annotation files
  data/mock_polyvore/        – synthetic Polyvore-format JSON files
  data/person1_crops/        – dummy 224×224 "wardrobe" crops (for service testing)

Usage:
    python scripts/generate_mock_data.py
    # Then train a smoke-test run:
    python scripts/train_classifier.py --config configs/config_mock.yaml
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
MOCK_DF_ROOT = Path("data/mock_deepfashion")
MOCK_PV_ROOT = Path("data/mock_polyvore")
CROPS_ROOT   = Path("data/person1_crops")

N_CATEGORIES   = 50
N_ATTRIBUTES   = 100          # reduced from 1000 for speed
N_TRAIN        = 800
N_VAL          = 100
N_TEST         = 100
N_TOTAL        = N_TRAIN + N_VAL + N_TEST

N_POLYVORE_ITEMS   = 500
N_POLYVORE_OUTFITS = 150
N_FITB_QUERIES     = 60

CATEGORY_NAMES = [f"category_{i:02d}" for i in range(N_CATEGORIES)]
ATTR_NAMES     = [f"attr_{i:03d}" for i in range(N_ATTRIBUTES)]

IMG_H, IMG_W = 224, 224

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def random_rgb_image() -> Image.Image:
    color = tuple(np.random.randint(50, 220, 3).tolist())
    img   = Image.new("RGB", (IMG_W, IMG_H), color)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# DeepFashion mock
# ──────────────────────────────────────────────────────────────────────────────

def generate_deepfashion():
    root = MOCK_DF_ROOT
    img_dir  = root / "img"
    anno_dir = root / "Anno"
    eval_dir = root / "Eval"
    for d in (img_dir, anno_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Generate images
    img_paths = []
    for i in range(N_TOTAL):
        cat_ix   = i % N_CATEGORIES
        sub_dir  = img_dir / f"cat{cat_ix:02d}"
        sub_dir.mkdir(exist_ok=True)
        rel_path = f"img/cat{cat_ix:02d}/item_{i:04d}.jpg"
        full_path = root / rel_path
        if not full_path.exists():
            random_rgb_image().save(full_path, quality=70)
        img_paths.append(rel_path)

    # list_category_cloth.txt
    with open(anno_dir / "list_category_cloth.txt", "w") as f:
        f.write(f"{N_CATEGORIES}\n")
        f.write("category_name cloth_type\n")
        for i, name in enumerate(CATEGORY_NAMES):
            cloth_type = (i % 3) + 1
            f.write(f"{name} {cloth_type}\n")

    # list_category_img.txt
    with open(anno_dir / "list_category_img.txt", "w") as f:
        f.write(f"{N_TOTAL}\n")
        f.write("image_name category_label\n")
        for idx, rel in enumerate(img_paths):
            cat_label = (idx % N_CATEGORIES) + 1  # 1-indexed
            f.write(f"{rel} {cat_label}\n")

    # list_attr_cloth.txt
    with open(anno_dir / "list_attr_cloth.txt", "w") as f:
        f.write(f"{N_ATTRIBUTES}\n")
        f.write("attribute_name attribute_type\n")
        attr_groups = ["black", "solid", "casual",  "stripe", "formal",
                       "white", "floral", "sporty", "graphic", "blue",
                       "green", "plaid", "vintage", "animal", "bohemian",
                       "red", "abstract", "streetwear", "yellow", "multi_color"]
        for i in range(N_ATTRIBUTES):
            name = attr_groups[i % len(attr_groups)] + f"_{i}"
            atype = (i % 5) + 1
            f.write(f"{name} {atype}\n")

    # list_attr_img.txt  (sparse: 3-5 active attributes per image)
    with open(anno_dir / "list_attr_img.txt", "w") as f:
        f.write(f"{N_TOTAL}\n")
        f.write("image_name " + " ".join(f"attr_{i:03d}" for i in range(N_ATTRIBUTES)) + "\n")
        for rel in img_paths:
            attr_vec = [-1] * N_ATTRIBUTES
            active = random.sample(range(N_ATTRIBUTES), k=random.randint(3, 5))
            for a in active:
                attr_vec[a] = 1
            f.write(rel + " " + " ".join(map(str, attr_vec)) + "\n")

    # list_eval_partition.txt
    splits = (["train"] * N_TRAIN + ["val"] * N_VAL + ["test"] * N_TEST)
    random.shuffle(splits)
    with open(eval_dir / "list_eval_partition.txt", "w") as f:
        f.write(f"{N_TOTAL}\n")
        f.write("image_name evaluation_status\n")
        for rel, sp in zip(img_paths, splits):
            f.write(f"{rel} {sp}\n")

    print(f"[OK] Mock DeepFashion created at {root} ({N_TOTAL} images)")


# ──────────────────────────────────────────────────────────────────────────────
# Polyvore Outfits mock
# ──────────────────────────────────────────────────────────────────────────────

GARMENT_TYPES = [
    "tops", "bottoms", "shoes", "bags", "outerwear",
    "dresses", "jewelry", "sunglasses", "hats", "scarves"
]

def generate_polyvore():
    root    = MOCK_PV_ROOT
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Generate item images and metadata
    item_ids = [f"item_{i:05d}" for i in range(N_POLYVORE_ITEMS)]
    metadata = {}
    for iid in item_ids:
        cat = random.choice(GARMENT_TYPES)
        metadata[iid] = {
            "name":              f"Mock {cat} item",
            "semantic_category": cat,
            "description":       f"A {cat} garment",
            "price":             round(random.uniform(10, 300), 2),
        }
        img_path = img_dir / f"{iid}.jpg"
        if not img_path.exists():
            random_rgb_image().save(img_path, quality=70)

    with open(root / "polyvore_item_metadata.json", "w") as f:
        json.dump(metadata, f)

    # Generate outfits
    def make_outfits(n: int, item_pool: list) -> list:
        outfits = []
        for i in range(n):
            k = random.randint(3, 6)
            chosen = random.sample(item_pool, k)
            outfits.append({
                "set_id": f"outfit_{i:05d}",
                "items": [{"item_id": iid, "index": j}
                          for j, iid in enumerate(chosen)]
            })
        return outfits

    splits      = {"train": 0.7, "valid": 0.15, "test": 0.15}
    all_outfits = make_outfits(N_POLYVORE_OUTFITS, item_ids)
    n_train = int(N_POLYVORE_OUTFITS * 0.7)
    n_valid = int(N_POLYVORE_OUTFITS * 0.15)

    for fname, data in [
        ("train.json", all_outfits[:n_train]),
        ("valid.json", all_outfits[n_train:n_train+n_valid]),
        ("test.json",  all_outfits[n_train+n_valid:]),
    ]:
        with open(root / fname, "w") as f:
            json.dump(data, f)

    # Fill-in-the-blank queries
    for split_name in ("train", "valid", "test"):
        queries = []
        for _ in range(N_FITB_QUERIES):
            context    = random.sample(item_ids, k=random.randint(2, 4))
            correct    = random.choice(item_ids)
            distractors= random.sample([i for i in item_ids if i != correct], k=3)
            answers    = [correct] + distractors
            queries.append({
                "question":       context,
                "answers":        answers,
                "blank_position": 0,
            })
        with open(root / f"fill_in_the_blank_{split_name}.json", "w") as f:
            json.dump(queries, f)

    print(f"[OK] Mock Polyvore created at {root} ({N_POLYVORE_ITEMS} items, "
          f"{N_POLYVORE_OUTFITS} outfits)")


# ──────────────────────────────────────────────────────────────────────────────
# Person 1 mock crops
# ──────────────────────────────────────────────────────────────────────────────

def generate_person1_crops(n: int = 10):
    CROPS_ROOT.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = random_rgb_image()
        img.save(CROPS_ROOT / f"my_garment_{i:02d}.jpg")
    print(f"[OK] {n} mock person1 crops created at {CROPS_ROOT}")


# ──────────────────────────────────────────────────────────────────────────────
# Mock config (points to mock data dirs, smaller batch/epoch counts)
# ──────────────────────────────────────────────────────────────────────────────

MOCK_CONFIG = """
paths:
  deepfashion_root: "data/mock_deepfashion"
  polyvore_root:    "data/mock_polyvore"
  person1_crops:    "data/person1_crops"
  checkpoints:      "outputs/checkpoints_mock"
  logs:             "outputs/logs_mock"
  results:          "outputs/results_mock"

classifier:
  backbone: "google/vit-base-patch16-224-in21k"
  image_size: 224
  embed_dim: 768
  num_categories: 50
  num_colors: 12
  num_patterns: 8
  num_styles: 6
  batch_size: 8
  num_epochs: 3
  warmup_epochs: 1
  head_lr: 2.0e-4
  backbone_lr: 1.0e-5
  backbone_unfreeze_epoch: 1
  weight_decay: 1.0e-4
  label_smoothing: 0.1
  grad_clip: 1.0
  random_erasing_p: 0.2
  mixup_alpha: 0.2
  color_jitter: [0.2, 0.2, 0.2, 0.05]
  scheduler: "cosine"
  min_lr: 1.0e-7
  loss_weights:
    category: 1.0
    color: 0.5
    pattern: 0.5
    style: 0.5
  topk: [1, 5]

compatibility:
  embed_dim: 768
  proj_dim: 64
  num_garment_types: 11
  batch_size: 16
  num_epochs: 3
  lr: 1.0e-3
  weight_decay: 1.0e-4
  loss: "bpr"
  triplet_margin: 0.2
  num_negatives: 2
  fitb_num_candidates: 4
  scheduler: "cosine"
  min_lr: 1.0e-6

seed: 42
num_workers: 0
device: "auto"
fp16: false
"""

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating mock datasets…\n")
    generate_deepfashion()
    generate_polyvore()
    generate_person1_crops(n=10)

    config_path = Path("configs/config_mock.yaml")
    with open(config_path, "w") as f:
        f.write(MOCK_CONFIG.strip())
    print(f"[OK] Mock config written to {config_path}")

    print("\n── Done ───────────────────────────────────────────────────────")
    print("To run a smoke-test training pipeline:")
    print("  python scripts/train_classifier.py --config configs/config_mock.yaml")
    print("  python scripts/train_compatibility.py \\")
    print("    --config configs/config_mock.yaml \\")
    print("    --classifier-ckpt outputs/checkpoints_mock/classifier_best.pth")
