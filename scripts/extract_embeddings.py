"""
scripts/extract_embeddings.py

Extracts and saves classifier embeddings for all crop images.
Outputs a JSON file mapping item_id to embedding + metadata.
Useful for Person 3 to build a FAISS index or pass to compatibility service.

Usage:
    python scripts/extract_embeddings.py
    python scripts/extract_embeddings.py --crops-dir data/person1_crops
    python scripts/extract_embeddings.py --crops-dir data/person1_crops --output outputs/results/embeddings.json
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from classifier.model import GarmentClassifier
from data.deepfashion_dataset import build_transforms, COLOR_LABELS, PATTERN_LABELS, STYLE_LABELS
from utils.common import load_config, load_checkpoint, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crops-dir",       default="data/person1_crops")
    p.add_argument("--classifier-ckpt", default="outputs/checkpoints/classifier_best.pth")
    p.add_argument("--config",          default="configs/config.yaml")
    p.add_argument("--output",          default="outputs/results/embeddings.json")
    p.add_argument("--save-npy",        action="store_true",
                   help="Also save embeddings as .npy for FAISS indexing")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = get_device(cfg)
    tfm    = build_transforms(cfg, "val")

    # ── Load category names ───────────────────────────────────────────────────
    cat_names_path = Path(cfg["paths"]["checkpoints"]) / "category_names.json"
    with open(cat_names_path) as f:
        cat_names = json.load(f)

    # ── Load classifier ───────────────────────────────────────────────────────
    print("Loading classifier...")
    classifier = GarmentClassifier(
        num_categories=cfg["classifier"]["num_categories"],
        pretrained_name=cfg["classifier"]["backbone"]
    ).to(device)
    ckpt = load_checkpoint(args.classifier_ckpt, device)
    classifier.load_state_dict(ckpt["state_dict"])
    classifier.eval()

    # ── Load crop images ──────────────────────────────────────────────────────
    crops_dir = Path(args.crops_dir)
    image_paths = sorted(
        list(crops_dir.glob("*.jpg")) +
        list(crops_dir.glob("*.png")) +
        list(crops_dir.glob("*.jpeg"))
    )

    if not image_paths:
        print(f"No images found in {crops_dir}")
        return

    print(f"Found {len(image_paths)} images. Extracting embeddings...")

    # ── Extract embeddings ────────────────────────────────────────────────────
    results     = {}
    all_embeddings = []
    all_item_ids   = []

    print(f"\n{'Item ID':<45} {'Category':<20} {'Color':<12} {'Pattern':<12} {'Style'}")
    print("-" * 100)

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            t   = tfm(img).unsqueeze(0).to(device)
            out = classifier(t)

            cat_idx  = int(out["logits_category"].argmax(-1))
            cat_name = cat_names[cat_idx]
            color    = COLOR_LABELS  [int(out["logits_color"]  .argmax(-1))]
            pattern  = PATTERN_LABELS[int(out["logits_pattern"].argmax(-1))]
            style    = STYLE_LABELS  [int(out["logits_style"]  .argmax(-1))]
            embedding = out["embedding"][0].cpu().numpy().tolist()

            # Top-5 categories with probabilities
            import torch.nn.functional as F
            probs = F.softmax(out["logits_category"][0], dim=-1)
            top5_vals, top5_ids = probs.topk(5)
            top5 = [(cat_names[i.item()], round(float(p), 4))
                    for i, p in zip(top5_ids, top5_vals)]

            item_id = img_path.stem
            results[item_id] = {
                "item_id":         item_id,
                "file":            img_path.name,
                "category":        cat_name,
                "category_idx":    cat_idx,
                "color":           color,
                "pattern":         pattern,
                "style":           style,
                "top5_categories": top5,
                "embedding":       embedding,   # 768-dim L2-normalised list
            }

            all_item_ids.append(item_id)
            all_embeddings.append(embedding)

            print(f"{img_path.name:<45} {cat_name:<20} {color:<12} {pattern:<12} {style}")

    print("-" * 100)
    print(f"Total: {len(results)} items processed")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEmbeddings saved to {out_path}")

    # ── Save .npy (for FAISS / Person 3 search module) ────────────────────────
    if args.save_npy:
        npy_path = out_path.with_suffix(".npy")
        ids_path = out_path.with_stem(out_path.stem + "_ids").with_suffix(".json")

        np.save(npy_path, np.array(all_embeddings, dtype=np.float32))
        with open(ids_path, "w") as f:
            json.dump(all_item_ids, f, indent=2)

        print(f"Embedding matrix saved to {npy_path}  shape: {np.array(all_embeddings).shape}")
        print(f"Item ID list saved to {ids_path}")
        print(f"\nPerson 3 (Search module) can load these directly into FAISS:")
        print(f"  import faiss, numpy as np, json")
        print(f"  embeddings = np.load('{npy_path}')")
        print(f"  item_ids   = json.load(open('{ids_path}'))")
        print(f"  index = faiss.IndexFlatIP(768)  # inner product = cosine on L2-normed vecs")
        print(f"  index.add(embeddings)")


if __name__ == "__main__":
    main()

    