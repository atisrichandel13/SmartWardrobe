"""
scripts/wardrobe_pairs.py

Generates compatible outfit pairs from wardrobe crop images.
For each item, finds the top-k most compatible items and outputs
a clean JSON + printed table of pairs with scores.

Usage:
    python scripts/wardrobe_pairs.py
    python scripts/wardrobe_pairs.py --crops-dir data/person1_crops --top-k 3
    python scripts/wardrobe_pairs.py --crops-dir data/person1_crops --top-k 5 --output outputs/results/wardrobe_pairs.json
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from classifier.model import GarmentClassifier
from compatibility.model import CompatibilityModel
from data.deepfashion_dataset import build_transforms, COLOR_LABELS, PATTERN_LABELS, STYLE_LABELS
from data.polyvore_dataset import category_to_type
from utils.common import load_config, load_checkpoint, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crops-dir",    default="data/person1_crops")
    p.add_argument("--top-k",        type=int, default=3)
    p.add_argument("--classifier-ckpt",  default="outputs/checkpoints/classifier_best.pth")
    p.add_argument("--compat-ckpt",      default="outputs/checkpoints/compatibility_final.pth")
    p.add_argument("--config",           default="configs/config.yaml")
    p.add_argument("--output",           default="outputs/results/wardrobe_pairs.json")
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

    # ── Load compatibility model ───────────────────────────────────────────────
    print("Loading compatibility model...")
    compat = CompatibilityModel(
        visual_dim=768,
        proj_dim=cfg["compatibility"]["proj_dim"],
        num_types=cfg["compatibility"]["num_garment_types"]
    ).to(device)
    ckpt2 = load_checkpoint(args.compat_ckpt, device)
    compat.load_state_dict(ckpt2["state_dict"])
    compat.eval()

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

    print(f"Found {len(image_paths)} images in {crops_dir}")

    # ── Extract embeddings + metadata ─────────────────────────────────────────
    print("Extracting embeddings...")
    items = []
    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            t   = tfm(img).unsqueeze(0).to(device)
            out = classifier(t)

            cat_idx = int(out["logits_category"].argmax(-1))
            cat_name = cat_names[cat_idx]
            color   = COLOR_LABELS  [int(out["logits_color"]  .argmax(-1))]
            pattern = PATTERN_LABELS[int(out["logits_pattern"].argmax(-1))]
            style   = STYLE_LABELS  [int(out["logits_style"]  .argmax(-1))]
            type_idx = torch.tensor(
                [category_to_type(cat_name)], dtype=torch.long, device=device
            )

            items.append({
                "item_id":   img_path.stem,        # filename without extension
                "file":      img_path.name,
                "category":  cat_name,
                "color":     color,
                "pattern":   pattern,
                "style":     style,
                "embedding": out["embedding"][0],  # tensor
                "type_idx":  type_idx,
            })

    # ── Compute compatibility scores ──────────────────────────────────────────
    print(f"\nComputing top-{args.top_k} compatible pairs...")
    print("=" * 70)

    results = []
    with torch.no_grad():
        for i, query in enumerate(items):
            scores = []
            for j, candidate in enumerate(items):
                if i == j:
                    continue
                score = compat.score(
                    query["embedding"].unsqueeze(0), query["type_idx"],
                    candidate["embedding"].unsqueeze(0), candidate["type_idx"]
                )
                scores.append({
                    "item_id":  candidate["item_id"],
                    "file":     candidate["file"],
                    "category": candidate["category"],
                    "color":    candidate["color"],
                    "score":    round(float(score), 4),
                })

            # Sort by score descending
            scores.sort(key=lambda x: x["score"], reverse=True)
            top_matches = scores[:args.top_k]

            # Print
            print(f"\nQuery: {query['file']}")
            print(f"  Category: {query['category']} | Color: {query['color']} | "
                  f"Pattern: {query['pattern']} | Style: {query['style']}")
            print(f"  Top-{args.top_k} compatible items:")
            for rank, m in enumerate(top_matches, 1):
                print(f"    {rank}. {m['file']} ({m['category']}) "
                      f"score={m['score']:.4f}")

            results.append({
                "query": {
                    "item_id":  query["item_id"],
                    "file":     query["file"],
                    "category": query["category"],
                    "color":    query["color"],
                    "pattern":  query["pattern"],
                    "style":    query["style"],
                },
                "top_matches": top_matches,
            })

    print("\n" + "=" * 70)

    # ── Save JSON output ──────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Pairs saved to {out_path}")


if __name__ == "__main__":
    main()

    