"""
scripts/evaluate.py

Standalone evaluation script for both modules.
Produces a formatted report with all metrics.

Usage:
    # Evaluate classifier on test set
    python scripts/evaluate.py --module classifier \\
        --ckpt outputs/checkpoints/classifier_best.pth

    # Evaluate compatibility model (FITB) on test set
    python scripts/evaluate.py --module compatibility \\
        --ckpt outputs/checkpoints/compatibility_best.pth \\
        --classifier-ckpt outputs/checkpoints/classifier_best.pth

    # Evaluate on personal wardrobe crops (real-world test)
    python scripts/evaluate.py --module classifier \\
        --ckpt outputs/checkpoints/classifier_best.pth \\
        --personal-crops data/person1_crops/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from sklearn.metrics import classification_report, f1_score
import numpy as np

from utils.common import load_config, get_device, get_logger, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True, choices=["classifier", "compatibility", "both"])
    p.add_argument("--ckpt",              help="Path to module checkpoint")
    p.add_argument("--classifier-ckpt",   help="Path to classifier checkpoint (for compat eval)")
    p.add_argument("--config",            default="configs/config.yaml")
    p.add_argument("--personal-crops",    default=None,
                   help="Directory of personal 224×224 wardrobe photos from Person 1")
    p.add_argument("--output-dir",        default="outputs/results")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Classifier evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(cfg, ckpt_path: str, device: torch.device, logger):
    from classifier.model import GarmentClassifier, MultiTaskLoss
    from data.deepfashion_dataset import (
        build_deepfashion_loaders, COLOR_LABELS, PATTERN_LABELS, STYLE_LABELS
    )
    from utils.common import topk_accuracy, AverageMeter

    logger.info("Loading test data…")
    _, _, test_loader, ann = build_deepfashion_loaders(cfg)

    model = GarmentClassifier(
        num_categories=cfg["classifier"]["num_categories"],
        pretrained_name=cfg["classifier"]["backbone"],
    ).to(device)
    ckpt = load_checkpoint(ckpt_path, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    top1_m = AverageMeter()
    top5_m = AverageMeter()
    all_pred = {"color": [], "pattern": [], "style": []}
    all_true = {"color": [], "pattern": [], "style": []}

    logger.info("Running test set evaluation…")
    with torch.no_grad():
        for batch in test_loader:
            imgs    = batch["image"].to(device)
            targets = {k: batch[k].to(device) for k in ("category", "color", "pattern", "style")}
            out     = model(imgs)

            t1, t5 = topk_accuracy(
                out["logits_category"], targets["category"],
                topk=(1, 5)
            )
            top1_m.update(t1, imgs.size(0))
            top5_m.update(t5, imgs.size(0))

            for attr in ("color", "pattern", "style"):
                preds = out[f"logits_{attr}"].argmax(1).cpu().numpy()
                trues = targets[attr].cpu().numpy()
                all_pred[attr].extend(preds)
                all_true[attr].extend(trues)

    # Per-attribute F1
    f1s = {}
    for attr in ("color", "pattern", "style"):
        f1s[attr] = f1_score(all_true[attr], all_pred[attr],
                             average="macro", zero_division=0) * 100

    results = {
        "test_top1_category":  round(top1_m.avg, 2),
        "test_top5_category":  round(top5_m.avg, 2),
        "test_f1_color":       round(f1s["color"], 2),
        "test_f1_pattern":     round(f1s["pattern"], 2),
        "test_f1_style":       round(f1s["style"], 2),
    }

    print("\n" + "="*60)
    print("CLASSIFIER TEST RESULTS")
    print("="*60)
    print(f"  Category Top-1 Accuracy : {results['test_top1_category']:.2f}%")
    print(f"  Category Top-5 Accuracy : {results['test_top5_category']:.2f}%")
    print(f"  Color    Macro-F1       : {results['test_f1_color']:.2f}%")
    print(f"  Pattern  Macro-F1       : {results['test_f1_pattern']:.2f}%")
    print(f"  Style    Macro-F1       : {results['test_f1_style']:.2f}%")
    print("="*60)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Compatibility evaluation (FITB)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_compatibility(cfg, ckpt_path: str, classifier_ckpt: str,
                           device: torch.device, logger):
    from compatibility.model import CompatibilityModel
    from compatibility.trainer import CompatibilityTrainer
    from classifier.model import GarmentClassifier
    from data.polyvore_dataset import PolyvoreFITBDataset, PolyvoreMetadata
    from data.deepfashion_dataset import build_transforms
    from torch.utils.data import DataLoader

    logger.info("Loading FITB test data…")
    meta    = PolyvoreMetadata(cfg["paths"]["polyvore_root"])
    tfm     = build_transforms(cfg, "val")
    fitb_ds = PolyvoreFITBDataset(meta, split="test", transform=tfm)
    fitb_loader = DataLoader(fitb_ds, batch_size=1, shuffle=False, num_workers=2)

    classifier = GarmentClassifier(
        num_categories=cfg["classifier"]["num_categories"],
        pretrained_name=cfg["classifier"]["backbone"],
    ).to(device)
    ckpt = load_checkpoint(classifier_ckpt, device)
    classifier.load_state_dict(ckpt["state_dict"])
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    cc = cfg["compatibility"]
    model = CompatibilityModel(
        visual_dim=cfg["classifier"]["embed_dim"],
        proj_dim=cc["proj_dim"],
        num_types=cc["num_garment_types"],
    ).to(device)
    ckpt = load_checkpoint(ckpt_path, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    trainer = CompatibilityTrainer(cfg, device)
    trainer.model = model
    trainer.attach_classifier(classifier)

    metrics = trainer._eval_fitb(fitb_loader)

    print("\n" + "="*60)
    print("COMPATIBILITY MODEL TEST RESULTS")
    print("="*60)
    print(f"  FITB Accuracy : {metrics['fitb_acc']:.2f}%")
    print(f"  FITB AUC      : {metrics['fitb_auc']:.2f}%")
    print("="*60)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Personal wardrobe evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_personal_crops(cfg, ckpt_path: str, crops_dir: str,
                             device: torch.device, logger):
    from classifier_service import ClassifierService
    from pathlib import Path
    import json

    logger.info(f"Running classifier on personal crops in {crops_dir}…")
    svc = ClassifierService(ckpt_path, device=device)

    # Load category names if available
    cat_names_path = Path(cfg["paths"]["checkpoints"]) / "category_names.json"
    if cat_names_path.exists():
        with open(cat_names_path) as f:
            svc.set_category_names(json.load(f))

    crops = sorted(Path(crops_dir).glob("*.jpg")) + sorted(Path(crops_dir).glob("*.png"))
    if not crops:
        logger.warning(f"No images found in {crops_dir}")
        return []

    print(f"\n{'='*60}")
    print(f"PERSONAL WARDROBE EVALUATION ({len(crops)} items)")
    print("="*60)
    results = []
    for img_path in crops:
        pred = svc.predict(str(img_path))
        results.append({"file": img_path.name, **pred})
        print(f"  {img_path.name:30s}  "
              f"cat={pred['category']:20s}  "
              f"color={pred['color']:12s}  "
              f"pattern={pred['pattern']:12s}  "
              f"style={pred['style']}")
    print("="*60)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = get_device(cfg)
    logger = get_logger("evaluate", cfg["paths"].get("logs", "outputs/logs"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.module in ("classifier", "both"):
        if not args.ckpt:
            print("ERROR: --ckpt required for classifier evaluation")
            sys.exit(1)
        r = evaluate_classifier(cfg, args.ckpt, device, logger)
        all_results["classifier"] = r

        if args.personal_crops:
            personal = evaluate_personal_crops(
                cfg, args.ckpt, args.personal_crops, device, logger
            )
            all_results["personal_crops"] = personal

    if args.module in ("compatibility", "both"):
        if not args.ckpt or not args.classifier_ckpt:
            print("ERROR: --ckpt and --classifier-ckpt required for compatibility eval")
            sys.exit(1)
        r = evaluate_compatibility(
            cfg, args.ckpt, args.classifier_ckpt, device, logger
        )
        all_results["compatibility"] = r

    results_path = out_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        # Filter out non-serialisable embedding lists for JSON output
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items() if k != "embedding"}
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            return obj
        json.dump(clean(all_results), f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
