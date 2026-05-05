"""
scripts/train_classifier.py

Train the ViT-B/16 garment classifier on DeepFashion.

Usage:
    python scripts/train_classifier.py --config configs/config.yaml

Before running:
    1. Download DeepFashion Category and Attribute Prediction benchmark
       from: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
       Extract to: data/deepfashion/
    2. Verify structure:
       data/deepfashion/
         img/
         Anno/list_category_img.txt
         Anno/list_category_cloth.txt
         Anno/list_attr_img.txt
         Anno/list_attr_cloth.txt
         Eval/list_eval_partition.txt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ModuleNotFoundError:
    HAS_TB = False
    print("Warning: tensorboard not installed. Run `pip install tensorboard` for loss curves.")

from classifier.trainer import ClassifierTrainer
from data.deepfashion_dataset import (
    build_deepfashion_loaders, DeepFashionAnnotations,
    COLOR_LABELS, PATTERN_LABELS, STYLE_LABELS,
)
from utils.common import load_config, get_device, set_seed, get_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--eval-only", action="store_true", help="Skip training, just evaluate")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = get_device(cfg)
    logger = get_logger("train_classifier", cfg["paths"]["logs"])
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info("Loading DeepFashion annotations…")
    train_loader, val_loader, test_loader, ann = build_deepfashion_loaders(cfg)
    logger.info(
        f"Dataset sizes: train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )

    # Save category names for service layer
    cat_names_path = Path(cfg["paths"]["checkpoints"]) / "category_names.json"
    cat_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cat_names_path, "w") as f:
        json.dump(ann.category_names, f, indent=2)
    logger.info(f"Category names saved to {cat_names_path}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = ClassifierTrainer(cfg, device)

    if args.resume:
        from utils.common import load_checkpoint
        ckpt = load_checkpoint(args.resume, device)
        trainer.model.load_state_dict(ckpt["state_dict"])
        logger.info(f"Resumed from {args.resume}")

    # ── TensorBoard ──────────────────────────────────────────────────────────
    tb_dir = Path(cfg["paths"]["logs"]) / "classifier_tb"
    writer = SummaryWriter(str(tb_dir)) if HAS_TB else None

    if not args.eval_only:
        logger.info("=== Starting training ===")
        history = trainer.train(train_loader, val_loader)

        if writer:
            for row in history:
                ep = row["epoch"]
                writer.add_scalar("train/loss",    row["train_loss"], ep)
                writer.add_scalar("val/top1",      row["val_top1"],   ep)
                writer.add_scalar("val/top5",      row["val_top5"],   ep)
                for attr in ("color", "pattern", "style"):
                    writer.add_scalar(f"val/f1_{attr}", row.get(attr, 0), ep)

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info("=== Test set evaluation ===")
    trainer.load_best()
    test_metrics = trainer._validate(test_loader, epoch=-1)
    logger.info(f"Test metrics: {test_metrics}")

    results_path = Path(cfg["paths"]["logs"]) / "classifier_test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test results saved to {results_path}")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()

    