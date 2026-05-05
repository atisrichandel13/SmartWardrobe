"""
scripts/train_compatibility.py

Train the type-aware compatibility model on Polyvore Outfits.

Usage:
    python scripts/train_compatibility.py --config configs/config.yaml \\
        --classifier-ckpt outputs/checkpoints/classifier_best.pth

Before running:
    1. Train classifier first (scripts/train_classifier.py)
    2. Download Polyvore Outfits (nondisjoint) from:
       https://github.com/mvasil/fashion-compatibility
       Extract to: data/polyvore_outfits/
    3. Verify structure:
       data/polyvore_outfits/
         images/
         polyvore_item_metadata.json
         train.json, valid.json, test.json
         fill_in_the_blank_train.json
         fill_in_the_blank_valid.json
         fill_in_the_blank_test.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ModuleNotFoundError:
    HAS_TB = False
    print("Warning: tensorboard not installed. Run `pip install tensorboard` for loss curves.")

from classifier.model import GarmentClassifier
from compatibility.trainer import CompatibilityTrainer, EmbeddingDataset
from data.polyvore_dataset import (
    build_polyvore_loaders, PolyvoreCompatibilityDataset,
    PolyvoreFITBDataset, PolyvoreMetadata,
)
from data.deepfashion_dataset import build_transforms
from utils.common import (
    load_config, get_device, set_seed, get_logger, load_checkpoint
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",           default="configs/config.yaml")
    p.add_argument("--classifier-ckpt",  required=True,
                   help="Path to best classifier checkpoint")
    p.add_argument("--skip-embedding-cache", action="store_true",
                   help="Recompute embeddings even if cache exists")
    return p.parse_args()


def load_classifier(cfg, ckpt_path: str, device: torch.device) -> GarmentClassifier:
    cc    = cfg["classifier"]
    model = GarmentClassifier(
        num_categories=cc["num_categories"],
        pretrained_name=cc["backbone"],
    ).to(device)
    ckpt = load_checkpoint(ckpt_path, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg)
    logger = get_logger("train_compatibility", cfg["paths"]["logs"])
    logger.info(f"Device: {device}")

    # ── Load classifier (frozen) ──────────────────────────────────────────────
    logger.info(f"Loading classifier from {args.classifier_ckpt}")
    classifier = load_classifier(cfg, args.classifier_ckpt, device)

    # ── Polyvore data ─────────────────────────────────────────────────────────
    logger.info("Loading Polyvore metadata…")
    root = cfg["paths"]["polyvore_root"]
    meta = PolyvoreMetadata(root)
    cc   = cfg["compatibility"]
    nw   = cfg.get("num_workers", 4)

    train_tfm = build_transforms(cfg, "train")
    eval_tfm  = build_transforms(cfg, "val")

    train_raw = PolyvoreCompatibilityDataset(
        meta, split="train", transform=train_tfm,
        num_negatives=cc["num_negatives"], seed=cfg.get("seed", 42)
    )
    fitb_ds = PolyvoreFITBDataset(meta, split="test", transform=eval_tfm)
    fitb_loader = DataLoader(fitb_ds, batch_size=1, shuffle=False, num_workers=nw)

    logger.info(f"Polyvore train pairs: {len(train_raw)}, FITB queries: {len(fitb_ds)}")

    # ── Pre-compute embeddings (caches on disk after first run) ───────────────
    logger.info("Pre-computing visual embeddings…")
    train_emb_ds = EmbeddingDataset(train_raw, classifier, device, batch_size=128)
    train_loader = DataLoader(
        train_emb_ds, batch_size=cc["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True,  # num_workers=0: tensors in RAM
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = CompatibilityTrainer(cfg, device)
    trainer.attach_classifier(classifier)

    # ── TensorBoard ──────────────────────────────────────────────────────────
    tb_dir = Path(cfg["paths"]["logs"]) / "compat_tb"
    writer = SummaryWriter(str(tb_dir)) if HAS_TB else None

    logger.info("=== Starting compatibility training ===")
    history = trainer.train(train_loader, fitb_loader)

    if writer:
        for row in history:
            ep = row["epoch"]
            writer.add_scalar("compat/train_loss",  row["train_loss"],  ep)
            writer.add_scalar("compat/fitb_acc",    row["fitb_acc"],    ep)
            writer.add_scalar("compat/fitb_auc",    row["fitb_auc"],    ep)

    # ── Final FITB report ─────────────────────────────────────────────────────
    trainer.load_best()
    final_metrics = trainer._eval_fitb(fitb_loader)
    logger.info(f"Final FITB metrics: {final_metrics}")

    results_path = Path(cfg["paths"]["logs"]) / "compatibility_test_results.json"
    with open(results_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()

    