"""
classifier/trainer.py

Two-stage ViT fine-tuning trainer for the garment classifier.

Stage 1 (epochs 0 → backbone_unfreeze_epoch-1):
  Only heads are trained. This stabilises the heads before opening the backbone.

Stage 2 (epochs backbone_unfreeze_epoch → end):
  Last 4 ViT blocks + heads are co-trained with separate learning rates
  (discriminative fine-tuning): backbone_lr << head_lr.

MixUp augmentation is applied in training.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

from classifier.model import GarmentClassifier, MultiTaskLoss
from utils.common import (
    AverageMeter, topk_accuracy, get_logger, save_checkpoint, load_checkpoint
)
from torch.amp import GradScaler
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast


class ClassifierTrainer:

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg    = cfg
        self.cc     = cfg["classifier"]
        self.device = device
        self.logger = get_logger("classifier", cfg["paths"]["logs"])

        # Model
        self.model = GarmentClassifier(
            num_categories=self.cc["num_categories"],
            pretrained_name=self.cc["backbone"],
        ).to(device)

        # Criterion
        self.criterion = MultiTaskLoss(
            loss_weights=self.cc["loss_weights"],
            label_smoothing=self.cc["label_smoothing"],
        ).to(device)

        # Mixed precision
        self.use_fp16 = cfg.get("fp16", False) and device.type == "cuda"
        self.scaler   = torch.amp.GradScaler("cuda", enabled=self.use_fp16) if self.use_fp16 else GradScaler("cpu", enabled=False)

        # MixUp alpha
        self.mixup_alpha = self.cc.get("mixup_alpha", 0.2)

        self.best_val_top1 = 0.0
        self.ckpt_dir = Path(cfg["paths"]["checkpoints"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Optimiser builder ─────────────────────────────────────────────────────

    def _build_optimizer(self, stage: int) -> AdamW:
        """
        Stage 1: only head params.
        Stage 2: backbone (low lr) + heads (high lr).
        """
        wd = self.cc["weight_decay"]
        if stage == 1:
            param_groups = [
                {"params": list(self.model.get_head_params()),
                 "lr": self.cc["head_lr"]}
            ]
        else:
            param_groups = [
                {"params": list(self.model.get_backbone_params()),
                 "lr": self.cc["backbone_lr"]},
                {"params": list(self.model.get_head_params()),
                 "lr": self.cc["head_lr"]},
            ]
        return AdamW(param_groups, weight_decay=wd)

    # ── MixUp ────────────────────────────────────────────────────────────────

    def _mixup(self, images: torch.Tensor, targets: dict):
        """
        Applies MixUp to a batch. Returns mixed images and a lambda scalar.
        Targets are kept as-is; loss function handles mixed labels via
        convex combination externally.
        """
        if self.mixup_alpha <= 0:
            return images, targets, 1.0

        lam  = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        B    = images.size(0)
        perm = torch.randperm(B, device=images.device)

        mixed = lam * images + (1 - lam) * images[perm]
        return mixed, targets, lam, perm

    def _mixup_loss(self, outputs, targets, lam, perm):
        """Convex combination of loss for original and permuted targets."""
        loss_a = self.criterion(outputs, targets)
        targets_b = {k: v[perm] if isinstance(v, torch.Tensor) else v
                     for k, v in targets.items()}
        loss_b = self.criterion(outputs, targets_b)
        total  = lam * loss_a["loss"] + (1 - lam) * loss_b["loss"]
        return {"loss": total, **{k: loss_a[k] for k in loss_a if k != "loss"}}

    # ── Train one epoch ───────────────────────────────────────────────────────

    def _train_epoch(self, loader, optimizer, epoch: int) -> dict:
        self.model.train()
        loss_m  = AverageMeter()
        top1_m  = AverageMeter()
        top5_m  = AverageMeter()
        t0 = time.time()

        for step, batch in enumerate(loader):
            images = batch["image"].to(self.device, non_blocking=True)
            targets = {k: batch[k].to(self.device, non_blocking=True)
                       for k in ("category", "color", "pattern", "style")}

            # MixUp
            if self.mixup_alpha > 0:
                images, targets, lam, perm = self._mixup(images, targets)
                with torch.amp.autocast("cuda", enabled=self.use_fp16):
                    outputs = self.model(images)
                    loss_dict = self._mixup_loss(outputs, targets, lam, perm)
            else:
                with torch.amp.autocast("cuda", enabled=self.use_fp16):
                    outputs   = self.model(images)
                    loss_dict = self.criterion(outputs, targets)

            optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss_dict["loss"]).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cc["grad_clip"])
            self.scaler.step(optimizer)
            self.scaler.update()

            B = images.size(0)
            loss_m.update(loss_dict["loss"].item(), B)

            with torch.no_grad():
                t1, t5 = topk_accuracy(
                    outputs["logits_category"], batch["category"].to(self.device),
                    topk=tuple(self.cc["topk"])
                )
            top1_m.update(t1, B)
            top5_m.update(t5, B)

            if step % 50 == 0:
                self.logger.info(
                    f"[Epoch {epoch}] step {step}/{len(loader)} | "
                    f"loss={loss_m.avg:.4f} | top1={top1_m.avg:.2f}% | "
                    f"top5={top5_m.avg:.2f}%"
                )

        self.logger.info(
            f"[Epoch {epoch}] TRAIN done in {time.time()-t0:.0f}s | "
            f"loss={loss_m.avg:.4f} top1={top1_m.avg:.2f}% top5={top5_m.avg:.2f}%"
        )
        return {"loss": loss_m.avg, "top1": top1_m.avg, "top5": top5_m.avg}

    # ── Validate ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader, epoch: int) -> dict:
        self.model.eval()
        loss_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()

        all_preds  = {"color": [], "pattern": [], "style": []}
        all_labels = {"color": [], "pattern": [], "style": []}

        for batch in loader:
            images  = batch["image"].to(self.device, non_blocking=True)
            targets = {k: batch[k].to(self.device, non_blocking=True)
                       for k in ("category", "color", "pattern", "style")}

            with torch.amp.autocast("cuda", enabled=self.use_fp16):
                outputs   = self.model(images)
                loss_dict = self.criterion(outputs, targets)

            B = images.size(0)
            loss_m.update(loss_dict["loss"].item(), B)
            t1, t5 = topk_accuracy(
                outputs["logits_category"], targets["category"],
                topk=tuple(self.cc["topk"])
            )
            top1_m.update(t1, B)
            top5_m.update(t5, B)

            for attr in ("color", "pattern", "style"):
                preds = outputs[f"logits_{attr}"].argmax(1).cpu().numpy()
                labs  = targets[attr].cpu().numpy()
                all_preds[attr].extend(preds)
                all_labels[attr].extend(labs)

        f1s = {}
        for attr in ("color", "pattern", "style"):
            f1s[attr] = f1_score(
                all_labels[attr], all_preds[attr],
                average="macro", zero_division=0
            ) * 100

        self.logger.info(
            f"[Epoch {epoch}] VAL | loss={loss_m.avg:.4f} | "
            f"top1={top1_m.avg:.2f}% | top5={top5_m.avg:.2f}% | "
            f"F1 color={f1s['color']:.1f} pattern={f1s['pattern']:.1f} "
            f"style={f1s['style']:.1f}"
        )
        return {
            "loss": loss_m.avg, "top1": top1_m.avg, "top5": top5_m.avg, **f1s
        }

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, train_loader, val_loader):
        cc     = self.cc
        epochs = cc["num_epochs"]
        unfreeze_at = cc.get("backbone_unfreeze_epoch", 5)

        # Stage 1 init
        self.model.freeze_backbone()
        optimizer = self._build_optimizer(stage=1)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=cc["min_lr"]
        )

        history = []
        for epoch in range(epochs):
            # Transition to Stage 2
            if epoch == unfreeze_at:
                self.logger.info("=== Switching to Stage 2: unfreezing last 4 blocks ===")
                self.model.unfreeze_last_n_blocks(n=4)
                optimizer = self._build_optimizer(stage=2)
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=epochs - unfreeze_at,
                    eta_min=cc["min_lr"]
                )

            train_metrics = self._train_epoch(train_loader, optimizer, epoch)
            val_metrics   = self._validate(val_loader, epoch)
            scheduler.step()

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                   **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append(row)

            if val_metrics["top1"] > self.best_val_top1:
                self.best_val_top1 = val_metrics["top1"]
                save_checkpoint(
                    {"epoch": epoch,
                     "state_dict": self.model.state_dict(),
                     "val_top1":   self.best_val_top1,
                     "cfg":        self.cfg},
                    str(self.ckpt_dir / "classifier_best.pth")
                )
                self.logger.info(
                    f"  [OK] Best model saved (val top1={self.best_val_top1:.2f}%)"
                )

        # Save final
        save_checkpoint(
            {"epoch": epochs - 1, "state_dict": self.model.state_dict()},
            str(self.ckpt_dir / "classifier_final.pth")
        )
        return history

    def load_best(self):
        ckpt = load_checkpoint(
            str(self.ckpt_dir / "classifier_best.pth"), self.device
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.logger.info(
            f"Loaded best classifier (val top1={ckpt.get('val_top1', '?'):.2f}%)"
        )
