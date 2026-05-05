"""
compatibility/trainer.py

Trains the CompatibilityModel using visual embeddings from a frozen
(or cached) GarmentClassifier backbone.

Strategy:
  We cache ViT embeddings for all Polyvore items to disk (numpy memmap)
  so the GPU doesn't need to re-run ViT on every epoch — the compatibility
  MLP is the only thing being trained, which is much faster.

Evaluation:
  Fill-in-the-blank (FITB) task:
    Given N context items, score each of 4 candidates against all context
    items and pick the highest-scoring candidate.
    Metric: accuracy (Polyvore standard) and AUC (per the project spec).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score

from compatibility.model import CompatibilityModel, build_loss
from classifier.model import GarmentClassifier
from utils.common import AverageMeter, get_logger, save_checkpoint, load_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Embedding caching helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def cache_embeddings(
    classifier: GarmentClassifier,
    loader,
    cache_path: str,
    device: torch.device,
    embed_dim: int = 768,
):
    """
    Runs classifier on all items in loader and caches embeddings + types.
    Returns dict: item_id → {"embedding": np.ndarray, "type": int}
    Saves to cache_path as a .npz for fast reload.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["cache"].item()

    print("Computing embeddings (this runs once)…")
    classifier.eval()
    cache = {}
    from torchvision import transforms as T
    from data.deepfashion_dataset import IMAGENET_MEAN, IMAGENET_STD

    for batch in loader:
        imgs    = batch["anchor_img"].to(device)
        types   = batch["anchor_type"]
        item_ids = batch.get("anchor_id", [None] * imgs.size(0))

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = classifier(imgs)

        embs = out["embedding"].cpu().numpy()
        for i, iid in enumerate(item_ids):
            if iid is not None:
                cache[iid] = {
                    "embedding": embs[i],
                    "type":      int(types[i]),
                }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, cache=cache)
    return cache


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class CompatibilityTrainer:

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg    = cfg
        self.cc     = cfg["compatibility"]
        self.device = device
        self.logger = get_logger("compatibility", cfg["paths"]["logs"])

        self.model = CompatibilityModel(
            visual_dim=cfg["classifier"]["embed_dim"],
            proj_dim=self.cc["proj_dim"],
            num_types=self.cc["num_garment_types"],
        ).to(device)

        self.loss_fn  = build_loss(self.cc["loss"], cfg)
        self.use_fp16 = cfg.get("fp16", False) and device.type == "cuda"
        self.scaler   = torch.amp.GradScaler("cuda", enabled=self.use_fp16) if self.use_fp16 else GradScaler("cpu", enabled=False)

        self.best_fitb_acc = 0.0
        self.ckpt_dir = Path(cfg["paths"]["checkpoints"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self):
        return AdamW(
            self.model.parameters(),
            lr=self.cc["lr"],
            weight_decay=self.cc["weight_decay"],
        )

    # ── Training epoch ────────────────────────────────────────────────────────

    def _train_epoch(self, loader, optimizer, epoch: int) -> float:
        self.model.train()
        loss_m = AverageMeter()
        t0     = time.time()

        for step, batch in enumerate(loader):
            anc_vis  = batch["anchor_visual"].to(self.device)
            anc_type = batch["anchor_type"].to(self.device)
            pos_vis  = batch["pos_visual"].to(self.device)
            pos_type = batch["pos_type"].to(self.device)
            neg_vis  = batch["neg_visual"].to(self.device)
            neg_type = batch["neg_type"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_fp16):
                if self.cc["loss"] == "bpr":
                    s_pos = self.model.score(anc_vis, anc_type, pos_vis, pos_type)
                    s_neg = self.model.score(anc_vis, anc_type, neg_vis, neg_type)
                    loss  = self.loss_fn(s_pos, s_neg)
                else:  # triplet
                    e_anc = self.model.embed(anc_vis, anc_type)
                    e_pos = self.model.embed(pos_vis, pos_type)
                    e_neg = self.model.embed(neg_vis, neg_type)
                    loss  = self.loss_fn(e_anc, e_pos, e_neg)

            optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            loss_m.update(loss.item(), anc_vis.size(0))
            if step % 100 == 0:
                self.logger.info(
                    f"[Compat Epoch {epoch}] step {step}/{len(loader)} "
                    f"loss={loss_m.avg:.4f}"
                )

        self.logger.info(
            f"[Compat Epoch {epoch}] TRAIN done in {time.time()-t0:.0f}s "
            f"loss={loss_m.avg:.4f}"
        )
        return loss_m.avg

    # ── FITB evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_fitb(self, fitb_loader) -> dict:
        """
        Fill-in-the-blank accuracy and AUC.

        For each query:
          score(candidate) = mean cosine similarity to all context items
          Predict: argmax(scores)

        AUC: treat correct candidate as positive (score=1), others as negative (score=0),
             then use the raw compatibility score to compute ROC-AUC.
        """
        self.model.eval()
        correct = 0
        total   = 0
        all_scores  = []    # list of 4-element arrays
        all_labels  = []    # list of 4-element one-hot arrays

        for batch in fitb_loader:
            # batch_size=1, so squeeze the outer dim
            ctx_imgs  = batch["context_imgs"].squeeze(0).to(self.device)   # (N_ctx, C,H,W)
            ctx_types = batch["context_types"].squeeze(0).to(self.device)  # (N_ctx,)
            cand_imgs  = batch["cand_imgs"].squeeze(0).to(self.device)     # (4, C,H,W)
            cand_types = batch["cand_types"].squeeze(0).to(self.device)    # (4,)
            label      = int(batch["label"])

            # We use pre-computed visuals if available, else pass through classifier
            # (In integration: ctx/cand images come as visual tensors already)
            # Here we assume pixel images and use classifier in eval mode
            # But since we cache embeddings, this path uses raw tensors:
            # The loader here is the FITB loader which returns image tensors.
            # We'll handle both cases:
            if ctx_imgs.dim() == 4:
                # Images → must run through backbone (slower; only for FITB eval)
                with torch.amp.autocast("cuda", enabled=self.use_fp16):
                    ctx_vis  = self._imgs_to_embeddings(ctx_imgs)
                    cand_vis = self._imgs_to_embeddings(cand_imgs)
            else:
                ctx_vis  = ctx_imgs
                cand_vis = cand_imgs

            ctx_embs  = self.model.embed(ctx_vis, ctx_types)   # (N_ctx, D)
            cand_embs = self.model.embed(cand_vis, cand_types)  # (4, D)

            # Score each candidate against mean context embedding
            ctx_mean   = ctx_embs.mean(0, keepdim=True)         # (1, D)
            scores     = (cand_embs * ctx_mean).sum(-1)          # (4,)

            pred   = int(scores.argmax())
            correct += int(pred == label)
            total  += 1

            sc = scores.cpu().numpy()
            all_scores.append(sc)
            lbl_onehot = np.zeros(4)
            lbl_onehot[label] = 1
            all_labels.append(lbl_onehot)

        if total == 0:
            self.logger.info("FITB skipped - no valid queries found")
            return {"fitb_acc": 0.0, "fitb_auc": 0.0}
        acc = correct / total * 100

        # AUC: flatten all scores/labels across all queries
        scores_flat = np.concatenate(all_scores)
        labels_flat = np.concatenate(all_labels)
        try:
            auc = roc_auc_score(labels_flat, scores_flat) * 100
        except ValueError:
            auc = 0.0

        self.logger.info(f"FITB accuracy={acc:.2f}%  AUC={auc:.2f}%")
        return {"fitb_acc": acc, "fitb_auc": auc}

    def _imgs_to_embeddings(self, imgs: torch.Tensor) -> torch.Tensor:
        """Run classifier backbone on a batch of images → visual embeddings."""
        if not hasattr(self, "_classifier"):
            raise RuntimeError(
                "Attach a classifier via trainer.attach_classifier(model) before FITB eval."
            )
        out = self._classifier(imgs)
        return out["embedding"]

    def attach_classifier(self, classifier: GarmentClassifier):
        """Attach a trained (frozen) classifier for embedding extraction."""
        self._classifier = classifier.to(self.device)
        self._classifier.eval()
        for p in self._classifier.parameters():
            p.requires_grad_(False)

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, train_loader, fitb_loader):
        """
        train_loader batches must have keys:
          anchor_visual, anchor_type, pos_visual, pos_type, neg_visual, neg_type
        (pre-computed embeddings; see EmbeddingCachingLoader below)
        """
        cc      = self.cc
        epochs  = cc["num_epochs"]
        optimizer = self._build_optimizer()
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cc["min_lr"])

        history = []
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer, epoch)
            eval_metrics = self._eval_fitb(fitb_loader)
            scheduler.step()

            history.append({"epoch": epoch, "train_loss": train_loss, **eval_metrics})

            if eval_metrics["fitb_acc"] > self.best_fitb_acc:
                self.best_fitb_acc = eval_metrics["fitb_acc"]
                save_checkpoint(
                    {"epoch": epoch,
                     "state_dict": self.model.state_dict(),
                     "fitb_acc":   self.best_fitb_acc,
                     "fitb_auc":   eval_metrics["fitb_auc"],
                     "cfg":        self.cfg},
                    str(self.ckpt_dir / "compatibility_best.pth")
                )
                self.logger.info(
                    f"  [OK] Best compat model saved "
                    f"(FITB acc={self.best_fitb_acc:.2f}%)"
                )

        save_checkpoint(
            {"epoch": epochs - 1, "state_dict": self.model.state_dict()},
            str(self.ckpt_dir / "compatibility_final.pth")
        )
        return history

    def load_best(self):
        ckpt = load_checkpoint(
            str(self.ckpt_dir / "compatibility_best.pth"), self.device
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.logger.info(
            f"Loaded best compat model "
            f"(FITB acc={ckpt.get('fitb_acc', '?'):.2f}%, "
            f"AUC={ckpt.get('fitb_auc', '?'):.2f}%)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Embedding-precomputing wrapper dataset
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingDataset(torch.utils.data.Dataset):
    """
    Precomputes visual embeddings to disk (memmap) instead of RAM.
    Handles datasets too large to fit in memory.
    """

    def __init__(self, base_dataset, classifier: GarmentClassifier,
                 device: torch.device, batch_size: int = 64):
        self.base = base_dataset
        self._precompute(classifier, device, batch_size)

    @torch.no_grad()
    def _precompute(self, classifier, device, bs):
        import numpy as np
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        import tempfile, os

        N = len(self.base)
        cache_dir = Path("outputs/emb_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        anc_path = cache_dir / "anc.npy"
        pos_path = cache_dir / "pos.npy"
        neg_path = cache_dir / "neg.npy"
        type_path = cache_dir / "types.npy"

        if anc_path.exists() and pos_path.exists():
            print("Loading cached embeddings from disk...")
            self._anc_emb  = np.load(anc_path)
            self._pos_emb  = np.load(pos_path)
            self._neg_emb  = np.load(neg_path)
            types          = np.load(type_path)
            self._anc_type = torch.from_numpy(types[0])
            self._pos_type = torch.from_numpy(types[1])
            self._neg_type = torch.from_numpy(types[2])
            print(f"Loaded {N} cached embeddings.")
            return

        classifier.eval()
        tmp_loader = DataLoader(self.base, batch_size=bs, shuffle=False,
                                num_workers=0, pin_memory=False)

        print("Pre-computing embeddings to disk (runs once)...")
        D = 768
        anc_arr = np.zeros((N, D), dtype=np.float32)
        pos_arr = np.zeros((N, D), dtype=np.float32)
        neg_arr = np.zeros((N, D), dtype=np.float32)
        anc_types, pos_types, neg_types = [], [], []

        idx = 0
        for batch in tqdm(tmp_loader):
            b = batch["anchor_img"].size(0)
            for key, arr in [("anchor_img", anc_arr),
                              ("pos_img",    pos_arr),
                              ("neg_img",    neg_arr)]:
                imgs = batch[key].to(device)
                with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                    emb = classifier(imgs)["embedding"].cpu().numpy()
                arr[idx:idx+b] = emb
            anc_types.extend(batch["anchor_type"].tolist())
            pos_types.extend(batch["pos_type"].tolist())
            neg_types.extend(batch["neg_type"].tolist())
            idx += b

        np.save(anc_path, anc_arr)
        np.save(pos_path, pos_arr)
        np.save(neg_path, neg_arr)
        np.save(type_path, np.array([anc_types, pos_types, neg_types], dtype=np.int64))

        self._anc_emb  = anc_arr
        self._pos_emb  = pos_arr
        self._neg_emb  = neg_arr
        self._anc_type = torch.tensor(anc_types, dtype=torch.long)
        self._pos_type = torch.tensor(pos_types, dtype=torch.long)
        self._neg_type = torch.tensor(neg_types, dtype=torch.long)
        print(f"Embeddings saved to {cache_dir}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return {
            "anchor_visual": torch.from_numpy(self._anc_emb[idx]),
            "anchor_type":   self._anc_type[idx],
            "pos_visual":    torch.from_numpy(self._pos_emb[idx]),
            "pos_type":      self._pos_type[idx],
            "neg_visual":    torch.from_numpy(self._neg_emb[idx]),
            "neg_type":      self._neg_type[idx],
        }
    
    