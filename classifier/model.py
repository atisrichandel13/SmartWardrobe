"""
classifier/model.py

Multi-task ViT-B/16 Classifier.

Architecture:
  - Backbone: google/vit-base-patch16-224-in21k  (768-dim CLS token)
  - Four separate linear classification heads:
      category (50 classes), color (12), pattern (8), style (6)
  - Returns both logits and the raw CLS embedding

Two-stage fine-tuning strategy (used by trainer):
  Stage 1 (epochs 0..backbone_unfreeze_epoch-1):
    - Only head parameters are updated (backbone frozen)
    - lr = head_lr
  Stage 2 (epochs backbone_unfreeze_epoch..end):
    - Last 4 transformer blocks + LayerNorm + heads are updated
    - backbone params: lr = backbone_lr, heads: lr = head_lr
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

from data.deepfashion_dataset import (
    COLOR_LABELS, PATTERN_LABELS, STYLE_LABELS
)


class GarmentClassifier(nn.Module):
    """
    ViT-B/16 backbone with four attribute classification heads.

    Args:
        num_categories (int): number of garment category classes
        pretrained_name (str): HuggingFace model id
        dropout (float): dropout before each head
    """

    def __init__(
        self,
        num_categories: int = 50,
        pretrained_name: str = "google/vit-base-patch16-224-in21k",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = 768  # ViT-B/16 CLS dim

        # ── Backbone ──────────────────────────────────────────────────────────
        self.vit = ViTModel.from_pretrained(pretrained_name)

        # ── Classification heads ──────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)

        self.head_category = nn.Linear(self.embed_dim, num_categories)
        self.head_color    = nn.Linear(self.embed_dim, len(COLOR_LABELS))
        self.head_pattern  = nn.Linear(self.embed_dim, len(PATTERN_LABELS))
        self.head_style    = nn.Linear(self.embed_dim, len(STYLE_LABELS))

        # Initialize heads with truncated normal (better than default uniform)
        for head in (self.head_category, self.head_color,
                     self.head_pattern, self.head_style):
            nn.init.trunc_normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    # ── Freeze helpers ────────────────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze all backbone parameters (Stage 1)."""
        for p in self.vit.parameters():
            p.requires_grad_(False)

    def unfreeze_last_n_blocks(self, n: int = 4):
        """
        Unfreeze the last n transformer blocks + final layernorm (Stage 2).
        ViT-B/16 has 12 encoder blocks indexed 0..11.
        """
        # Unfreeze pooler and layernorm
        for module in (self.vit.layernorm, self.vit.pooler):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(True)

        # Unfreeze last n encoder blocks
        blocks = self.vit.encoder.layer
        total  = len(blocks)
        for block in blocks[total - n:]:
            for p in block.parameters():
                p.requires_grad_(True)

    def get_backbone_params(self):
        """Yield trainable backbone parameter groups."""
        for name, p in self.vit.named_parameters():
            if p.requires_grad:
                yield p

    def get_head_params(self):
        """All head parameters (always trainable)."""
        for head in (self.head_category, self.head_color,
                     self.head_pattern, self.head_style):
            yield from head.parameters()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        Args:
            pixel_values: (B, 3, 224, 224)
        Returns dict:
            embedding:  (B, 768) – raw CLS token, L2-normalised
            logits_category: (B, num_categories)
            logits_color:    (B, 12)
            logits_pattern:  (B, 8)
            logits_style:    (B, 6)
        """
        outputs = self.vit(pixel_values=pixel_values)
        cls = outputs.last_hidden_state[:, 0, :]   # (B, 768) – CLS token

        z = self.dropout(cls)
        logits = {
            "logits_category": self.head_category(z),
            "logits_color":    self.head_color(z),
            "logits_pattern":  self.head_pattern(z),
            "logits_style":    self.head_style(z),
        }
        # L2-normalise the embedding for downstream use (cosine compatibility)
        embedding = nn.functional.normalize(cls, p=2, dim=-1)
        return {"embedding": embedding, **logits}


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Weighted sum of cross-entropy losses for each attribute head.
    Label smoothing is applied to all heads.
    """

    def __init__(
        self,
        loss_weights: dict,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.weights = loss_weights
        ce = lambda: nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.ce_category = ce()
        self.ce_color    = ce()
        self.ce_pattern  = ce()
        self.ce_style    = ce()

    def forward(self, outputs: dict, targets: dict) -> dict:
        l_cat  = self.ce_category(outputs["logits_category"], targets["category"])
        l_col  = self.ce_color   (outputs["logits_color"],    targets["color"])
        l_pat  = self.ce_pattern (outputs["logits_pattern"],  targets["pattern"])
        l_sty  = self.ce_style   (outputs["logits_style"],    targets["style"])

        w = self.weights
        total = (w["category"] * l_cat + w["color"] * l_col
                 + w["pattern"] * l_pat + w["style"] * l_sty)
        return {
            "loss":        total,
            "loss_cat":    l_cat.item(),
            "loss_color":  l_col.item(),
            "loss_pattern":l_pat.item(),
            "loss_style":  l_sty.item(),
        }
