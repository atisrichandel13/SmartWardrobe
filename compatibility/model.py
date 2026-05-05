"""
compatibility/model.py

Type-Aware Compatibility Embedding Model.

Architecture (inspired by Type-Aware Net / SCE-Net):
─────────────────────────────────────────────────────
Given two garment items (a, b) with visual embeddings from ViT and their
garment type indices, we compute a compatibility score as follows:

  1. Type embedding: learned e_type ∈ R^{type_embed_dim} per garment type.
  2. Projection: concat(visual_embed, type_embed) → MLP → proj ∈ R^{proj_dim}
     (L2-normalised). Separate MLP weights per garment-type pair are expensive;
     instead we use a type-conditioned FiLM (Feature-wise Linear Modulation)
     layer that scales and shifts the visual embedding before projection.
  3. Compatibility score: cosine_similarity(proj_a, proj_b)

Training losses available:
  BPR  (Bayesian Personalised Ranking): log-sigmoid of score difference
        between positive and negative pair. Fast, stable, recommended.
  Triplet: max(0, margin - sim_pos + sim_neg), standard alternative.

Reference:
  Vasileva et al., Learning Type-Aware Embeddings for Fashion Compatibility,
  ECCV 2018.  https://arxiv.org/abs/1803.09196
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.polyvore_dataset import NUM_TYPES


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on garment type.
    γ and β are produced by a small MLP from the type embedding.
    """

    def __init__(self, visual_dim: int, type_embed_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(type_embed_dim, visual_dim)
        self.beta_net  = nn.Linear(type_embed_dim, visual_dim)
        nn.init.ones_(self.gamma_net.weight)
        nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, x: torch.Tensor, type_embed: torch.Tensor) -> torch.Tensor:
        γ = self.gamma_net(type_embed)
        β = self.beta_net(type_embed)
        return γ * x + β


class CompatibilityEmbedder(nn.Module):
    """
    Maps (visual_embedding, garment_type) → L2-normalised compatibility embedding.

    Args:
        visual_dim (int):     input visual embedding dimension (768 for ViT-B/16)
        proj_dim (int):       output compatibility embedding dimension (64)
        num_types (int):      number of garment types (11 for Polyvore)
        type_embed_dim (int): dimension of learnable type embedding
        dropout (float):      dropout in MLP
    """

    def __init__(
        self,
        visual_dim: int = 768,
        proj_dim: int = 64,
        num_types: int = NUM_TYPES,
        type_embed_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.type_embed = nn.Embedding(num_types, type_embed_dim)

        self.film = FiLMLayer(visual_dim, type_embed_dim)

        hidden_dim = 256
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Init
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, visual: torch.Tensor, type_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual   : (B, visual_dim) – L2-normalised ViT CLS embeddings
            type_idx : (B,) – integer garment type indices
        Returns:
            (B, proj_dim) – L2-normalised compatibility embeddings
        """
        te  = self.type_embed(type_idx)           # (B, type_embed_dim)
        x   = self.film(visual, te)               # (B, visual_dim)
        out = self.mlp(x)                         # (B, proj_dim)
        return F.normalize(out, p=2, dim=-1)


class CompatibilityModel(nn.Module):
    """
    Wraps CompatibilityEmbedder and provides score computation.
    """

    def __init__(self, visual_dim: int = 768, proj_dim: int = 64, **kwargs):
        super().__init__()
        self.embedder = CompatibilityEmbedder(visual_dim, proj_dim, **kwargs)

    def embed(self, visual: torch.Tensor, type_idx: torch.Tensor) -> torch.Tensor:
        return self.embedder(visual, type_idx)

    def score(
        self,
        visual_a: torch.Tensor, type_a: torch.Tensor,
        visual_b: torch.Tensor, type_b: torch.Tensor,
    ) -> torch.Tensor:
        """Returns cosine similarity scores ∈ [-1, 1], shape (B,)."""
        ea = self.embed(visual_a, type_a)
        eb = self.embed(visual_b, type_b)
        return (ea * eb).sum(dim=-1)   # dot product of unit vectors = cosine sim

    def forward(self, visual: torch.Tensor, type_idx: torch.Tensor) -> torch.Tensor:
        return self.embed(visual, type_idx)


# ──────────────────────────────────────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────────────────────────────────────

class BPRLoss(nn.Module):
    """
    Bayesian Personalised Ranking loss.
    Maximises score(anchor, positive) - score(anchor, negative).
    Loss = -mean(log σ(s_pos - s_neg))
    """

    def forward(
        self,
        s_pos: torch.Tensor,  # (B,)
        s_neg: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        return -F.logsigmoid(s_pos - s_neg).mean()


class TripletLoss(nn.Module):
    """Standard cosine triplet loss."""

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,    # (B, D)
        positive: torch.Tensor,  # (B, D)
        negative: torch.Tensor,  # (B, D)
    ) -> torch.Tensor:
        sim_pos = (anchor * positive).sum(-1)
        sim_neg = (anchor * negative).sum(-1)
        loss = F.relu(self.margin - sim_pos + sim_neg)
        return loss.mean()


def build_loss(loss_name: str, cfg: dict) -> nn.Module:
    if loss_name == "bpr":
        return BPRLoss()
    elif loss_name == "triplet":
        return TripletLoss(margin=cfg["compatibility"]["triplet_margin"])
    raise ValueError(f"Unknown loss: {loss_name}")
