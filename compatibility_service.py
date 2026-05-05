"""
compatibility_service.py

Public API for the Compatibility module.
Person 3 imports this and wraps in FastAPI.

Usage:
    from compatibility_service import CompatibilityService

    svc = CompatibilityService("outputs/checkpoints/compatibility_best.pth")

    # Rank wardrobe items by compatibility with a query item
    ranked = svc.rank_compatibility(query_item, wardrobe_items)
    # ranked = [
    #   {"item_id": "...", "score": 0.91, "metadata": {...}},
    #   ...sorted by score descending...
    # ]
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from compatibility.model import CompatibilityModel
from data.polyvore_dataset import category_to_type
from utils.common import get_device, load_config, load_checkpoint


class CompatibilityService:
    """
    Thin wrapper around a trained CompatibilityModel for inference.
    Accepts embeddings (produced by ClassifierService) + type info.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/config.yaml",
        device: Union[str, torch.device, None] = None,
    ):
        cfg = load_config(config_path)
        self.cfg = cfg
        cc = cfg["compatibility"]

        if device is None:
            self.device = get_device(cfg)
        else:
            self.device = torch.device(device)

        self.model = CompatibilityModel(
            visual_dim=cfg["classifier"]["embed_dim"],
            proj_dim=cc["proj_dim"],
            num_types=cc["num_garment_types"],
        ).to(self.device)

        ckpt = load_checkpoint(checkpoint_path, self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def rank_compatibility(
        self,
        query_item: dict,
        wardrobe_items: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rank wardrobe items by compatibility with a query garment.

        Args:
            query_item: dict with keys:
                "embedding"  – list[float] of length 768 (from ClassifierService)
                "category"   – str garment category name
                optionally "item_id" (str)
            wardrobe_items: list of dicts, same structure as query_item,
                plus any extra metadata keys (passed through unchanged)
            top_k: return only top-k results (None = return all)

        Returns:
            List of wardrobe item dicts sorted by compatibility score (desc),
            with an added "score" key (float in [-1, 1]).
        """
        if not wardrobe_items:
            return []

        query_emb  = self._emb_tensor(query_item["embedding"])      # (1, D)
        query_type = self._type_tensor(query_item.get("category", "other"))  # (1,)

        ward_embs  = torch.stack([
            self._emb_tensor(it["embedding"])[0] for it in wardrobe_items
        ])                                                            # (N, D)
        ward_types = torch.tensor(
            [category_to_type(it.get("category", "other")) for it in wardrobe_items],
            dtype=torch.long, device=self.device,
        )                                                             # (N,)

        # Expand query to match wardrobe size for batched scoring
        N = len(wardrobe_items)
        q_emb_exp  = query_emb.expand(N, -1)
        q_type_exp = query_type.expand(N)

        scores = self.model.score(
            q_emb_exp, q_type_exp,
            ward_embs, ward_types,
        ).cpu().numpy()                                               # (N,)

        # Sort descending
        order = np.argsort(scores)[::-1]
        if top_k is not None:
            order = order[:top_k]

        results = []
        for rank, idx in enumerate(order):
            item = dict(wardrobe_items[idx])   # shallow copy
            item["score"] = round(float(scores[idx]), 4)
            item["rank"]  = rank + 1
            results.append(item)

        return results

    @torch.no_grad()
    def score_pair(self, item_a: dict, item_b: dict) -> float:
        """Return scalar compatibility score for a single pair."""
        ea  = self._emb_tensor(item_a["embedding"])
        ta  = self._type_tensor(item_a.get("category", "other"))
        eb  = self._emb_tensor(item_b["embedding"])
        tb  = self._type_tensor(item_b.get("category", "other"))
        return float(self.model.score(ea, ta, eb, tb).item())

    def _emb_tensor(self, embedding: list | np.ndarray) -> torch.Tensor:
        """Convert embedding (list or array) to (1, D) tensor on device."""
        arr = np.asarray(embedding, dtype=np.float32)
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def _type_tensor(self, category_str: str) -> torch.Tensor:
        type_idx = category_to_type(category_str)
        return torch.tensor([type_idx], dtype=torch.long, device=self.device)
