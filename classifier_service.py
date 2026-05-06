"""
classifier_service.py

Public API for the Classifier module.
Person 3 imports this and wraps in FastAPI.

Usage:
    from classifier_service import ClassifierService

    svc = ClassifierService("outputs/checkpoints/classifier_best.pth")
    result = svc.predict("path/to/224x224_crop.jpg")
    # result = {
    #     "category":  "blouse",
    #     "category_idx": 3,
    #     "color":     "blue",
    #     "pattern":   "solid",
    #     "style":     "casual",
    #     "embedding": [0.021, -0.004, ...],   # 768-dim list
    #     "top5_categories": [("blouse", 0.82), ("shirt", 0.11), ...]
    # }
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from classifier.model import GarmentClassifier
from data.deepfashion_dataset import (
    PATTERN_LABELS, STYLE_LABELS,
    IMAGENET_MEAN, IMAGENET_STD,
)
from utils.common import get_device, load_config, load_checkpoint


_EVAL_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class ClassifierService:
    """
    Thin wrapper around a trained GarmentClassifier for inference.
    Thread-safe (read-only after init).

    Color is detected via a rule-based RGB pixel averaging approach
    that filters out background pixels (dark or white) before computing
    the dominant garment color. This is more reliable than the trained
    color head which suffers from noisy DeepFashion attribute labels.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/config.yaml",
        device: Union[str, torch.device, None] = None,
    ):
        cfg = load_config(config_path)
        self.cfg = cfg
        cc  = cfg["classifier"]

        if device is None:
            self.device = get_device(cfg)
        else:
            self.device = torch.device(device)

        self.model = GarmentClassifier(
            num_categories=cc["num_categories"],
            pretrained_name=cc["backbone"],
        ).to(self.device)

        ckpt = load_checkpoint(checkpoint_path, self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        # Category names (loaded from DeepFashion annotations at init)
        self._category_names: list[str] | None = None

    def set_category_names(self, names: list[str]):
        """Optionally inject DeepFashion category names for human-readable output."""
        self._category_names = names

    # ── Color detection ───────────────────────────────────────────────────────

    def _detect_color_rgb(self, image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]) -> str:
        """
        Rule-based color detection using foreground pixel averaging.
        Filters out dark backgrounds (all channels < 30) and white/light
        backgrounds (all channels > 220) before computing average RGB.
        Works correctly with both black-background and white-background
        segmented garment images from Person 1's pipeline.
        """
        # Convert to PIL
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, torch.Tensor):
            # Assumes normalised tensor — denormalise first
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            img_t = image.cpu() * std + mean
            img_t = (img_t.clamp(0, 1) * 255).byte()
            pil_img = Image.fromarray(img_t.permute(1, 2, 0).numpy())
        else:
            pil_img = image.convert("RGB")

        arr = np.array(pil_img).reshape(-1, 3).astype(float)

        # Filter out background pixels
        mask = (arr.max(axis=1) > 30) & (arr.min(axis=1) < 220)
        arr_fg = arr[mask] if mask.sum() >= 100 else arr

        r, g, b = arr_fg.mean(axis=0)

        # Color rules
        if max(r, g, b) < 80:
            return "black"
        if min(r, g, b) > 200:
            return "white"
        if max(r, g, b) - min(r, g, b) < 20:
            return "black" if max(r, g, b) < 120 else "gray"
        if b > r and b > g and b > 80:
            return "blue"
        if r > g and r > b:
            return "red" if r > 150 else "brown"
        if g > r and g > b:
            return "green"
        if b > r and b > g:
            return "blue"
        if r > 150 and g > 150 and b < 100:
            return "yellow"
        if r > 150 and b > 150 and g < 100:
            return "purple"
        if r > 180 and g > 100 and b < 80:
            return "orange"
        if r > 200 and g > 150 and b > 150:
            return "pink"
        return "multicolor"

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
    ) -> dict:
        """
        Run inference on a single garment image.

        Args:
            image: one of:
              - str / Path  → loaded from disk (must be 224x224 crop from Person 1)
              - PIL.Image
              - np.ndarray  (HxWx3, uint8)
              - torch.Tensor (3x224x224, already normalised)

        Returns:
            dict with keys: category, category_idx, color, pattern, style,
                            embedding (list[float], len=768),
                            top5_categories (list of (name, prob) tuples)
        """
        # Detect color from raw image before normalisation
        color = self._detect_color_rgb(image)

        tensor = self._to_tensor(image)               # (1, 3, 224, 224)
        tensor = tensor.to(self.device)

        outputs = self.model(tensor)

        # ── Category ──────────────────────────────────────────────────────────
        cat_logits  = outputs["logits_category"]      # (1, 50)
        cat_probs   = F.softmax(cat_logits, dim=-1)[0]
        cat_idx     = int(cat_probs.argmax())
        cat_name    = (self._category_names[cat_idx]
                       if self._category_names else str(cat_idx))

        top5_vals, top5_ids = cat_probs.topk(5)
        top5 = [
            (self._category_names[i.item()] if self._category_names else str(i.item()),
             round(float(p), 4))
            for i, p in zip(top5_ids, top5_vals)
        ]

        # ── Attributes ────────────────────────────────────────────────────────
        pattern = PATTERN_LABELS[int(outputs["logits_pattern"].argmax(-1))]
        style   = STYLE_LABELS  [int(outputs["logits_style"]  .argmax(-1))]

        # ── Embedding ─────────────────────────────────────────────────────────
        embedding = outputs["embedding"][0].cpu().numpy().tolist()

        return {
            "category":         cat_name,
            "category_idx":     cat_idx,
            "color":            color,       # rule-based RGB detection
            "pattern":          pattern,
            "style":            style,
            "embedding":        embedding,   # 768-dim, L2-normalised
            "top5_categories":  top5,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        images: list[Union[str, Path, Image.Image]],
        batch_size: int = 32,
    ) -> list[dict]:
        """Run predict() on a list of images efficiently."""
        results = []
        for i in range(0, len(images), batch_size):
            chunk = images[i: i + batch_size]
            for img in chunk:
                results.append(self.predict(img))
        return results

    def _to_tensor(self, image) -> torch.Tensor:
        """Normalise any image input format → (1, 3, 224, 224) tensor."""
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return _EVAL_TRANSFORM(image).unsqueeze(0)
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    