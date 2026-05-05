"""
tests/test_all.py

Unit + integration tests. Run with: pytest tests/test_all.py -v

Tests are designed to work WITHOUT real datasets and WITHOUT internet access.
All ViT models are constructed from a local config (no HuggingFace download).
No GPU required.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ──────────────────────────────────────────────────────────────────────────────
# Offline ViT factory
# Build a tiny ViT locally so tests never hit huggingface.co
# ──────────────────────────────────────────────────────────────────────────────
from transformers import ViTConfig, ViTModel

def _make_offline_vit() -> ViTModel:
    """
    Constructs a tiny ViT-B/16-compatible model from a local config dict.
    hidden_size=768 to match real ViT-B/16 CLS dim; fewer layers/heads for speed.
    """
    cfg = ViTConfig(
        hidden_size=768,
        num_hidden_layers=2,          # real=12; 2 is enough for shape tests
        num_attention_heads=12,
        intermediate_size=1024,
        image_size=224,
        patch_size=16,
        num_channels=3,
    )
    return ViTModel(cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def cfg():
    """Minimal config for CPU tests."""
    return {
        "classifier": {
            "backbone":              "google/vit-base-patch16-224-in21k",
            "image_size":            224,
            "embed_dim":             768,
            "num_categories":        10,
            "num_colors":            12,
            "num_patterns":          8,
            "num_styles":            6,
            "batch_size":            2,
            "num_epochs":            1,
            "warmup_epochs":         0,
            "head_lr":               2e-4,
            "backbone_lr":           1e-5,
            "backbone_unfreeze_epoch": 1,
            "weight_decay":          1e-4,
            "label_smoothing":       0.1,
            "grad_clip":             1.0,
            "random_erasing_p":      0.0,
            "mixup_alpha":           0.0,
            "color_jitter":          [0.2, 0.2, 0.2, 0.05],
            "scheduler":             "cosine",
            "min_lr":                1e-7,
            "loss_weights":          {"category": 1.0, "color": 0.5,
                                     "pattern": 0.5, "style": 0.5},
            "topk":                  [1, 5],
        },
        "compatibility": {
            "embed_dim":          768,
            "proj_dim":           32,
            "num_garment_types":  11,
            "batch_size":         4,
            "num_epochs":         1,
            "lr":                 1e-3,
            "weight_decay":       1e-4,
            "loss":               "bpr",
            "triplet_margin":     0.2,
            "num_negatives":      2,
            "fitb_num_candidates": 4,
            "scheduler":          "cosine",
            "min_lr":             1e-6,
        },
        "paths": {
            "checkpoints": "/tmp/person2_test_ckpts",
            "logs":        "/tmp/person2_test_logs",
        },
        "seed":        42,
        "num_workers": 0,
        "device":      "cpu",
        "fp16":        False,
    }


@pytest.fixture(scope="session")
def dummy_images():
    """Batch of 2 random 224×224 images."""
    return torch.randn(2, 3, 224, 224)


# ──────────────────────────────────────────────────────────────────────────────
# DeepFashion annotation parsing
# ──────────────────────────────────────────────────────────────────────────────

class TestDeepFashionAnnotations:

    def test_keyword_match(self):
        from data.deepfashion_dataset import _keyword_match, COLOR_KEYWORDS
        assert _keyword_match("solid_black_color", COLOR_KEYWORDS) == 0  # black
        assert _keyword_match("floral_pattern",    COLOR_KEYWORDS) is None

    def test_label_vocabularies(self):
        from data.deepfashion_dataset import COLOR_LABELS, PATTERN_LABELS, STYLE_LABELS
        assert len(COLOR_LABELS)   == 12
        assert len(PATTERN_LABELS) == 8
        assert len(STYLE_LABELS)   == 6

    def test_transforms_shape(self):
        from data.deepfashion_dataset import build_transforms
        cfg_mini = {
            "classifier": {
                "image_size": 224, "color_jitter": [0.2, 0.2, 0.2, 0.05],
                "random_erasing_p": 0.0,
            }
        }
        from PIL import Image
        img = Image.new("RGB", (320, 480), (120, 80, 200))
        for split in ("train", "val", "test"):
            tfm = build_transforms(cfg_mini, split)
            t   = tfm(img)
            assert t.shape == (3, 224, 224), f"Wrong shape for split={split}: {t.shape}"


# ──────────────────────────────────────────────────────────────────────────────
# GarmentClassifier model
# ──────────────────────────────────────────────────────────────────────────────

class TestGarmentClassifier:

    @pytest.fixture(scope="class")
    def model(self):
        from classifier.model import GarmentClassifier
        m = GarmentClassifier.__new__(GarmentClassifier)
        torch.nn.Module.__init__(m)
        import torch.nn as nn
        m.embed_dim = 768
        m.vit     = _make_offline_vit()
        m.dropout = nn.Dropout(0.1)
        m.head_category = nn.Linear(768, 10)
        m.head_color    = nn.Linear(768, 12)
        m.head_pattern  = nn.Linear(768, 8)
        m.head_style    = nn.Linear(768, 6)
        return m

    def test_output_shapes(self, model, dummy_images):
        model.eval()
        with torch.no_grad():
            out = model(dummy_images)
        assert out["embedding"].shape         == (2, 768)
        assert out["logits_category"].shape   == (2, 10)
        assert out["logits_color"].shape      == (2, 12)
        assert out["logits_pattern"].shape    == (2, 8)
        assert out["logits_style"].shape      == (2, 6)

    def test_embedding_normalized(self, model, dummy_images):
        model.eval()
        with torch.no_grad():
            out = model(dummy_images)
        norms = out["embedding"].norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_freeze_unfreeze(self, model):
        model.freeze_backbone()
        frozen = sum(1 for p in model.vit.parameters() if not p.requires_grad)
        total  = sum(1 for p in model.vit.parameters())
        assert frozen == total, "All backbone params should be frozen"

        model.unfreeze_last_n_blocks(n=4)
        unfrozen = sum(1 for p in model.vit.parameters() if p.requires_grad)
        assert unfrozen > 0, "Some backbone params should be unfrozen after Stage 2"

    def test_head_params_count(self, model):
        head_params = list(model.get_head_params())
        assert len(head_params) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Multi-task loss
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiTaskLoss:

    def test_loss_finite(self, cfg):
        from classifier.model import GarmentClassifier, MultiTaskLoss
        import torch.nn as nn

        # Build model offline (no HF download)
        model = GarmentClassifier.__new__(GarmentClassifier)
        torch.nn.Module.__init__(model)
        model.embed_dim    = 768
        model.vit          = _make_offline_vit()
        model.dropout      = nn.Dropout(0.1)
        model.head_category = nn.Linear(768, 10)
        model.head_color    = nn.Linear(768, 12)
        model.head_pattern  = nn.Linear(768, 8)
        model.head_style    = nn.Linear(768, 6)

        criterion = MultiTaskLoss(
            loss_weights=cfg["classifier"]["loss_weights"],
            label_smoothing=0.1,
        )
        imgs = torch.randn(2, 3, 224, 224)
        targets = {
            "category": torch.randint(0, 10, (2,)),
            "color":    torch.randint(0, 12, (2,)),
            "pattern":  torch.randint(0,  8, (2,)),
            "style":    torch.randint(0,  6, (2,)),
        }
        with torch.no_grad():
            out = model(imgs)
        loss_dict = criterion(out, targets)
        assert torch.isfinite(loss_dict["loss"])
        assert loss_dict["loss"].item() > 0


# ──────────────────────────────────────────────────────────────────────────────
# Compatibility model
# ──────────────────────────────────────────────────────────────────────────────

class TestCompatibilityModel:

    @pytest.fixture(scope="class")
    def model(self):
        from compatibility.model import CompatibilityModel
        return CompatibilityModel(visual_dim=768, proj_dim=32, num_types=11)

    def test_embed_shape(self, model):
        visuals  = torch.randn(4, 768)
        types    = torch.randint(0, 11, (4,))
        embs     = model.embed(visuals, types)
        assert embs.shape == (4, 32)

    def test_embed_normalized(self, model):
        visuals = torch.randn(4, 768)
        types   = torch.randint(0, 11, (4,))
        embs    = model.embed(visuals, types)
        norms   = embs.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_score_range(self, model):
        va = torch.randn(4, 768)
        ta = torch.randint(0, 11, (4,))
        vb = torch.randn(4, 768)
        tb = torch.randint(0, 11, (4,))
        scores = model.score(va, ta, vb, tb)
        assert scores.shape == (4,)
        assert (scores >= -1.01).all() and (scores <= 1.01).all()


# ──────────────────────────────────────────────────────────────────────────────
# BPR and Triplet losses
# ──────────────────────────────────────────────────────────────────────────────

class TestLosses:

    def test_bpr_loss(self):
        from compatibility.model import BPRLoss
        loss_fn = BPRLoss()
        s_pos = torch.tensor([0.8, 0.7, 0.9])
        s_neg = torch.tensor([0.2, 0.1, 0.3])
        loss  = loss_fn(s_pos, s_neg)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_triplet_loss(self):
        from compatibility.model import TripletLoss
        loss_fn = TripletLoss(margin=0.2)
        B, D = 4, 32
        anc = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        pos = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        neg = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        loss = loss_fn(anc, pos, neg)
        assert torch.isfinite(loss)
        assert loss.item() >= 0


# ──────────────────────────────────────────────────────────────────────────────
# Polyvore type mapping
# ──────────────────────────────────────────────────────────────────────────────

class TestPolyvoreTypes:

    def test_type_mapping(self):
        from data.polyvore_dataset import category_to_type, NUM_TYPES
        assert category_to_type("tops")     == 0
        assert category_to_type("jeans")    == 1
        assert category_to_type("sneakers") == 2
        assert category_to_type("handbags") == 3
        assert category_to_type("jackets")  == 4
        assert category_to_type("dresses")  == 5
        assert category_to_type("unknown_thing") == 10   # fallback: other

    def test_num_types(self):
        from data.polyvore_dataset import NUM_TYPES
        assert NUM_TYPES == 11


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

class TestUtils:

    def test_topk_accuracy(self):
        from utils.common import topk_accuracy
        logits = torch.tensor([[0.1, 0.9, 0.0],
                               [0.8, 0.1, 0.1]])
        labels = torch.tensor([1, 0])   # both should be top-1 correct
        top1, top5_or_3 = topk_accuracy(logits, labels, topk=(1, 3))
        assert top1 == 100.0

    def test_average_meter(self):
        from utils.common import AverageMeter
        m = AverageMeter()
        m.update(10.0, n=2)
        m.update(20.0, n=2)
        assert m.avg == 15.0
        assert m.count == 4

    def test_save_load_checkpoint(self, cfg):
        from utils.common import save_checkpoint, load_checkpoint
        state = {"epoch": 5, "val_top1": 72.3}
        path  = "/tmp/test_ckpt.pth"
        save_checkpoint(state, path)
        loaded = load_checkpoint(path, torch.device("cpu"))
        assert loaded["epoch"]   == 5
        assert loaded["val_top1"] == 72.3


# ──────────────────────────────────────────────────────────────────────────────
# ClassifierService (inference API)
# ──────────────────────────────────────────────────────────────────────────────

class TestClassifierService:

    def test_to_tensor_from_pil(self):
        """Service correctly normalises a PIL image."""
        from classifier_service import ClassifierService, _EVAL_TRANSFORM
        from PIL import Image
        img = Image.new("RGB", (224, 224), (128, 64, 200))
        t   = _EVAL_TRANSFORM(img)
        assert t.shape == (3, 224, 224)

    def test_predict_dict_keys(self, cfg, device):
        """Full forward pass through service returns expected keys."""
        from classifier.model import GarmentClassifier
        from classifier_service import ClassifierService
        from PIL import Image
        import torch.nn as nn
        import yaml

        # Build model offline
        model = GarmentClassifier.__new__(GarmentClassifier)
        torch.nn.Module.__init__(model)
        model.embed_dim     = 768
        model.vit           = _make_offline_vit()
        model.dropout       = nn.Dropout(0.1)
        model.head_category = nn.Linear(768, cfg["classifier"]["num_categories"])
        model.head_color    = nn.Linear(768, 12)
        model.head_pattern  = nn.Linear(768, 8)
        model.head_style    = nn.Linear(768, 6)

        ckpt_path = "/tmp/test_classifier_svc.pth"
        torch.save({"state_dict": model.state_dict()}, ckpt_path)

        cfg_path = "/tmp/test_svc_cfg.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        # Patch ClassifierService to skip from_pretrained
        svc = ClassifierService.__new__(ClassifierService)
        svc.cfg    = cfg
        svc.device = device
        svc.model  = model.to(device)
        svc.model.eval()
        svc._category_names = None

        img = Image.new("RGB", (224, 224), (100, 150, 200))
        out = svc.predict(img)

        assert "category"        in out
        assert "color"           in out
        assert "pattern"         in out
        assert "style"           in out
        assert "embedding"       in out
        assert "top5_categories" in out
        assert len(out["embedding"]) == 768


# ──────────────────────────────────────────────────────────────────────────────
# CompatibilityService (inference API)
# ──────────────────────────────────────────────────────────────────────────────

class TestCompatibilityService:

    def test_rank_compatibility(self, cfg, device):
        from compatibility.model import CompatibilityModel
        from compatibility_service import CompatibilityService
        import yaml

        model = CompatibilityModel(
            visual_dim=768, proj_dim=32, num_types=11
        )
        ckpt_path = "/tmp/test_compat_svc.pth"
        torch.save({"state_dict": model.state_dict()}, ckpt_path)

        cfg_path = "/tmp/test_compat_cfg.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        svc = CompatibilityService(ckpt_path, config_path=cfg_path, device=device)

        dummy_emb = np.random.randn(768).tolist()
        query = {"embedding": dummy_emb, "category": "tops", "item_id": "q0"}
        wardrobe = [
            {"embedding": np.random.randn(768).tolist(),
             "category":  cat, "item_id": f"w{i}"}
            for i, cat in enumerate(["bottoms", "shoes", "bags", "outerwear"])
        ]

        ranked = svc.rank_compatibility(query, wardrobe, top_k=3)
        assert len(ranked) == 3
        assert "score" in ranked[0]
        assert "rank"  in ranked[0]
        # Verify descending order
        scores = [r["score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_score_pair(self, cfg, device):
        from compatibility.model import CompatibilityModel
        from compatibility_service import CompatibilityService
        import yaml

        model     = CompatibilityModel(visual_dim=768, proj_dim=32, num_types=11)
        ckpt_path = "/tmp/test_pair_svc.pth"
        torch.save({"state_dict": model.state_dict()}, ckpt_path)

        cfg_path = "/tmp/test_pair_cfg.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        svc = CompatibilityService(ckpt_path, config_path=cfg_path, device=device)
        a = {"embedding": np.random.randn(768).tolist(), "category": "tops"}
        b = {"embedding": np.random.randn(768).tolist(), "category": "bottoms"}
        score = svc.score_pair(a, b)
        assert -1.01 <= score <= 1.01
