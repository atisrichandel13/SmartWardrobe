"""
utils/common.py
Shared utilities: config, device, seeding, logging, metrics helpers.
"""

import os
import random
import logging
import yaml
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ─── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_nested(cfg: dict, *keys, default=None):
    """Safe nested dict access: get_nested(cfg, 'classifier', 'lr')"""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ─── Device ────────────────────────────────────────────────────────────────────

def get_device(cfg: dict) -> torch.device:
    spec = cfg.get("device", "auto")
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


# ─── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, log_dir: str | None = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


# ─── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


# ─── Metric helpers ─────────────────────────────────────────────────────────────

def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """
    Computes top-k accuracy for each k in topk.
    output: (N, C) logits, target: (N,) integer labels.
    Returns list of float accuracies in [0, 100].
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / batch_size * 100).item())
        return res


class AverageMeter:
    """Tracks running mean of a scalar (loss, accuracy, etc.)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.avg:.4f}"
