# utils/seed.py
# Global seeding & (optional) deterministic backend setup for fair comparisons across runs.
from __future__ import annotations
import os, random, numpy as np
from typing import Optional

def set_global_seed(seed: int = 42, deterministic: bool = False):
    """
    Set seeds for Python, NumPy, and PyTorch (if available).
    In deterministic mode, enable cudnn deterministic flags (slower but repeatable).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False     # type: ignore
    except Exception:
        pass

def seed_from_env(env_var: str = "SEED", default: int = 42, deterministic: bool = False) -> int:
    try:
        s = int(os.environ.get(env_var, default))
    except Exception:
        s = default
    set_global_seed(s, deterministic=deterministic)
    return s
