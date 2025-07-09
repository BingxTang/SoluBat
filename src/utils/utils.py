"""
utils.py – Utility helper functions
-------------------------------------------------
* get_device(): returns available device (cuda/mps/cpu)
* find_latest_ckpt(): finds the latest epoch*.pt checkpoint in a directory
"""

import re
from pathlib import Path
from typing import Optional

import torch

# Device selection
def get_device() -> torch.device:
    """Return the best available device: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Checkpoint utilities
_ckpt_pat = re.compile(r"epoch(\d+)\.pt$")

def find_latest_ckpt(ckpt_dir: Path) -> Optional[Path]:
    """Return the checkpoint file with the highest epoch number in the directory."""
    ckpts = []
    for p in ckpt_dir.glob("epoch*.pt"):
        m = _ckpt_pat.search(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0], reverse=True)
    return ckpts[0][1]
