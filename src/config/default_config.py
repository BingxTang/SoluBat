"""
default_config.py – Global unified configuration

Only keys actually used by code are retained.
CLI arguments (--key value) can override defaults via get_cfg().
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Default settings
default_config: Dict[str, Any] = {
    # Paths
    "data_root":      "src/data/Data",
    "fold_root_dir":  "src/data/Data/Datasets/NetSolP",
    "train_dir":      "src/data/Data/train",
    "val_dir":        "src/data/Data/val",
    "test_dir":       "src/data/Data/Datasets/NetSolP/test",

    # Cross-validation
    "num_folds":      5,
    "fold_idx":       1,
    "train_flow":     "single",
    "run_name":       "exp",

    # Data filtering
    "min_len":        0,
    "max_len":        2000,

    # Model parameters
    "pssm_feat_dim":  42,
    "seq_embed_dim":  256,
    "mamba_dim":      512,
    "pp_glb_dim":     30,

    # Training hyperparameters
    "batch_size":          8,
    "lr":                  7e-6,
    "lr_factor":           0.5,
    "lr_patience":         2,
    "num_workers":         4,
    "max_epochs":          45,
    "early_stop_patience": 15,
    "seed":                618,
    "class_weight":        "auto",

    # Miscellaneous
    "pssm_norm":          True,
    "abl_mode":           "seq-pssm-pp",
    "gate_alpha":         0.5,
    "gate_temperature":   2.0,
}


def get_cfg(extra_cli: list[str] | None = None) -> Dict[str, Any]:
    """
    Parse CLI overrides and return updated config dictionary.
    Pass remaining args from argparse.parse_known_args().
    """
    parser = argparse.ArgumentParser(add_help=False)
    for key, val in default_config.items():
        arg_type = type(val) if not isinstance(val, bool) else lambda x: x.lower() == "true"
        parser.add_argument(f"--{key}", default=val, type=arg_type)

    cli_args, _ = parser.parse_known_args(extra_cli)
    cfg = vars(cli_args)

    # Resolve paths
    for key in ("data_root", "fold_root_dir", "train_dir", "val_dir", "test_dir"):
        cfg[key] = str(Path(cfg[key]).resolve())

    return cfg
