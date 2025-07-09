from __future__ import annotations
# --------------------------- train.py ---------------------------
"""
Standard training script (train/val/test version)
============================================================
- All hyperparameters are from default_config.py and can be overridden via CLI.
- Configuration snapshot printed for training, validation, and testing.
"""
import sys, os, json, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config.default_config import get_cfg
from src.data.dataset import ProteinDataset, protein_collate_fn
from src.models.model import SoluBat
from src.training.trainer import Trainer
from src.utils.run_manager import RunManager

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

def _evaluate(model: torch.nn.Module,
              loader: DataLoader,
              device: torch.device,
              cfg: dict) -> dict[str, float | int | list]:
    """Run inference on loader and return full metrics dictionary."""
    model.eval()
    preds, labels = [], []
    abl = cfg.get("abl_mode", "seq+pssm+pp")
    with torch.no_grad():
        for batch in loader:
            pssm = batch["pssm"].to(device) if "pssm" in abl else torch.zeros_like(batch["pssm"]).to(device)
            if cfg.get("pssm_norm", True) and "pssm" in abl:
                pssm = _normalize_pssm(pssm)

            pp_glb = batch["pp_glb"].to(device) if "pp" in abl else torch.zeros_like(batch["pp_glb"]).to(device)

            logits = model(pssm, batch["seq"].to(device), pp_glb)
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "conf_mat": cm,
        "report": classification_report(labels, preds, digits=4)
    }


def _normalize_pssm(pssm):
    mu = pssm.mean(dim=1, keepdim=True)
    std = pssm.std(dim=1, keepdim=True).clamp_(1e-5)
    return torch.clamp((pssm - mu) / std, -3, 3)

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Standard Trainer & Evaluator")
    parser.add_argument("--cfg-file", type=str, default=None, help="Path to JSON/YAML config file")
    args, unknown = parser.parse_known_args()

    # -------- Load configuration --------
    cfg = get_cfg(unknown)
    if args.cfg_file:
        cfg.update(json.load(open(args.cfg_file, "r", encoding="utf-8")))
    cfg.setdefault("run_name", "exp")

    rm = RunManager(cfg, redirect_print=True)
    rm.logger.info("===== Configuration Snapshot =====\n" +
                   json.dumps({k: cfg[k] for k in
                               ("train_dir", "val_dir", "test_dir", "abl_mode", "batch_size",
                                "num_workers") if k in cfg}, indent=2))

    # --------------------------- Data ---------------------------
    train_ds = ProteinDataset(cfg["train_dir"], cfg)
    val_ds = ProteinDataset(cfg["val_dir"], cfg)
    test_ds = ProteinDataset(cfg["test_dir"], cfg)
    rm.logger.info(f"Sizes  Train / Val / Test = {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    # --------------------------- Training ---------------------------
    model = SoluBat(seq_vocab_size=21, cfg=cfg)
    trainer = Trainer(model, train_ds, val_ds, cfg, rm)
    trainer.fit()  # best_model.pt saved under checkpoints/

    # --------------------------- Testing ---------------------------
    best_ckpt = rm.run_dir / "checkpoints/best_model.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
        rm.logger.info("Loaded best_model.pt for testing.")
    else:
        rm.logger.warning("best_model.pt not found; using latest weights for evaluation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], collate_fn=protein_collate_fn)
    metrics = _evaluate(model, test_loader, device, cfg)

    # log & print
    rm.log_metrics(step=0, phase="test", metrics=metrics)
    rm.logger.info("\n" + metrics["report"])
    rm.logger.info(
        f"[Test]  ACC={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
        f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")

    rm.finish_fold({k: float(v) for k, v in metrics.items()
                    if k not in {"conf_mat", "report"}})
    rm.close()
