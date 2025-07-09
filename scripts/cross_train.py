from __future__ import annotations
# ---------------------------  cross_train.py  ---------------------------

import sys, os, json, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from src.config.default_config import get_cfg
from src.data.dataset import ProteinDataset, protein_collate_fn
from src.models.model import SoluBat
from src.training.trainer import Trainer
from src.utils.run_manager import RunManager

# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

def _build_dataset(cfg: dict, fold_idx: int, *, train: bool):
    root = Path(cfg["fold_root_dir"])
    if train:
        parts = [root / f"train_fold{i}" for i in range(cfg["num_folds"]) if i != fold_idx]
        dss = [ProteinDataset(str(p), cfg) for p in parts]
        return ConcatDataset(dss)
    else:
        part = root / f"train_fold{fold_idx}"
        return ProteinDataset(str(part), cfg)

def _evaluate(model: torch.nn.Module,
              loader: DataLoader,
              device: torch.device,
              cfg: dict) -> dict[str, float | int | list]:
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

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _normalize_pssm(pssm):
    mu = pssm.mean(dim=1, keepdim=True)
    std = pssm.std(dim=1, keepdim=True).clamp_(1e-5)
    return torch.clamp((pssm - mu) / std, -3, 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cross-/Single-Fold Trainer & Evaluator")
    parser.add_argument("--cfg-file", type=str, default=None, help="Path to JSON/YAML config file")
    parser.add_argument("--fold-idx", type=int, default=6, help="Fold index: -1=all folds; others specify a single fold")
    args, unknown = parser.parse_known_args()

    # load configuration
    cfg = get_cfg(unknown)
    if args.cfg_file:
        cfg.update(json.load(open(args.cfg_file, "r", encoding="utf-8")))
    cfg["fold_idx"] = args.fold_idx
    cfg.setdefault("run_name", "exp")

    folds = range(cfg["num_folds"]) if args.fold_idx < 0 else [args.fold_idx]

    for f in folds:
        cfg_f = {**cfg, "fold_idx": f, "train_flow": ("cv" if args.fold_idx < 0 else "single")}
        rm = RunManager(cfg_f, fold_idx=f, redirect_print=True)

        rm.logger.info("===== Configuration Snapshot =====\n" +
                       json.dumps({k: cfg_f[k] for k in
                                   ("train_flow", "fold_idx", "num_folds",
                                    "train_dir", "val_dir", "test_dir",
                                    "fold_root_dir", "abl_mode")}, indent=2, ensure_ascii=False))

        train_ds = _build_dataset(cfg_f, f, train=True)
        val_ds   = _build_dataset(cfg_f, f, train=False)
        test_ds  = ProteinDataset(cfg_f["test_dir"], cfg_f)
        rm.logger.info(f"Sizes  Train / Val / Test = {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

        model = SoluBat(seq_vocab_size=21, cfg=cfg_f)
        trainer = Trainer(model, train_ds, val_ds, cfg_f, rm)
        trainer.fit()  # best_model.pt saved under checkpoints/

        best_ckpt = rm.run_dir / "checkpoints/best_model.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
            rm.logger.info("Loaded best_model.pt for testing.")
        else:
            rm.logger.warning("best_model.pt not found; using latest weights for evaluation.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg_f["batch_size"],
            shuffle=False,
            num_workers=cfg_f["num_workers"],
            collate_fn=protein_collate_fn
        )
        metrics = _evaluate(model, test_loader, device, cfg_f)

        # log & print
        rm.log_metrics(step=0, phase="test", metrics=metrics)
        rm.logger.info("\n" + metrics["report"])
        rm.logger.info(
            f"[Fold {f}]  ACC={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
            f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
        )

        rm.finish_fold({k: float(v) for k, v in metrics.items() if k not in {"conf_mat", "report"}})
        rm.close()
