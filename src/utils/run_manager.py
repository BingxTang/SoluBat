from __future__ import annotations

"""
run_manager.py – Unified logging, TensorBoard, CSV, and JSONL manager

- Uses matplotlib Agg backend
- CSV writing excludes 'conf_mat' and 'report' fields
"""
import matplotlib
matplotlib.use("Agg")

import atexit
import csv
import json
import logging
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict

import numpy as np
import torch
from rich.logging import RichHandler
from torch.utils.tensorboard import SummaryWriter

__all__ = ["RunManager"]

class _JsonlWriter:
    def __init__(self, path: Path):
        self.fp = path.open("a", encoding="utf-8")
    def write(self, obj: dict):
        json.dump(obj, self.fp, ensure_ascii=False)
        self.fp.write("\n")
        self.fp.flush()
    def close(self):
        self.fp.close()

class RunManager:
    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        fold_idx: int | None = None,
        redirect_print: bool = True,
        root_dir: Path | None = None
    ):
        self.cfg = cfg
        self.fold_idx = fold_idx if fold_idx is not None else cfg.get("fold_idx")

        ts = time.strftime("%Y%m%d_%H%M%S")
        base = "runs"
        name = f"{ts}_{cfg.get('run_name','exp')}"
        self.root_dir = (Path(base) / name if root_dir is None else root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        if cfg.get("train_flow") == "cv":
            self.run_dir = self.root_dir / f"fold{self.fold_idx}"
        else:
            self.run_dir = self.root_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        cfg_path = self.run_dir / "config.json"
        if not cfg_path.exists():
            cfg_path.write_text(
                json.dumps(cfg, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

        self.logger = self._init_logger(redirect_print)
        self.logger.info(f"Run directory: {self.run_dir}")

        self.tb = SummaryWriter(log_dir=str(self.run_dir / "tflogs"))
        self.jsonl = _JsonlWriter(self.run_dir / "metrics.jsonl")
        self._csv_fp = (self.run_dir / "metrics.csv").open(
            "a", newline="", encoding="utf-8"
        )
        self._csv_writer: csv.DictWriter | None = None
        atexit.register(self.close)

    def _init_logger(self, redirect_print: bool) -> logging.Logger:
        fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
        handlers = [
            RichHandler(rich_tracebacks=True, markup=True, show_path=False),
            logging.FileHandler(self.run_dir / "train.log", encoding="utf-8"),
        ]
        logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
        name = f"run.fold{self.fold_idx}" if self.fold_idx is not None else "run.single"
        logger = logging.getLogger(name)
        if redirect_print:
            import builtins
            builtins.print = lambda *a, **k: logger.info(
                " ".join(str(x) for x in a)
            )
        return logger

    def save_ckpt(self, name: str, state_dict: Any):
        torch.save(state_dict, self.run_dir / "checkpoints" / name)

    def log_metrics(
        self,
        *,
        step: int,
        phase: str,
        metrics: dict[str, float | int | np.ndarray | str]
    ):
        # TensorBoard scalars
        for k, v in metrics.items():
            if k in {"conf_mat", "report"}:
                continue
            self.tb.add_scalar(f"{phase}/{k}", float(v), step)

        # JSONL output
        safe_json = {"step": step, "phase": phase}
        safe_json.update({k: v for k, v in metrics.items() if k != "conf_mat"})
        self.jsonl.write(safe_json)

        # CSV output
        safe_csv = {k: v for k, v in safe_json.items() if k != "report"}
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_fp, fieldnames=list(safe_csv.keys())
            )
            self._csv_writer.writeheader()
        row = {k: safe_csv.get(k, "") for k in self._csv_writer.fieldnames}
        self._csv_writer.writerow(row)
        self._csv_fp.flush()

        # Confusion matrix image
        cm = metrics.get("conf_mat")
        if cm is not None:
            try:
                from sklearn.metrics import ConfusionMatrixDisplay
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6,6))
                ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", colorbar=False)
                fig.tight_layout()
                fig.savefig(
                    self.run_dir / f"confusion_matrix_{phase}_{step}.png",
                    dpi=150
                )
                plt.close(fig)
            except ImportError:
                self.logger.warning(
                    "scikit-learn not installed, cannot plot confusion matrix."
                )

    def finish_fold(self, best_metrics: dict[str, float | int]) -> None:
        if self.cfg.get("train_flow") != "cv":
            return
        summary_path = self.root_dir / "cv_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            summary = {"folds": {}}
        summary["folds"][str(self.fold_idx)] = best_metrics
        if len(summary["folds"]) == self.cfg["num_folds"]:
            keys = best_metrics.keys()
            summary["mean"] = {k: mean(summary["folds"][f][k] for f in summary["folds"]) for k in keys}
            summary["std"]  = {k: stdev(summary["folds"][f][k] for f in summary["folds"]) for k in keys}
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    def close(self) -> None:
        try:
            self.tb.close()
        except Exception:
            pass
        try:
            self.jsonl.close()
        except Exception:
            pass
        try:
            self._csv_fp.close()
        except Exception:
            pass
