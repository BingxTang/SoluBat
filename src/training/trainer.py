from __future__ import annotations

"""
trainer.py – Training and validation loop driven by cfg

- Hyperparameters from cfg only
- Uses new RunManager.log_metrics with keyword args
"""
import sys, os, time
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import pynvml
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import classification_metrics
from src.data.dataset import protein_collate_fn
from src.utils.run_manager import RunManager
from src.utils.utils import get_device, find_latest_ckpt

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_ds,
        val_ds,
        cfg: Dict[str, Any],
        run_mgr: RunManager,
        resume: bool = False,
    ):
        self.cfg = cfg
        self.rm = run_mgr
        self.device = get_device()

        self.model = model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            mode="max",
            factor=cfg.get("lr_factor", 0.5),
            patience=cfg.get("lr_patience", 2),
            verbose=True
        )
        self.scaler = GradScaler()
        self.crit = torch.nn.CrossEntropyLoss()

        self.early_stop_patience = cfg["early_stop_patience"]
        self.no_improve_epochs = 0

        self.rm.logger.info("[Data] building DataLoader...")
        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            collate_fn=protein_collate_fn,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            collate_fn=protein_collate_fn,
            pin_memory=True
        )
        self.rm.logger.info(f"[Data] ready: Train={len(train_ds)}  Val={len(val_ds)}")

        self.start_epoch = 0
        if resume:
            latest = find_latest_ckpt(self.rm.run_dir / "checkpoints")
            if latest:
                self.start_epoch = self._load_ckpt(latest) + 1
                self.rm.logger.info(f"[Resume] from {latest}, next epoch {self.start_epoch}")

        self._t0_wall = datetime.datetime.now()
        pynvml.nvmlInit()
        self._nv_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._energy_j = 0.0

        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        self.rm.log_metrics(step=0, phase="meta", metrics={"param_M": total_params})

    def _step(self, batch: dict, train: bool = True):
        pssm = batch["pssm"].to(self.device)
        seq = batch["seq"].to(self.device)
        pp_glb = batch["pp_glb"].to(self.device)
        label = batch["label"].to(self.device)

        abl = self.cfg["abl_mode"]
        if "seq" not in abl:
            seq = torch.zeros_like(seq)
        if "pssm" not in abl:
            pssm = torch.zeros_like(pssm)
        if "pp" not in abl:
            pp_glb = torch.zeros_like(pp_glb)

        if self.cfg.get("pssm_norm", True) and "pssm" in abl:
            mu = pssm.mean(dim=1, keepdim=True)
            std = pssm.std(dim=1, keepdim=True).clamp_(1e-5)
            pssm = torch.clamp((pssm - mu) / std, -3, 3)

        with autocast():
            logits = self.model(pssm, seq, pp_glb)
            loss = self.crit(logits, label)
            preds = logits.argmax(dim=1)
        return loss, preds, label

    def _save_ckpt(self, epoch: int, metrics: Dict[str, Any]):
        state = {
            "model": (
                self.model.module.state_dict()
                if isinstance(self.model, torch.nn.DataParallel)
                else self.model.state_dict()
            ),
            "optim": self.optim.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "cfg": self.cfg,
        }
        self.rm.save_ckpt(f"epoch{epoch}.pt", state)

    def _load_ckpt(self, path: Path) -> int:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optim"])
        return ckpt["epoch"]

    def fit(self):
        best_acc = 0.0
        for epoch in range(self.start_epoch, self.cfg["max_epochs"]):
            t0 = time.time()

            # Training
            self.model.train()
            tloss, t_true, t_pred = 0.0, [], []
            for batch in tqdm(self.train_loader, desc=f"train {epoch}", ncols=100):
                batch_t0 = time.time()
                self.optim.zero_grad(set_to_none=True)
                loss, p, y = self._step(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

                batch_time = time.time() - batch_t0
                power_w = pynvml.nvmlDeviceGetPowerUsage(self._nv_handle) / 1000
                self._energy_j += power_w * batch_time

                tloss += loss.item() * len(y)
                t_pred.extend(p.cpu().numpy())
                t_true.extend(y.cpu().numpy())

            tr_metrics = classification_metrics(t_true, t_pred)
            tr_metrics["loss"] = tloss / len(self.train_loader.dataset)
            self.rm.log_metrics(step=epoch, phase="train", metrics=tr_metrics)

            # Validation
            self.model.eval()
            vloss, v_true, v_pred = 0.0, [], []
            with torch.no_grad():
                for batch in self.val_loader:
                    loss, p, y = self._step(batch, train=False)
                    vloss += loss.item() * len(y)
                    v_pred.extend(p.cpu().numpy())
                    v_true.extend(y.cpu().numpy())
            val_metrics = classification_metrics(v_true, v_pred)
            val_metrics["loss"] = vloss / len(self.val_loader.dataset)
            self.rm.log_metrics(step=epoch, phase="val", metrics=val_metrics)

            self.scheduler.step(val_metrics["accuracy"])
            improved = val_metrics["accuracy"] > best_acc
            best_acc = max(best_acc, val_metrics["accuracy"])
            self.no_improve_epochs = 0 if improved else self.no_improve_epochs + 1
            if improved:
                self.rm.save_ckpt("best_model.pt", self.model.state_dict())
                self.rm.logger.info(f"▲ New best ACC {best_acc:.4f}")
            else:
                self.rm.logger.info(f"■ No gain ({self.no_improve_epochs}/{self.early_stop_patience})")

            self._save_ckpt(epoch, val_metrics)
            self.rm.logger.info(
                f"[{epoch:03d}] TrAcc={tr_metrics['accuracy']:.4f} "
                f"VaAcc={val_metrics['accuracy']:.4f}  Δt={time.time()-t0:.1f}s"
            )

            if self.no_improve_epochs >= self.early_stop_patience:
                self.rm.logger.info("[EarlyStopping] stopping training.")
                break
            torch.cuda.empty_cache()

            # Log cost metrics
            delta = datetime.datetime.now() - self._t0_wall
            gpu_h = delta.total_seconds() / 3600
            kwh = self._energy_j / 3.6e6
            usd = kwh * self.cfg.get("usd_per_kwh", 0.207)
            co2 = kwh * self.cfg.get("co2_factor", 0.445)
            cost_metrics = {"gpu_h": gpu_h, "kwh": kwh, "usd": usd, "kgco2": co2}
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            cost_metrics["peak_mem_gb"] = peak_mem
            self.rm.log_metrics(step=0, phase="cost", metrics=cost_metrics)
            self.rm.logger.info(
                f"[Cost] GPU-h={gpu_h:.2f}  kWh={kwh:.2f}  USD={usd:.2f}  CO₂={co2:.3f} kg"
            )
