#!/usr/bin/env python
# -------------------------------- eval_pro_tsne.py --------------------------------

from __future__ import annotations
import sys, os, argparse, json, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, brier_score_loss
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.default_config import get_cfg
from src.data.dataset import ProteinDataset, protein_collate_fn
from src.models.model import SoluBat

# ---------------- utils ----------------

VOCAB_LIST = list("ACDEFGHIKLMNPQRSTVWY")
IDX2AA = {i+1: aa for i, aa in enumerate(VOCAB_LIST)}

def decode_seq(seq_tensor: torch.Tensor) -> str:
    return "".join(IDX2AA.get(idx, "X") for idx in seq_tensor.cpu().numpy() if idx > 0)

def normalize_pssm(pssm: torch.Tensor) -> torch.Tensor:
    mu = pssm.mean(dim=1, keepdim=True)
    std = pssm.std(dim=1, keepdim=True).clamp_(1e-5)
    return torch.clamp((pssm - mu) / std, -3, 3)

# ---------------- main eval ----------------

def evaluate(cfg: dict, ckpt_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.get("out_dir", "eval_out")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> All evaluation outputs directory: {out_dir}\n")

    test_ds = ProteinDataset(cfg["test_dir"], cfg)
    loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                        shuffle=False, num_workers=cfg["num_workers"],
                        collate_fn=protein_collate_fn, pin_memory=True)

    model = SoluBat(seq_vocab_size=21, cfg=cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()
    print(model, "\n")

    # ===============================================================================

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(">>> Model parameter statistics")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}\n")

    # ===============================================================================

    preds, labels, logits_all, pooled_all, seq_cache = [], [], [], [], []
    abl = cfg.get("abl_mode", "seq+pssm+pp")
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            pssm = batch["pssm"].to(device) if "pssm" in abl else torch.zeros_like(batch["pssm"]).to(device)
            pp = batch["pp_glb"].to(device) if "pp" in abl else torch.zeros_like(batch["pp_glb"]).to(device)

            if cfg.get("pssm_norm", True) and "pssm" in abl:
                pssm = normalize_pssm(pssm)

            out = model(pssm, seq, pp, return_res_logits=True)
            logits, _ = out if isinstance(out, tuple) else (out, None)

            _, _, h_enc = model.encoder(seq, pssm)
            pooled = h_enc.mean(1)

            pooled_all.append(pooled.cpu().numpy())
            logits_all.append(logits.cpu())
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(batch["label"].numpy())
            seq_cache.extend([decode_seq(s) for s in batch["seq"]])

    pooled_all = np.concatenate(pooled_all, 0)
    logits_all = torch.cat(logits_all)
    labels_arr = np.array(labels)
    num_classes = logits_all.size(1)
    probs = logits_all.softmax(1).numpy()

    acc = accuracy_score(labels_arr, preds)
    prec = precision_score(labels_arr, preds, average="macro", zero_division=0)
    rec = recall_score(labels_arr, preds, average="macro", zero_division=0)
    f1 = f1_score(labels_arr, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(labels_arr, preds)
    if num_classes == 2:
        auc_val = roc_auc_score(labels_arr, probs[:, 1])
    else:
        y_bin = label_binarize(labels_arr, classes=list(range(num_classes)))
        auc_val = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")

    metrics = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "mcc": mcc, "auc": auc_val}
    print(">>> Basic evaluation metrics")
    print(json.dumps(metrics, indent=2))

    print("\n>>> Additional statistics")

    kappa = cohen_kappa_score(labels_arr, preds)
    metrics['cohen_kappa'] = kappa
    print(f"Cohen's Kappa: {kappa:.4f}")

    balanced_acc = balanced_accuracy_score(labels_arr, preds)
    metrics['balanced_accuracy'] = balanced_acc
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    cm = confusion_matrix(labels_arr, preds)
    total_samples = np.sum(cm)
    specificities = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_samples - (tp + fn + fp)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities[f'specificity_class_{i}'] = specificity
        print(f"Specificity (Class {i}): {specificity:.4f}")
    metrics.update(specificities)

    if num_classes == 2:
        brier = brier_score_loss(labels_arr, probs[:, 1])
        metrics['brier_score'] = brier
        print(f"Brier Score: {brier:.4f}")

        pearson_corr, pearson_p = pearsonr(labels_arr, probs[:, 1])
        metrics['pearson_corr'] = pearson_corr
        metrics['pearson_p_value'] = pearson_p
        print(f"Pearson Correlation: {pearson_corr:.4f}, p-value: {pearson_p:.4g}")

        spearman_corr, spearman_p = spearmanr(labels_arr, probs[:, 1])
        metrics['spearman_corr'] = spearman_corr
        metrics['spearman_p_value'] = spearman_p
        print(f"Spearman Correlation: {spearman_corr:.4f}, p-value: {spearman_p:.4g}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n>>> All metrics saved to: {out_dir/'metrics.json'}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    cm_df = pd.DataFrame(cm,
                         index=[f"True_{i}" for i in range(num_classes)],
                         columns=[f"Pred_{i}" for i in range(num_classes)])
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    probs = logits_all.softmax(1).numpy()
    roc_curve_data = []
    pr_curve_data = []

    if num_classes == 2:
        fpr, tpr, roc_thresholds = roc_curve(labels_arr, probs[:, 1])
        precision, recall, pr_thresholds = precision_recall_curve(labels_arr, probs[:, 1])

        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "roc_curve.png", dpi=300)
        plt.close()

        plt.figure()
        plt.plot(recall, precision, label='PR Curve')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.tight_layout()
        plt.savefig(out_dir / "pr_curve.png", dpi=300)
        plt.close()

        min_len = min(len(fpr), len(tpr), len(roc_thresholds) + 1)
        roc_df = pd.DataFrame({
            "fpr": fpr[:min_len],
            "tpr": tpr[:min_len],
            "threshold": np.r_[roc_thresholds, np.nan][:min_len]
        })
        roc_df.to_csv(out_dir / "roc_curve.csv", index=False)

        pr_df = pd.DataFrame({
            "precision": precision,
            "recall": recall,
            "threshold": np.append(pr_thresholds, np.nan)
        })
        pr_df.to_csv(out_dir / "pr_curve.csv", index=False)

    else:
        y_bin = label_binarize(labels_arr, classes=list(range(num_classes)))
        for i in range(num_classes):
            fpr, tpr, roc_thresholds = roc_curve(y_bin[:, i], probs[:, i])
            precision, recall, pr_thresholds = precision_recall_curve(y_bin[:, i], probs[:, i])

            plt.figure()
            plt.plot(fpr, tpr, label=f'Class {i} AUC = {auc(fpr, tpr):.3f}')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve (Class {i})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"roc_curve_class{i}.png", dpi=300)
            plt.close()

            plt.figure()
            plt.plot(recall, precision, label=f'Class {i} PR Curve')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve (Class {i})")
            plt.tight_layout()
            plt.savefig(out_dir / f"pr_curve_class{i}.png", dpi=300)
            plt.close()

            roc_df = pd.DataFrame({
                "fpr": fpr,
                "tpr": tpr,
                "threshold": np.append(roc_thresholds, np.nan)
            })
            roc_df.to_csv(out_dir / f"roc_curve_class{i}.csv", index=False)

            pr_df = pd.DataFrame({
                "precision": precision,
                "recall": recall,
                "threshold": np.append(pr_thresholds, np.nan)
            })
            pr_df.to_csv(out_dir / f"pr_curve_class{i}.csv", index=False)

    perp = float(cfg.get("tsne_perp", 30))
    print(f"\n>>> Running t-SNE (perplexity={perp}) ...")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=0, init="pca")
    emb2d = tsne.fit_transform(pooled_all)

    plt.figure(figsize=(6,5))
    cmap = plt.get_cmap("tab10")
    for cls in np.unique(labels_arr):
        idx = labels_arr == cls
        plt.scatter(emb2d[idx,0], emb2d[idx,1], s=18, alpha=.8,
                    label=f"class {cls}", color=cmap(int(cls)%10))
    plt.legend(); plt.title("t-SNE of pooled features")
    plt.tight_layout()
    plt.savefig(out_dir / "tsne_embeddings.png", dpi=300)
    plt.close()
    print(f"t-SNE plot saved: {out_dir/'tsne_embeddings.png'}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate checkpoint + t-SNE")
    parser.add_argument("--ckpt", required=True, help="Checkpoint file (.pt)")
    parser.add_argument("--test_dir", required=True, help="Test data directory")
    parser.add_argument("--out_dir", default="eval_out", help="Output directory")
    parser.add_argument("--tsne_perp", type=float, default=30, help="t-SNE perplexity")
    args, unknown = parser.parse_known_args()

    cfg = get_cfg(unknown)
    cfg["test_dir"] = args.test_dir
    cfg["out_dir"]  = args.out_dir
    cfg["tsne_perp"] = args.tsne_perp

    evaluate(cfg, Path(args.ckpt).expanduser())
