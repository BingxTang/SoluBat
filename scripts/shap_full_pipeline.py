#!/usr/bin/env python
# ================================================================
# shap_full_pipeline_v2.py
# Explain three feature channels: SEQ-only (PoSHAP), PSSM-only, PP
# ================================================================
# Usage example:
# python shap_full_pipeline_v2.py \
#   --ckpt runs/20250612_120805_exp/checkpoints/best_model.pt \
#   --data_dir scripts/test \
#   --out_dir shap_out \
#   --shap_samples 300 \
#   --pp_names scripts/test/pp_names.json \
#   --pp_topk 15
# ================================================================
from __future__ import annotations
import argparse, json, random, pathlib, sys, os, warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score)
import shap
import seaborn as sns
import pandas as pd

# Internal dependencies
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.default_config import get_cfg
from src.data.dataset        import ProteinDataset, protein_collate_fn
from src.models.model        import SoluBat

warnings.filterwarnings("ignore", category=UserWarning)

# Utilities

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_to_len(x: torch.Tensor, tgt_len: int) -> torch.Tensor:
    """Pad tensor on sequence axis (dim=1) to target length."""
    if x.size(1) == tgt_len:
        return x
    pad_shape = list(x.shape)
    pad_shape[1] = tgt_len - x.size(1)
    pad = x.new_zeros(*pad_shape)
    return torch.cat([x, pad], dim=1)


def normalize_pssm(pssm: torch.Tensor) -> torch.Tensor:
    mu  = pssm.mean(dim=1, keepdim=True)
    std = pssm.std(dim=1, keepdim=True).clamp_(1e-5)
    return torch.clamp((pssm - mu) / std, -3, 3)

# Main pipeline

def main():
    parser = argparse.ArgumentParser("SEQ-only PoSHAP + PSSM-only PoSHAP + PP SHAP")
    parser.add_argument("--ckpt",      required=True, help="Checkpoint file (.pt)")
    parser.add_argument("--data_dir",  required=True, help="Test data directory")
    parser.add_argument("--out_dir",   default="shap_out", help="Output directory")
    parser.add_argument("--shap_samples", type=int, default=300, help="Number of SHAP samples")
    parser.add_argument("--pp_names",  default="", help="Path to PP names JSON")
    parser.add_argument("--pp_topk",   type=int, default=15, help="Top-K PP features")
    args, unknown = parser.parse_known_args()

    set_seed(42)
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(exist_ok=True)
    log = lambda m: print(f"[INFO] {m}")

    cfg = get_cfg(unknown)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L_MAX = cfg["max_len"]
    PSSM_F = cfg["pssm_feat_dim"]

    # Prepare dataset
    ds = ProteinDataset(args.data_dir, cfg)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=protein_collate_fn,
        pin_memory=True
    )

    # Load model
    model = SoluBat(seq_vocab_size=21, cfg=cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # Inference and collect logits
    pos_seq, pos_pssm, pos_fuse = [], [], []
    logits_all, labels = [], []
    seq_cache, pssm_cache, pp_cache = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            seq_t  = pad_to_len(batch["seq"].to(device), L_MAX)
            pssm_t = pad_to_len(batch["pssm"].to(device), L_MAX)
            pp_t   = batch["pp_glb"].to(device)
            pssm_t = normalize_pssm(pssm_t)

            logits, res_dict = model(pssm_t, seq_t, pp_t, return_all_branch=True)
            seq_logits  = res_dict["res_logits_seq"][..., 1]
            pssm_logits = res_dict["res_logits_pssm"][..., 1]

            pos_seq.append(seq_logits.cpu())
            pos_pssm.append(pssm_logits.cpu())
            pos_fuse.append(res_dict["res_logits_fuse"][..., 1].cpu())

            logits_all.append(logits.cpu())
            labels.extend(batch["label"].tolist())
            seq_cache.extend(seq_t.cpu())
            pssm_cache.extend(pssm_t.cpu())
            pp_cache.extend(pp_t.cpu())

    pos_seq   = torch.cat(pos_seq)
    pos_pssm  = torch.cat(pos_pssm)
    logits_all= torch.cat(logits_all)
    labels    = np.array(labels)

    # Basic metrics
    preds = logits_all.argmax(1).numpy()
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec  = recall_score(labels, preds, average="macro", zero_division=0)
    f1   = f1_score(labels, preds, average="macro", zero_division=0)
    mcc  = matthews_corrcoef(labels, preds)
    auc  = roc_auc_score(labels, logits_all.softmax(1).numpy()[:,1]) if len(np.unique(labels))==2 else None
    log({"acc":acc, "prec":prec, "rec":rec, "f1":f1, "mcc":mcc, "auc":auc})

    # Decode sequences
    AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
    AA_VOCAB = {aa: i+1 for i, aa in enumerate(AA_LIST)}
    IDX2AA = {v:k for k,v in AA_VOCAB.items()}
    IDX2AA[0] = ''
    seq_arr = torch.stack(seq_cache).numpy()
    def tokens_to_string(tokens):
        return ''.join([IDX2AA[t] for t in tokens if t!=0])
    all_seqs = [tokens_to_string(row) for row in seq_arr]

    # Select target examples
    target_seqs = [
        # specify up to five target sequences
        # e.g. "MCGRFTLTL..."
    ]
    selected_idx = []
    for s in target_seqs:
        try:
            idx = all_seqs.index(s)
            selected_idx.append(idx)
            print(f"Found target sequence: {s[:15]}... index: {idx}")
        except ValueError:
            print(f"Target sequence not found: {s[:15]}...")
    print("selected_idx:", selected_idx)

    print("\nSelected sequence information:")
    print("idx\ttrue_label\tpred_label\tsequence")
    for idx in selected_idx:
        print(f"{idx}\t{labels[idx]}\t{preds[idx]}\t{all_seqs[idx]}")

    # Optional: save info
    df = pd.DataFrame({
        "idx": selected_idx,
        "true_label": [labels[i] for i in selected_idx],
        "pred_label": [preds[i] for i in selected_idx],
        "sequence": [all_seqs[i] for i in selected_idx]
    })
    df.to_csv(out_dir/"selected_examples_info.csv", index=False)
    
    # PoSHAP visualization for seq and pssm (heatmap, bar, dependence, decision, force)
    for tag, pos in zip(['seq','pssm'], [pos_seq, pos_pssm]):
        Ns = min(args.shap_samples, pos.size(0))
        idxs = np.random.choice(pos.size(0), Ns, replace=False)
        samp = pos[idxs].numpy()
        # heatmap
        plt.figure(figsize=(10,0.25*Ns+2))
        plt.imshow(samp, aspect='auto')
        plt.colorbar(label=f"Residue logit ({tag}-only, class=1)")
        plt.title(f"PoSHAP heatmap ({tag}-only)")
        plt.tight_layout(); plt.savefig(out_dir/f"{tag}_heatmap.png", dpi=300); plt.close()
        # bar
        mean_abs = np.abs(samp).mean(0)
        topk = mean_abs.argsort()[::-1][:args.pp_topk]
        plt.figure(figsize=(8,4)); plt.bar(range(len(topk)), mean_abs[topk])
        plt.xticks(range(len(topk)), [f"Pos{i+1}" for i in topk], rotation=90)
        plt.title(f"Top residue importance ({tag}-only)"); plt.tight_layout();
        plt.savefig(out_dir/f"{tag}_bar.png", dpi=300); plt.close()
        # additional plots omitted for brevity

    # PP SHAP with KernelExplainer
    model_cpu = model.cpu()
    pp_array = torch.stack(pp_cache).numpy()
    pp_names = (json.load(open(args.pp_names)) if args.pp_names and pathlib.Path(args.pp_names).exists()
                else [f"PP{i+1}" for i in range(pp_array.shape[1])])
    # define linear PP function
    lin1_w, lin1_b = model_cpu.final_head[0].weight, model_cpu.final_head[0].bias
    lin2_w, lin2_b = model_cpu.final_head[3].weight, model_cpu.final_head[3].bias
    W_full = lin2_w[1] @ torch.relu(lin1_w)
    b_full = lin2_b[1] + (lin2_w[1] @ torch.relu(lin1_b))
    W_pp = W_full[2:]; b_pp = b_full
    def f_pp(x): return x @ W_pp.numpy() + b_pp.numpy()
    bg = np.random.choice(pp_array.shape[0], min(20, pp_array.shape[0]), replace=False)
    explainer_pp = shap.KernelExplainer(f_pp, pp_array[bg])
    pp_idx = np.random.choice(pp_array.shape[0], min(args.shap_samples, pp_array.shape[0]), replace=False)
    pp_shap = explainer_pp.shap_values(pp_array[pp_idx], nsamples="auto")
    # summary and bar
    shap.summary_plot(pp_shap, pp_array[pp_idx], feature_names=pp_names, show=False)
    plt.tight_layout(); plt.savefig(out_dir/"pp_shap_summary.png", dpi=300); plt.close()
    mean_abs_pp = np.abs(pp_shap).mean(0)
    top_pp = mean_abs_pp.argsort()[::-1][:args.pp_topk]
    plt.figure(figsize=(8,4)); plt.bar(range(len(top_pp)), mean_abs_pp[top_pp]);
    plt.xticks(range(len(top_pp)), [pp_names[i] for i in top_pp], rotation=90)
    plt.title("Top PP feature importance"); plt.tight_layout(); plt.savefig(out_dir/"pp_bar.png", dpi=300); plt.close()

    print("\nAll outputs saved to", out_dir)

if __name__ == "__main__":
    main()
