from __future__ import annotations

"""
Dataset loader driven by cfg
-------------------------------------------------
- Receives cfg parameters from caller
- Filters sequences by min_len / max_len from cfg
- Requires data_seqs.npy, data_pssm.npy, data_labels.npy, data_pps.npy
- Optional data_mutations.npy for sample IDs
"""
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

VOCAB = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
PAD_IDX = 0

class ProteinDataset(Dataset):
    """Loads sequences, PSSM, and PP features from .npy files."""

    def __init__(self, dir_path: str | Path, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.max_len = cfg["max_len"]
        self.min_len = cfg.get("min_len", 0)

        dir_path = Path(dir_path)
        self.seqs   = np.load(dir_path / "data_seqs.npy", allow_pickle=True)
        self.pssm   = np.load(dir_path / "data_pssm.npy", allow_pickle=True)
        self.labels = np.load(dir_path / "data_labels.npy")
        self.pp_glb = np.load(dir_path / "data_pps.npy", allow_pickle=True)

        id_file = dir_path / "data_mutations.npy"
        if id_file.exists():
            self.ids = np.load(id_file, allow_pickle=True)
        else:
            self.ids = None

        assert len(self.seqs) == len(self.pssm) == len(self.labels) == len(self.pp_glb), \
            f"Inconsistent sample counts in {dir_path}"

        self.indices = [i for i, s in enumerate(self.seqs)
                        if self.min_len <= len(s.decode() if isinstance(s, bytes) else s) <= self.max_len]

    @staticmethod
    def _encode_seq(s: str) -> torch.Tensor:
        """Encode amino acid sequence into integer tensor."""
        return torch.tensor([VOCAB.get(a, PAD_IDX) for a in s], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = self.indices[idx]
        raw = self.seqs[real_idx]
        seq_str = raw.decode() if isinstance(raw, bytes) else raw
        L = len(seq_str)

        seq_tensor = self._encode_seq(seq_str)
        pssm_arr = self.pssm[real_idx][:L, :]
        pp_vec = torch.from_numpy(self.pp_glb[real_idx]).float()

        sample = {
            "seq": seq_tensor,
            "pssm": torch.from_numpy(pssm_arr.copy()),
            "pp_glb": pp_vec,
            "label": torch.tensor(self.labels[real_idx], dtype=torch.long),
        }
        if self.ids is not None:
            sample["id"] = self.ids[real_idx]
        return sample


def protein_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function to pad sequences and PSSM arrays into a batch."""
    seqs = [b["seq"] for b in batch]
    pssms = [b["pssm"] for b in batch]
    pps = [b["pp_glb"] for b in batch]
    labels = [b["label"] for b in batch]
    ids_present = "id" in batch[0]

    max_len = max(len(s) for s in seqs)
    feat_dim = pssms[0].shape[1]

    seq_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    pssm_pad = torch.zeros(len(batch), max_len, feat_dim)
    for i, (s, p) in enumerate(zip(seqs, pssms)):
        seq_pad[i, :len(s)] = s
        pssm_pad[i, :p.shape[0]] = p

    batch_dict = {
        "seq": seq_pad,
        "pssm": pssm_pad,
        "pp_glb": torch.stack(pps),
        "label": torch.stack(labels),
    }
    if ids_present:
        batch_dict["id"] = [b["id"] for b in batch]

    return batch_dict
