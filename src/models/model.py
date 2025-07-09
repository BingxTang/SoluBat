from __future__ import annotations
import torch
import torch.nn as nn
from src.config.default_config import get_cfg
from src.models.mamba2simple import BiMambaBlock

# Embedding and gating modules
class SeqEmbedding(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
    def forward(self, x):
        return self.emb(x)

class PSSMEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, pssm):
        return self.proj(pssm)

class PosEmbedding(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(max_len + 1, dim, padding_idx=0)
    def forward(self, seq):
        pos_idx = torch.arange(seq.size(1), device=seq.device).unsqueeze(0) + 1
        pos_idx = pos_idx * (seq != 0).long()
        return self.emb(pos_idx)

class SEGating(nn.Module):
    def __init__(self, dim: int, T: float = 1.0, alpha: float = .9):
        super().__init__()
        self.T, self.alpha = T, alpha
        self.linear = nn.Linear(dim * 2, 1)
    def forward(self, s, p):
        g = torch.sigmoid(self.linear(torch.cat([s, p], -1)) / self.T)
        return self.alpha * g * s + (1 - self.alpha * g) * p

# Dual-branch encoder: Seq-only, PSSM-only, and fusion
class DualBranchEncoder(nn.Module):
    def __init__(
        self,
        vocab: int,
        pssm_feat_dim: int,
        seq_embed_dim: int,
        d_model: int,
        n_layers: int,
        gate_T: float,
        gate_alpha: float,
        max_len: int
    ):
        super().__init__()
        self.seq_emb = SeqEmbedding(vocab, seq_embed_dim)
        self.pssm_emb = PSSMEmbedding(pssm_feat_dim, seq_embed_dim)
        self.pos_emb = PosEmbedding(max_len, seq_embed_dim)

        # Seq-only branch
        self.seq_proj_in = nn.Linear(seq_embed_dim, d_model, bias=False)
        self.seq_blocks = nn.ModuleList([BiMambaBlock(d_model) for _ in range(n_layers)])

        # PSSM-only branch
        self.pssm_proj_in = nn.Linear(seq_embed_dim, d_model, bias=False)
        self.pssm_blocks = nn.ModuleList([BiMambaBlock(d_model) for _ in range(n_layers)])

        # Fusion branch
        self.gate = SEGating(seq_embed_dim, gate_T, gate_alpha)
        self.fuse_proj_in = nn.Linear(seq_embed_dim, d_model, bias=False)
        self.fuse_blocks = nn.ModuleList([BiMambaBlock(d_model) for _ in range(n_layers)])

    def forward(self, seq, pssm):
        # Seq-only path
        seq_x = self.seq_emb(seq) + self.pos_emb(seq)
        seq_x = self.seq_proj_in(seq_x)
        for blk in self.seq_blocks:
            seq_x = blk(seq_x)

        # PSSM-only path
        pssm_x = self.pssm_emb(pssm)
        pssm_x = self.pssm_proj_in(pssm_x)
        for blk in self.pssm_blocks:
            pssm_x = blk(pssm_x)

        # Fusion path
        s = self.seq_emb(seq) + self.pos_emb(seq)
        p = self.pssm_emb(pssm)
        fuse_x = self.gate(s, p)
        fuse_x = self.fuse_proj_in(fuse_x)
        for blk in self.fuse_blocks:
            fuse_x = blk(fuse_x)

        return seq_x, pssm_x, fuse_x

# Main model: SoluBat with dual-branch outputs
class SoluBat(nn.Module):
    def __init__(self, seq_vocab_size, cfg=None, n_classes=2):
        super().__init__()
        cfg = cfg or get_cfg([])
        self.pp_dim = cfg["pp_glb_dim"]
        self.encoder = DualBranchEncoder(
            vocab=seq_vocab_size,
            pssm_feat_dim=cfg["pssm_feat_dim"],
            seq_embed_dim=cfg["seq_embed_dim"],
            d_model=cfg["mamba_dim"],
            n_layers=cfg.get("mam_n_layer", 6),
            gate_T=cfg.get("gate_temperature", 1.0),
            gate_alpha=cfg.get("gate_alpha", 0.9),
            max_len=cfg["max_len"]
        )
        d_model = cfg["mamba_dim"]

        # Residue scoring heads
        self.residue_score_seq   = nn.Linear(d_model, n_classes, bias=False)
        self.residue_score_pssm  = nn.Linear(d_model, n_classes, bias=False)
        self.residue_score_fuse  = nn.Linear(d_model, n_classes, bias=False)

        in_dim = n_classes + self.pp_dim
        self.final_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(in_dim, n_classes)
        )

    def forward(
        self, pssm, seq, pp_glb=None,
        return_res_logits=False, return_all_branch=False
    ):
        seq_h, pssm_h, fuse_h = self.encoder(seq, pssm)

        res_logits_seq  = self.residue_score_seq(seq_h)
        res_logits_pssm = self.residue_score_pssm(pssm_h)
        res_logits_fuse = self.residue_score_fuse(fuse_h)

        fused_logits = res_logits_fuse.sum(1)
        if pp_glb is None:
            pp_glb = fused_logits.new_zeros(fused_logits.size(0), self.pp_dim)
        fused = torch.cat([fused_logits, pp_glb], dim=-1)
        logits = self.final_head(fused)

        if return_all_branch:
            return logits, {
                "res_logits_fuse": res_logits_fuse,
                "res_logits_seq": res_logits_seq,
                "res_logits_pssm": res_logits_pssm
            }
        if return_res_logits:
            return logits, res_logits_fuse
        return logits
