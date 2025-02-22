import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
from functools import partial

import copy

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class RNNModel(nn.Module):
    def __init__(self,
                 in_channels=42,
                 n_layers=2,
                 conv1d_feature_size=256,
                 conv1d_kernel_size=3,
                 avgpool1d_kernel_size=2,
                 gru_hidden_size=256,
                 fully_connected_layer_size=32,
                 dropout_prob=0.5):  # Added dropout probability parameter
        super(RNNModel, self).__init__()
        self.CONV1D_FEATURE_SIZE = conv1d_feature_size
        self.CONV1D_KERNEL_SIZE = conv1d_kernel_size
        self.AVGPOOL1D_KERNEL_SIZE = avgpool1d_kernel_size
        self.GRU_HIDDEN_SIZE = gru_hidden_size
        self.FULLY_CONNECTED_LAYER_SIZE = fully_connected_layer_size

        self.c1 = nn.Conv1d(in_channels, self.CONV1D_FEATURE_SIZE, self.CONV1D_KERNEL_SIZE)
        self.p1 = nn.AvgPool1d(self.AVGPOOL1D_KERNEL_SIZE)
        self.c2 = nn.Conv1d(self.CONV1D_FEATURE_SIZE, self.CONV1D_FEATURE_SIZE, self.CONV1D_KERNEL_SIZE)
        self.p2 = nn.AvgPool1d(self.AVGPOOL1D_KERNEL_SIZE)
        self.gru = nn.GRU(self.CONV1D_FEATURE_SIZE, self.GRU_HIDDEN_SIZE, n_layers, dropout=dropout_prob)  # Added dropout
        self.fc = nn.Linear(self.GRU_HIDDEN_SIZE, self.FULLY_CONNECTED_LAYER_SIZE)
        self.dropout = nn.Dropout(dropout_prob)  # Added dropout
        self.out_act = nn.Sigmoid()

        self.gru_layers = n_layers
        self.classification_head = nn.Linear(self.FULLY_CONNECTED_LAYER_SIZE, 2)  # Modified to 1 output

    def forward(self, inputs):
        h0 = Variable(torch.zeros(self.gru_layers, inputs.size(0), self.GRU_HIDDEN_SIZE).cuda())

        c = self.c1(inputs)
        p = self.p1(c)
        c = self.c2(p)
        p = self.p2(c)

        p = p.transpose(1, 2).transpose(0, 1)

        p = F.relu(p)

        output, hidden = self.gru(p, h0)

        output = self.dropout(output)

        output = F.relu(self.fc(output.mean(dim=0)))

        return output
