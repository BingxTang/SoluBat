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
                 dropout_prob=0.5):  # Add dropout probability parameter
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
        self.gru = nn.GRU(self.CONV1D_FEATURE_SIZE, self.GRU_HIDDEN_SIZE, n_layers, dropout=dropout_prob)  # Add dropout
        self.fc = nn.Linear(self.GRU_HIDDEN_SIZE, self.FULLY_CONNECTED_LAYER_SIZE)
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout
        self.out_act = nn.Sigmoid()

        self.gru_layers = n_layers
        self.classification_head = nn.Linear(self.FULLY_CONNECTED_LAYER_SIZE, 2)  # Change to 1 output

    def forward(self, inputs):
        h0 = Variable(torch.zeros(self.gru_layers, inputs.size(0), self.GRU_HIDDEN_SIZE).cuda())

        inputs = inputs.transpose(1, 2)
        c = self.c1(inputs)
        p = self.p1(c)
        c = self.c2(p)
        p = self.p2(c)

        p = p.transpose(1, 2).transpose(0, 1)

        p = F.relu(p)

        output, hidden = self.gru(p, h0)

        output = self.dropout(output)

        output = F.relu(self.fc(output.mean(dim=0)))

        #logits = self.classification_head(output)
        #output = self.bn(output)
        return output


class Mamaba(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        dropout_prob: float = 0.1,  # Add dropout probability parameter
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.fc_layers = nn.ModuleList(
            [nn.Linear(2 * d_model, d_model) for i in range(n_layer)]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

        # Add a fully connected layer for binary classification
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer
        #self.bn = nn.BatchNorm1d(d_model)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer, fc in zip(self.layers, self.fc_layers):
            h_f, res_f = layer(hidden_states, residual)  # Get information from left to right
            h_f = self.norm_f(h_f)
            res_flip = residual.flip([1]) if residual is not None else None
            h_b, res_b = layer(hidden_states.flip([1]), res_flip)  # Get information from right to left
            h_b = self.norm_f(h_b)
            #h_combined = torch.cat([h_f, h_b.flip([1])], dim=-1)
            #hidden_states = fc(h_combined)
            hidden_states = h_f + h_b.flip([1])
            residual = res_f + res_b.flip([1])

        hidden_states = self.dropout(hidden_states)  # Add dropout to the output after pooling

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )

        # Average pooling over all time steps
        hidden_states = hidden_states.mean(dim=1)

        return hidden_states


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.fc_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.fc_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.fc_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.fc_out(context)
        return output, attn


class SoluBat(nn.Module):
    def __init__(self,
                 mam_d_model,
                 mam_n_layer,
                 mam_d_intermediate,
                 mam_vocab_size,
                 mam_ssm_cfg=None,
                 mam_attn_layer_idx=None,
                 mam_attn_cfg=None,
                 mam_norm_epsilon=1e-5,
                 mam_rms_norm=False,
                 mam_initializer_cfg=None,
                 mam_fused_add_norm=False,
                 mam_residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 mam_dropout_prob=0.5,
                 rnn_in_channels=42,
                 rnn_n_layers=1,
                 rnn_conv1d_feature_size=200,
                 rnn_conv1d_kernel_size=3,
                 rnn_avgpool1d_kernel_size=3,
                 rnn_gru_hidden_size=200,
                 rnn_fully_connected_layer_size=32,
                 rnn_dropout_prob=0.5):  # Add RNN model parameters
        super(SoluBat, self).__init__()
        self.RNN = RNNModel(in_channels=rnn_in_channels,
                            n_layers=rnn_n_layers,
                            conv1d_feature_size=rnn_conv1d_feature_size,
                            conv1d_kernel_size=rnn_conv1d_kernel_size,
                            avgpool1d_kernel_size=rnn_avgpool1d_kernel_size,
                            gru_hidden_size=rnn_gru_hidden_size,
                            fully_connected_layer_size=rnn_fully_connected_layer_size,
                            dropout_prob=rnn_dropout_prob)  # Pass RNN model parameters
        self.mamaba = Mamaba(mam_d_model, mam_n_layer, mam_d_intermediate, mam_vocab_size, mam_ssm_cfg, mam_attn_layer_idx, mam_attn_cfg,
                             mam_norm_epsilon, mam_rms_norm, mam_initializer_cfg, mam_fused_add_norm, mam_residual_in_fp32, device,
                             dtype, mam_dropout_prob)

        self.fc = nn.Linear(rnn_fully_connected_layer_size + mam_d_model, 2)
        self.multi_head_attention = MultiHeadAttention(rnn_fully_connected_layer_size + mam_d_model, num_heads=8)
        self.dropout = nn.Dropout(0.2)

    def forward(self, pssm_input, seq_input):
        RNN_output = self.RNN(pssm_input)
        mamaba_output = self.mamaba(seq_input)

        combined_logits = torch.cat((RNN_output, mamaba_output), dim=1)
        combined_logits = self.dropout(combined_logits)

        attn_output, _ = self.multi_head_attention(combined_logits, combined_logits, combined_logits)
        attn_output = self.dropout(attn_output)

        output = self.fc(attn_output.squeeze(1))
        return output
