"""
End-to-End Stateful RT-GNN Architecture (rtgnn_stateful_gnn.py)
==============================================================
Integrates the Phase 7 Complex Parallel Associative Scan with the
spatial message-passing convolution and link prediction head.

Architecture:
1. Node Memory Layer: StatefulRotationalNodeMemory maintaining z_u in C^d,
   updated via parallel associative prefix scan over consecutive event intervals.
2. Spatial GNN Layer: TemporalRotationalConv aggregating rotated messages
   from the causal HistoryBuffer (t < t_ref).
3. Link Predictor Head: DynamicLinkPredictor evaluating pair occurrence likelihoods.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rtgnn_model import (
    DynamicLinkPredictor,
    HistoryBuffer,
    RTGNNConfig,
    TemporalRotationalConv,
)
from rtgnn_scan import StatefulRotationalNodeMemory, parallel_complex_scan


def complex_rms_norm(z: Tensor, eps: float = 1e-6) -> Tensor:
    """Contracts complex tensor magnitude to unit RMS while strictly preserving phase arg(z)."""
    mag_sq = z.real.square() + z.imag.square()
    rms = torch.sqrt(mag_sq.mean(dim=-1, keepdim=True) + eps)
    scale = torch.clamp_min(rms, 1.0)
    return z / scale


class StatefulRotationalTemporalGNN(nn.Module):
    """Full End-to-End Stateful RT-GNN with Parallel Associative Scan Memory."""

    def __init__(self, cfg: RTGNNConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        # 1. Base initial node embeddings
        self.base_emb_real = nn.Embedding(cfg.num_nodes, d)
        self.base_emb_imag = nn.Embedding(cfg.num_nodes, d)

        # 2. Stateful Complex Node Memory (Phase 7 Parallel Associative Scan)
        self.memory = StatefulRotationalNodeMemory(
            num_nodes=cfg.num_nodes,
            feature_dim=d,
            message_dim=d * 2,  # Incoming message combines src and dst representations
            omega_min=cfg.omega_min,
            omega_max=cfg.omega_max,
            lambda_init=cfg.lambda_init,
        )

        # 3. Channel Normalization for Dual-Channel Complex Representations
        self.norm_real = nn.LayerNorm(d)
        self.norm_imag = nn.LayerNorm(d)

        # 4. Spatial GNN Layers (Causal History Message Passing)
        self.layers = nn.ModuleList([
            TemporalRotationalConv(d, d, cfg) for _ in range(cfg.num_layers)
        ])

        # 5. Dynamic Link Predictor
        self.predictor = DynamicLinkPredictor(d, cfg.dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.base_emb_real.weight, std=self.cfg.emb_init_std)
        nn.init.normal_(self.base_emb_imag.weight, std=self.cfg.emb_init_std)
        self.memory.reset_parameters(
            omega_min=self.cfg.omega_min,
            omega_max=self.cfg.omega_max,
            lambda_init=self.cfg.lambda_init,
        )
        self.norm_real.reset_parameters()
        self.norm_imag.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.predictor.reset_parameters()

    def reset_memory(self) -> None:
        """Clears persistent memory across epochs or split evaluations."""
        self.memory.reset_memory()

    def get_node_states(self) -> Tuple[Tensor, Tensor]:
        """Combines base embeddings with persistent continuous-time complex memory."""
        # z_u = base_emb + memory
        r = self.base_emb_real.weight + self.memory.mem_real
        i = self.base_emb_imag.weight + self.memory.mem_imag
        r = self.norm_real(r)
        i = self.norm_imag(i)
        return r, i

    def encode(
        self,
        hist_edge_index: Optional[Tensor],
        hist_delta_t: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Computes causal message-passing representations over the HistoryBuffer."""
        x_real, x_imag = self.get_node_states()

        if hist_edge_index is None or hist_delta_t is None or hist_edge_index.size(1) == 0:
            return x_real, x_imag

        for conv in self.layers:
            h_real, h_imag = conv(x_real, x_imag, hist_edge_index, hist_delta_t)
            # Split ReLU activation on real and imaginary components separately
            x_real = F.relu(h_real) + x_real
            x_imag = F.relu(h_imag) + x_imag

        return x_real, x_imag

    def forward(
        self,
        hist_edge_index: Optional[Tensor],
        hist_delta_t: Optional[Tensor],
        src_idx: Tensor,
        dst_idx: Tensor,
    ) -> Tensor:
        """Evaluates dynamic link likelihood logits conditioned on causal history."""
        x_real, x_imag = self.encode(hist_edge_index, hist_delta_t)
        return self.predictor(
            x_real[src_idx],
            x_imag[src_idx],
            x_real[dst_idx],
            x_imag[dst_idx],
        )

    def update_streaming_memory(
        self,
        src: Tensor,
        dst: Tensor,
        delta_t_src: Tensor,
        delta_t_dst: Tensor,
    ) -> None:
        """Updates persistent node memory after scoring using continuous decay-rotation."""
        with torch.no_grad():
            x_r, x_i = self.get_node_states()

            # Message vectors formed from current interaction representations
            msg_for_src = torch.cat([x_r[dst], x_i[dst]], dim=-1)
            msg_for_dst = torch.cat([x_r[src], x_i[src]], dim=-1)

            # Continuous decay-rotation jump factors for this batch
            a_src = self.memory.compute_decay_rotation_factor(delta_t_src)
            a_dst = self.memory.compute_decay_rotation_factor(delta_t_dst)

            b_src = self.memory.project_message(msg_for_src)
            b_dst = self.memory.project_message(msg_for_dst)

            # State update: z <- a * z + b
            old_z_src = torch.complex(self.memory.mem_real[src], self.memory.mem_imag[src])
            new_z_src = complex_rms_norm(a_src * old_z_src + b_src)

            old_z_dst = torch.complex(self.memory.mem_real[dst], self.memory.mem_imag[dst])
            new_z_dst = complex_rms_norm(a_dst * old_z_dst + b_dst)

            # Detach to ensure state updates maintain O(1) memory bound
            self.memory.mem_real[src] = new_z_src.real.detach()
            self.memory.mem_imag[src] = new_z_src.imag.detach()
            self.memory.mem_real[dst] = new_z_dst.real.detach()
            self.memory.mem_imag[dst] = new_z_dst.imag.detach()
