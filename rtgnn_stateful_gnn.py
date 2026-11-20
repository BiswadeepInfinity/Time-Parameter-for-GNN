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
            # Sliced active-node representation: normalizes only active interaction nodes
            # (mathematically equivalent to full-table LayerNorm with zero redundant ops)
            r_dst = self.norm_real(self.base_emb_real(dst) + self.memory.mem_real[dst])
            i_dst = self.norm_imag(self.base_emb_imag(dst) + self.memory.mem_imag[dst])
            r_src = self.norm_real(self.base_emb_real(src) + self.memory.mem_real[src])
            i_src = self.norm_imag(self.base_emb_imag(src) + self.memory.mem_imag[src])

            # Message vectors formed from current interaction representations
            msg_for_src = torch.cat([r_dst, i_dst], dim=-1)
            msg_for_dst = torch.cat([r_src, i_src], dim=-1)

            # Continuous decay-rotation jump factors for this batch
            a_src = self.memory.compute_decay_rotation_factor(delta_t_src)
            a_dst = self.memory.compute_decay_rotation_factor(delta_t_dst)

            b_src = self.memory.project_message(msg_for_src)
            b_dst = self.memory.project_message(msg_for_dst)

            # Concatenate all interaction updates in this batch
            nodes = torch.cat([src, dst])
            a = torch.cat([a_src, a_dst])
            b = torch.cat([b_src, b_dst])

            # Check if all node appearances are unique
            num_unique = torch.unique(nodes).size(0)
            if num_unique == nodes.size(0):
                old_z = torch.complex(
                    self.memory.mem_real[nodes], self.memory.mem_imag[nodes]
                )
                new_z = complex_rms_norm(a * old_z + b)
                self.memory.mem_real[nodes] = new_z.real.detach()
                self.memory.mem_imag[nodes] = new_z.imag.detach()
            else:
                # Associative intra-batch compounding for duplicate active nodes:
                # Groups events by node and applies (a2*a1, a2*b1 + b2) in chronological order
                perm = torch.argsort(nodes, stable=True)
                s_nodes = nodes[perm]
                s_a = a[perm]
                s_b = b[perm]

                mask = torch.cat([
                    torch.tensor([True], device=nodes.device),
                    s_nodes[1:] != s_nodes[:-1],
                ])
                unique_nodes = s_nodes[mask]

                idx = torch.arange(len(s_nodes), device=nodes.device)
                starts = torch.where(mask)[0]
                group_id = torch.cumsum(mask.long(), dim=0) - 1
                intra_idx = idx - starts[group_id]
                max_occurrences = intra_idx.max().item() + 1

                comb_a = s_a[mask].clone()
                comb_b = s_b[mask].clone()

                for step in range(1, max_occurrences):
                    step_mask = (intra_idx == step)
                    if not step_mask.any():
                        continue
                    target_u = group_id[step_mask]
                    step_a = s_a[step_mask]
                    step_b = s_b[step_mask]

                    cur_a = comb_a[target_u]
                    cur_b = comb_b[target_u]
                    comb_a[target_u] = step_a * cur_a
                    comb_b[target_u] = step_a * cur_b + step_b

                old_z = torch.complex(
                    self.memory.mem_real[unique_nodes],
                    self.memory.mem_imag[unique_nodes],
                )
                new_z = complex_rms_norm(comb_a * old_z + comb_b)
                self.memory.mem_real[unique_nodes] = new_z.real.detach()
                self.memory.mem_imag[unique_nodes] = new_z.imag.detach()
