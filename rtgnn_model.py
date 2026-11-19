"""
RT-GNN Model Architecture (rtgnn_model.py)
=========================================
Strictly adheres to RT-GNN Implementation Blueprint v2:
- Closed-form complex rotation and exponential decay over continuous time gaps Delta_t.
- Semigroup property T(a) o T(b) = T(a + b): theta is applied once outside the time operator.
- HistoryBuffer: strictly past-event message-passing graph (prevents label leakage).
- Predictor outputs raw logits; loss is BCEWithLogitsLoss.
- PyG message returns one tensor (real || imag) for robust cross-version compatibility.
- Split activation: ReLU applied separately to real and imaginary components.
- Completely parameter-driven via RTGNNConfig (DO NOT HARDCODE ANY VALUE).

Accuracy Upgrades (v3 — leak-free):
  1. Query-Time Rotation on SOURCE only (causal — delta_t_src known before the edge)
  2. Destination Popularity Prior (global learnable bias per node)
  3. Multi-Harmonic Fourier Time Encoder
  4. Edge Feature Projection (172-D wiki edge features → d//2)
  5. Dot-Product Shortcut in Predictor (bilinear interaction term alongside MLP)
  6. Wider 2-Hidden-Layer MLP Predictor
  7. Attention Aggregation in GNN layer (replaces mean — learns which neighbours matter)

Accuracy Upgrades (v4 — stateless persistent memory, O(1) per event):
  8. SemiGroupNodeMemory: lightweight persistent complex EMA memory per node.
     Updated as z_u ← T(dt) * z_u + B * msg, NO edge history stored.
     Strictly O(1) per event, semigroup-compatible, zero leakage.
     Keeps stateless paradigm: no sequential scan needed (B * msg is a linear
     projection of a normalised embedding, not the true destination).
  9. history_batches raised from 8 → 32 (more graph context for GNN layers).

What was tried and REMOVED because it inflated MRR to ~1.0 against TGB random negatives:
  - SrcDstAffinityTable: This is EdgeBank in disguise. After several epochs every
    training positive has count>0 while every random negative has count=0.
    Perfect binary signal → MRR = 1.0.
  - InteractionMemory: Same problem. memory[src] converges to emb[true_dst].
    MLP learns "if memory ≈ dst_emb → positive". MRR = 1.0.
  Both track past positive edges explicitly, which trivially separates them from
  randomly-sampled negatives. They are non-parametric EdgeBanks, not GNN improvements.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


@dataclass
class RTGNNConfig:
    """Hyperparameter configuration dataclass for RT-GNN."""
    num_nodes: int                     # Taken dynamically from dataset
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    emb_init_std: float = 0.1
    omega_min: float = 1e-2            # Set from data: omega_min * dt_p99 < pi
    omega_max: float = 1e1
    lambda_init: float = 0.1           # Initial decay rate in normalised time units
    bidirectional: bool = True         # Messages flow both ways along each event
    history_batches: int = 32          # Past batches kept as the message graph (v4: 8→32)
    lr: float = 5e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    grad_clip: Optional[float] = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Stateless Performance Upgrades (O(1) continuous time):
    use_query_rotation: bool = True     # 1. Query-time rotation T_dt(z) on src
    use_dst_bias: bool = True           # 2. Destination popularity / frequency prior
    use_harmonic_time: bool = True      # 3. Multi-harmonic Fourier time encodings
    num_harmonics: int = 8              # Number of harmonic octave bands
    use_dot_shortcut: bool = False      # Bilinear dot-product shortcut in predictor
    use_attn_aggr: bool = False         # False -> robust mean aggregation
    # v4: Lightweight O(1) persistent complex node memory (SemiGroupNodeMemory)
    use_node_memory: bool = True        # 8. Stateless persistent complex EMA memory


class TemporalRotationEngine(nn.Module):
    """Computes continuous-time complex rotation and decay.

    Formula:
        T_dt(z) = z * exp((-lambda + i*omega) * dt)

    Semigroup property: T(a) o T(b) = T(a+b). Theta is NOT in the time operator.
    """

    def __init__(self, dim: int, cfg: RTGNNConfig):
        super().__init__()
        self._cfg = cfg
        self.omega = nn.Parameter(torch.empty(dim))
        self.raw_lambda = nn.Parameter(torch.empty(dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        c = self._cfg
        with torch.no_grad():
            self.omega.copy_(
                torch.logspace(
                    math.log10(c.omega_min),
                    math.log10(c.omega_max),
                    self.omega.numel(),
                )
            )
            self.raw_lambda.fill_(math.log(math.expm1(c.lambda_init)))

    @property
    def lambda_decay(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda)

    def forward(
        self, h_real: torch.Tensor, h_imag: torch.Tensor, delta_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = delta_t.clamp_min(0.0)
        if dt.dim() == 1 or dt.dim() == h_real.dim() - 1:
            dt = dt.unsqueeze(-1)
        decay = torch.exp(-self.lambda_decay * dt)
        operator = torch.polar(decay, self.omega * dt)
        z = torch.complex(h_real, h_imag) * operator
        return z.real, z.imag


class TemporalRotationalConv(MessagePassing):
    """Continuous-time rotational graph convolution layer.

    Upgrade 7: Optionally uses ATTENTION aggregation instead of mean,
    so the model learns which past neighbours are more informative.
    Uses a lightweight dot-product attention over (query=dst, key=rotated_src).
    """

    def __init__(self, in_channels: int, out_channels: int, cfg: RTGNNConfig):
        use_attn = getattr(cfg, "use_attn_aggr", False)
        # 'add' aggr + manual softmax == attention; 'mean' aggr == plain mean
        super().__init__(aggr="add" if use_attn else "mean")
        self.out_channels = out_channels
        self.use_attn = use_attn
        self.lin_real = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_imag = nn.Linear(in_channels, out_channels, bias=False)
        self.theta = nn.Parameter(torch.empty(out_channels))
        self.rotation = TemporalRotationEngine(out_channels, cfg)
        if use_attn:
            # Simple scalar attention: score = dot(q_src, rotated_msg) / sqrt(d)
            self.attn_q = nn.Linear(out_channels, out_channels, bias=False)
            self.attn_k = nn.Linear(out_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_real.weight)
        nn.init.xavier_uniform_(self.lin_imag.weight)
        nn.init.uniform_(self.theta, -math.pi, math.pi)
        self.rotation.reset_parameters()
        if self.use_attn:
            nn.init.xavier_uniform_(self.attn_q.weight)
            nn.init.xavier_uniform_(self.attn_k.weight)

    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        edge_index: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_proj = torch.cat([self.lin_real(x_real), self.lin_imag(x_imag)], dim=-1)
        out = self.propagate(
            edge_index,
            x=x_proj,
            delta_t=delta_t,
            size=(x_proj.size(0), x_proj.size(0)),
        )
        return out[:, : self.out_channels], out[:, self.out_channels :]

    def message(self, x_j: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        d = self.out_channels
        r, i = self.rotation(x_j[:, :d], x_j[:, d:], delta_t)
        z = torch.complex(r, i) * torch.polar(torch.ones_like(self.theta), self.theta)
        msg = torch.cat([z.real, z.imag], dim=-1)   # [E, 2d]
        if self.use_attn:
            # Lightweight self-attention weight per edge: softmax handled by 'add' aggr
            # We scale message by tanh(dot(k, q) / sqrt(d)) to keep magnitudes bounded
            k = self.attn_k(z.real)                  # [E, d]  key from rotated msg
            # q is computed on z.real as well (we don't have dst features here directly)
            q = self.attn_q(z.real)                  # [E, d]
            score = torch.tanh(
                (k * q).sum(dim=-1, keepdim=True) / math.sqrt(d)
            )  # [E, 1]
            msg = msg * score                         # scale message by attention weight
        return msg


class SemiGroupNodeMemory(nn.Module):
    """Lightweight O(1) persistent complex node memory for the stateless RT-GNN.

    Maintains a per-node complex state z_u in C^d, updated after each batch
    without storing any edge history (leak-free) via:

        z_u ← T(dt) * z_u + proj(msg_u)

    where T(dt) = exp((-lambda + i*omega) * dt) is the semigroup time operator
    and msg_u = mean of [emb_real[u] || emb_imag[u]] of involved nodes,
    projected through a learnable linear map B.

    This is O(1) per event and obeys the semigroup property:
        T(a) o T(b) = T(a+b)

    Critically: msg_u is built from CURRENT BATCH node indices and their static
    base embeddings — it does NOT encode the destination identity, so it cannot
    degenerate into EdgeBank.
    """

    def __init__(self, num_nodes: int, feature_dim: int, cfg: RTGNNConfig):
        super().__init__()
        self.num_nodes = num_nodes
        self.d = feature_dim
        # Learnable input projection B: 2d → d (projects base emb concat into memory space)
        self.msg_proj_real = nn.Linear(2 * feature_dim, feature_dim, bias=False)
        self.msg_proj_imag = nn.Linear(2 * feature_dim, feature_dim, bias=False)
        # Shared time operator for memory decay-rotation
        self.time_op = TemporalRotationEngine(feature_dim, cfg)
        # Norm scale to prevent memory explosion
        self.norm_scale = nn.Parameter(torch.ones(1))
        # Persistent (non-gradient) complex memory buffers
        self.register_buffer("mem_real", torch.zeros(num_nodes, feature_dim))
        self.register_buffer("mem_imag", torch.zeros(num_nodes, feature_dim))

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.msg_proj_real.weight)
        nn.init.xavier_uniform_(self.msg_proj_imag.weight)
        self.time_op.reset_parameters()
        nn.init.ones_(self.norm_scale)

    def reset_memory(self) -> None:
        """Clears persistent memory (called at start of each epoch)."""
        self.mem_real.zero_()
        self.mem_imag.zero_()

    @torch.no_grad()
    def update(
        self,
        node_ids: torch.Tensor,       # [K] unique node indices active in this batch
        base_emb_real: torch.Tensor,  # [N, d] full embedding table (detached)
        base_emb_imag: torch.Tensor,  # [N, d]
        delta_t: torch.Tensor,        # [K] normalised elapsed time for each node
    ) -> None:
        """Updates memory for active nodes after batch scoring (no gradient)."""
        emb_r = base_emb_real[node_ids].detach()   # [K, d]
        emb_i = base_emb_imag[node_ids].detach()   # [K, d]
        # Build input message from base embeddings (no destination identity)
        msg_in = torch.cat([emb_r, emb_i], dim=-1)  # [K, 2d]
        m_r = self.msg_proj_real(msg_in).detach()   # [K, d]
        m_i = self.msg_proj_imag(msg_in).detach()   # [K, d]

        # Decay-rotate existing memory
        old_r = self.mem_real[node_ids]  # [K, d]
        old_i = self.mem_imag[node_ids]  # [K, d]
        decayed_r, decayed_i = self.time_op(old_r, old_i, delta_t)

        # Accumulate: z ← T(dt) * z + B * emb
        new_r = decayed_r + m_r
        new_i = decayed_i + m_i

        # Soft norm to bound magnitude (prevents runaway growth)
        mag = (new_r.square() + new_i.square()).mean(dim=-1, keepdim=True).sqrt().clamp_min(1.0)
        scale = self.norm_scale.abs().clamp_min(1e-3)
        new_r = new_r / mag * scale
        new_i = new_i / mag * scale

        self.mem_real[node_ids] = new_r
        self.mem_imag[node_ids] = new_i

    def get(
        self, node_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns current memory state for given nodes. Read-only (no gradient)."""
        return self.mem_real[node_ids].detach(), self.mem_imag[node_ids].detach()


class HarmonicTimeEncoder(nn.Module):
    """Multi-harmonic Fourier temporal feature encoder."""

    def __init__(self, num_harmonics: int = 8, out_dim: int = 32):
        super().__init__()
        self.num_harmonics = num_harmonics
        freqs = (2.0 ** torch.arange(num_harmonics, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(2 * num_harmonics, out_dim)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        dt = delta_t.clamp_min(0.0)
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        phases = dt * self.freqs
        harmonics = torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1)
        return self.proj(harmonics)


class DynamicLinkPredictor(nn.Module):
    """Predicts link logits.

    Upgrade 5: Optional dot-product shortcut: adds Re(<src_complex, conj(dst_complex)>) directly
    to the MLP output.
    Upgrade 6: 2-hidden-layer MLP (d → 2d → d → 1) for higher representation capacity.
    """

    def __init__(
        self,
        dim: int,
        dropout: float,
        time_dim: int = 0,
        use_dot_shortcut: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.time_dim = time_dim
        self.use_dot_shortcut = use_dot_shortcut
        in_dim = 4 * dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        if use_dot_shortcut:
            self.dot_scale = nn.Parameter(torch.ones(1))

    def reset_parameters(self) -> None:
        for m in self.net:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        if self.use_dot_shortcut:
            nn.init.ones_(self.dot_scale)

    def forward(
        self,
        src_r: torch.Tensor,
        src_i: torch.Tensor,
        dst_r: torch.Tensor,
        dst_i: torch.Tensor,
        time_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats = [src_r, src_i, dst_r, dst_i]
        if self.time_dim > 0:
            feats.append(
                time_feat if time_feat is not None
                else torch.zeros(src_r.size(0), self.time_dim, device=src_r.device)
            )
        logit = self.net(torch.cat(feats, dim=-1)).squeeze(-1)
        if self.use_dot_shortcut:
            dot = (src_r * dst_r + src_i * dst_i).sum(dim=-1)
            logit = logit + self.dot_scale * dot / math.sqrt(self.dim)
        return logit


class RotationalTemporalGNN(nn.Module):
    """End-to-End Rotational Temporal Graph Neural Network (RT-GNN).

    v4: Integrates SemiGroupNodeMemory for O(1) persistent cross-batch state.
    Memory is updated with no_grad after each batch, keeping the stateless
    (O(1) per event, semigroup-compliant) paradigm intact.
    """

    def __init__(self, cfg: RTGNNConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.cfg = cfg
        self.emb_real = nn.Embedding(cfg.num_nodes, d)
        self.emb_imag = nn.Embedding(cfg.num_nodes, d)
        self.layers = nn.ModuleList([
            TemporalRotationalConv(d, d, cfg) for _ in range(cfg.num_layers)
        ])

        # Upgrade 1: Query-Time Rotation Engine (src only)
        self.query_rotation = (
            TemporalRotationEngine(d, cfg) if getattr(cfg, "use_query_rotation", True) else None
        )

        # Upgrade 2: Destination Popularity Prior
        self.dst_bias = (
            nn.Embedding(cfg.num_nodes, 1) if getattr(cfg, "use_dst_bias", True) else None
        )

        # Upgrade 3: Multi-Harmonic Fourier Time Encoder
        if getattr(cfg, "use_harmonic_time", True):
            time_dim = d // 2
            self.time_encoder = HarmonicTimeEncoder(
                num_harmonics=getattr(cfg, "num_harmonics", 8), out_dim=time_dim
            )
        else:
            time_dim = 0
            self.time_encoder = None

        # Upgrade 8 (v4): Lightweight O(1) persistent complex node memory
        if getattr(cfg, "use_node_memory", True):
            self.node_memory = SemiGroupNodeMemory(cfg.num_nodes, d, cfg)
            # Memory fuser: projects (emb + mem) → d for each of real and imag
            self.mem_fuse_real = nn.Linear(2 * d, d, bias=True)
            self.mem_fuse_imag = nn.Linear(2 * d, d, bias=True)
        else:
            self.node_memory = None
            self.mem_fuse_real = None
            self.mem_fuse_imag = None

        self.predictor = DynamicLinkPredictor(
            dim=d,
            dropout=cfg.dropout,
            time_dim=time_dim,
            use_dot_shortcut=getattr(cfg, "use_dot_shortcut", False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.emb_real.weight, std=self.cfg.emb_init_std)
        nn.init.normal_(self.emb_imag.weight, std=self.cfg.emb_init_std)
        for layer in self.layers:
            layer.reset_parameters()
        if self.query_rotation is not None:
            self.query_rotation.reset_parameters()
        if self.dst_bias is not None:
            nn.init.zeros_(self.dst_bias.weight)
        if self.time_encoder is not None:
            self.time_encoder.reset_parameters()
        if self.node_memory is not None:
            self.node_memory.reset_parameters()
            nn.init.xavier_uniform_(self.mem_fuse_real.weight)
            nn.init.zeros_(self.mem_fuse_real.bias)
            nn.init.xavier_uniform_(self.mem_fuse_imag.weight)
            nn.init.zeros_(self.mem_fuse_imag.bias)
        self.predictor.reset_parameters()

    def reset_memory(self) -> None:
        """Clears persistent node memory (called at epoch start)."""
        if self.node_memory is not None:
            self.node_memory.reset_memory()

    def _fuse_memory(
        self, x_real: torch.Tensor, x_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuses base embeddings with persistent memory state."""
        if self.node_memory is None:
            return x_real, x_imag
        # Read all node memory (no gradient — memory is updated separately)
        all_ids = torch.arange(x_real.size(0), device=x_real.device)
        mem_r, mem_i = self.node_memory.get(all_ids)
        # Fuse: cat([emb, mem]) → Linear → d
        fused_r = F.relu(self.mem_fuse_real(torch.cat([x_real, mem_r], dim=-1)))
        fused_i = F.relu(self.mem_fuse_imag(torch.cat([x_imag, mem_i], dim=-1)))
        # Residual connection keeps gradient flowing into base embeddings
        return fused_r + x_real, fused_i + x_imag

    def encode(
        self,
        hist_edge_index: Optional[torch.Tensor],
        hist_delta_t: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_real = self.emb_real.weight
        x_imag = self.emb_imag.weight
        # Fuse in persistent memory before GNN message passing
        x_real, x_imag = self._fuse_memory(x_real, x_imag)
        if hist_edge_index is None or hist_delta_t is None:
            return x_real, x_imag
        for conv in self.layers:
            h_real, h_imag = conv(x_real, x_imag, hist_edge_index, hist_delta_t)
            x_real = F.relu(h_real) + x_real
            x_imag = F.relu(h_imag) + x_imag
        return x_real, x_imag

    def update_memory(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        delta_t_src: torch.Tensor,
        delta_t_dst: torch.Tensor,
    ) -> None:
        """Updates persistent memory for src and dst nodes after scoring.

        Called AFTER the forward pass + loss.backward() + optimizer.step()
        so memory never leaks into the current batch's gradient computation.
        """
        if self.node_memory is None:
            return
        # Unique active nodes
        all_nodes = torch.cat([src, dst]).unique()
        # Build a delta_t for each: use max(dt_src, dt_dst) per node
        # For simplicity, scatter-reduce to get per-node elapsed time
        N = self.cfg.num_nodes
        node_dt = torch.zeros(N, device=src.device)
        node_dt.scatter_reduce_(0, src, delta_t_src, reduce="amax", include_self=True)
        node_dt.scatter_reduce_(0, dst, delta_t_dst, reduce="amax", include_self=True)
        per_node_dt = node_dt[all_nodes]

        self.node_memory.update(
            all_nodes,
            self.emb_real.weight,
            self.emb_imag.weight,
            per_node_dt,
        )

    def forward(
        self,
        hist_edge_index: Optional[torch.Tensor],
        hist_delta_t: Optional[torch.Tensor],
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        delta_t_src: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluates link logits strictly conditioned on earlier history.

        delta_t_dst and edge_feat are intentionally absent — destination
        elapsed time and query edge features are unknown at prediction time.
        Only delta_t_src (source recency) is causally available.
        """
        x_real, x_imag = self.encode(hist_edge_index, hist_delta_t)

        src_r = x_real[src_idx]
        src_i = x_imag[src_idx]
        dst_r = x_real[dst_idx]
        dst_i = x_imag[dst_idx]

        # Upgrade 1: Query-Time Rotation — SOURCE ONLY
        if self.query_rotation is not None and delta_t_src is not None:
            src_r, src_i = self.query_rotation(src_r, src_i, delta_t_src)

        # Upgrade 3: Harmonic Time Features from source recency
        time_feat = None
        if self.time_encoder is not None and delta_t_src is not None:
            time_feat = self.time_encoder(delta_t_src)

        # Scoring MLP
        logits = self.predictor(
            src_r, src_i, dst_r, dst_i, time_feat=time_feat
        )

        # Upgrade 2: Destination Popularity Prior
        if self.dst_bias is not None:
            logits = logits + self.dst_bias(dst_idx).squeeze(-1)

        return logits


class HistoryBuffer:
    """Maintains last `history_batches` of events for strictly causal message graphs."""

    def __init__(self, cfg: RTGNNConfig):
        self._bidirectional = cfg.bidirectional
        self._chunks = deque(maxlen=cfg.history_batches)

    def clear(self) -> None:
        self._chunks.clear()

    def push(self, src: torch.Tensor, dst: torch.Tensor, t: torch.Tensor) -> None:
        self._chunks.append((src.detach(), dst.detach(), t.detach()))

    def graph(
        self, t_ref: float, time_scale: float
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self._chunks:
            return None, None
        src = torch.cat([c[0] for c in self._chunks])
        dst = torch.cat([c[1] for c in self._chunks])
        t   = torch.cat([c[2] for c in self._chunks])
        delta_t = ((t_ref - t) / time_scale).clamp_min(0.0).float()
        edge_index = torch.stack([src, dst])
        if self._bidirectional:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            delta_t    = torch.cat([delta_t, delta_t])
        return edge_index, delta_t
