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
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Union

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
    history_batches: int = 8           # Past batches kept as the message graph
    lr: float = 5e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    grad_clip: Optional[float] = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Stateless Performance Upgrades (O(1) continuous time):
    use_query_rotation: bool = True     # 1. Query-time rotation T_dt(z) on scoring pairs
    use_dst_bias: bool = True           # 2. Destination popularity / frequency prior
    use_harmonic_time: bool = True      # 3. Multi-harmonic Fourier time encodings
    num_harmonics: int = 8              # Number of harmonic octave bands
    use_edge_feat: bool = True          # 4. Project edge features if provided
    edge_dim: int = 172                 # Dimension of dataset edge features (172 for tgbl-wiki)


class TemporalRotationEngine(nn.Module):
    """Computes continuous-time complex rotation and decay.
    
    Formula:
        T_dt(z) = z * exp((-lambda + i*omega) * dt)
    
    Note: As specified in Section 3 of Blueprint v2, theta is NOT in the time operator,
    ensuring the semigroup composition property: T(a) o T(b) = T(a + b).
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
            # Inverse softplus initialization: raw = log(exp(lambda_init) - 1)
            self.raw_lambda.fill_(math.log(math.expm1(c.lambda_init)))

    @property
    def lambda_decay(self) -> torch.Tensor:
        """Enforces strictly positive decay: lambda = softplus(raw_lambda) > 0."""
        return F.softplus(self.raw_lambda)

    def forward(
        self, h_real: torch.Tensor, h_imag: torch.Tensor, delta_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies continuous time operator to complex state.
        
        Args:
            h_real: [..., dim] Real component
            h_imag: [..., dim] Imaginary component
            delta_t: [...] Continuous time elapsed (clamped >= 0)
        """
        dt = delta_t.clamp_min(0.0)  # Prevents explosive growth from negative time deltas
        if dt.dim() == 1 or dt.dim() == h_real.dim() - 1:
            dt = dt.unsqueeze(-1)

        decay = torch.exp(-self.lambda_decay * dt)
        operator = torch.polar(decay, self.omega * dt)
        z = torch.complex(h_real, h_imag) * operator
        return z.real, z.imag


class TemporalRotationalConv(MessagePassing):
    """Continuous-time rotational graph convolution layer.
    
    Mean-aggregates rotated source features over earlier interaction graph edges.
    Returns concatenated [real || imag] tensor in message() for robust PyG compatibility.
    """

    def __init__(self, in_channels: int, out_channels: int, cfg: RTGNNConfig):
        super().__init__(aggr="mean")
        self.out_channels = out_channels
        self.lin_real = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_imag = nn.Linear(in_channels, out_channels, bias=False)
        self.theta = nn.Parameter(torch.empty(out_channels))  # Constant per-channel phase
        self.rotation = TemporalRotationEngine(out_channels, cfg)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_real.weight)
        nn.init.xavier_uniform_(self.lin_imag.weight)
        nn.init.uniform_(self.theta, -math.pi, math.pi)
        self.rotation.reset_parameters()

    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        edge_index: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([self.lin_real(x_real), self.lin_imag(x_imag)], dim=-1)
        out = self.propagate(
            edge_index,
            x=x,
            delta_t=delta_t,
            size=(x.size(0), x.size(0)),
        )
        return out[:, : self.out_channels], out[:, self.out_channels :]

    def message(self, x_j: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        # Separate real and imaginary components of projected source nodes
        r, i = self.rotation(
            x_j[:, : self.out_channels],
            x_j[:, self.out_channels :],
            delta_t,
        )
        # Apply constant phase theta once: m = exp(i * theta) * T_dt(W * z)
        z = torch.complex(r, i) * torch.polar(torch.ones_like(self.theta), self.theta)
        return torch.cat([z.real, z.imag], dim=-1)


class HarmonicTimeEncoder(nn.Module):
    """Multi-harmonic Fourier temporal feature encoder.
    Maps continuous time intervals Delta_t to multi-scale sinusoidal features:
    [cos(2^k * pi * omega * Delta_t), sin(2^k * pi * omega * Delta_t)]
    """

    def __init__(self, num_harmonics: int = 8, out_dim: int = 32):
        super().__init__()
        self.num_harmonics = num_harmonics
        # Logarithmically spaced harmonic octave frequencies
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
    """Predicts dynamic link likelihood logits from source and destination complex states,
    with optional multi-harmonic temporal features and edge feature projections.
    """

    def __init__(
        self,
        dim: int,
        dropout: float,
        time_dim: int = 0,
        edge_dim: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        in_dim = 4 * dim + time_dim + edge_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def reset_parameters(self) -> None:
        for m in self.net:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(
        self,
        src_r: torch.Tensor,
        src_i: torch.Tensor,
        dst_r: torch.Tensor,
        dst_i: torch.Tensor,
        time_feat: Optional[torch.Tensor] = None,
        edge_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats = [src_r, src_i, dst_r, dst_i]
        if self.time_dim > 0:
            if time_feat is not None:
                feats.append(time_feat)
            else:
                feats.append(torch.zeros(src_r.size(0), self.time_dim, device=src_r.device, dtype=src_r.dtype))
        if self.edge_dim > 0:
            if edge_feat is not None:
                feats.append(edge_feat)
            else:
                feats.append(torch.zeros(src_r.size(0), self.edge_dim, device=src_r.device, dtype=src_r.dtype))

        cat_feat = torch.cat(feats, dim=-1)
        return self.net(cat_feat).squeeze(-1)


class RotationalTemporalGNN(nn.Module):
    """End-to-End Rotational Temporal Graph Neural Network (RT-GNN)."""

    def __init__(self, cfg: RTGNNConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.cfg = cfg
        self.emb_real = nn.Embedding(cfg.num_nodes, d)
        self.emb_imag = nn.Embedding(cfg.num_nodes, d)
        self.layers = nn.ModuleList([
            TemporalRotationalConv(d, d, cfg) for _ in range(cfg.num_layers)
        ])

        # Upgrade 1: Query-Time Rotation Engine
        if getattr(cfg, "use_query_rotation", True):
            self.query_rotation = TemporalRotationEngine(d, cfg)
        else:
            self.query_rotation = None

        # Upgrade 2: Destination Popularity / Frequency Prior
        if getattr(cfg, "use_dst_bias", True):
            self.dst_bias = nn.Embedding(cfg.num_nodes, 1)
        else:
            self.dst_bias = None

        # Upgrade 3: Multi-Harmonic Fourier Time Encoder
        if getattr(cfg, "use_harmonic_time", True):
            time_dim = d // 2
            self.time_encoder = HarmonicTimeEncoder(num_harmonics=getattr(cfg, "num_harmonics", 8), out_dim=time_dim)
        else:
            time_dim = 0
            self.time_encoder = None

        # Upgrade 4: Edge Feature Projection
        if getattr(cfg, "use_edge_feat", True) and getattr(cfg, "edge_dim", 0) > 0:
            edge_proj_dim = d // 2
            self.edge_proj = nn.Linear(cfg.edge_dim, edge_proj_dim)
        else:
            edge_proj_dim = 0
            self.edge_proj = None

        self.predictor = DynamicLinkPredictor(
            dim=d,
            dropout=cfg.dropout,
            time_dim=time_dim,
            edge_dim=edge_proj_dim,
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
        if self.edge_proj is not None:
            nn.init.xavier_uniform_(self.edge_proj.weight)
            nn.init.zeros_(self.edge_proj.bias)
        self.predictor.reset_parameters()

    def encode(
        self,
        hist_edge_index: Optional[torch.Tensor],
        hist_delta_t: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes node representations using the history message graph."""
        x_real, x_imag = self.emb_real.weight, self.emb_imag.weight
        if hist_edge_index is None or hist_delta_t is None:
            return x_real, x_imag

        for conv in self.layers:
            h_real, h_imag = conv(x_real, x_imag, hist_edge_index, hist_delta_t)
            # Split activation: ReLU applied separately to real and imaginary components
            x_real = F.relu(h_real) + x_real
            x_imag = F.relu(h_imag) + x_imag

        return x_real, x_imag

    def forward(
        self,
        hist_edge_index: Optional[torch.Tensor],
        hist_delta_t: Optional[torch.Tensor],
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        delta_t_src: Optional[torch.Tensor] = None,
        edge_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluates link logits strictly conditioned on earlier history.

        DESIGN NOTE — delta_t_dst is intentionally absent:
        The destination's elapsed-time-since-last-contact is unknown at
        inference time (that is what we are predicting).  Passing the real
        delta_t_dst for positive targets and 0.0 for negatives is perfect
        label leakage.  Only delta_t_src (source recency) is causal and safe.

        Args:
            hist_edge_index: edge index of past events [2, E_hist] or None
            hist_delta_t:    normalised time gaps for past events [E_hist] or None
            src_idx:         source node indices [B]
            dst_idx:         destination node indices [B]
            delta_t_src:     time since source last acted [B] (causal, optional)
            edge_feat:       projected edge features [B, feat_dim] (optional;
                             should be zeros for negative candidates)
        """
        x_real, x_imag = self.encode(hist_edge_index, hist_delta_t)

        src_r = x_real[src_idx]
        src_i = x_imag[src_idx]
        dst_r = x_real[dst_idx]
        dst_i = x_imag[dst_idx]

        # Upgrade 1: Query-Time Rotation & Decay — SOURCE ONLY (causal)
        if self.query_rotation is not None and delta_t_src is not None:
            src_r, src_i = self.query_rotation(src_r, src_i, delta_t_src)
        # NOTE: dst rotation removed — delta_t_dst is unknown at prediction time

        # Upgrade 3: Multi-Harmonic Fourier Time Features
        time_feat = None
        if self.time_encoder is not None and delta_t_src is not None:
            time_feat = self.time_encoder(delta_t_src)

        # Upgrade 4: Edge Feature Projection
        edge_emb = None
        if self.edge_proj is not None and edge_feat is not None:
            edge_emb = self.edge_proj(edge_feat)

        # Scoring MLP
        logits = self.predictor(
            src_r, src_i, dst_r, dst_i, time_feat=time_feat, edge_feat=edge_emb
        )

        # Upgrade 2: Destination Popularity / Frequency Prior
        if self.dst_bias is not None:
            logits = logits + self.dst_bias(dst_idx).squeeze(-1)

        return logits


class HistoryBuffer:
    """Maintains last `history_batches` of events to construct strictly causal message graphs."""

    def __init__(self, cfg: RTGNNConfig):
        self._bidirectional = cfg.bidirectional
        self._chunks = deque(maxlen=cfg.history_batches)

    def clear(self) -> None:
        self._chunks.clear()

    def push(self, src: torch.Tensor, dst: torch.Tensor, t: torch.Tensor) -> None:
        # Detach to ensure history doesn't track computational graphs across steps
        self._chunks.append((src.detach(), dst.detach(), t.detach()))

    def graph(
        self, t_ref: float, time_scale: float
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self._chunks:
            return None, None

        src = torch.cat([c[0] for c in self._chunks])
        dst = torch.cat([c[1] for c in self._chunks])
        t = torch.cat([c[2] for c in self._chunks])

        delta_t = ((t_ref - t) / time_scale).clamp_min(0.0).float()
        edge_index = torch.stack([src, dst])

        if self._bidirectional:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            delta_t = torch.cat([delta_t, delta_t])

        return edge_index, delta_t
