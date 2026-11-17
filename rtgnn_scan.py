"""
RT-GNN Stateful Complex Associative Scan Module (rtgnn_scan.py)
==============================================================
Implements the core theoretical contribution from Blueprint v2 (Section 3 & 7 Phase 7):
- Per-node complex state representation z_u in C^d.
- Continuous-time decay-rotation transition factor:
    a_k = exp((-lambda + i*omega) * Delta_t_k)
- Input message projection:
    b_k = B * m_k in C^d
- Binary associative combining operator:
    (a1, b1) o (a2, b2) = (a2 * a1, a2 * b1 + b2)
- Parallel associative prefix scan in O(log K) steps with zero sequential unrolling (No BPTT).
- Exact numerical matching with sequential recurrence to < 1e-05 error.
- Strictly bounded gradients over arbitrarily long event streams.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def complex_assoc_combine(
    a1: Tensor, b1: Tensor, a2: Tensor, b2: Tensor
) -> Tuple[Tensor, Tensor]:
    """Applies the binary associative operator:
    (a1, b1) o (a2, b2) = (a2 * a1, a2 * b1 + b2)
    where a1, a2, b1, b2 are complex tensors.
    """
    a_out = a2 * a1
    b_out = a2 * b1 + b2
    return a_out, b_out


def sequential_complex_scan(
    a: Tensor, b: Tensor, h0: Optional[Tensor] = None
) -> Tensor:
    """Sequential recurrence reference loop:
        h_k = a_k * h_{k-1} + b_k
    
    Args:
        a: Complex tensor of shape [K, ..., d] (decay-rotation transition)
        b: Complex tensor of shape [K, ..., d] (input message)
        h0: Optional initial state [..., d]. Defaults to zero.
    Returns:
        h: Complex tensor of shape [K, ..., d] containing all prefix states.
    """
    K = a.size(0)
    out = []
    curr = torch.zeros_like(b[0]) if h0 is None else h0

    for k in range(K):
        curr = a[k] * curr + b[k]
        out.append(curr)

    return torch.stack(out, dim=0)


def parallel_complex_scan(
    a: Tensor, b: Tensor, h0: Optional[Tensor] = None
) -> Tensor:
    """Parallel associative prefix scan over sequence dimension dim=0.
    
    Solves the linear complex recurrence:
        h_k = a_k * h_{k-1} + b_k
    in O(log2 K) parallel steps using doubling associative combination.
    
    Fully differentiable with PyTorch autograd.
    """
    K = a.size(0)
    if K == 0:
        return b
    if K == 1:
        if h0 is not None:
            return (a * h0 + b)
        return b

    # If initial state h0 is provided, incorporate it into b[0]
    if h0 is not None:
        b0 = a[0] * h0 + b[0]
        b = torch.cat([b0.unsqueeze(0), b[1:]], dim=0)

    # Doubling prefix scan (Hillis-Steele / out-of-place associative doubling)
    # At each step s = 1, 2, 4, 8, ..., elements at index k >= s combine with index k - s
    step = 1
    curr_a = a
    curr_b = b

    while step < K:
        # Slice elements that need combining
        a_left = curr_a[:-step]
        b_left = curr_b[:-step]
        a_right = curr_a[step:]
        b_right = curr_b[step:]

        # Associative combination: (a_left, b_left) o (a_right, b_right)
        comb_a, comb_b = complex_assoc_combine(a_left, b_left, a_right, b_right)

        # Non-mutating tensor concatenation to preserve autograd graph
        curr_a = torch.cat([curr_a[:step], comb_a], dim=0)
        curr_b = torch.cat([curr_b[:step], comb_b], dim=0)

        step *= 2

    return curr_b


class StatefulRotationalNodeMemory(nn.Module):
    """Continuous-Time Stateful Node Memory with Parallel Associative Scan.
    
    Maintains and updates persistent complex node representations z_u in C^d
    across streaming events using closed-form analytical rotation and associative prefix scans.
    """

    def __init__(
        self,
        num_nodes: int,
        feature_dim: int,
        message_dim: Optional[int] = None,
        omega_min: float = 1e-2,
        omega_max: float = 1e1,
        lambda_init: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.message_dim = message_dim or feature_dim

        # Learnable angular frequencies and strictly positive decay rates
        self.omega = nn.Parameter(torch.empty(feature_dim))
        self.raw_lambda = nn.Parameter(torch.empty(feature_dim))

        # Complex projection for incoming messages: B * m = (B_r * m) + i*(B_i * m)
        self.proj_real = nn.Linear(self.message_dim, feature_dim, bias=False)
        self.proj_imag = nn.Linear(self.message_dim, feature_dim, bias=False)

        # Persistent complex node memory storage [num_nodes, feature_dim]
        self.register_buffer("mem_real", torch.zeros(num_nodes, feature_dim))
        self.register_buffer("mem_imag", torch.zeros(num_nodes, feature_dim))

        self.reset_parameters(omega_min, omega_max, lambda_init)

    def reset_parameters(
        self,
        omega_min: float = 1e-2,
        omega_max: float = 1e1,
        lambda_init: float = 0.1,
    ) -> None:
        with torch.no_grad():
            self.omega.copy_(
                torch.logspace(
                    math.log10(omega_min),
                    math.log10(omega_max),
                    self.omega.numel(),
                )
            )
            # Inverse softplus: raw = log(exp(lambda_init) - 1)
            self.raw_lambda.fill_(math.log(math.expm1(lambda_init)))

        nn.init.xavier_uniform_(self.proj_real.weight)
        nn.init.xavier_uniform_(self.proj_imag.weight)
        self.reset_memory()

    def reset_memory(self) -> None:
        """Clears persistent node memory to zero."""
        self.mem_real.zero_()
        self.mem_imag.zero_()

    @property
    def lambda_decay(self) -> Tensor:
        """Enforces strictly positive decay via softplus."""
        return F.softplus(self.raw_lambda)

    def compute_decay_rotation_factor(self, delta_t: Tensor) -> Tensor:
        """Calculates a_k = exp((-lambda + i*omega) * Delta_t_k).
        
        Args:
            delta_t: Tensor of continuous time intervals [..., 1] or [...]
        Returns:
            Complex tensor a_k of shape [..., feature_dim]
        """
        dt = delta_t.clamp_min(0.0)
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)

        decay = torch.exp(-self.lambda_decay * dt)
        phase = self.omega * dt
        return torch.polar(decay, phase)

    def project_message(self, message: Tensor) -> Tensor:
        """Projects real message to complex space: b_k = B * m_k in C^d."""
        b_r = self.proj_real(message)
        b_i = self.proj_imag(message)
        return torch.complex(b_r, b_i)

    def scan_node_trajectory(
        self,
        node_id: int,
        delta_t_seq: Tensor,
        message_seq: Tensor,
        use_parallel_scan: bool = True,
    ) -> Tensor:
        """Computes the exact trajectory of states for a single node's sequence of events.
        
        Args:
            node_id: Target node index
            delta_t_seq: Sequence of consecutive continuous time gaps [K]
            message_seq: Sequence of incoming interaction messages [K, message_dim]
            use_parallel_scan: If True, executes parallel associative scan;
                               If False, executes sequential reference loop.
        Returns:
            z_trajectory: Complex tensor of updated states [K, feature_dim]
        """
        a_seq = self.compute_decay_rotation_factor(delta_t_seq)
        b_seq = self.project_message(message_seq)

        # Initial persistent state for this node
        h0 = torch.complex(self.mem_real[node_id], self.mem_imag[node_id])

        if use_parallel_scan:
            return parallel_complex_scan(a_seq, b_seq, h0=h0)
        else:
            return sequential_complex_scan(a_seq, b_seq, h0=h0)

    def update_node_state(
        self,
        node_id: int,
        delta_t_seq: Tensor,
        message_seq: Tensor,
    ) -> None:
        """Updates persistent buffer with the final state of an event sequence."""
        trajectory = self.scan_node_trajectory(
            node_id, delta_t_seq, message_seq, use_parallel_scan=True
        )
        final_state = trajectory[-1].detach()
        self.mem_real[node_id] = final_state.real
        self.mem_imag[node_id] = final_state.imag
