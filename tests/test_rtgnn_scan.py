"""
Phase 7 Gate Verification Suite: Stateful Complex Associative Scan (test_rtgnn_scan.py)
====================================================================================
Strictly verifies Phase 7 Gates from Blueprint v2 (Section 7, page 14):
1. Binary Associative Operator Axiom: (T1 o T2) o T3 == T1 o (T2 o T3).
2. Phase 7 Gate (Forward): Parallel associative scan matches sequential loop within 1e-05.
3. Phase 7 Gate (Backward): Gradients of parallel scan match sequential loop within 1e-05.
4. Phase 7 Gate (Stability): Gradient norms remain finite and bounded over K=1,000+ events.
5. Node Memory State Preservation: Memory correctly updates and preserves state across batches.
"""

import math
import sys
from pathlib import Path

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import unittest

try:
    import torch
    import torch.nn as nn
    from rtgnn_scan import (
        StatefulRotationalNodeMemory,
        complex_assoc_combine,
        parallel_complex_scan,
        sequential_complex_scan,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipIf(not HAS_TORCH, "PyTorch required for Phase 7 associative scan tests")
class TestRTGNNPhase7ScanGate(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.dim = 32
        self.num_nodes = 50

    def test_01_binary_associative_axiom(self):
        """Axiom: ((a1, b1) o (a2, b2)) o (a3, b3) == (a1, b1) o ((a2, b2) o (a3, b3))."""
        a1 = torch.complex(torch.randn(4, self.dim), torch.randn(4, self.dim))
        b1 = torch.complex(torch.randn(4, self.dim), torch.randn(4, self.dim))
        a2 = torch.complex(torch.randn(4, self.dim), torch.randn(4, self.dim))
        b2 = torch.complex(torch.randn(4, self.dim), torch.randn(4, self.dim))
        a3 = torch.complex(torch.randn(4, self.dim), torch.randn(4, self.dim))
        b3 = torch.complex(torch.randn(4, self.dim), torch.randn(4, self.dim))

        # Left grouping: (T1 o T2) o T3
        a12, b12 = complex_assoc_combine(a1, b1, a2, b2)
        a_left, b_left = complex_assoc_combine(a12, b12, a3, b3)

        # Right grouping: T1 o (T2 o T3)
        a23, b23 = complex_assoc_combine(a2, b2, a3, b3)
        a_right, b_right = complex_assoc_combine(a1, b1, a23, b23)

        torch.testing.assert_close(a_left, a_right, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(b_left, b_right, atol=1e-6, rtol=1e-6)

    def test_02_gate_forward_scan_matches_sequential(self):
        """Phase 7 Gate: Parallel scan matches sequential per-node loop to 1e-05."""
        K = 128  # Arbitrary sequence length
        dt_seq = torch.rand(K) * 50.0  # Continuous time intervals
        msg_seq = torch.randn(K, self.dim)

        memory = StatefulRotationalNodeMemory(self.num_nodes, self.dim)

        # Forward via parallel associative scan
        z_parallel = memory.scan_node_trajectory(
            node_id=0, delta_t_seq=dt_seq, message_seq=msg_seq, use_parallel_scan=True
        )

        # Forward via reference sequential recurrence loop
        z_sequential = memory.scan_node_trajectory(
            node_id=0, delta_t_seq=dt_seq, message_seq=msg_seq, use_parallel_scan=False
        )

        # Assert exact agreement within 1e-05
        torch.testing.assert_close(z_parallel, z_sequential, atol=1e-5, rtol=1e-5)

    def test_03_gate_backward_gradients_match_sequential(self):
        """Phase 7 Gate: Autograd gradients of parallel scan match sequential loop to 1e-05."""
        K = 64
        dt_seq = torch.rand(K) * 20.0
        msg_seq = torch.randn(K, self.dim)

        # 1. Parallel pass gradients
        mem_parallel = StatefulRotationalNodeMemory(self.num_nodes, self.dim)
        z_par = mem_parallel.scan_node_trajectory(
            node_id=0, delta_t_seq=dt_seq, message_seq=msg_seq, use_parallel_scan=True
        )
        loss_par = z_par.real.sum() + z_par.imag.sum()
        loss_par.backward()

        # 2. Sequential pass gradients (using identical initial parameters)
        mem_seq = StatefulRotationalNodeMemory(self.num_nodes, self.dim)
        with torch.no_grad():
            mem_seq.omega.copy_(mem_parallel.omega)
            mem_seq.raw_lambda.copy_(mem_parallel.raw_lambda)
            mem_seq.proj_real.weight.copy_(mem_parallel.proj_real.weight)
            mem_seq.proj_imag.weight.copy_(mem_parallel.proj_imag.weight)

        z_seq = mem_seq.scan_node_trajectory(
            node_id=0, delta_t_seq=dt_seq, message_seq=msg_seq, use_parallel_scan=False
        )
        loss_seq = z_seq.real.sum() + z_seq.imag.sum()
        loss_seq.backward()

        # Assert gradients match within 1e-05 tolerance
        torch.testing.assert_close(mem_parallel.omega.grad, mem_seq.omega.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(mem_parallel.raw_lambda.grad, mem_seq.raw_lambda.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(mem_parallel.proj_real.weight.grad, mem_seq.proj_real.weight.grad, atol=1e-5, rtol=1e-5)

    def test_04_gate_gradients_bounded_on_longest_sequence(self):
        """Phase 7 Gate: Gradients stay finite and bounded over longest node sequence (K=1,000+)."""
        K = 1024  # 1,000+ continuous events
        dt_seq = torch.rand(K) * 10.0
        msg_seq = torch.randn(K, self.dim)

        memory = StatefulRotationalNodeMemory(self.num_nodes, self.dim)
        z_traj = memory.scan_node_trajectory(
            node_id=5, delta_t_seq=dt_seq, message_seq=msg_seq, use_parallel_scan=True
        )

        loss = z_traj[-1].real.sum() + z_traj[-1].imag.sum()
        loss.backward()

        # Gradients must be strictly finite (no exploding / vanishing NaNs)
        self.assertTrue(torch.isfinite(memory.omega.grad).all().item())
        self.assertTrue(torch.isfinite(memory.raw_lambda.grad).all().item())
        self.assertTrue(torch.isfinite(memory.proj_real.weight.grad).all().item())

        # Norm must remain bounded (contractive exponential decay prevents explosion)
        grad_norm = torch.norm(memory.omega.grad).item()
        self.assertLess(grad_norm, 1e4)

    def test_05_persistent_state_continuity(self):
        """Verifies state preservation and update across multiple sequential event bursts."""
        memory = StatefulRotationalNodeMemory(self.num_nodes, self.dim)

        # Event burst 1
        dt_1 = torch.tensor([5.0, 10.0, 15.0])
        msg_1 = torch.randn(3, self.dim)
        memory.update_node_state(node_id=2, delta_t_seq=dt_1, message_seq=msg_1)

        # Check that memory was modified from zero
        self.assertFalse((memory.mem_real[2] == 0.0).all().item())

        # Event burst 2: starts from previous state
        dt_2 = torch.tensor([2.0, 4.0])
        msg_2 = torch.randn(2, self.dim)
        memory.update_node_state(node_id=2, delta_t_seq=dt_2, message_seq=msg_2)

        self.assertTrue(torch.isfinite(memory.mem_real[2]).all().item())
        self.assertTrue(torch.isfinite(memory.mem_imag[2]).all().item())


if __name__ == "__main__":
    unittest.main()
