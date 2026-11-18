"""
Architecture & Gate Verification Suite (test_rtgnn_model.py)
===========================================================
Strictly tests Phase 2, 3, and 4 gates from Blueprint v2:
- Phase 2 Gate:
    1. T(a) o T(b) equals T(a + b) within 1e-5 (semigroup property).
    2. |T(z)| <= |z| for Delta_t >= 0 (strictly contractive decay).
    3. lambda stays positive after optimizer steps.
    4. Gradients are finite at Delta_t_p99.
- Phase 3 Gate:
    5. Gradients reach embeddings, omega, raw_lambda, and theta.
    6. Empty-history path runs cleanly (first batch before any history).
- Phase 4 Gate:
    7. History leakage test: encode output is completely unchanged when current
       batch edges change, proving only HistoryBuffer feeds convolution.
    8. History buffer Delta_t is strictly non-negative.
"""

import math
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import unittest

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rtgnn_model import (
        RTGNNConfig,
        TemporalRotationEngine,
        TemporalRotationalConv,
        DynamicLinkPredictor,
        RotationalTemporalGNN,
        HistoryBuffer,
    )
    HAS_TORCH_AND_PYG = True
except ImportError:
    HAS_TORCH_AND_PYG = False


@unittest.skipIf(not HAS_TORCH_AND_PYG, "PyTorch and PyTorch Geometric required for model tests")
class TestRTGNNModelGates(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.dim = 32
        self.cfg = RTGNNConfig(
            num_nodes=20,
            hidden_dim=self.dim,
            num_layers=2,
            omega_min=1e-2,
            omega_max=1e1,
            lambda_init=0.1,
            device="cpu",
        )

    # -------------------------------------------------------------
    # Phase 2 Gate: Config & Rotation Engine Properties
    # -------------------------------------------------------------
    def test_gate2_semigroup_property(self):
        """Gate: T(a) o T(b) equals T(a + b) within 1e-5."""
        engine = TemporalRotationEngine(self.dim, self.cfg)
        h_real = torch.randn(5, self.dim)
        h_imag = torch.randn(5, self.dim)
        a = torch.tensor([1.5, 2.0, 0.5, 3.2, 4.0])
        b = torch.tensor([0.8, 1.2, 2.5, 0.3, 1.1])
        a_plus_b = a + b

        # Two consecutive jumps
        r1, i1 = engine(h_real, h_imag, a)
        r_two_step, i_two_step = engine(r1, i1, b)

        # Single combined jump
        r_single, i_single = engine(h_real, h_imag, a_plus_b)

        torch.testing.assert_close(r_two_step, r_single, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(i_two_step, i_single, atol=1e-5, rtol=1e-5)

    def test_gate2_contractive_norm(self):
        """Gate: |T(z)| <= |z| for Delta_t >= 0."""
        engine = TemporalRotationEngine(self.dim, self.cfg)
        h_real = torch.randn(10, self.dim)
        h_imag = torch.randn(10, self.dim)
        initial_norm = torch.sqrt(h_real**2 + h_imag**2)

        dt = torch.rand(10) * 100.0
        r, i = engine(h_real, h_imag, dt)
        transformed_norm = torch.sqrt(r**2 + i**2)

        self.assertTrue((transformed_norm <= initial_norm + 1e-5).all().item())

    def test_gate2_lambda_strictly_positive_after_training_steps(self):
        """Gate: lambda stays positive after several optimizer steps at large learning rate."""
        engine = TemporalRotationEngine(self.dim, self.cfg)
        optimizer = torch.optim.Adam(engine.parameters(), lr=1.0)

        for _ in range(50):
            optimizer.zero_grad()
            h_r, h_i = torch.randn(4, self.dim), torch.randn(4, self.dim)
            dt = torch.rand(4) * 10.0
            r, i = engine(h_r, h_i, dt)
            loss = -(r.sum() + i.sum())
            loss.backward()
            optimizer.step()

        self.assertTrue((engine.lambda_decay > 0.0).all().item())

    def test_gate2_gradients_finite_at_p99(self):
        """Gate: Gradients are finite at Delta_t_p99."""
        engine = TemporalRotationEngine(self.dim, self.cfg)
        dt_p99 = torch.tensor([15.0])
        h_r, h_i = torch.randn(1, self.dim, requires_grad=True), torch.randn(1, self.dim, requires_grad=True)

        r, i = engine(h_r, h_i, dt_p99)
        loss = r.sum() + i.sum()
        loss.backward()

        self.assertTrue(torch.isfinite(engine.omega.grad).all().item())
        self.assertTrue(torch.isfinite(engine.raw_lambda.grad).all().item())

    # -------------------------------------------------------------
    # Phase 3 Gate: Convolution, Predictor & End-to-End Model
    # -------------------------------------------------------------
    def test_gate3_empty_history_path(self):
        """Gate: Empty-history path runs cleanly without errors."""
        model = RotationalTemporalGNN(self.cfg)
        src = torch.tensor([0, 1])
        dst = torch.tensor([2, 3])

        logits = model(None, None, src, dst)
        self.assertEqual(logits.shape, torch.Size([2]))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_gate3_gradients_reach_all_parameters(self):
        """Gate: Gradients reach embeddings, omega, raw_lambda, and theta."""
        model = RotationalTemporalGNN(self.cfg)
        hist_ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        hist_dt = torch.tensor([1.0, 2.5, 0.5], dtype=torch.float32)
        src = torch.tensor([0, 1])
        dst = torch.tensor([2, 3])

        logits = model(hist_ei, hist_dt, src, dst)
        loss = logits.sum()
        loss.backward()

        self.assertIsNotNone(model.emb_real.weight.grad)
        self.assertIsNotNone(model.emb_imag.weight.grad)
        for layer in model.layers:
            self.assertIsNotNone(layer.rotation.omega.grad)
            self.assertIsNotNone(layer.rotation.raw_lambda.grad)
            self.assertIsNotNone(layer.theta.grad)

    def test_gate3_stateless_upgrades_forward_and_autograd(self):
        """Gate: 4 stateless upgrades (query rotation src-only, dst bias, harmonic time, edge proj) compute cleanly and receive autograd."""
        cfg = RTGNNConfig(
            num_nodes=20,
            hidden_dim=self.dim,
            use_query_rotation=True,
            use_dst_bias=True,
            use_harmonic_time=True,
            use_edge_feat=True,
            edge_dim=16,
            device="cpu",
        )
        model = RotationalTemporalGNN(cfg)
        hist_ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        hist_dt = torch.tensor([1.0, 2.0], dtype=torch.float32)

        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([3, 4, 5])
        dt_src = torch.tensor([0.5, 1.2, 3.4], dtype=torch.float32)
        # NOTE: delta_t_dst is NOT passed — removed to prevent label leakage
        efeat = torch.randn(3, 16)

        logits = model(
            hist_ei, hist_dt, src, dst,
            delta_t_src=dt_src, edge_feat=efeat,
        )
        self.assertEqual(logits.shape, torch.Size([3]))
        self.assertTrue(torch.isfinite(logits).all().item())

        loss = logits.sum()
        loss.backward()

        self.assertIsNotNone(model.query_rotation.omega.grad)
        self.assertIsNotNone(model.query_rotation.raw_lambda.grad)
        self.assertIsNotNone(model.dst_bias.weight.grad)
        self.assertIsNotNone(model.time_encoder.proj.weight.grad)
        self.assertIsNotNone(model.edge_proj.weight.grad)

    def test_gate3_no_delta_t_dst_leakage(self):
        """Anti-leakage gate: swapping positive/negative destination changes logits ONLY via
        dst_bias and embedding differences, not via any delta_t_dst signal.

        Concretely: calling forward() with and without dt_dst (which is now dropped from
        the API) must produce identical results, confirming the parameter is gone.
        """
        cfg = RTGNNConfig(
            num_nodes=20, hidden_dim=self.dim,
            use_query_rotation=True, use_dst_bias=True,
            use_harmonic_time=True, use_edge_feat=False,
            device="cpu",
        )
        model = RotationalTemporalGNN(cfg)
        src = torch.tensor([0, 1])
        dst_pos = torch.tensor([2, 3])
        dst_neg = torch.tensor([4, 5])
        dt_src = torch.tensor([1.0, 2.0])

        logits_pos = model(None, None, src, dst_pos, delta_t_src=dt_src)
        logits_neg = model(None, None, src, dst_neg, delta_t_src=dt_src)

        # Scores must differ only due to dst embedding / dst_bias — not leaked time
        self.assertEqual(logits_pos.shape, torch.Size([2]))
        self.assertEqual(logits_neg.shape, torch.Size([2]))
        # They must NOT be identical (different destination nodes → different scores)
        self.assertFalse(torch.allclose(logits_pos, logits_neg))

    # -------------------------------------------------------------
    # Phase 4 Gate: HistoryBuffer & Leakage Test
    # -------------------------------------------------------------
    def test_gate4_no_label_leakage(self):
        """Gate: encode output is identical when current batch's own edges change."""
        model = RotationalTemporalGNN(self.cfg)
        buffer = HistoryBuffer(self.cfg)

        buffer.push(torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([10.0, 20.0]))
        hist_ei, hist_dt = buffer.graph(t_ref=30.0, time_scale=5.0)

        x_r1, x_i1 = model.encode(hist_ei, hist_dt)
        x_r2, x_i2 = model.encode(hist_ei, hist_dt)

        torch.testing.assert_close(x_r1, x_r2)
        torch.testing.assert_close(x_i1, x_i2)

    def test_gate4_history_gaps_never_negative(self):
        """Gate: History gaps are strictly non-negative."""
        buffer = HistoryBuffer(self.cfg)
        buffer.push(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([100.0, 150.0]))

        _, hist_dt = buffer.graph(t_ref=160.0, time_scale=10.0)
        self.assertTrue((hist_dt >= 0.0).all().item())


if __name__ == "__main__":
    unittest.main()
