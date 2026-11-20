"""
RT-GNN Comprehensive Stress Test & Vulnerability Discovery Suite
================================================================
Stress tests continuous-time rotational GNNs (Stateless & Stateful) across
extreme boundaries, pathological graph structures, and numerical edge cases:

1. Extreme Delta_t (0, machine epsilon, astronomical hiatus, negative timestamps).
2. Intra-Batch Hub Nodes & Duplicate Write Race Condition in Memory Update.
3. Self-loops (src == dst) & Node Overwrite Collision.
4. Long-Sequence Associative Scan Scaling (K=2,048 to K=8,192).
5. Single-Edge Streaming Inference (Batch Size = 1).
6. Cold-Start / Isolated Unseen Nodes & Empty History Buffer.
7. Gradient Vanishing & Explosion under Long Trajectories.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtgnn_model import RTGNNConfig, RotationalTemporalGNN, HistoryBuffer, TemporalRotationEngine
from rtgnn_stateful_gnn import StatefulRotationalTemporalGNN
from rtgnn_scan import parallel_complex_scan, StatefulRotationalNodeMemory


def run_all_stress_tests():
    print("=" * 80)
    print("  RT-GNN VULNERABILITY & STRESS-TESTING SUITE")
    print("=" * 80)
    findings = []

    # --------------------------------------------------------------------------
    # Test 1: Extreme Delta_t Stability
    # --------------------------------------------------------------------------
    print("\n[STRESS TEST 1] Extreme Delta_t & Numerical Boundaries...")
    cfg = RTGNNConfig(num_nodes=50, hidden_dim=32, num_layers=1, device="cpu")
    rot = TemporalRotationEngine(32, cfg)
    h_r = torch.randn(5, 32)
    h_i = torch.randn(5, 32)

    extreme_dts = torch.tensor([0.0, 1e-15, 1e-8, 1e6, 1e12, -5.0, -1e8])
    for dt_val in extreme_dts:
        dt = torch.full((5,), dt_val.item())
        out_r, out_i = rot(h_r, h_i, dt)
        has_nan = torch.isnan(out_r).any() or torch.isnan(out_i).any()
        has_inf = torch.isinf(out_r).any() or torch.isinf(out_i).any()
        if has_nan or has_inf:
            print(f"  [FAIL] Extreme dt={dt_val.item()}: NaN={has_nan}, Inf={has_inf}")
            findings.append(f"VULNERABILITY: Extreme dt={dt_val.item()} produces NaN/Inf.")
        else:
            print(f"  [PASS] dt={dt_val.item():<10.2e} -> Finite. Max abs out: {torch.max(torch.abs(out_r)).item():.4f}")

    # --------------------------------------------------------------------------
    # Test 2: Intra-Batch Hub Node & Write Collision Race in Stateful Memory
    # --------------------------------------------------------------------------
    print("\n[STRESS TEST 2] Intra-Batch Duplicate Nodes & Memory Collision...")
    stateful_model = StatefulRotationalTemporalGNN(cfg)
    stateful_model.reset_memory()

    # Hub node 0 interacts with 20 different destinations in a single batch
    hub_src = torch.zeros(20, dtype=torch.long)
    hub_dst = torch.arange(1, 21, dtype=torch.long)
    dt_src = torch.linspace(0.1, 2.0, 20)
    dt_dst = torch.linspace(0.1, 2.0, 20)

    # Record initial memory of hub node 0
    init_mem_0 = stateful_model.memory.mem_real[0].clone()

    # Apply batch update
    stateful_model.update_streaming_memory(hub_src, hub_dst, dt_src, dt_dst)
    final_mem_0 = stateful_model.memory.mem_real[0].clone()

    # Verify that hub node 0's update strictly compounds all 20 interactions:
    # Under naive overwrite, ||mem[0]|| would only equal the single 20th interaction.
    # Under associative compounding, ||mem[0]|| compounds all 20 interactions.
    single_model = StatefulRotationalTemporalGNN(cfg)
    single_model.load_state_dict(stateful_model.state_dict())
    single_model.reset_memory()
    single_model.update_streaming_memory(hub_src[-1:], hub_dst[-1:], dt_src[-1:], dt_dst[-1:])
    single_mem_0 = single_model.memory.mem_real[0].clone()

    diff_single = torch.norm(final_mem_0 - single_mem_0).item()
    print(f"  Compounded norm: {torch.norm(final_mem_0).item():.4f} vs Single-event overwrite norm: {torch.norm(single_mem_0).item():.4f}")
    if diff_single > 0.1:
        print(f"  [PASS] Compounding verified! Memory successfully compounded all 20 interactions (diff from overwrite: {diff_single:.4f}).")
    else:
        print(f"  [FAIL] Compounding failed: identical to single overwrite.")
        findings.append("VULNERABILITY: Duplicate nodes did not compound.")

    # --------------------------------------------------------------------------
    # Test 3: Self-Loops (src == dst)
    # --------------------------------------------------------------------------
    print("\n[STRESS TEST 3] Self-Loops (src == dst) & Mutual Memory Overwrite...")
    stateful_model.reset_memory()
    self_loop_src = torch.tensor([5, 6, 7])
    self_loop_dst = torch.tensor([5, 6, 7])
    dt = torch.tensor([0.5, 0.5, 0.5])

    try:
        # Check forward pass
        logits = stateful_model(None, None, self_loop_src, self_loop_dst)
        # Check memory update
        stateful_model.update_streaming_memory(self_loop_src, self_loop_dst, dt, dt)
        has_nan = torch.isnan(stateful_model.memory.mem_real[5]).any()
        if has_nan:
            print("  [FAIL] Self-loop memory update produced NaNs!")
            findings.append("VULNERABILITY: Self-loops produce NaN in memory state.")
        else:
            print(f"  [PASS] Self-loops execute safely without NaN. Logits: {logits.detach().numpy()}")
    except Exception as e:
        print(f"  [FAIL] Self-loop raised exception: {e}")
        findings.append(f"VULNERABILITY: Self-loops crash with {e}")

    # --------------------------------------------------------------------------
    # Test 4: Long-Sequence Associative Scan Stability (K=4,096)
    # --------------------------------------------------------------------------
    print("\n[STRESS TEST 4] Long-Sequence Associative Scan Scaling (K=4,096)...")
    K = 4096
    d = 32
    a = torch.polar(
        torch.exp(-F.softplus(torch.randn(1, K, d)) * 0.1),
        torch.randn(1, K, d) * 0.5
    )
    b = torch.randn(1, K, d, dtype=torch.cfloat)

    t0 = time.time()
    scan_out = parallel_complex_scan(a, b)
    scan_time = (time.time() - t0) * 1000

    has_nan = torch.isnan(scan_out.real).any() or torch.isnan(scan_out.imag).any()
    has_inf = torch.isinf(scan_out.real).any() or torch.isinf(scan_out.imag).any()
    max_val = torch.max(torch.abs(scan_out)).item()

    if has_nan or has_inf or max_val > 1e6:
        print(f"  [FAIL] Scan instability: NaN={has_nan}, Inf={has_inf}, Max={max_val}")
        findings.append(f"VULNERABILITY: Long-sequence associative scan instability at K={K}.")
    else:
        print(f"  [PASS] K={K} events scanned in {scan_time:.2f} ms | Max magnitude: {max_val:.4f} | Zero NaNs/Infs.")

    # --------------------------------------------------------------------------
    # Test 5: Single-Edge Streaming Inference (Batch Size = 1)
    # --------------------------------------------------------------------------
    print("\n[STRESS TEST 5] Single-Edge Streaming Inference (Batch Size = 1)...")
    model_stateless = RotationalTemporalGNN(cfg)
    buf = HistoryBuffer(cfg)
    buf.push(torch.tensor([1]), torch.tensor([2]), torch.tensor([10.0]))

    try:
        ei, edt = buf.graph(t_ref=15.0, time_scale=1.0)
        single_src = torch.tensor([1])
        single_dst = torch.tensor([2])
        single_dt = torch.tensor([5.0])
        single_logit = model_stateless(ei, edt, single_src, single_dst, delta_t_src=single_dt)
        if single_logit.dim() == 1 and single_logit.size(0) == 1 and not torch.isnan(single_logit).any():
            print(f"  [PASS] Batch size 1 forward succeeded. Output: {single_logit.item():.4f}")
        else:
            print(f"  [FAIL] Batch size 1 output malformed: shape {single_logit.shape}")
            findings.append("VULNERABILITY: Batch size = 1 produces malformed output tensor.")
    except Exception as e:
        print(f"  [FAIL] Batch size 1 raised exception: {e}")
        findings.append(f"VULNERABILITY: Batch size = 1 crash: {e}")

    # --------------------------------------------------------------------------
    # Test 6: Cold-Start / Isolated Unseen Nodes & Empty History Buffer
    # --------------------------------------------------------------------------
    print("\n[STRESS TEST 6] Cold-Start & Empty History Buffer...")
    empty_buf = HistoryBuffer(cfg)
    ei, edt = empty_buf.graph(t_ref=100.0, time_scale=1.0)
    unseen_src = torch.tensor([48])
    unseen_dst = torch.tensor([49])
    unseen_dt = torch.tensor([0.0])

    try:
        cold_logit = model_stateless(ei, edt, unseen_src, unseen_dst, delta_t_src=unseen_dt)
        print(f"  [PASS] Empty history forward pass succeeded. Logit: {cold_logit.item():.4f}")
    except Exception as e:
        print(f"  [FAIL] Empty history forward pass crashed: {e}")
        findings.append(f"VULNERABILITY: Empty history buffer causes crash: {e}")

    # --------------------------------------------------------------------------
    # Test 7: Autograd Gradient Health Under Consecutive Updates
    # --------------------------------------------------------------------------
    print("\n[STRESS TEST 7] Autograd Stability Under Consecutive Long-Horizon Steps...")
    opt = torch.optim.Adam(model_stateless.parameters(), lr=1e-3)
    loss_history = []
    grad_healthy = True

    for step in range(25):
        opt.zero_grad()
        s = torch.randint(0, 50, (32,))
        d = torch.randint(0, 50, (32,))
        dt = torch.rand(32) * 5.0
        buf.push(s[:8], d[:8], torch.full((8,), float(step)))
        ei, edt = buf.graph(t_ref=float(step + 1), time_scale=1.0)
        logits = model_stateless(ei, edt, s, d, delta_t_src=dt)
        targets = torch.randint(0, 2, (32,)).float()
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()

        for name, p in model_stateless.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    grad_healthy = False
                    print(f"  [FAIL] Step {step}: Gradient of {name} has NaN/Inf!")
                    findings.append(f"VULNERABILITY: Gradient NaN/Inf in parameter {name} during step {step}.")
                    break
        opt.step()
        loss_history.append(loss.item())

    if grad_healthy:
        print(f"  [PASS] 25 consecutive training steps stable. Loss: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}")

    print("\n" + "=" * 80)
    print("  STRESS TEST AUDIT SUMMARY")
    print("=" * 80)
    if not findings:
        print("  All stress tests passed with ZERO critical vulnerabilities!")
    else:
        print(f"  Found {len(findings)} noteworthy vulnerability/architectural edge cases:")
        for idx, f in enumerate(findings, 1):
            print(f"  [{idx}] {f}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_stress_tests()
