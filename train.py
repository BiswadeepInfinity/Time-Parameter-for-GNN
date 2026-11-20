"""
RT-GNN End-to-End Training & Evaluation Script (train.py)
========================================================
Strictly follows RT-GNN Implementation Blueprint v2:
- HistoryBuffer isolates earlier events so message passing never leaks labels.
- Evaluates with BCEWithLogitsLoss on prediction logits.
- Official TGB MRR evaluation protocol with fixed negative sets.
- Tracks peak VRAM (torch.cuda.max_memory_allocated) and throughput (edges/sec).
- Configurable via CLI / Config (DO NOT HARDCODE ANY VALUE).
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rtgnn_data import LoaderConfig, build_loaders
from rtgnn_model import HistoryBuffer, RTGNNConfig, RotationalTemporalGNN


def train_one_epoch(
    model: RotationalTemporalGNN,
    bundle,
    buffer: HistoryBuffer,
    optimizer: torch.optim.Optimizer,
    cfg: RTGNNConfig,
) -> float:
    """Trains the model for one chronological epoch over the stream."""
    model.train()
    buffer.clear()
    model.reset_memory()
    total_loss, count = 0.0, 0

    for batch in bundle.train:
        batch = batch.to(cfg.device)

        # 1. Retrieve causal history graph prior to scoring current batch
        hist_ei, hist_dt = buffer.graph(batch.t[0].item(), bundle.time_scale)

        # 2. Get combined prediction pairs, temporal deltas & ground truth
        src, dst, dt_src, labels = batch.link_prediction_inputs()

        # 3. Model forward pass on logits with all leak-free upgrades
        logits = model(
            hist_ei, hist_dt, src, dst, delta_t_src=dt_src,
        )
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # 4. Optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # 5. Update persistent node memory AFTER the optimizer step (leak-free)
        #    Uses batch.delta_t_src/dst (unnormalised, from the original batch)
        if hasattr(model, "update_memory"):
            with torch.no_grad():
                model.update_memory(
                    batch.src.to(cfg.device),
                    batch.dst.to(cfg.device),
                    batch.delta_t_src.to(cfg.device),
                    batch.delta_t_dst.to(cfg.device),
                )

        # 6. Push current batch to history buffer ONLY after scoring
        buffer.push(batch.src, batch.dst, batch.t)

        total_loss += loss.item()
        count += 1

    return total_loss / max(1, count)


@torch.no_grad()
def evaluate(
    model: RotationalTemporalGNN,
    bundle,
    buffer: HistoryBuffer,
    split: str,
    dataset_name: str,
    device: str,
) -> float:
    """Evaluates MRR using TGB's official fixed negative sets with vectorized tensor scoring."""
    try:
        from tgb.linkproppred.evaluate import Evaluator
    except ImportError as e:
        raise RuntimeError("tgb is required for evaluation. Install via pip install py-tgb.") from e

    model.eval()
    ds = bundle.tgb_dataset
    if ds is None:
        raise ValueError(f"Cannot evaluate: bundle.tgb_dataset is None for dataset '{dataset_name}'.")

    load_fn = getattr(ds, f"load_{split}_ns", None)
    if load_fn is not None:
        load_fn()

    evaluator = Evaluator(name=dataset_name)
    metric = ds.eval_metric
    sampler = ds.negative_sampler

    mrr_list = []
    loader = getattr(bundle, split)

    for batch in loader:
        batch = batch.to(device)

        # 1. Causal history graph
        hist_ei, hist_dt = buffer.graph(batch.t[0].item(), bundle.time_scale)
        xr, xi = model.encode(hist_ei, hist_dt)

        # 2. Query TGB official negative candidate sets
        negs = sampler.query_batch(
            batch.src.cpu().numpy(),
            batch.dst.cpu().numpy(),
            batch.t.cpu().numpy(),
            split_mode=split,
        )

        # 3. Predict scores for positive + negative candidates
        # Handles inhomogeneous candidate lengths (e.g. 998 vs 999) seamlessly
        all_src_list = []
        all_dst_list = []
        all_dt_src_list = []
        offsets = [0]
        for i, neg in enumerate(negs):
            cand = [batch.dst[i].item(), *neg]
            k = len(cand)
            all_src_list.extend([batch.src[i].item()] * k)
            all_dst_list.extend(cand)
            all_dt_src_list.extend([batch.delta_t_src[i].item()] * k)
            # NOTE: delta_t_dst intentionally NOT included — the destination's
            # elapsed time is unknown at prediction time (that is what we predict).
            # Passing real delta_t_dst for positives and 0.0 for negatives is
            # a perfect label-leakage signal and was the root cause of MRR=0.8981.
            offsets.append(offsets[-1] + k)

        all_src_t = torch.tensor(all_src_list, dtype=torch.long, device=device)
        all_dst_t = torch.tensor(all_dst_list, dtype=torch.long, device=device)
        all_dt_src_t = torch.tensor(all_dt_src_list, dtype=torch.float32, device=device)

        # Vectorized scoring with leak-free stateless upgrades
        src_r, src_i = xr[all_src_t], xi[all_src_t]
        dst_r, dst_i = xr[all_dst_t], xi[all_dst_t]

        if model.query_rotation is not None:
            src_r, src_i = model.query_rotation(src_r, src_i, all_dt_src_t)

        time_feat = None
        if model.time_encoder is not None:
            time_feat = model.time_encoder(all_dt_src_t)

        all_logits = model.predictor(src_r, src_i, dst_r, dst_i, time_feat=time_feat)
        if model.dst_bias is not None:
            all_logits = all_logits + model.dst_bias(all_dst_t).squeeze(-1)

        all_scores = all_logits.cpu().numpy()

        # 4. TGB Evaluator MRR metric calculation
        for i in range(len(negs)):
            y_i = all_scores[offsets[i] : offsets[i + 1]]
            eval_res = evaluator.eval({
                "y_pred_pos": y_i[:1],
                "y_pred_neg": y_i[1:],
                "eval_metric": [metric],
            })
            mrr_list.append(float(eval_res[metric]))

        # Maintain causal progression in history buffer + persistent memory
        buffer.push(batch.src, batch.dst, batch.t)
        if hasattr(model, "update_memory"):
            model.update_memory(
                batch.src.to(device),
                batch.dst.to(device),
                batch.delta_t_src.to(device),
                batch.delta_t_dst.to(device),
            )

    return float(np.mean(mrr_list)) if mrr_list else 0.0


def main():
    parser = argparse.ArgumentParser(description="RT-GNN v2 Training Pipeline")
    parser.add_argument("--dataset_name", type=str, default="tgbl-wiki")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--history_batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_negatives", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overfit_slice", type=int, default=None, help="Phase 5 Gate A: overfit test on N edges")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    print("=" * 65)
    print("RT-GNN Training Pipeline Initialization (Blueprint v2)")
    print(f"Device: {args.device} | Dataset: {args.dataset_name}")
    if args.overfit_slice is not None:
        print(f"Mode: Phase 5 Gate A (Overfitting Sanity Check on {args.overfit_slice} edges)")
    print("=" * 65)

    # 1. Build Data Pipeline
    loader_cfg = LoaderConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        device=args.device,
        seed=args.seed,
    )
    bundle = build_loaders(loader_cfg)

    # If overfit_slice is requested, slice training edges to the first N
    if args.overfit_slice is not None:
        N = min(args.overfit_slice, bundle.train.sources.size(0))
        bundle.train.sources = bundle.train.sources[:N]
        bundle.train.destinations = bundle.train.destinations[:N]
        bundle.train.timestamps = bundle.train.timestamps[:N]
        bundle.train.edge_indices = bundle.train.edge_indices[:N]
        if bundle.train.edge_feats is not None:
            bundle.train.edge_feats = bundle.train.edge_feats[:N]
        bundle.train.num_edges = N

    # 2. Configure Model
    # Set omega_min dynamically so that omega_min * dt_p99 < pi (Section 3 of Blueprint v2)
    omega_min = min(1e-2, (0.5 * 3.14159265) / max(bundle.dt_p99, 1e-4))
    model_cfg = RTGNNConfig(
        num_nodes=bundle.num_nodes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        omega_min=omega_min,
        history_batches=args.history_batches,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        device=args.device,
    )

    model = RotationalTemporalGNN(model_cfg).to(args.device)
    buffer = HistoryBuffer(model_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.weight_decay
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Time scale: {bundle.time_scale:.4f}s | dt_p99: {bundle.dt_p99:.4f} | omega_min: {omega_min:.6f}")

    # Reset GPU memory stats
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # 3. Training & Evaluation Loop
    for epoch in range(model_cfg.epochs):
        bundle.reset_state()
        if hasattr(model, "reset_memory"):
            model.reset_memory()
        t0 = time.time()
        bundle.train.set_epoch(epoch)
        loss = train_one_epoch(model, bundle, buffer, optimizer, model_cfg)
        elapsed = time.time() - t0
        edges_per_sec = bundle.train.num_edges / max(elapsed, 1e-6)

        print(
            f"Epoch {epoch + 1:02d}/{model_cfg.epochs:02d} | "
            f"Loss: {loss:.4f} | "
            f"Time: {elapsed:.2f}s ({edges_per_sec:.1f} edges/s)"
        )

        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  Peak VRAM: {peak_vram:.4f} GB")

        # Evaluate on validation split (causal continuation)
        if bundle.tgb_dataset is not None and args.overfit_slice is None:
            val_mrr = evaluate(model, bundle, buffer, "val", args.dataset_name, args.device)
            print(f"  Val MRR: {val_mrr:.4f}")

    # Final Test Evaluation
    if bundle.tgb_dataset is not None and args.overfit_slice is None:
        test_mrr = evaluate(model, bundle, buffer, "test", args.dataset_name, args.device)
        print("=" * 65)
        print(f"Final Test MRR: {test_mrr:.4f}")
        print("=" * 65)
    elif args.overfit_slice is not None:
        print("=" * 65)
        print(f"Phase 5 Gate A Overfit Check Complete. Final Loss: {loss:.6f}")
        if loss < 0.1:
            print(">>> Gate A PASSED: Loss descended close to zero! No architecture bugs detected.")
        else:
            print(">>> Gate A Notice: Loss did not reach < 0.1, check learning rate / epochs.")
        print("=" * 65)


if __name__ == "__main__":
    main()
