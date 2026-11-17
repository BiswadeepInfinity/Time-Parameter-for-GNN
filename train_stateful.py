"""
Stateful RT-GNN Training & Evaluation Pipeline (train_stateful.py)
=================================================================
Trains and benchmarks the Stateful Rotational Temporal GNN on tgbl-wiki
with per-node continuous-time complex memory (Phase 7).
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rtgnn_data import LoaderConfig, build_loaders
from rtgnn_model import HistoryBuffer, RTGNNConfig
from rtgnn_stateful_gnn import StatefulRotationalTemporalGNN


def train_stateful_epoch(
    model: StatefulRotationalTemporalGNN,
    bundle,
    buffer: HistoryBuffer,
    optimizer: torch.optim.Optimizer,
    cfg: RTGNNConfig,
) -> float:
    """Trains the stateful RT-GNN for one chronological epoch over the stream."""
    model.train()
    buffer.clear()
    total_loss, count = 0.0, 0

    for batch in bundle.train:
        batch = batch.to(cfg.device)

        # 1. Causal history graph (t < t_ref)
        hist_ei, hist_dt = buffer.graph(batch.t[0].item(), bundle.time_scale)

        # 2. Extract link prediction pairs
        src, dst, labels = batch.link_prediction_pairs()

        # 3. Model forward pass on logits
        logits = model(hist_ei, hist_dt, src, dst)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # 4. Optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # 5. Update per-node complex memory states
        model.update_streaming_memory(
            batch.src, batch.dst, batch.delta_t_src, batch.delta_t_dst
        )

        # 6. Push to history buffer only after scoring
        buffer.push(batch.src, batch.dst, batch.t)

        total_loss += loss.item()
        count += 1
        if count % 15 == 0 or count == 1:
            print(f"    Batch {count:02d}/{len(bundle.train):02d} | Step Loss: {loss.item():.4f}", flush=True)

    return total_loss / max(1, count)


@torch.no_grad()
def evaluate_stateful(
    model: StatefulRotationalTemporalGNN,
    bundle,
    buffer: HistoryBuffer,
    split: str,
    dataset_name: str,
    device: str,
) -> float:
    """Evaluates the stateful model using TGB official fixed negative sets."""
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

        # Causal history graph
        hist_ei, hist_dt = buffer.graph(batch.t[0].item(), bundle.time_scale)
        xr, xi = model.encode(hist_ei, hist_dt)

        negs = sampler.query_batch(
            batch.src.cpu().numpy(),
            batch.dst.cpu().numpy(),
            batch.t.cpu().numpy(),
            split_mode=split,
        )

        all_src_list = []
        all_dst_list = []
        offsets = [0]
        for i, neg in enumerate(negs):
            cand = [batch.dst[i].item(), *neg]
            all_src_list.extend([batch.src[i].item()] * len(cand))
            all_dst_list.extend(cand)
            offsets.append(offsets[-1] + len(cand))

        all_src_t = torch.tensor(all_src_list, dtype=torch.long, device=device)
        all_dst_t = torch.tensor(all_dst_list, dtype=torch.long, device=device)
        all_scores = model.predictor(
            xr[all_src_t], xi[all_src_t], xr[all_dst_t], xi[all_dst_t]
        ).cpu().numpy()

        for i in range(len(negs)):
            y_i = all_scores[offsets[i] : offsets[i + 1]]
            eval_res = evaluator.eval({
                "y_pred_pos": y_i[:1],
                "y_pred_neg": y_i[1:],
                "eval_metric": [metric],
            })
            mrr_list.append(float(eval_res[metric]))

        # Update streaming memory and history buffer
        model.update_streaming_memory(
            batch.src, batch.dst, batch.delta_t_src, batch.delta_t_dst
        )
        buffer.push(batch.src, batch.dst, batch.t)

    return float(np.mean(mrr_list)) if mrr_list else 0.0


def main():
    parser = argparse.ArgumentParser(description="Stateful RT-GNN Training Pipeline")
    parser.add_argument("--dataset_name", type=str, default="tgbl-wiki")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--history_batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("  Stateful RT-GNN Training Pipeline (Phase 7 Parallel Associative Scan)")
    print(f"  Device: {args.device} | Dataset: {args.dataset_name} | Epochs: {args.epochs}")
    print("=" * 70)

    # 1. Data Pipeline in 'event' mode for exact per-event gaps
    loader_cfg = LoaderConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        batch_size=args.batch_size,
        delta_t_mode="event",  # Blueprint v2 Phase 7 requirement
        device=args.device,
        seed=args.seed,
    )
    bundle = build_loaders(loader_cfg)

    # 2. Model Configuration
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

    model = StatefulRotationalTemporalGNN(model_cfg).to(args.device)
    buffer = HistoryBuffer(model_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.weight_decay
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Fitted Time Scale: {bundle.time_scale:.4f}s | dt_p99: {bundle.dt_p99:.4f}")

    # 3. Multi-Epoch Training Loop
    for epoch in range(model_cfg.epochs):
        bundle.reset_state()
        model.reset_memory()
        t0 = time.time()
        bundle.train.set_epoch(epoch)
        loss = train_stateful_epoch(model, bundle, buffer, optimizer, model_cfg)
        elapsed = time.time() - t0
        edges_per_sec = bundle.train.num_edges / max(elapsed, 1e-6)

        print(
            f"Epoch {epoch + 1:02d}/{model_cfg.epochs:02d} | "
            f"Loss: {loss:.4f} | "
            f"Time: {elapsed:.2f}s ({edges_per_sec:.1f} edges/s)"
        )

        if bundle.tgb_dataset is not None:
            val_mrr = evaluate_stateful(
                model, bundle, buffer, "val", args.dataset_name, args.device
            )
            print(f"  Val MRR: {val_mrr:.4f}")

    # Final Test Evaluation
    if bundle.tgb_dataset is not None:
        test_mrr = evaluate_stateful(
            model, bundle, buffer, "test", args.dataset_name, args.device
        )
        print("=" * 70)
        print(f"  Stateful RT-GNN Final Test MRR: {test_mrr:.4f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
