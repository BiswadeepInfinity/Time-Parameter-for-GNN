"""
RT-GNN Empirical Benchmark Runner (run_benchmark.py)
===================================================
Executes multi-seed training and evaluation on TGB datasets (default: tgbl-wiki).
Measures:
  - Exact Test & Val MRR via TGB's official Evaluator
  - Processing throughput (edges/second)
  - Peak RAM footprint (MB via tracemalloc)
  - Mean +/- std across multiple random seeds

All results are saved directly to 'benchmark_results.json' with zero hand-typed metrics.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from rtgnn_data import LoaderConfig, build_loaders
from rtgnn_model import HistoryBuffer, RTGNNConfig, RotationalTemporalGNN
from rtgnn_stateful_gnn import StatefulRotationalTemporalGNN
from train import train_one_epoch, evaluate as evaluate_stateless
from train_stateful import train_stateful_epoch, evaluate_stateful


def run_single_seed_stateless(
    seed: int,
    dataset_name: str,
    data_root: str,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    device: str,
    num_negatives: int = 3,
) -> Dict[str, float]:
    """Runs Stateless RT-GNN for a single random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader_cfg = LoaderConfig(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=batch_size,
        delta_t_mode="event",
        num_negatives=num_negatives,
        device=device,
        seed=seed,
    )
    bundle = build_loaders(loader_cfg)

    omega_min = min(1e-2, (0.5 * 3.14159265) / max(bundle.dt_p99, 1e-4))
    cfg = RTGNNConfig(
        num_nodes=bundle.num_nodes,
        hidden_dim=hidden_dim,
        omega_min=omega_min,
        history_batches=16,
        use_dot_shortcut=True,
        epochs=epochs,
        device=device,
    )

    model = RotationalTemporalGNN(cfg).to(device)
    buffer = HistoryBuffer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    tracemalloc.start()
    epoch_times = []
    final_loss = 0.0

    for epoch in range(epochs):
        bundle.reset_state()
        model.reset_memory()
        bundle.train.set_epoch(epoch)
        t0 = time.time()
        final_loss = train_one_epoch(model, bundle, buffer, optimizer, cfg)
        epoch_times.append(time.time() - t0)

    val_mrr = evaluate_stateless(model, bundle, buffer, "val", dataset_name, device)
    test_mrr = evaluate_stateless(model, bundle, buffer, "test", dataset_name, device)

    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    avg_time = float(np.mean(epoch_times)) if epoch_times else 1.0
    throughput = bundle.train.num_edges / max(avg_time, 1e-6)

    return {
        "seed": seed,
        "final_train_loss": float(final_loss),
        "val_mrr": float(val_mrr),
        "test_mrr": float(test_mrr),
        "avg_epoch_sec": avg_time,
        "throughput_edges_sec": float(throughput),
        "peak_ram_mb": float(peak_bytes / (1024 * 1024)),
    }


def run_single_seed_stateful(
    seed: int,
    dataset_name: str,
    data_root: str,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    device: str,
    num_negatives: int = 3,
) -> Dict[str, float]:
    """Runs Stateful RT-GNN (Associative Scan) for a single random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader_cfg = LoaderConfig(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=batch_size,
        delta_t_mode="event",
        num_negatives=num_negatives,
        device=device,
        seed=seed,
    )
    bundle = build_loaders(loader_cfg)

    omega_min = min(1e-2, (0.5 * 3.14159265) / max(bundle.dt_p99, 1e-4))
    cfg = RTGNNConfig(
        num_nodes=bundle.num_nodes,
        hidden_dim=hidden_dim,
        num_layers=1,
        history_batches=8,
        omega_min=omega_min,
        epochs=epochs,
        device=device,
    )

    model = StatefulRotationalTemporalGNN(cfg).to(device)
    buffer = HistoryBuffer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    tracemalloc.start()
    epoch_times = []
    final_loss = 0.0

    for epoch in range(epochs):
        bundle.reset_state()
        model.reset_memory()
        bundle.train.set_epoch(epoch)
        t0 = time.time()
        final_loss = train_stateful_epoch(model, bundle, buffer, optimizer, cfg)
        epoch_times.append(time.time() - t0)

    val_mrr = evaluate_stateful(model, bundle, buffer, "val", dataset_name, device)
    test_mrr = evaluate_stateful(model, bundle, buffer, "test", dataset_name, device)

    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    avg_time = float(np.mean(epoch_times)) if epoch_times else 1.0
    throughput = bundle.train.num_edges / max(avg_time, 1e-6)

    return {
        "seed": seed,
        "final_train_loss": float(final_loss),
        "val_mrr": float(val_mrr),
        "test_mrr": float(test_mrr),
        "avg_epoch_sec": avg_time,
        "throughput_edges_sec": float(throughput),
        "peak_ram_mb": float(peak_bytes / (1024 * 1024)),
    }


def aggregate_seed_results(runs: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculates mean and sample standard deviation (ddof=1) across seeds."""
    test_mrrs = [r["test_mrr"] for r in runs]
    val_mrrs = [r["val_mrr"] for r in runs]
    throughputs = [r["throughput_edges_sec"] for r in runs]
    rams = [r["peak_ram_mb"] for r in runs]

    ddof = 1 if len(runs) > 1 else 0
    return {
        "mrr_mean": float(np.mean(test_mrrs)),
        "mrr_std": float(np.std(test_mrrs, ddof=ddof)),
        "val_mrr_mean": float(np.mean(val_mrrs)),
        "val_mrr_std": float(np.std(val_mrrs, ddof=ddof)),
        "edges_per_sec": float(np.mean(throughputs)),
        "peak_ram_mb": float(np.mean(rams)),
        "seeds": runs,
    }


def main():
    parser = argparse.ArgumentParser(description="RT-GNN Empirical Benchmark Runner")
    parser.add_argument("--dataset_name", type=str, default="tgbl-wiki")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--models", nargs="+", choices=["stateless", "stateful"], default=["stateless", "stateful"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_negatives", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default="benchmark_results.json")

    args = parser.parse_args()

    print("=" * 80)
    print("  RT-GNN Empirical Benchmark Execution (TGB Protocol)")
    print(f"  Dataset: {args.dataset_name} | Epochs: {args.epochs} | Seeds: {args.seeds} | Device: {args.device}")
    print("=" * 80)

    out_path = Path(args.output_json)
    existing_models = {}
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                existing_models = prev_data.get("models", {})
        except Exception:
            existing_models = {}

    benchmark_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "dataset": args.dataset_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "device": args.device,
        "models": existing_models,
    }

    for model_name in args.models:
        print(f"\n[RUNNING BENCHMARK] Model: {model_name.upper()} across {len(args.seeds)} seeds...")
        runs = []
        for s in args.seeds:
            print(f"  -> Seed {s}...")
            if model_name == "stateless":
                res = run_single_seed_stateless(
                    seed=s,
                    dataset_name=args.dataset_name,
                    data_root=args.data_root,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden_dim,
                    device=args.device,
                    num_negatives=args.num_negatives,
                )
            else:
                res = run_single_seed_stateful(
                    seed=s,
                    dataset_name=args.dataset_name,
                    data_root=args.data_root,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden_dim,
                    device=args.device,
                    num_negatives=args.num_negatives,
                )
            runs.append(res)
            print(f"     Test MRR: {res['test_mrr']:.4f} | Val MRR: {res['val_mrr']:.4f} | Speed: {res['throughput_edges_sec']:.1f} edges/s")

        agg = aggregate_seed_results(runs)
        benchmark_record["models"][model_name] = agg
        print(f"  => {model_name.upper()} Summary: MRR = {agg['mrr_mean']:.4f} +/- {agg['mrr_std']:.4f} | Speed = {agg['edges_per_sec']:.1f} edges/s")

    out_path = Path(args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_record, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  Benchmark successfully completed and written to: {out_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
