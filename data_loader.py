"""
RT-GNN Streaming Temporal Data Pipeline (data_loader.py)
======================================================
Compatibility bridge and alias for Blueprint v2 implementation in rtgnn_data.py.

This file provides full backward compatibility with v1 names:
- RTGNNTemporalDataLoader
- DataLoaderConfig
- TemporalEdgeBatch

While executing with full Blueprint v2 compliance via rtgnn_data.py:
- Native PyTorch and NumPy fallback (no crash if torch is not installed yet).
- Official TGB key names ('sources', 'destinations', 'timestamps').
- Official train / val / test masks.
- Atomic scatter-reduce timestamp updates.
- Scaled Delta_t calculation.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Iterator, Optional, Tuple, Union

# Re-export Blueprint v2 classes and methods from rtgnn_data
from rtgnn_data import (
    DataBundle,
    LoaderConfig,
    SplitLoader,
    TemporalBatch,
    build_loaders,
    compute_time_scale_and_p99,
    parse_arguments,
    HAS_TORCH,
    torch,
)

# Backward-compatibility aliases for v1 imports
DataLoaderConfig = LoaderConfig
TemporalEdgeBatch = TemporalBatch


class RTGNNTemporalDataLoader:
    """Backward-compatible wrapper mapping v1 API to Blueprint v2 SplitLoader."""

    def __init__(
        self,
        config: Optional[LoaderConfig] = None,
        data_dict: Optional[Dict[str, Any]] = None,
        num_nodes: Optional[int] = None,
    ) -> None:
        self.config = config or LoaderConfig()
        self.bundle = build_loaders(
            config=self.config,
            raw_data=data_dict,
            num_nodes=num_nodes,
        )
        self.loader = self.bundle.train
        self.num_nodes = self.bundle.num_nodes
        self.num_edges = self.bundle.num_edges
        self.time_scale = self.bundle.time_scale
        self.dt_p99 = self.bundle.dt_p99

    def reset_timestamps(self) -> None:
        self.bundle.reset_state()

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[TemporalBatch]:
        return iter(self.loader)


if __name__ == "__main__":
    cfg = parse_arguments()
    print("=" * 65)
    print("RT-GNN Streaming Temporal Data Pipeline (v2 / data_loader)")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
    print("=" * 65)

    try:
        bundle = build_loaders(cfg)
        print(f"Dataset: {cfg.dataset_name}")
        print(f"Total Edges: {bundle.num_edges} | Unique Nodes: {bundle.num_nodes}")
        print(f"Train batches: {len(bundle.train)} | Val batches: {len(bundle.val)} | Test batches: {len(bundle.test)}")
        print(f"Fitted Time Scale: {bundle.time_scale:.4f} s | Delta_t p99: {bundle.dt_p99:.4f}")

        bundle.train.set_epoch(0)
        for b_idx, batch in enumerate(bundle.train):
            first_seen_pct = (~batch.has_history_src).numpy().astype(float).mean() * 100.0
            print(
                f"[Batch {b_idx + 1:02d}] Size: {batch.src.size(0)} | "
                f"First-seen src: {first_seen_pct:.1f}% | "
                f"Delta_t mean: {batch.delta_t_src.mean().item():.4f}"
            )
            if b_idx + 1 >= (cfg.max_batches or 5):
                break

    except ImportError as e:
        print(f"\n[Note] {e}")
        print("Running synthetic dry-run verification for Section 5 first-run checklist...")
        import numpy as np

        N = 10000
        V = 500
        s = np.random.randint(0, V, N)
        d = np.random.randint(0, V, N)
        t = np.sort(np.random.rand(N) * 100000.0)
        m_tr = np.zeros(N, dtype=bool)
        m_val = np.zeros(N, dtype=bool)
        m_te = np.zeros(N, dtype=bool)
        m_tr[:7000] = True
        m_val[7000:8500] = True
        m_te[8500:] = True

        syn_bundle = build_loaders(
            cfg,
            raw_data={"sources": s, "destinations": d, "timestamps": t},
            train_mask=torch.as_tensor(m_tr),
            val_mask=torch.as_tensor(m_val),
            test_mask=torch.as_tensor(m_te),
            num_nodes=V,
        )
        print(f"Synthetic edges: {syn_bundle.num_edges} (7000 train, 1500 val, 1500 test)")
        print(f"Fitted Time Scale: {syn_bundle.time_scale:.4f} s | Delta_t p99: {syn_bundle.dt_p99:.4f}")

        syn_bundle.train.set_epoch(0)
        for b_idx, batch in enumerate(syn_bundle.train):
            first_seen_pct = (~batch.has_history_src).numpy().astype(float).mean() * 100.0
            print(
                f"[Synthetic Batch {b_idx + 1:02d}] "
                f"First-seen src: {first_seen_pct:.1f}% | "
                f"Delta_t src min: {batch.delta_t_src.min().item():.4f}, "
                f"mean: {batch.delta_t_src.mean().item():.4f}, "
                f"max: {batch.delta_t_src.max().item():.4f}"
            )
            if b_idx + 1 >= (cfg.max_batches or 5):
                break
