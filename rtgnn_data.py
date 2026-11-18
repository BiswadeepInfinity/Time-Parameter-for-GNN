"""
RT-GNN Streaming Temporal Data Pipeline (rtgnn_data.py)
======================================================
Strictly adheres to RT-GNN Implementation Blueprint v2:
- Supports TGB's official keys: 'sources', 'destinations', 'timestamps', etc.
- Uses official train / val / test masks.
- Configurable delta_t calculation ('batch' vs 'event' mode).
- Unseen node policy: 'zero_dt' (Delta_t = 0, has_history = False) or 'origin'.
- Time normalization by train-split median positive Delta_t.
- Deterministic negative destination sampling for training: (seed, epoch, split).
- Cross-split timestamp continuity: val is primed by train, test is primed by val.
- Atomic / maximum timestamp updates per batch (no fancy indexing overwrite race).
- Structured TemporalBatch container with link_prediction_pairs() method.
- Complete parameterization via LoaderConfig (DO NOT HARDCODE ANY VALUE).
- Compatible with native PyTorch and lightweight numpy fallback for Phase 0 testing.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class Tensor:
        """Lightweight NumPy-backed Tensor shim for Phase 0 offline verification."""
        def __init__(self, data: Any, dtype: Any = None):
            if isinstance(data, Tensor):
                self._arr = data._arr.copy()
            elif isinstance(data, np.ndarray):
                self._arr = data.astype(dtype) if dtype else data
            else:
                self._arr = np.array(data, dtype=dtype)

        @property
        def ndim(self) -> int:
            return self._arr.ndim

        @property
        def shape(self) -> Tuple[int, ...]:
            return self._arr.shape

        @property
        def dtype(self) -> Any:
            return self._arr.dtype

        def size(self, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
            if dim is None:
                return self._arr.shape
            return self._arr.shape[dim]

        def numel(self) -> int:
            return self._arr.size

        def item(self) -> Any:
            return self._arr.item()

        def tolist(self) -> list:
            return self._arr.tolist()

        def numpy(self) -> np.ndarray:
            return self._arr

        def to(self, device: Any = None, dtype: Any = None) -> Tensor:
            new_arr = self._arr if dtype is None else self._arr.astype(dtype)
            return Tensor(new_arr)

        def clamp_min(self, min_val: float) -> Tensor:
            return Tensor(np.maximum(self._arr, min_val))

        def zero_(self) -> Tensor:
            self._arr.fill(0)
            return self

        def view(self, *shape: int) -> Tensor:
            return Tensor(self._arr.reshape(*shape))

        def repeat_interleave(self, repeats: int) -> Tensor:
            return Tensor(np.repeat(self._arr, repeats, axis=0))

        def nonzero(self, as_tuple: bool = False) -> Any:
            nz = np.nonzero(self._arr)
            res = tuple(Tensor(x) for x in nz)
            return res if as_tuple else Tensor(np.column_stack(nz))

        def min(self) -> Tensor:
            return Tensor(np.min(self._arr))

        def max(self) -> Tensor:
            return Tensor(np.max(self._arr))

        def mean(self) -> Tensor:
            return Tensor(np.mean(self._arr))

        def float(self) -> Tensor:
            return Tensor(self._arr.astype(np.float32))

        def __getitem__(self, item: Any) -> Tensor:
            if isinstance(item, Tensor):
                item = item._arr
            elif isinstance(item, tuple):
                item = tuple(x._arr if isinstance(x, Tensor) else x for x in item)
            res = self._arr[item]
            return Tensor(res)

        def __setitem__(self, key: Any, val: Any) -> None:
            if isinstance(key, Tensor):
                key = key._arr
            elif isinstance(key, tuple):
                key = tuple(x._arr if isinstance(x, Tensor) else x for x in key)
            if isinstance(val, Tensor):
                val = val._arr
            self._arr[key] = val

        def __add__(self, other: Any) -> Tensor:
            o = other._arr if isinstance(other, Tensor) else other
            return Tensor(self._arr + o)

        def __sub__(self, other: Any) -> Tensor:
            o = other._arr if isinstance(other, Tensor) else other
            return Tensor(self._arr - o)

        def __mul__(self, other: Any) -> Tensor:
            o = other._arr if isinstance(other, Tensor) else other
            return Tensor(self._arr * o)

        def __truediv__(self, other: Any) -> Tensor:
            o = other._arr if isinstance(other, Tensor) else other
            return Tensor(self._arr / o)

        def __gt__(self, other: Any) -> Tensor:
            o = other._arr if isinstance(other, Tensor) else other
            return Tensor(self._arr > o)

        def __ge__(self, other: Any) -> Tensor:
            o = other._arr if isinstance(other, Tensor) else other
            return Tensor(self._arr >= o)

        def __invert__(self) -> Tensor:
            return Tensor(~self._arr)

        def __len__(self) -> int:
            return len(self._arr)

        def scatter_reduce_(self, dim: int, index: Tensor, src: Tensor, reduce: str = "amax", include_self: bool = True) -> Tensor:
            idx = index._arr
            vals = src._arr
            if reduce == "amax":
                np.maximum.at(self._arr, idx, vals)
            elif reduce == "amin":
                np.minimum.at(self._arr, idx, vals)
            elif reduce == "sum":
                np.add.at(self._arr, idx, vals)
            return self

    class _TorchShim:
        long = np.int64
        float32 = np.float32
        float64 = np.float64
        bool = np.bool_

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        @staticmethod
        def as_tensor(data: Any, dtype: Any = None) -> Tensor:
            return Tensor(data, dtype=dtype)

        @staticmethod
        def tensor(data: Any, dtype: Any = None) -> Tensor:
            return Tensor(data, dtype=dtype)

        @staticmethod
        def zeros(shape: Any, dtype: Any = float32, **kwargs: Any) -> Tensor:
            if isinstance(shape, int):
                shape = (shape,)
            return Tensor(np.zeros(shape, dtype=dtype))

        @staticmethod
        def ones(shape: Any, dtype: Any = float32, **kwargs: Any) -> Tensor:
            if isinstance(shape, int):
                shape = (shape,)
            return Tensor(np.ones(shape, dtype=dtype))

        @staticmethod
        def zeros_like(t: Tensor, **kwargs: Any) -> Tensor:
            return Tensor(np.zeros_like(t._arr))

        @staticmethod
        def arange(end: int, dtype: Any = long, **kwargs: Any) -> Tensor:
            return Tensor(np.arange(end, dtype=dtype))

        @staticmethod
        def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
            arrs = [t._arr for t in tensors]
            return Tensor(np.concatenate(arrs, axis=dim))

        @staticmethod
        def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
            arrs = [t._arr for t in tensors]
            return Tensor(np.stack(arrs, axis=dim))

        @staticmethod
        def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
            return Tensor(np.where(condition._arr, x._arr, y._arr))

        @staticmethod
        def randint(low: int, high: int, size: Tuple[int, ...], generator: Any = None, dtype: Any = long) -> Tensor:
            if generator is not None and hasattr(generator, "_rng"):
                arr = generator._rng.integers(low, high, size=size, dtype=dtype)
            else:
                arr = np.random.randint(low, high, size=size, dtype=dtype)
            return Tensor(arr)

        class Generator:
            def __init__(self):
                self._rng = np.random.default_rng(0)
            def manual_seed(self, seed: int):
                self._rng = np.random.default_rng(seed)
                return self

        def device(self, name: str) -> str:
            return name

    torch = _TorchShim()


@dataclass
class LoaderConfig:
    """Configuration dataclass for RT-GNN Data Pipeline."""
    dataset_name: str = "tgbl-wiki"
    data_root: str = "data"
    batch_size: int = 2048
    delta_t_mode: str = "batch"       # 'batch' or 'event'
    unseen_policy: str = "zero_dt"    # 'zero_dt' or 'origin'
    normalize_dt: bool = True
    time_scale: Optional[float] = None
    scale_statistic: str = "median"   # 'median' or 'mean'
    num_negatives: int = 1            # Negatives per positive edge (train only)
    drop_last: bool = False
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_batches: Optional[int] = None # For dry runs / testing


@dataclass
class TemporalBatch:
    """Batch container emitted by the temporal data loader."""
    src: Tensor                 # [B] Source node indices
    dst: Tensor                 # [B] Destination node indices
    t: Tensor                   # [B] Raw continuous timestamps
    delta_t_src: Tensor         # [B] Scaled Delta_t for source nodes
    delta_t_dst: Tensor         # [B] Scaled Delta_t for destination nodes
    has_history_src: Tensor     # [B] Boolean flag: whether source was seen prior
    has_history_dst: Tensor     # [B] Boolean flag: whether destination was seen prior
    edge_idx: Tensor            # [B] Original dataset edge index
    neg_dst: Optional[Tensor] = None # [B, num_negatives] or [B] (train only)
    edge_feat: Optional[Tensor] = None # Optional edge features

    def to(self, device: Any) -> TemporalBatch:
        """Move all batch tensors to device."""
        return TemporalBatch(
            src=self.src.to(device),
            dst=self.dst.to(device),
            t=self.t.to(device),
            delta_t_src=self.delta_t_src.to(device),
            delta_t_dst=self.delta_t_dst.to(device),
            has_history_src=self.has_history_src.to(device),
            has_history_dst=self.has_history_dst.to(device),
            edge_idx=self.edge_idx.to(device),
            neg_dst=self.neg_dst.to(device) if self.neg_dst is not None else None,
            edge_feat=self.edge_feat.to(device) if self.edge_feat is not None else None,
        )

    def link_prediction_pairs(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Constructs combined prediction pairs and binary classification labels.
        
        Returns:
            combined_src: Tensor of shape [B * (1 + num_negatives)]
            combined_dst: Tensor of shape [B * (1 + num_negatives)]
            labels: Tensor of shape [B * (1 + num_negatives)] (1.0 for pos, 0.0 for neg)
        """
        B = self.src.size(0)
        if self.neg_dst is None:
            raise ValueError("Cannot call link_prediction_pairs() on a batch without neg_dst (e.g. val/test).")

        neg_flat = self.neg_dst.view(-1)
        num_negs = neg_flat.size(0) // B
        dev = self.src.device if hasattr(self.src, "device") else "cpu"

        if num_negs == 1:
            combined_src = torch.cat([self.src, self.src], dim=0)
            combined_dst = torch.cat([self.dst, neg_flat], dim=0)
            labels = torch.cat([
                torch.ones(B, dtype=torch.float32, device=dev),
                torch.zeros(B, dtype=torch.float32, device=dev),
            ], dim=0)
        else:
            expanded_src = self.src.repeat_interleave(num_negs)
            combined_src = torch.cat([self.src, expanded_src], dim=0)
            combined_dst = torch.cat([self.dst, neg_flat], dim=0)
            labels = torch.cat([
                torch.ones(B, dtype=torch.float32, device=dev),
                torch.zeros(B * num_negs, dtype=torch.float32, device=dev),
            ], dim=0)

        return combined_src, combined_dst, labels

    def link_prediction_inputs(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Constructs combined prediction pairs, temporal deltas, and ground truth labels.

        DESIGN NOTE — Zero Label Leakage Guarantee:
        1. delta_t_dst is NOT returned: the destination's elapsed time is unknown
           at inference time.
        2. edge_feat is NOT returned: at link prediction time (u, ?, t), the edge has
           not occurred yet. Feeding real features for positives and zeros for
           negatives creates a trivial binary classifier cheat code.

        Returns:
            combined_src:    LongTensor [B * (1 + num_negatives)]
            combined_dst:    LongTensor [B * (1 + num_negatives)]
            combined_dt_src: FloatTensor [B * (1 + num_negatives)]  — source elapsed time
            labels:          FloatTensor [B * (1 + num_negatives)]  — 1=positive, 0=negative
        """
        B = self.src.size(0)
        if self.neg_dst is None:
            raise ValueError(
                "Cannot call link_prediction_inputs() on a batch without neg_dst (e.g. val/test)."
            )

        neg_flat = self.neg_dst.view(-1)
        num_negs = neg_flat.size(0) // B
        dev = self.src.device if hasattr(self.src, "device") else "cpu"

        if num_negs == 1:
            combined_src = torch.cat([self.src, self.src], dim=0)
            combined_dst = torch.cat([self.dst, neg_flat], dim=0)
            combined_dt_src = torch.cat([self.delta_t_src, self.delta_t_src], dim=0)
            labels = torch.cat([
                torch.ones(B, dtype=torch.float32, device=dev),
                torch.zeros(B, dtype=torch.float32, device=dev),
            ], dim=0)
        else:
            expanded_src = self.src.repeat_interleave(num_negs)
            combined_src = torch.cat([self.src, expanded_src], dim=0)
            combined_dst = torch.cat([self.dst, neg_flat], dim=0)
            expanded_dt_src = self.delta_t_src.repeat_interleave(num_negs)
            combined_dt_src = torch.cat([self.delta_t_src, expanded_dt_src], dim=0)
            labels = torch.cat([
                torch.ones(B, dtype=torch.float32, device=dev),
                torch.zeros(B * num_negs, dtype=torch.float32, device=dev),
            ], dim=0)

        return combined_src, combined_dst, combined_dt_src, labels


class SplitLoader:
    """Sequential, chronological edge batch iterator for a single split (train/val/test)."""

    def __init__(
        self,
        split_name: str,
        sources: Tensor,
        destinations: Tensor,
        timestamps: Tensor,
        edge_indices: Tensor,
        edge_feats: Optional[Tensor],
        num_nodes: int,
        config: LoaderConfig,
        time_scale: float,
        last_seen_table: Tensor,
        seen_mask_table: Tensor,
        train_dst_set: Optional[Tensor] = None,
    ) -> None:
        self.split_name = split_name
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_indices = edge_indices
        self.edge_feats = edge_feats
        self.num_nodes = num_nodes
        self.config = config
        self.time_scale = time_scale
        self.last_seen_table = last_seen_table
        self.seen_mask_table = seen_mask_table
        self.train_dst_set = train_dst_set

        self.num_edges = self.sources.size(0)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Sets current epoch for deterministic negative sampling."""
        self.current_epoch = epoch

    def __len__(self) -> int:
        batch_size = self.config.batch_size
        if self.config.drop_last:
            n = self.num_edges // batch_size
        else:
            n = (self.num_edges + batch_size - 1) // batch_size
        if self.config.max_batches is not None:
            return min(n, self.config.max_batches)
        return n

    def __iter__(self) -> Iterator[TemporalBatch]:
        batch_size = self.config.batch_size
        num_edges = self.num_edges
        limit = num_edges - batch_size + 1 if self.config.drop_last else num_edges

        rng = None
        if self.split_name == "train" and self.config.num_negatives > 0:
            split_salt = {"train": 1000, "val": 2000, "test": 3000}.get(self.split_name, 0)
            neg_seed = (self.config.seed + self.current_epoch * 10007 + split_salt) & 0x7FFFFFFF
            rng = torch.Generator().manual_seed(neg_seed)

        batches_yielded = 0

        for start_idx in range(0, limit, batch_size):
            if self.config.max_batches is not None and batches_yielded >= self.config.max_batches:
                break

            end_idx = min(start_idx + batch_size, num_edges)
            cur_bsz = end_idx - start_idx
            if cur_bsz <= 0:
                break

            b_src = self.sources[start_idx:end_idx]
            b_dst = self.destinations[start_idx:end_idx]
            b_t = self.timestamps[start_idx:end_idx]
            b_eidx = self.edge_indices[start_idx:end_idx]
            b_efeat = self.edge_feats[start_idx:end_idx] if self.edge_feats is not None else None

            # Calculate Delta_t and history flags
            dt_src, hist_src, dt_dst, hist_dst = self._compute_deltas(b_src, b_dst, b_t)

            # Normalization by time_scale
            if self.config.normalize_dt and self.time_scale > 0:
                dt_src = dt_src / self.time_scale
                dt_dst = dt_dst / self.time_scale

            # Negative destination sampling strictly from destination candidate pool (train split only)
            b_neg_dst = None
            if self.split_name == "train" and self.config.num_negatives > 0 and rng is not None:
                pool = self.train_dst_set
                if pool is not None and pool.size(0) > 0:
                    pool_sz = pool.size(0)
                    if self.config.num_negatives == 1:
                        rand_idx = torch.randint(
                            0, pool_sz, (cur_bsz,), generator=rng, dtype=torch.long
                        )
                        b_neg_dst = pool[rand_idx]
                    else:
                        rand_idx = torch.randint(
                            0, pool_sz, (cur_bsz, self.config.num_negatives), generator=rng, dtype=torch.long
                        )
                        b_neg_dst = pool[rand_idx]
                else:
                    if self.config.num_negatives == 1:
                        b_neg_dst = torch.randint(
                            0, self.num_nodes, (cur_bsz,), generator=rng, dtype=torch.long
                        )
                    else:
                        b_neg_dst = torch.randint(
                            0, self.num_nodes, (cur_bsz, self.config.num_negatives), generator=rng, dtype=torch.long
                        )

            # Update last-seen timestamps using atomic maximums
            self._update_last_seen(b_src, b_dst, b_t)

            batch = TemporalBatch(
                src=b_src,
                dst=b_dst,
                t=b_t,
                delta_t_src=dt_src.to(dtype=torch.float32),
                delta_t_dst=dt_dst.to(dtype=torch.float32),
                has_history_src=hist_src,
                has_history_dst=hist_dst,
                edge_idx=b_eidx,
                neg_dst=b_neg_dst,
                edge_feat=b_efeat,
            )

            batches_yielded += 1
            yield batch

    def _compute_deltas(
        self, b_src: Tensor, b_dst: Tensor, b_t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Calculates Delta_t and has_history for src and dst according to delta_t_mode."""
        if self.config.delta_t_mode == "batch":
            src_prev = self.last_seen_table[b_src]
            src_seen = self.seen_mask_table[b_src]
            dst_prev = self.last_seen_table[b_dst]
            dst_seen = self.seen_mask_table[b_dst]

            if self.config.unseen_policy == "zero_dt":
                dt_src = torch.where(src_seen, b_t - src_prev, torch.zeros_like(b_t))
                dt_dst = torch.where(dst_seen, b_t - dst_prev, torch.zeros_like(b_t))
            elif self.config.unseen_policy == "origin":
                dt_src = torch.where(src_seen, b_t - src_prev, b_t)
                dt_dst = torch.where(dst_seen, b_t - dst_prev, b_t)
            else:
                raise ValueError(f"Unknown unseen_policy: {self.config.unseen_policy}")

            dt_src = dt_src.clamp_min(0.0)
            dt_dst = dt_dst.clamp_min(0.0)
            return dt_src, src_seen, dt_dst, dst_seen

        elif self.config.delta_t_mode == "event":
            B = b_src.size(0)
            dt_src = torch.zeros(B, dtype=b_t.dtype)
            hist_src = torch.zeros(B, dtype=torch.bool)
            dt_dst = torch.zeros(B, dtype=b_t.dtype)
            hist_dst = torch.zeros(B, dtype=torch.bool)

            local_seen: Dict[int, Tuple[float, bool]] = {}
            for i in range(B):
                u = int(b_src[i].item())
                v = int(b_dst[i].item())
                ti = float(b_t[i].item())

                # Source
                if u in local_seen:
                    t_prev, seen = local_seen[u]
                else:
                    t_prev = float(self.last_seen_table[u].item())
                    seen = bool(self.seen_mask_table[u].item())

                hist_src[i] = seen
                if seen:
                    dt_src[i] = max(0.0, ti - t_prev)
                else:
                    dt_src[i] = 0.0 if self.config.unseen_policy == "zero_dt" else ti
                local_seen[u] = (ti, True)

                # Destination
                if v in local_seen:
                    t_prev_v, seen_v = local_seen[v]
                else:
                    t_prev_v = float(self.last_seen_table[v].item())
                    seen_v = bool(self.seen_mask_table[v].item())

                hist_dst[i] = seen_v
                if seen_v:
                    dt_dst[i] = max(0.0, ti - t_prev_v)
                else:
                    dt_dst[i] = 0.0 if self.config.unseen_policy == "zero_dt" else ti
                local_seen[v] = (ti, True)

            return dt_src, hist_src, dt_dst, hist_dst
        else:
            raise ValueError(f"Unknown delta_t_mode: {self.config.delta_t_mode}")

    def _update_last_seen(self, b_src: Tensor, b_dst: Tensor, b_t: Tensor) -> None:
        """Updates last-seen timestamps keeping maximum timestamp per node without race conditions."""
        all_nodes = torch.cat([b_src, b_dst], dim=0)
        all_times = torch.cat([b_t, b_t], dim=0)

        self.last_seen_table.scatter_reduce_(
            dim=0, index=all_nodes, src=all_times, reduce="amax", include_self=True
        )
        self.seen_mask_table[all_nodes] = True


@dataclass
class DataBundle:
    """Holds train, val, test loaders, fitted statistics, and dataset metadata."""
    train: SplitLoader
    val: SplitLoader
    test: SplitLoader
    time_scale: float
    dt_p99: float
    num_nodes: int
    num_edges: int
    tgb_dataset: Optional[Any] = None

    def reset_state(self) -> None:
        """Resets the historical timestamp state (e.g. before train epoch 0)."""
        self.train.last_seen_table.zero_()
        self.train.seen_mask_table.zero_()


def compute_time_scale_and_p99(
    sources: Tensor,
    destinations: Tensor,
    timestamps: Tensor,
    num_nodes: int,
    config: LoaderConfig,
) -> Tuple[float, float]:
    """Computes train split median positive Delta_t and 99th percentile gap."""
    temp_last_seen = torch.zeros(num_nodes, dtype=torch.float64)
    temp_seen = torch.zeros(num_nodes, dtype=torch.bool)
    positive_deltas: List[float] = []

    B = config.batch_size
    N = sources.size(0)

    for i in range(0, N, B):
        end = min(i + B, N)
        b_src = sources[i:end]
        b_t = timestamps[i:end]

        for j in range(b_src.size(0)):
            u = int(b_src[j].item())
            tj = float(b_t[j].item())
            if bool(temp_seen[u].item()):
                gap = tj - float(temp_last_seen[u].item())
                if gap > 0.0:
                    positive_deltas.append(gap)

        b_dst = destinations[i:end]
        all_nodes = torch.cat([b_src, b_dst], dim=0)
        all_times = torch.cat([b_t, b_t], dim=0)
        temp_last_seen.scatter_reduce_(dim=0, index=all_nodes, src=all_times, reduce="amax", include_self=True)
        temp_seen[all_nodes] = True

    if not positive_deltas:
        return 1.0, 1.0

    deltas_arr = np.array(positive_deltas, dtype=np.float64)
    if config.time_scale is not None:
        scale = float(config.time_scale)
    elif config.scale_statistic == "median":
        scale = float(np.median(deltas_arr))
    elif config.scale_statistic == "mean":
        scale = float(np.mean(deltas_arr))
    else:
        scale = float(np.median(deltas_arr))

    if scale <= 0.0 or math.isnan(scale):
        scale = 1.0

    normalized_deltas = deltas_arr / scale
    dt_p99 = float(np.percentile(normalized_deltas, 99))
    return scale, dt_p99


def build_loaders(
    config: Optional[LoaderConfig] = None,
    raw_data: Optional[Dict[str, Any]] = None,
    train_mask: Optional[Tensor] = None,
    val_mask: Optional[Tensor] = None,
    test_mask: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> DataBundle:
    """Builds and returns DataBundle for train, val, and test splits."""
    if config is None:
        config = LoaderConfig()

    tgb_dataset = None

    if raw_data is None:
        try:
            from tgb.linkproppred.dataset import LinkPropPredDataset
        except ImportError as err:
            raise ImportError(
                "TGB library ('py-tgb') is required. Install it using: pip install py-tgb"
            ) from err

        print(f"Downloading/loading TGB dataset '{config.dataset_name}' into '{config.data_root}'...")
        tgb_dataset = LinkPropPredDataset(name=config.dataset_name, root=config.data_root)
        data = tgb_dataset.full_data

        src_key = "sources" if "sources" in data else "src"
        dst_key = "destinations" if "destinations" in data else "dst"
        t_key = "timestamps" if "timestamps" in data else "t"

        sources = torch.as_tensor(data[src_key], dtype=torch.long)
        destinations = torch.as_tensor(data[dst_key], dtype=torch.long)
        timestamps = torch.as_tensor(data[t_key], dtype=torch.float64)

        edge_idx_key = "edge_idxs" if "edge_idxs" in data else "edge_idx"
        if edge_idx_key in data:
            edge_indices = torch.as_tensor(data[edge_idx_key], dtype=torch.long)
        else:
            edge_indices = torch.arange(sources.size(0), dtype=torch.long)

        edge_feat = None
        if "edge_feat" in data and data["edge_feat"] is not None:
            edge_feat = torch.as_tensor(data["edge_feat"], dtype=torch.float32)

        train_mask = torch.as_tensor(tgb_dataset.train_mask, dtype=torch.bool)
        val_mask = torch.as_tensor(tgb_dataset.val_mask, dtype=torch.bool)
        test_mask = torch.as_tensor(tgb_dataset.test_mask, dtype=torch.bool)
    else:
        src_key = "sources" if "sources" in raw_data else "src"
        dst_key = "destinations" if "destinations" in raw_data else "dst"
        t_key = "timestamps" if "timestamps" in raw_data else "t"

        sources = torch.as_tensor(raw_data[src_key], dtype=torch.long)
        destinations = torch.as_tensor(raw_data[dst_key], dtype=torch.long)
        timestamps = torch.as_tensor(raw_data[t_key], dtype=torch.float64)

        if "edge_idx" in raw_data:
            edge_indices = torch.as_tensor(raw_data["edge_idx"], dtype=torch.long)
        elif "edge_idxs" in raw_data:
            edge_indices = torch.as_tensor(raw_data["edge_idxs"], dtype=torch.long)
        else:
            edge_indices = torch.arange(sources.size(0), dtype=torch.long)

        edge_feat = None
        if "edge_feat" in raw_data and raw_data["edge_feat"] is not None:
            edge_feat = torch.as_tensor(raw_data["edge_feat"], dtype=torch.float32)

        total_edges = sources.size(0)
        if train_mask is None:
            train_mask = torch.as_tensor(np.ones(total_edges, dtype=np.bool_))
        if val_mask is None:
            val_mask = torch.as_tensor(np.zeros(total_edges, dtype=np.bool_))
        if test_mask is None:
            test_mask = torch.as_tensor(np.zeros(total_edges, dtype=np.bool_))

    total_edges = sources.size(0)
    if num_nodes is None:
        num_nodes = max(int(sources.max().item()), int(destinations.max().item())) + 1

    def get_split(mask: Tensor):
        idxs = mask.nonzero(as_tuple=True)[0]
        ef = edge_feat[idxs] if edge_feat is not None else None
        return sources[idxs], destinations[idxs], timestamps[idxs], edge_indices[idxs], ef

    tr_s, tr_d, tr_t, tr_ei, tr_ef = get_split(train_mask)
    val_s, val_d, val_t, val_ei, val_ef = get_split(val_mask)
    te_s, te_d, te_t, te_ei, te_ef = get_split(test_mask)

    time_scale, dt_p99 = compute_time_scale_and_p99(tr_s, tr_d, tr_t, num_nodes, config)

    last_seen_table = torch.zeros(num_nodes, dtype=torch.float64)
    seen_mask_table = torch.zeros(num_nodes, dtype=torch.bool)

    try:
        train_dst_set = torch.unique(tr_d)
    except Exception:
        train_dst_set = Tensor(np.unique(tr_d.numpy()))

    train_loader = SplitLoader(
        "train", tr_s, tr_d, tr_t, tr_ei, tr_ef, num_nodes, config, time_scale, last_seen_table, seen_mask_table,
        train_dst_set=train_dst_set,
    )
    val_loader = SplitLoader(
        "val", val_s, val_d, val_t, val_ei, val_ef, num_nodes, config, time_scale, last_seen_table, seen_mask_table,
        train_dst_set=train_dst_set,
    )
    test_loader = SplitLoader(
        "test", te_s, te_d, te_t, te_ei, te_ef, num_nodes, config, time_scale, last_seen_table, seen_mask_table,
        train_dst_set=train_dst_set,
    )

    return DataBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        time_scale=time_scale,
        dt_p99=dt_p99,
        num_nodes=num_nodes,
        num_edges=total_edges,
        tgb_dataset=tgb_dataset,
    )


def parse_arguments() -> LoaderConfig:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(description="RT-GNN Streaming Temporal Edge Data Loader (v2)")
    parser.add_argument("--dataset_name", type=str, default="tgbl-wiki")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--delta_t_mode", type=str, choices=["batch", "event"], default="batch")
    parser.add_argument("--unseen_policy", type=str, choices=["zero_dt", "origin"], default="zero_dt")
    parser.add_argument("--no_normalize_dt", action="store_true")
    parser.add_argument("--time_scale", type=float, default=None)
    parser.add_argument("--scale_statistic", type=str, choices=["median", "mean"], default="median")
    parser.add_argument("--num_negatives", type=int, default=1)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_batches", type=int, default=None, help="Stop after max_batches (for testing)")

    args = parser.parse_args()
    return LoaderConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        batch_size=args.batch_size,
        delta_t_mode=args.delta_t_mode,
        unseen_policy=args.unseen_policy,
        normalize_dt=not args.no_normalize_dt,
        time_scale=args.time_scale,
        scale_statistic=args.scale_statistic,
        num_negatives=args.num_negatives,
        drop_last=args.drop_last,
        seed=args.seed,
        device=args.device,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    cfg = parse_arguments()
    print("=" * 65)
    print("RT-GNN Streaming Temporal Data Pipeline (v2)")
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
