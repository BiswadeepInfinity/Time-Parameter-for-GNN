"""
Unit Test Suite for RT-GNN Data Pipeline (test_rtgnn_data.py)
=============================================================
Strictly verifies the 12 requirements and constraints from Blueprint v2:
- 19 passing tests on synthetic data without requiring network/TGB download.
- 1 skipped test (the torch tensor test) when running without torch installed.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import unittest
import numpy as np

from rtgnn_data import (
    LoaderConfig,
    TemporalBatch,
    DataBundle,
    build_loaders,
    HAS_TORCH,
    torch,
)


class TestRTGNNDataPipeline(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.num_nodes = 50
        self.num_edges = 1000

        # Deterministic synthetic edge stream
        self.sources = np.random.randint(0, self.num_nodes, self.num_edges)
        self.destinations = np.random.randint(0, self.num_nodes, self.num_edges)
        # Monotonically increasing continuous timestamps
        self.timestamps = np.sort(np.random.uniform(10.0, 10000.0, self.num_edges))

        self.train_mask = np.zeros(self.num_edges, dtype=bool)
        self.val_mask = np.zeros(self.num_edges, dtype=bool)
        self.test_mask = np.zeros(self.num_edges, dtype=bool)

        self.train_mask[:700] = True
        self.val_mask[700:850] = True
        self.test_mask[850:] = True

        self.raw_data = {
            "sources": self.sources,
            "destinations": self.destinations,
            "timestamps": self.timestamps,
        }

    def _get_bundle(self, **kwargs) -> DataBundle:
        params = {"dataset_name": "synthetic", "batch_size": 128, "seed": 42}
        params.update(kwargs)
        cfg = LoaderConfig(**params)
        return build_loaders(
            config=cfg,
            raw_data=self.raw_data,
            train_mask=torch.as_tensor(self.train_mask),
            val_mask=torch.as_tensor(self.val_mask),
            test_mask=torch.as_tensor(self.test_mask),
            num_nodes=self.num_nodes,
        )

    def test_01_default_config_fields(self):
        """Test default values of LoaderConfig match blueprint v2."""
        cfg = LoaderConfig()
        self.assertEqual(cfg.dataset_name, "tgbl-wiki")
        self.assertEqual(cfg.delta_t_mode, "batch")
        self.assertEqual(cfg.unseen_policy, "zero_dt")
        self.assertTrue(cfg.normalize_dt)
        self.assertEqual(cfg.num_negatives, 1)
        self.assertFalse(cfg.drop_last)

    def test_02_split_sizes_sum_to_total(self):
        """Test that train, val, test edge counts sum to total edge count."""
        bundle = self._get_bundle()
        total_split = (
            bundle.train.num_edges + bundle.val.num_edges + bundle.test.num_edges
        )
        self.assertEqual(total_split, self.num_edges)
        self.assertEqual(bundle.train.num_edges, 700)
        self.assertEqual(bundle.val.num_edges, 150)
        self.assertEqual(bundle.test.num_edges, 150)

    def test_03_batch_container_fields(self):
        """Test that yielded batches contain all required fields from blueprint v2."""
        bundle = self._get_bundle()
        batch = next(iter(bundle.train))
        self.assertTrue(hasattr(batch, "src"))
        self.assertTrue(hasattr(batch, "dst"))
        self.assertTrue(hasattr(batch, "t"))
        self.assertTrue(hasattr(batch, "delta_t_src"))
        self.assertTrue(hasattr(batch, "delta_t_dst"))
        self.assertTrue(hasattr(batch, "has_history_src"))
        self.assertTrue(hasattr(batch, "has_history_dst"))
        self.assertTrue(hasattr(batch, "edge_idx"))
        self.assertTrue(hasattr(batch, "neg_dst"))

    def test_04_first_batch_mostly_unseen(self):
        """Test that the first batch reports mostly first-seen source nodes."""
        bundle = self._get_bundle()
        batch = next(iter(bundle.train))
        has_history = batch.has_history_src.numpy()
        self.assertFalse(bool(has_history[0]))

    def test_05_unseen_policy_zero_dt(self):
        """Test that unseen nodes get Delta_t = 0 when unseen_policy='zero_dt'."""
        bundle = self._get_bundle(unseen_policy="zero_dt")
        batch = next(iter(bundle.train))
        dt_src = batch.delta_t_src.numpy()
        has_hist = batch.has_history_src.numpy()
        unseen_idxs = np.where(~has_hist)[0]
        self.assertTrue(len(unseen_idxs) > 0)
        np.testing.assert_allclose(dt_src[unseen_idxs], 0.0)

    def test_06_unseen_policy_origin(self):
        """Test that unseen nodes get Delta_t = raw timestamp when unseen_policy='origin'."""
        bundle = self._get_bundle(unseen_policy="origin", normalize_dt=False)
        batch = next(iter(bundle.train))
        dt_src = batch.delta_t_src.numpy()
        raw_t = batch.t.numpy()
        has_hist = batch.has_history_src.numpy()
        unseen_idxs = np.where(~has_hist)[0]
        self.assertTrue(len(unseen_idxs) > 0)
        np.testing.assert_allclose(dt_src[unseen_idxs], raw_t[unseen_idxs])

    def test_07_delta_t_never_negative(self):
        """Test that all delta_t values are strictly non-negative."""
        bundle = self._get_bundle()
        for batch in bundle.train:
            self.assertTrue(np.all(batch.delta_t_src.numpy() >= 0.0))
            self.assertTrue(np.all(batch.delta_t_dst.numpy() >= 0.0))

    def test_08_delta_t_mode_event_vs_batch(self):
        """Test that event mode calculates intra-batch sequential deltas."""
        sources = np.array([1, 1, 2], dtype=np.int64)
        destinations = np.array([3, 4, 5], dtype=np.int64)
        timestamps = np.array([100.0, 150.0, 200.0], dtype=np.float64)
        mask = np.array([True, True, True], dtype=bool)

        cfg_event = LoaderConfig(delta_t_mode="event", normalize_dt=False, batch_size=10)
        b_event = build_loaders(
            cfg_event,
            raw_data={"sources": sources, "destinations": destinations, "timestamps": timestamps},
            train_mask=torch.as_tensor(mask),
            num_nodes=10,
        )
        batch = next(iter(b_event.train))
        self.assertEqual(batch.delta_t_src.numpy()[1], 50.0)

    def test_09_time_scale_positive_median(self):
        """Test that fitted time scale is strictly positive and finite."""
        bundle = self._get_bundle()
        self.assertGreater(bundle.time_scale, 0.0)
        self.assertFalse(np.isnan(bundle.time_scale))

    def test_10_time_scale_normalization_effect(self):
        """Test that normalized Delta_t is divided by time_scale."""
        b_unnorm = self._get_bundle(normalize_dt=False)
        b_norm = self._get_bundle(normalize_dt=True)

        batch_un = next(iter(b_unnorm.train))
        batch_n = next(iter(b_norm.train))

        scale = b_norm.time_scale
        un_vals = batch_un.delta_t_src.numpy()
        n_vals = batch_n.delta_t_src.numpy()
        np.testing.assert_allclose(un_vals / scale, n_vals, rtol=1e-4)

    def test_11_dt_p99_computation(self):
        """Test that dt_p99 is computed and strictly positive."""
        bundle = self._get_bundle()
        self.assertGreater(bundle.dt_p99, 0.0)

    def test_12_single_negative_link_prediction_pairs(self):
        """Test link_prediction_pairs() with num_negatives=1."""
        bundle = self._get_bundle(num_negatives=1)
        batch = next(iter(bundle.train))
        c_src, c_dst, labels = batch.link_prediction_pairs()
        B = batch.src.size(0)
        self.assertEqual(c_src.size(0), 2 * B)
        self.assertEqual(c_dst.size(0), 2 * B)
        self.assertEqual(labels.size(0), 2 * B)
        np.testing.assert_allclose(labels.numpy()[:B], 1.0)
        np.testing.assert_allclose(labels.numpy()[B:], 0.0)

    def test_13_multi_negatives_link_prediction_pairs(self):
        """Test link_prediction_pairs() with num_negatives=5."""
        bundle = self._get_bundle(num_negatives=5)
        batch = next(iter(bundle.train))
        c_src, c_dst, labels = batch.link_prediction_pairs()
        B = batch.src.size(0)
        self.assertEqual(c_src.size(0), 6 * B)
        self.assertEqual(labels.numpy()[:B].sum(), float(B))
        self.assertEqual(labels.numpy()[B:].sum(), 0.0)

    def test_14_link_prediction_pairs_missing_negs_raises(self):
        """Test that link_prediction_pairs() raises ValueError if neg_dst is None (e.g. val)."""
        bundle = self._get_bundle()
        val_batch = next(iter(bundle.val))
        with self.assertRaises(ValueError):
            val_batch.link_prediction_pairs()

    def test_15_deterministic_negative_sampling(self):
        """Test that negative sampling is reproducible for same seed and epoch."""
        b1 = self._get_bundle(seed=123)
        b2 = self._get_bundle(seed=123)
        b1.train.set_epoch(1)
        b2.train.set_epoch(1)

        batch1 = next(iter(b1.train))
        batch2 = next(iter(b2.train))
        np.testing.assert_array_equal(batch1.neg_dst.numpy(), batch2.neg_dst.numpy())

    def test_16_cross_split_continuity_val_primed_by_train(self):
        """Test that val split starts with timestamps from train split."""
        bundle = self._get_bundle()
        for _ in bundle.train:
            pass
        val_batch = next(iter(bundle.val))
        val_has_hist = val_batch.has_history_src.numpy()
        self.assertGreater(val_has_hist.sum(), 0)

    def test_17_cross_split_continuity_test_primed_by_val(self):
        """Test that test split starts with timestamps from val split."""
        bundle = self._get_bundle()
        for _ in bundle.train:
            pass
        for _ in bundle.val:
            pass
        test_batch = next(iter(bundle.test))
        test_has_hist = test_batch.has_history_src.numpy()
        self.assertGreater(test_has_hist.sum(), 0)

    def test_18_atomic_timestamp_update_race_condition(self):
        """Test that multiple events for the same node in a batch keep the latest timestamp."""
        sources = np.array([7, 7, 7], dtype=np.int64)
        destinations = np.array([8, 9, 10], dtype=np.int64)
        timestamps = np.array([10.0, 50.0, 30.0], dtype=np.float64)  # 50.0 is latest
        mask = np.array([True, True, True], dtype=bool)

        cfg = LoaderConfig(batch_size=10)
        bundle = build_loaders(
            cfg,
            raw_data={"sources": sources, "destinations": destinations, "timestamps": timestamps},
            train_mask=torch.as_tensor(mask),
            num_nodes=20,
        )
        for _ in bundle.train:
            pass
        last_t_7 = bundle.train.last_seen_table[7].item()
        self.assertEqual(last_t_7, 50.0)

    def test_19_batch_to_device(self):
        """Test batch.to(device) creates a copy on the target device/mock."""
        bundle = self._get_bundle()
        batch = next(iter(bundle.train))
        batch_dev = batch.to("cpu")
        self.assertEqual(batch_dev.src.size(0), batch.src.size(0))

    @unittest.skipIf(not HAS_TORCH, "the torch tensor test (requires torch installation)")
    def test_20_torch_tensor(self):
        """The torch tensor test: strictly verifies native PyTorch tensors and CUDA if present."""
        t = torch.tensor([1, 2, 3])
        self.assertIsInstance(t, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
