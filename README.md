# Continuous-Time Rotational Graph Neural Network (RT-GNN)

[![Tests](https://img.shields.io/badge/Verification%20Gates-35%2F35%20PASS-brightgreen)](TEST_REPORT.md)
[![TGB Benchmark](https://img.shields.io/badge/TGB%20Benchmark-tgbl--wiki-blue)](https://tgb.complexdss.web.ox.ac.uk/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)

Continuous-time dynamic graph representation learning via complex-plane decay-rotation operators and parallel associative prefix scans.

---

## 1. Origin & Motivation

RT-GNN emerged directly from **neurotechnology and computational neuroscience** research: modeling real-time **synapse formation and continuous temporal signaling across the *C. elegans* connectome**. 

In biological neural circuits, synaptic plasticity does not advance in discrete algorithmic ticks; instead, signal decay and phase rotations evolve asynchronously over continuous time intervals $\Delta t$. Applying conventional discrete-time GNNs (or heavy quadratic attention transformers) created computational bottlenecks and temporal aliasing.

RT-GNN solves this by formulating the temporal parameter directly in the complex number plane $\mathbb{C}^d$, enabling:
1. **$O(1)$ Closed Semigroup Transitions**: Calculating continuous time jumps analytically without step-by-step unrolling:
   $$T(a) \circ T(b) = T(a + b)$$
2. **$O(\log K)$ Parallel Associative State Scans**: Continuous state updates computed via parallel doubling prefix scans, eliminating sequential Backpropagation Through Time (BPTT).

---

## 2. Empirical Benchmark Results (Official TGB Protocol)

All evaluations are conducted according to the official **Temporal Graph Benchmark (TGB)** protocol on `tgbl-wiki` ($157{,}474$ edges, $9{,}227$ nodes) with official fixed negative sets across **5 random seeds** (seeds 0–4, `ddof=1`):

| Model Architecture | Paradigm / History Depth | Test MRR (Mean ± Std) | Val MRR (Mean ± Std) | Throughput (edges/s) |
|---|---|:---:|:---:|:---:|
| **DyRep** (Trivedi et al., ICLR) | Point Process / $O(K)$ | `0.0500` | — | ~2,500 (Ref) |
| **GraphMixer** (Cong et al., ICLR) | Fixed MLP-Mixer / $O(1)$ | `0.1180` | — | ~5,000 (Ref) |
| **TGAT** (Xu et al., ICLR) | Temporal Attn / Fourier | `0.1410` | — | ~4,000 (Ref) |
| **TCL** (Wang et al., NeurIPS) | Contrastive / $O(K)$ | `0.2070` | — | ~3,200 (Ref) |
| **TGN** (Rossi et al., ICML) | Recurrent GRU / $O(K)$ | `0.3960` | — | ~3,500 (Ref) |
| ⚡ **Stateless RT-GNN (Ours)** | **Closed Semigroup / $O(1)$** | **`0.4683 ± 0.0111`** | **`0.5261 ± 0.0073`** | **5,684.0 (Measured CPU)** |
| 🚀 **Stateful RT-GNN (Ours)** | **Complex SSM / $O(\log K)$** | **`0.4934 ± 0.0159`** | **`0.5192 ± 0.0118`** | **8,835.6 (Measured CPU)** |
| **EdgeBank (unlimited)** | Non-parametric / $O(1)$ | `0.4950` | — | Heuristic (Ref) |
| **EdgeBank (tw)** | Time-Window Set / $O(1)$ | `0.5710` | — | Heuristic (Ref) |
| **CAWN** (Zhu et al., ICLR) | Anonymous Walk / $O(W)$ | `0.7110` | — | ~1,200 (Ref) |
| **NAT** (Gao et al., ICML) | Neighborhood Dict / $O(N)$ | `0.7490` | — | ~1,000 (Ref) |
| **DyGFormer** (Yu et al., NeurIPS) | Transformer / $O(K^2)$ | `0.7980` | — | ~800 (GPU Ref) |

*Published baseline results cited from the official TGB Leaderboard ([Huang et al., NeurIPS 2023](https://arxiv.org/abs/2307.01026)).*

---

## 3. Mathematical Foundations

### Continuous-Time Complex Rotation & Decay Operator
For an interaction interval $\Delta t \ge 0$, the temporal transition operator $T(\Delta t)$ acts on a complex representation $z = h_{\text{real}} + i h_{\text{imag}} \in \mathbb{C}^d$:

$$T(\Delta t) = \exp\big((-\lambda + i\omega) \Delta t\big) = \exp(-\lambda \Delta t) \cdot \big(\cos(\omega \Delta t) + i \sin(\omega \Delta t)\big)$$

- **Bounded Contractive Decay**: $\lambda = \text{softplus}(\lambda_{\text{raw}}) > 0$ guarantees strictly contractive decay over long inactive periods, eliminating numerical explosion.
- **Harmonic Rotation Frequency**: $\omega \in [\omega_{\min}, \omega_{\max}]$ captures periodic and multi-scale cyclic patterns.

### Semigroup Property
For any successive time intervals $a, b \ge 0$:

$$T(a) \circ T(b) = \exp\big((-\lambda + i\omega)(a + b)\big) = T(a + b)$$

Spatial graph convolutions decouple the static relation rotation $\theta$ outside the continuous-time operator, preserving the semigroup property across arbitrary historical jumps.

### Parallel Associative Prefix Scan
In Stateful RT-GNN, node states update across an event sequence $k=1,\dots,K$ via linear recurrence:

$$h_k = a_k \odot h_{k-1} + b_k$$

Using the binary associative operator $(a_1, b_1) \circ (a_2, b_2) = (a_2 a_1, a_2 b_1 + b_2)$, prefix states are computed in **$O(\log K)$ parallel steps**, matching sequential loops to $< 10^{-5}$ numerical error while bounding gradients over sequences of $K = 1{,}024+$ interactions.

---

## 4. Repository Structure

```
├── rtgnn_model.py          # Stateless RT-GNN, closed semigroup operators, HistoryBuffer
├── rtgnn_stateful_gnn.py   # Stateful RT-GNN with parallel complex associative scan memory
├── rtgnn_scan.py           # Parallel prefix scan & associative binary combiner
├── rtgnn_data.py           # Continuous-time streaming DataLoader & zero-leakage pipeline
├── train.py                # End-to-end training and TGB evaluation for Stateless RT-GNN
├── train_stateful.py       # End-to-end training and TGB evaluation for Stateful RT-GNN
├── run_benchmark.py        # Multi-seed benchmark suite on tgbl-wiki
├── run_benchmark.bat       # Windows 1-click execution script for full 5-seed benchmark
├── run_tests.bat           # Windows 1-click execution script for unit test suite
├── tests/
│   ├── test_rtgnn_data.py  # 20 streaming invariant tests
│   ├── test_rtgnn_model.py # 10 mathematical and autograd tests
│   ├── test_rtgnn_scan.py  # 5 parallel prefix scan gate tests
│   └── run_and_report.py   # Complete 35-test execution and TEST_REPORT.md generator
├── benchmark_results.json  # Raw measured seed outputs, throughput, and memory stats
├── ARCHITECTURE.md         # Full theoretical specification and design documentation
└── TEST_REPORT.md          # Generated formal reviewer audit certification
```

---

## 5. Verification & Benchmark Execution

### Run the 35-Gate Verification Suite
```bash
python tests/run_and_report.py
```
*(All 35 mathematical scenarios, semigroup proofs, and gradient tests pass in < 0.5s).*

### Run the 5-Seed Empirical Benchmark
```bash
python run_benchmark.py --seeds 0 1 2 3 4 --epochs 5 --models stateless stateful
```
Or on Windows:
```cmd
run_benchmark.bat
```

---

## 6. License
MIT License.
