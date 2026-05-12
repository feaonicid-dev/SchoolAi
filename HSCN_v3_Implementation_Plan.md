# HSCN v3 Implementation Plan

50+ mechanisms: 6-level adaptive compute, MLA+YOCO+IHA, Gated DeltaNet, DeepSeek V3 MoE, Neural ODE, BitNet, ChebyGLU, FP8, QAT, Muon, 7-phase curriculum on TPU v5e/v6e.

---

## Architecture

```
d_model=768, n_layers=12 (4 self + 8 cross), n_heads_q=12
n_experts=4 (top-2) + 2 shared, n_clusters=2
d_ff: T1=2304, T2=2304, T3=1024
vocab=256256 (Gemma + special), max_seq=4096 (infinite via DeltaNet + Ring)
Total: ~808M, Active: ~150M (18.5%)
```

### Data Flow

```
INPUT TOKENS [B, L]
  │
  ▼ Matryoshka Embed (√d scaled) → x_full [B,L,768], x_router [B,L,192]
  │ + 16 memory tokens
  │
  ▼ 4 self-decoder: MLA + iRoPE + QK-Norm → cache KV (YOCO)
  │   Sliding window (W=512) on self-decoder
  │
  ▼ 8 cross-decoder: cross-attn into cached KV
  │   Per layer (grows 4→12 in Phase 1):
  │
  ├─ 1. Sequence Mixer
  │    ├─ MLA + iRoPE + QK-Norm + IHA (optional) + Ring Attn + Prefix-LM
  │    └─ Gated DeltaNet (layers 2,5,8): delta-rule O(N) linear attention
  │
  ├─ 2. Tier1 FFN: xSwiGLU (always fires) [ChebyGLU optional]
  │
  ├─ 3. MoD Gate: 60% fire → MoE, 40% skip
  │
  ├─ 4. MoE: top-2 of 4 + 2 shared (DeepSeek V3 bias balancing)
  │    └─ Dense first+last layers
  │    └─ Inside each expert:
  │       ├─ UCR: ternary {-1,0,1}
  │       │    v=-1→SKIP, v=0→BitNet T2+T3, v=1→NeuralODE+T2+T3
  │       ├─ Verifier: confidence score
  │       │    ≥0.9→emit, <0.9→MoR-inner RMV retry (max 2)
  │       └─ Expert output
  │
  ├─ 5. Early Exit (layers 4,8)
  └─ 6. MoR-outer (layers 4,8): block repeat (max 4, lax.scan)
  │
  ▼ Final RMSNorm
  ├─→ Logits (weight-tied, softcap=30) + 4 MTP heads
  ├─→ Verifier score + Depth predictor + PRM head
```

### Inference

```
System 1: token → model → sample → next (MLA absorbed mode, INT4 KV, SSM carry-over)
System 2: verifier alarm / MoR retry>1 → MCTS → best path
ULTRAPLAN: <plan>→<step>→<observe>→verify→<replan>
Steering: verifier alarm → honesty vector at layer 6
Speculative: MTP heads 1-3 draft → full verify
Server: continuous batching + prefix caching
```

---

## 50+ Mechanisms

### Adaptive Compute (6)
1. MoD — Mixture-of-Depth gating (60% capacity)
2. UCR — Unified Complexity Router (ternary {-1,0,1})
3. MoR-Inner — Expert-level RMV retry (max 2)
4. MoR-Outer — Block repeat via lax.scan (max 4)
5. Early Exit — Exit ramps at layers 4, 8
6. Expert MoE — Token-choice top-k + DeepSeek V3 bias balancing

### Attention & Sequence Mixing (11)
7. MLA — Multi-Head Latent Attention (compressed KV)
8. YOCO — You Only Cache Once (shared KV)
9. Decoupled RoPE — Position separate from cached c_kv
10. iRoPE — Interleaved RoPE/NoPE (Llama 4, 64× context)
11. YaRN — Context extension magnitude scaling
12. IHA — Interleaved Head Attention (cross-head pseudo-heads)
13. Ring Attention — Shard sequence across devices
14. Sliding Window — Local+global alternating (Gemma 2)
15. Prefix-LM — Bidirectional prefix, causal generation
16. QK-Norm — RMSNorm on Q/K (Qwen3/OLMo)
17. Gated DeltaNet — Delta-rule linear attention (Qwen3-Next)

### FFN & Activation (4)
18. 3-Tier FFN — T1(shared)→T2(cluster)→T3(expert)
19. xSwiGLU — Gated linear unit with learnable β
20. BitNet xIELU — Ternary weights + trainable activation
21. ChebyGLU — Chebyshev basis + cubic gate (optional)

### Quantization & Precision (4)
22. BitNet 1.58b — Ternary {-1,0,+1} + STE
23. FP8 Training — E4M3/E5M2 matmul (v6e+)
24. QAT — int4/int8/FP4 for deployment
25. INT4 KV Cache — Inference memory compression

### Routing (3)
26. Expert/Token-Choice Routing — Configurable
27. DeepSeek V3 Load Balancing — Per-expert bias (no aux loss)
28. Router Tau Annealing — 2.0→0.1 temperature decay

### Neural ODE (2)
29. Neural ODE — Adjoint method, O(1) memory
30. StableDeltaHypernet — Low-rank ΔW=U@V (rank 8)

### Memory & Context (3)
31. Memory Compression Tokens — 16 learnable vectors
32. SSM State Carry-Over — DeltaNet state across chunks
33. NTK-Aware RoPE Scaling — Context extension

### Inference (7)
34. MLA Absorbed Mode — W_uk into W_q for long context
35. MCTS — UCB tree search
36. System 1/2 — Fast/slow cognitive loop
37. Speculative Decoding — MTP draft + verify
38. ULTRAPLAN — Plan/step/observe/replan
39. Activation Steering — Honesty vector injection
40. Inference Server — Continuous batching + prefix caching

### Training (16)
41. 7-Phase Curriculum — Progressive mechanism activation
42. EMA — Shadow params (decay=0.999)
43. GRPO — Group Relative Policy Optimization (Phase 7)
44. Multi-Token Prediction — 4 output heads
45. Self-Distillation — KL at phase transitions (EMA teacher)
46. Cross-Distillation — Pre-computed teacher logits (Qwen3.5-2B)
47. QDoRA — Quantized Direction-Magnitude LoRA
48. rsLoRA — Rank-Stabilized LoRA
49. Muon Optimizer — Momentum + Orthogonalization
50. Gradient Accumulation — lax.scan micro-batch
51. Gradient Checkpointing — jax.remat
52. Pipeline Parallelism — 1F1B / DualPipe
53. Progressive Depth — 4→8→12 layer growth
54. Depth Predictor — Auxiliary depth prediction
55. Token-Level Curriculum — Progressive difficulty
56. Router Warmup — Freeze non-router at transitions

### Data (7)
57. MinHash Dedup — 128-perm, Jaccard 0.7
58. Quality Filtering — Length/char ratio/perplexity
59. Contamination Detection — N-gram benchmark overlap
60. Replay Buffer — Re-sample high-loss tokens
61. Self-Play Data — Generate+verify synthetic (Phase 6+)
62. Optimized Packing — Multi-doc with attention masks
63. Custom Tokenizer — BPE + ULTRAPLAN + thinking tokens

### Stability (5)
64. Logit Softcap — Cap at 30 (Gemma 2)
65. Embed Scaling — √d_model (Gemma/T5)
66. Residual Scaling — 1/√(2·n_layers) (DeepNorm)
67. Kill Switches — NaN rollback + loss spike
68. MHC — Manifold-Constrained Hyper-Connections (optional)

---

## File Structure

```
hscn/
├── config.py                    # All hyperparameters (frozen dataclass)
├── model/
│   ├── embedding.py             # Matryoshka + memory tokens + √d scale
│   ├── attention.py             # MLA + iRoPE + YaRN + QK-Norm + YOCO + Prefix-LM + IHA + Ring + Sliding
│   ├── iha.py                   # Interleaved Head Attention (pseudo-heads)
│   ├── ssm.py                   # Mamba-2 (legacy)
│   ├── deltanet.py              # Gated DeltaNet (default SSM)
│   ├── norms.py                 # RMSNorm (optional bias)
│   ├── ffn.py                   # xSwiGLU + BitNet xIELU + TieredFFNBlock
│   ├── bitnet.py                # BitNet 1.58b ternary linear
│   ├── moe.py                   # Token-choice + DeepSeek V3 bias balancing
│   ├── ucr.py                   # Unified Complexity Router
│   ├── neural_ode.py            # StableDeltaHypernet + ODE + adjoint
│   ├── mod.py                   # MoD gating
│   ├── mor.py                   # MoR-inner + MoR-outer
│   ├── verifier.py              # Active Verifier + RMV
│   ├── early_exit.py            # Exit ramps
│   ├── prm.py                   # Process Reward Model
│   ├── memory.py                # Memory tokens + SSM carry-over
│   ├── mtp_heads.py             # Multi-token prediction
│   ├── depth_predictor.py       # Depth predictor
│   ├── kv_cache.py              # Model-level KV cache
│   ├── ring_attention.py        # Ring Attention
│   ├── sliding_window.py        # Sliding window mask
│   └── hscn.py                  # Main model assembly
├── training/
│   ├── losses.py                # All losses (L_task, L_mtp, L_matryoshka, L_ucr, L_depth, L_thinking, L_factuality)
│   ├── curriculum.py            # 7-phase curriculum + annealing
│   ├── token_curriculum.py      # Token-level difficulty
│   ├── adapters.py              # QDoRA + rsLoRA
│   ├── optimizer.py             # Per-tier AdamW + multi-transform
│   ├── muon.py                  # Muon optimizer
│   ├── distillation.py          # Self-distillation (EMA teacher)
│   ├── cross_distill.py         # Cross-distill pre-computation
│   ├── cross_distillation.py    # Cross-distill runtime
│   ├── ema.py                   # EMA shadow params
│   ├── sharding.py              # DP + FSDP + FSDP+TP
│   ├── checkpointing.py         # Orbax save/load
│   ├── gradient_accumulation.py # Accumulation utilities
│   ├── remat.py                 # jax.remat utilities
│   ├── fp8.py                   # FP8 training
│   ├── qat.py                   # QAT (int4/int8/FP4)
│   ├── pipeline_parallel.py     # 1F1B / DualPipe
│   ├── grpo.py                  # GRPO trainer
│   ├── tpu_setup.py             # TPU environment detection
│   └── trainer.py               # Main training loop
├── inference/
│   ├── generate.py              # Autoregressive + SSM carry-over + thinking
│   ├── kv_cache.py              # INT4 KV quantization
│   ├── mla_absorbed.py          # MLA absorbed mode
│   ├── mcts.py                  # MCTS
│   ├── system2.py               # System 1/2 loop
│   ├── speculative.py           # Speculative decoding
│   ├── agentic.py               # ULTRAPLAN
│   ├── steering.py              # Activation steering
│   └── server.py                # Inference server
├── data/
│   ├── pipeline.py              # Data loading + mixing
│   ├── tokenizer.py             # Custom BPE + special tokens
│   ├── cot_generator.py         # CoT trace generation
│   ├── curation.py              # Dedup + quality + contamination
│   └── efficiency.py            # Replay + self-play + packing
└── evaluation/
    ├── eval.py                  # Perplexity harness
    └── benchmarks.py            # Downstream benchmarks
```

---

## Implementation Steps

### Step 1: Config + Skeleton
`config.py` — HSCNConfig frozen dataclass with all 658 lines of params. `__init__.py` files. `requirements.txt`.

Key derived methods: `is_ssm_layer()`, `is_dense_layer()`, `get_current_phase()`, `get_router_tau()`, `get_active_layers()`, `is_mechanism_active()` (Python bool), `mechanism_scale()` (JAX-compatible), `with_tpu_overrides()`, `effective_batch_size`, `total_params_estimate`.

### Step 2: Core Components (no routing)
`embedding.py`, `attention.py`, `iha.py`, `ssm.py`, `deltanet.py`, `norms.py`, `ffn.py`, `bitnet.py`, `ring_attention.py`, `sliding_window.py`

### Step 3: Routing Mechanisms
`ucr.py`, `moe.py`, `mod.py`, `mor.py`, `verifier.py`, `neural_ode.py`

### Step 4: Main Model Assembly
`hscn.py`, `early_exit.py`, `prm.py`, `memory.py`, `mtp_heads.py`, `depth_predictor.py`, `kv_cache.py`

### Step 5: Training Infrastructure
`losses.py`, `curriculum.py`, `token_curriculum.py`, `adapters.py`, `optimizer.py`, `muon.py`, `distillation.py`, `cross_distill.py`, `cross_distillation.py`, `ema.py`, `sharding.py`, `checkpointing.py`, `gradient_accumulation.py`, `remat.py`, `fp8.py`, `qat.py`, `pipeline_parallel.py`, `grpo.py`, `tpu_setup.py`, `trainer.py`

### Step 6: Data Pipeline
`pipeline.py`, `tokenizer.py`, `cot_generator.py`, `curation.py`, `efficiency.py`

### Step 7: Inference
`generate.py`, `kv_cache.py`, `mla_absorbed.py`, `mcts.py`, `system2.py`, `speculative.py`, `agentic.py`, `steering.py`, `server.py`

### Step 8: Evaluation
`eval.py`, `benchmarks.py`

### Step 9: Entry Points
`train_v6e32.py`, `kaggle_tpu_v5e8.py`, `kaggle_gpu_teacher.py`

### Step 10: ChebyGLU Integration (optional)
`cheby_glu_jax.py` — Drop-in replacement for xSwiGLU in any tier. Chebyshev polynomial basis (order 3 default) + cubic tanh³ gate + partition-of-unity sech² residual. Pallas TPU kernel for order=3. Orthogonality regularizer. Separate optimizer mask for Chebyshev coefficients.

### Step 11: Known Bug Patches (applied at runtime)
- `hscn.py`: CE loss → `take_along_axis` (avoids one_hot vocab OOM)
- `losses.py`: Same CE fix + factuality p_correct fix
- `mtp_heads.py`: Same CE fix
- `trainer.py`: `donate_argnums=(0,1)` for memory, `batch['labels']` → `batch['input_ids'][:, 1:]`
- `optimizer.py`: `create_param_labels(params, config)` lambda fix
- `iha.py`: Manual einsum replaces `jax.nn.dot_product_attention` (GQA assertion)
- `sharding.py`: Mesh axis name `'dp'` not `'data'`
- `attention.py`: Missing `RMSNorm` import, IHA out_dim/v shape fixes
- `ema.py`: `p.astype(jnp.float32)` not `jnp.float32(p)`
- `grpo.py`: `jnp.broadcast_to` not `.broadcast_to`
- `system2.py`, `generate.py`, `steering.py`: 4-tuple unpack from `model.apply()`
- `hscn.py`: remat policy → `nothing_saveable` for memory

### Step 12: Architecture Spec
This document.

---

## Loss Function

```
L_total = L_task (CE with take_along_axis)
        + 0.25 × L_mtp
        + 0.1  × L_matryoshka
        + 0.1  × L_thinking (Phase 1+, Qwen3 style)
        + 0.05 × L_factuality (anti-hallucination)
        + 0.05 × L_early_exit
        + 0.01 × L_mor_entropy
        + 0.01 × L_mod_diversity
        + 0.01 × L_ucr_entropy
        + 0.001 × L_z_loss
        + 0.1  × L_depth_predict (Phase 1)
        + L_cross_distill (Phases 1-2, decaying)
        + L_self_distill (phase transitions, decaying)
        + 0.1  × L_verifier (Phase 3+)
        + 0.1  × L_prm (Phase 5+)
        + L_grpo (Phase 7+)
```

## Compute Summary

| Metric | 150M (v6e-8) | 500M (v6e-8) | 808M (v6e-32) |
|---|---|---|---|
| Total params | ~150M | ~510M | ~808M |
| Active/token | ~50M | ~150M | ~150M |
| Memory/chip | ~4 GB | ~9 GB | ~13 GB |
| 30B tokens | ~9h | ~25h | ~11h |
| Cost ($2-3/hr) | ~$20 | ~$65 | ~$30 |

## 7-Phase Curriculum

| Phase | Steps | Mechanisms Activated |
|---|---|---|
| 1 (0-20%) | 0-10K | Attention + DeltaNet + T1 FFN + Progressive Depth + Depth Predictor + Cross-Distill + Matryoshka + Thinking |
| 2 (20-35%) | 10K-17.5K | + MoE + MTP + T2/T3 FFN + Cross-Distill fading |
| 3 (35-50%) | 17.5K-25K | + UCR + BitNet + Neural ODE + MoR-Inner + MoR-Outer + Verifier |
| 4 (50-70%) | 25K-35K | + MoD gating |
| 5 (70-80%) | 35K-40K | + PRM + Steering extraction |
| 6 (80-90%) | 40K-45K | + PRM training + CoT data + Self-Play data |
| 7 (90-100%) | 45K-50K | + GRPO + QDoRA + rsLoRA |
