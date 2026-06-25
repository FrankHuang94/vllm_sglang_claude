# Transformer Architecture Internals for Inference Engineers

> **Audience & scope:** This is a PRIMARY reference file. It covers every architectural component of a modern decoder-only transformer at the level of detail an inference engineer needs to *optimize* it — not to train it. For each component we give the math, the inference-time cost model, the data layout in GPU memory, and the optimization levers that vLLM and SGLang actually pull. Cross-references point to the systems files (03–11) where these components become engineering problems.

---

## Table of Contents

1. The Attention Mechanism — Full Mathematical Derivation
2. Attention Variants: MHA, MQA, GQA, MLA, Sliding Window
3. FlashAttention Mechanics
4. Feed-Forward Networks and Mixture of Experts
5. Positional Encoding and Long Context (RoPE, YaRN, ALiBi)
6. KV Cache Data Structures and Memory Layout
7. KV Cache Quantization
8. Sampling and Decoding Strategies
9. Speculative and Tree-Based Decoding (foundations)
10. Constrained / Structured Decoding (foundations)
11. Quantization Fundamentals for Inference

---

## 1. The Attention Mechanism — Full Mathematical Derivation

### 1.1 Scaled dot-product attention

Given an input sequence represented as a matrix `X ∈ ℝ^{n×d_model}` (n tokens, each a `d_model`-dimensional vector), self-attention first computes three linear projections:

```
Q = X · W_Q      W_Q ∈ ℝ^{d_model × d_k}     Q ∈ ℝ^{n × d_k}
K = X · W_K      W_K ∈ ℝ^{d_model × d_k}     K ∈ ℝ^{n × d_k}
V = X · W_V      W_V ∈ ℝ^{d_model × d_v}     V ∈ ℝ^{n × d_v}
```

Attention is then:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

Step by step, with shapes and cost:

1. **Score matrix** `S = Q·Kᵀ ∈ ℝ^{n×n}`. This is a GEMM costing `O(n² · d_k)` FLOPs and — crucially for FlashAttention — producing an `n×n` matrix that, if materialized in HBM, costs `O(n²)` memory.
2. **Scaling** by `1/√d_k`. The division stabilizes gradients/logits: without it, for large `d_k` the dot products grow with variance proportional to `d_k`, pushing softmax into saturated regions with vanishing gradients (and, at inference, into numerically extreme regimes). `√d_k` normalizes the variance of each score back to O(1).
3. **Softmax** along each row: `P = softmax(S)`, `P ∈ ℝ^{n×n}`, each row summing to 1. Numerically this is always computed as `softmax(s)_i = exp(s_i − max_j s_j) / Σ_k exp(s_k − max_j s_j)` — the max-subtraction prevents `exp` overflow. This *online* reformulation is the seed of FlashAttention (§3).
4. **Value aggregation** `O = P·V ∈ ℝ^{n×d_v}`. Another GEMM, `O(n² · d_v)` FLOPs.

**Total complexity:** `O(n² · d)` compute and (naively) `O(n²)` memory for the score matrix. The quadratic-in-sequence-length scaling is *the* reason long context is hard, and the reason prefill (which processes all n tokens at once) is compute-bound while decode (n grows by 1 per step, query is a single row) is bandwidth-bound.

### 1.2 Causal masking

For autoregressive decoding, token `i` may only attend to tokens `j ≤ i`. This is enforced by adding a mask `M` to the scores before softmax, with `M_{ij} = 0` for `j ≤ i` and `M_{ij} = −∞` for `j > i`. The `−∞` entries become 0 after `exp`. In an inference kernel the mask is never materialized as a dense matrix; it is applied implicitly by simply not iterating over future positions (in decode, there *are* no future positions — the query is the last token).

### 1.3 The decode-time specialization

During decode, the new token's query is a *single row* `q ∈ ℝ^{1×d_k}`. The keys and values are the *entire history* `K ∈ ℝ^{n×d_k}`, `V ∈ ℝ^{n×d_v}` retrieved from the KV cache. So the decode attention is:

```
o = softmax( q·Kᵀ / √d_k ) · V          q·Kᵀ ∈ ℝ^{1×n}
```

This is a matrix–vector product (GEMV) against the cached K and V. The FLOPs are `O(n·d)` but the bytes moved are `O(n·d)` as well (you read all of K and V once) → arithmetic intensity ≈ 1, memory-bound. This is exactly the regime PagedAttention's kernel (File 03) and FlashInfer's decode kernel (File 10) are built to serve efficiently against a *non-contiguous, paged* KV cache.

---

## 2. Attention Variants

The single biggest lever on KV cache size — and therefore on serving concurrency and cost — is the choice of attention variant, because it sets `num_kv_heads` and the per-token cache bytes. Inference engineers must understand all five live in 2024–2026 production.

### 2.1 Multi-Head Attention (MHA)

The original Transformer (Vaswani et al. 2017) uses `h` independent attention heads, each operating in a `d_k = d_model / h` subspace:

```
head_i = Attention(X·W_Q^i, X·W_K^i, X·W_V^i)        i = 1..h
MultiHead(X) = Concat(head_1, …, head_h) · W_O
```

Each head learns a different attention pattern (syntactic, positional, coreference, etc.) in its own subspace. **Inference cost:** MHA stores K and V for *all* `h` heads → `num_kv_heads = h`, the maximum possible KV cache. GPT-2, GPT-3, OPT, LLaMA-1, and most pre-2023 models use MHA.

### 2.2 Multi-Query Attention (MQA)

Shazeer 2019 ("Fast Transformer Decoding") observed that the KV cache, not the query, is the inference bottleneck. **MQA shares a single K head and a single V head across all `h` query heads.**

```
num_kv_heads = 1     →     KV cache shrinks by factor of h vs MHA
```

For a model with 64 query heads, MQA cuts the KV cache by 64×. The decode step also reads far less from HBM, directly improving the memory-bound decode latency. The cost is a small quality degradation (all query heads must agree on one set of keys/values) and some training instability. Used in PaLM, Falcon, and (effectively) Gemini-class models.

### 2.3 Grouped-Query Attention (GQA)

Ainslie et al. 2023 (arXiv 2305.13245) interpolates between MHA and MQA. Query heads are partitioned into `G` groups; each group shares one K/V head:

```
G = h    →   MHA (one KV head per query head)
G = 1    →   MQA (one KV head total)
1 < G < h →  GQA
```

**LLaMA-2/3 70B uses `G = 8`** (8 KV heads, 64 query heads) → **8× KV cache reduction vs MHA** with quality nearly indistinguishable from MHA. GQA is now the default for essentially all large open models because it sits at the sweet spot: most of MQA's memory savings, almost none of the quality loss.

**Critical inference constraint:** under tensor parallelism (File 05) `num_kv_heads` must be divisible by the TP degree, or KV heads must be replicated. With 8 KV heads, TP ≤ 8 keeps one KV head per GPU; TP=16 would require replicating KV heads across GPU pairs. This constraint shapes deployment topology.

### 2.4 Multi-head Latent Attention (MLA)

DeepSeek-V2/V3's MLA is the most aggressive production KV-cache compression. Instead of caching K and V per head, it caches a **single low-rank latent vector** per token and reconstructs K and V on the fly.

```
c_KV = x · W_DKV          # down-projection to a small latent dim d_c  (d_c ≪ h·d_head)
K_C  = c_KV · W_UK        # up-projection to keys
V_C  = c_KV · W_UV        # up-projection to values
# Cache stores only c_KV  (size d_c per token), not K_C/V_C
```

For DeepSeek-V2: `d_c = 512` while MHA-equivalent storage would be `num_kv_heads·d_head·2 = 128·128·... ` — the realized reduction is roughly **93% vs MHA at equivalent model capacity** (≈64× on the raw K/V tensors). Because RoPE (a position-dependent rotation, §5) cannot be absorbed into the static `W_UK`/`W_UV` up-projections, MLA uses **decoupled RoPE**: a small separate set of RoPE-carrying key dimensions `K_R` is concatenated with the reconstructed `K_C`. At inference, the up-projection matrices can be *absorbed* into the query projection (`W_Q_absorbed = W_Q · W_UKᵀ`), eliminating one matmul per layer. The full MLA treatment, including vLLM's `DeepseekV2Attention` implementation and weight absorption, is in File 15.

### 2.5 Sliding Window Attention (SWA)

Mistral 7B popularized SWA: each token attends only to the previous `w` tokens (e.g. `w = 4096`).

```
Complexity:  O(n · w)   instead of  O(n²)
KV cache:    capped at w tokens per layer  (a rolling buffer)
```

The obvious worry — losing long-range dependencies — is mitigated by *depth*: information propagates `w` tokens per layer, so after `L` layers the effective receptive field is `L·w` tokens. Modern designs (e.g. some Gemma and Llama variants) *interleave* SWA layers with full-attention layers to get bounded KV cache on most layers while preserving true long-range attention on a few. For the inference engine, SWA means the KV cache for SWA layers is a fixed-size ring buffer — `--sliding-window` in vLLM triggers block-manager logic that frees blocks older than `w` (File 03).

### 2.6 Summary table

| Variant | num_kv_heads | KV cache vs MHA | Quality | Representative models |
|---|---|---|---|---|
| MHA | h | 1× | baseline | GPT-2/3, OPT, LLaMA-1 |
| MQA | 1 | 1/h | slight ↓ | PaLM, Falcon |
| GQA | G (e.g. 8) | G/h | ≈ MHA | LLaMA-2/3, Mistral, Qwen |
| MLA | latent d_c | ~0.07× | ≈ MHA | DeepSeek-V2/V3 |
| SWA | h (capped len) | w/n per layer | slight ↓ | Mistral, parts of Gemma |

---

## 3. FlashAttention Mechanics

FlashAttention (Dao et al. 2022, arXiv 2205.14135; FA-2 arXiv 2307.08691; FA-3 2024) is not a different *mathematical* attention — it computes the exact same softmax(QKᵀ)V — but a different *IO schedule* that avoids materializing the `n×n` score matrix in HBM. It is mandatory in any serious inference system; vLLM and SGLang both default to FlashAttention/FlashInfer kernels.

### 3.1 The IO problem

Standard attention reads/writes the `n×n` matrices `S` and `P` to HBM:

```
Naive IO ≈ Θ(n² + n·d)    bytes to/from HBM
```

For `n = 8192`, the `2n²` term (≈128 MB at FP16) dwarfs the `2n·d` term (≈2 MB). Attention is therefore *memory-bound on HBM traffic*, not compute-bound, in the naive implementation — the GPU spends its time writing and re-reading the giant score matrix.

### 3.2 Tiling + online softmax

FlashAttention tiles Q into row blocks and K/V into column blocks sized to fit in **SRAM** (on-chip shared memory, ~192 KB/SM on A100, ~228 KB on H100). For each Q tile it streams through the K/V tiles, computing partial attention and combining results *without ever writing the full S or P to HBM*.

The mathematical enabler is **online softmax** (the Milakov–Gimelshein running-max trick). Maintain a running max `m`, running denominator `ℓ`, and running output accumulator `O`. For each new K/V block producing local scores `S_new`:

```
m_new = max(m_old, rowmax(S_new))
ℓ_new = exp(m_old − m_new)·ℓ_old + rowsum(exp(S_new − m_new))
O_new = exp(m_old − m_new)·O_old + exp(S_new − m_new)·V_new
```

The `exp(m_old − m_new)` correction factor rescales the previously accumulated output and denominator whenever a larger max is discovered, keeping everything numerically stable in a single streaming pass. At the end, `O = O / ℓ`.

### 3.3 IO complexity result

By keeping tiles in SRAM of size `M`, HBM traffic drops to:

```
FlashAttention IO ≈ Θ(n² · d / M)
```

For `M ≈ 192 KB` and `d = 128`, this is roughly `n·d` rather than `n²` for the regime `n ≫ √M` — a reduction of one factor of `n`. The practical result is **2–4× speedup** over naive attention and the elimination of the `O(n²)` memory blowup, which is what makes 100K+ context prefill feasible at all.

### 3.4 FlashAttention-2 and -3

**FA-2** repartitions work so that the *query* dimension is parallelized across warps (FA-1 parallelized over K/V), reduces the number of non-matmul FLOPs (rescalings), and cuts shared-memory bank conflicts. It reaches ~70–75% of A100 peak FLOP/s.

**FA-3** is H100-specific. It exploits the **TMA** (Tensor Memory Accelerator) for asynchronous bulk SMEM loads and **WGMMA** (warp-group matrix-multiply) instructions, using producer–consumer **warp specialization**: some warps fetch data while others compute, overlapping HBM/SMEM transfer with tensor-core math. It adds **FP8** support with incoherent processing for accuracy. FA-3 reaches up to ~740 TFLOP/s in BF16 on H100 (vs FA-2's ~312-class numbers on A100), a major step. These hardware mechanics are detailed in File 10.

### 3.5 Paged variants

Standard FlashAttention assumes contiguous K/V. Inference engines need attention over a **paged, non-contiguous** KV cache (File 03). vLLM's original paged-attention kernel and the **FlashInfer** library (arXiv 2501.01005) extend the FlashAttention idea to gather K/V from physical blocks via a block table, support variable-length batches (`indptr` ragged layout), and special-case the decode (single-query) path. FlashInfer is now the default high-performance backend in both vLLM and SGLang on NVIDIA hardware.

---

## 4. Feed-Forward Networks and Mixture of Experts

After attention, each transformer block applies a position-wise feed-forward network (FFN). For inference, FFNs are pure GEMMs and are usually the largest share of FLOPs and weight memory.

### 4.1 Dense FFN and gated variants

The classic FFN expands then contracts the hidden dimension:

```
FFN(x) = W_2 · activation(W_1 · x)      d_ff ≈ 4 · d_model   (GELU/ReLU)
```

Modern models (LLaMA, Mistral, Qwen) use the **SwiGLU** gated variant (Shazeer 2020):

```
FFN(x) = W_2 · ( SiLU(W_1·x) ⊙ (W_3·x) )       SiLU(z) = z·σ(z)
```

SwiGLU uses **three** weight matrices (`W_1` gate, `W_3` up, `W_2` down) with a smaller `d_ff ≈ (8/3)·d_model` to keep parameter count comparable. The element-wise product `⊙` of the gate and up projections is the "gating." **GeGLU** (BLOOM, PaLM) is identical with GELU instead of SiLU. For inference, the gate/up matmuls and the SiLU⊙multiply are prime **kernel-fusion** targets — vLLM and SGLang fuse `SiLU(gate) ⊙ up` into a single kernel to avoid an HBM round-trip (File 10).

### 4.2 Mixture of Experts (MoE)

MoE (Shazeer et al. 2017; Switch Transformer, Fedus et al. 2022) replaces the single FFN with `E` independent expert FFNs and a **router** that sends each token to only `k` of them:

```
router_logits = x · W_router            ∈ ℝ^E
experts        = TopK(router_logits, k)  # the k highest-scoring experts
gate           = softmax(router_logits[experts])
y = Σ_{e ∈ experts}  gate_e · FFN_e(x)
```

Only `k/E` of the FFN parameters are *activated* per token, so MoE decouples *total* parameters (capacity) from *active* parameters (compute per token). Switch uses top-1; Mixtral-8×7B uses top-2 of 8; DeepSeek-V3 uses top-8 of 256 (plus shared experts); Qwen-MoE and GLaM vary.

**Why MoE is hard for inference, not training:**

- **Memory:** *All* `E` experts must reside in VRAM even though only `k` run per token. Mixtral-8×7B has ~47B parameters in memory but only ~13B active per token — you pay the full memory bill for the sparse compute benefit.
- **Load imbalance:** Routing is data-dependent and rarely uniform. Hot experts get oversubscribed (risking OOM or token-dropping at a capacity factor), cold experts waste GPU cycles. Training adds an auxiliary load-balancing loss; inference must cope with whatever routing the model produces.
- **Routing overhead:** Grouping tokens by expert, the gather/scatter, and (in distributed settings) the all-to-all communication add latency that dense models don't have.

vLLM implements a **`FusedMoE`** Triton kernel that groups tokens by expert, pads each group to a tensor-core-friendly multiple, and runs a stacked per-expert matmul, supporting GPTQ/AWQ/FP8 quantized expert weights (Files 06, 15).

### 4.3 Expert Parallelism (EP)

When experts don't fit on one GPU, **expert parallelism** distributes them: each GPU (or group) holds a subset of experts, and an **all-to-all** routes each token's hidden state to the GPU(s) hosting its selected experts, then a second all-to-all returns the results.

```
EP_degree × TP_degree = total GPUs for the MoE layers
DeepSeek-V3:  EP128 — 256 experts across 128 GPUs (2 experts/GPU), top-8 routing
```

EP is orthogonal to TP (TP splits dense layers and attention; EP splits experts). The all-to-all communication volume is `batch · k · d_model · bytes` per dispatch and again per combine; for large batches this dominates MoE-layer latency. DeepSeek's DualPipe overlaps this communication with computation. Full EP mechanics and the DeepSeek-V3 case study are in Files 05 and 15.

---

## 5. Positional Encoding and Long Context

Self-attention is permutation-invariant — without positional information, "dog bites man" and "man bites dog" are identical to the model. Positional encoding injects order. The choice profoundly affects how an inference engine handles context-window extension.

### 5.1 RoPE (Rotary Position Embedding)

RoPE (Su et al., arXiv 2104.09864) is the dominant scheme in modern models (LLaMA, Mistral, Qwen, DeepSeek, Gemma). It encodes **absolute** position as a **rotation** in 2D subspaces of the query/key vectors, such that the dot product of a query at position `m` and a key at position `n` depends only on the **relative** offset `m − n`.

The `d`-dimensional head is split into `d/2` pairs `(2i, 2i+1)`. Each pair is rotated by an angle `m·θ_i` where:

```
θ_i = base^(−2i/d)        base = 10000  (typically)
```

Low-index pairs rotate fast (high frequency, capture local position); high-index pairs rotate slowly (low frequency, capture long-range position). For position `m`, the rotation applied to pair `i` is:

```
[ x_{2i}  ]     [ cos(mθ_i)  −sin(mθ_i) ] [ x_{2i}  ]
[ x_{2i+1}]  =  [ sin(mθ_i)   cos(mθ_i) ] [ x_{2i+1}]
```

The key inference property: because rotation is applied to Q and K *before* the dot product, and rotations compose, `⟨R_m q, R_n k⟩ = g(q, k, m−n)` — relative position falls out naturally with **no learned positional parameters** and no extra KV cache. The inference engine applies RoPE inside or right after the QKV projection; vLLM and SGLang **fuse** the rotation into the QKV kernel to avoid a separate HBM pass (File 10).

### 5.2 RoPE context extension: NTK, YaRN, LongRoPE

A model trained with RoPE at context length `L_train` degrades when asked to extrapolate to `L > L_train`, because high positions produce rotation angles never seen in training. Several **training-free or light-finetune** extensions rescale the frequencies:

- **Position Interpolation (PI):** linearly compress positions `m → m·(L_train/L)` so the max angle stays in-distribution. Simple but blurs fine positional resolution.
- **NTK-aware scaling:** instead of scaling positions uniformly, scale the RoPE `base` so high-frequency (local) dimensions are barely touched while low-frequency (global) dimensions stretch. Preserves local resolution better than PI.
- **YaRN** (Peng et al., arXiv 2309.00071): a refined NTK scheme that applies different interpolation to different frequency bands ("NTK-by-parts") plus an attention-temperature correction. Achieves strong 8–32× extension with minimal fine-tuning; widely used.
- **LongRoPE:** searches for per-dimension rescaling factors (evolutionary search) to push to 2M+ tokens.
- **Dynamic NTK:** adjusts the scaling factor *at runtime* as the sequence grows, avoiding degradation on short sequences.

**Inference engineer responsibility:** the engine must apply the exact `rope_scaling` configuration the model was extended with. vLLM and SGLang expose a `rope_scaling` config (e.g. `{"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192}`), and `--max-model-len` must be set consistently with the scaling factor or outputs degrade silently. Mismatched RoPE scaling is one of the most common "the model got dumber after deployment" bugs.

### 5.3 ALiBi (Attention with Linear Biases)

Press et al. 2022 take a different route: no rotation, no learned embedding. ALiBi adds a **linear, head-specific bias** to the attention logits proportional to the query–key distance:

```
score_{ij} = q_i · k_j − slope_h · |i − j|
```

Each head `h` has a fixed geometric `slope_h`. Nearer tokens get a smaller penalty; distant tokens are linearly down-weighted. ALiBi generalizes to longer sequences than seen in training essentially for free (no frequency saturation problem). Used in BLOOM and MPT. For inference it's a cheap additive bias — no rotation kernel — but it is **not** RoPE-compatible, so engines treat ALiBi models on a separate code path. ALiBi has largely lost ground to RoPE+YaRN in new models, but remains relevant for serving the installed base.

---

## 6. KV Cache Data Structures and Memory Layout

How the KV cache is laid out in HBM determines coalesced-access efficiency, which directly affects the memory-bound decode latency.

### 6.1 Logical layout

A naive (contiguous, per-request) KV cache for one layer is a tensor:

```
[2, num_kv_heads, max_seq_len, head_dim]      # the leading 2 = K and V
```

Across the whole model it's `[num_layers, 2, num_kv_heads, max_seq_len, head_dim]`. Two design choices matter:

- **Interleaved vs separate K/V:** storing K and V in one tensor (interleaved per layer) vs two tensors. Separate tensors simplify the attention kernel's gather; interleaving can improve locality for fused write of K and V at generation time. Different backends choose differently.
- **Layout of the seq_len and head_dim axes:** for coalesced reads in the decode kernel, the `head_dim` (the contiguous inner dimension a warp reads together) should be the fastest-varying axis so that consecutive threads read consecutive addresses. Paged layouts (File 03) store `[num_blocks, 2, num_kv_heads, block_size, head_dim]` so each physical block is contiguous and head_dim-major.

### 6.2 Dtype

Typical KV dtype is **FP16 or BF16 (2 bytes)**. Quantized KV (§7) uses **FP8 (1 byte)** or **INT8 (1 byte)** with per-token or per-channel scales, and experimental **INT4** with group scales.

### 6.3 Worked sizes (recap and extension)

Using `per_token = 2 · num_layers · num_kv_heads · head_dim · bytes`:

- **LLaMA-3 8B** (32 layers, 8 KV heads, 128 head_dim, FP16): `2·32·8·128·2 = 131,072 B = 128 KB/token`. At 4096 context → 512 MB/request. On an 80 GB GPU after ~16 GB weights, ~64 GB KV → ~128 full-context requests (more in practice since few requests fill the window).
- **LLaMA-3 70B** (80 layers, 8 KV heads, 128 head_dim, FP16): `320 KB/token`. At 100K context → 32 GB/request.

### 6.4 Decode memory-access pattern

Every decode step reads, per layer, all cached K and V for that sequence:

```
bytes_read_per_step = 2 · num_kv_heads · head_dim · bytes · context_len · num_layers
```

For LLaMA-3 70B at 1,000 tokens context: `2·8·128·2·1000·80 ≈ 327 MB` → ~164 µs at 2 TB/s for KV reads alone, on top of weight reads. The attention kernel must read these blocks with maximal coalescing; the paged kernel's challenge is doing so when blocks are physically scattered (File 03 §PagedAttention kernel, File 10 §paged attention).

---

## 7. KV Cache Quantization

Because KV cache is the concurrency bottleneck, shrinking it with quantization buys directly more concurrent requests (and faster decode, since fewer bytes are read).

- **FP16 → INT8:** halves KV memory and KV read bandwidth. Quantize per-token (one scale per token vector) or per-channel (one scale per head-dim channel). Per-channel handles outlier channels better; per-token is cheaper to compute online during decode.
- **INT4 KV (4-bit):** with group quantization (a scale per small group of channels). Research-stage for general use because reconstruction error in K (which feeds softmax) can shift attention distributions.
- **KIVI** (arXiv 2402.02750): asymmetric scheme — **2-bit keys, 4-bit values**, with per-channel quantization for K and per-token for V, motivated by the observation that K has more salient outlier channels than V. Reports <0.5% perplexity degradation while reaching very low bit-widths.
- **FP8 KV cache:** the production sweet spot on H100. SGLang and vLLM both support `--kv-cache-dtype fp8_e5m2` (or `e4m3`). Per-tensor or per-layer scaling; <0.5% perplexity impact in practice; 2× memory and bandwidth reduction. FP8 avoids the integer↔float conversion overhead of INT8 on Hopper tensor cores.

Quality rule of thumb: INT8/FP8 KV is nearly free in quality and a clear win for memory-constrained serving; INT4 and below require validation on the target workload. KV quantization composes with weight quantization (§11) and is covered operationally in Files 06, 13, 15.

---

## 8. Sampling and Decoding Strategies

After the final layer produces logits `z ∈ ℝ^{vocab}` for the next token, a sampling strategy selects the actual token. Sampling runs on the GPU every decode step for every sequence in the batch, so its kernel efficiency matters; but more importantly, the *choice* of strategy interacts with batching, KV cache (parallel sampling shares prompt KV), and speculative decoding (acceptance is a sampling operation).

### 8.1 Greedy decoding

`token = argmax(z)`. Deterministic, fastest (a single reduction over the vocab), zero variance. Used for reproducible evaluation and as the *draft* sampler in many speculative-decoding setups (greedy maximizes agreement with the target).

### 8.2 Temperature sampling

Scale logits by temperature `T` before softmax:

```
p_i = softmax(z_i / T)
```

`T → 0` approaches greedy (the max dominates); `T = 1` is the model's native distribution; `T → ∞` approaches uniform. Numerically the kernel subtracts `max(z)` before `exp` for stability. Temperature is applied *before* top-k/top-p truncation in most implementations, though order conventions vary (vLLM applies penalties, then top-k, then top-p, then temperature, then sample — order matters for reproducibility).

### 8.3 Top-k and top-p (nucleus) sampling

- **Top-k:** keep the `k` highest logits, set the rest to `−∞`, renormalize, sample. Typical `k = 40–100`. Bounds the candidate set to a fixed size — easy to vectorize.
- **Top-p (nucleus, Holtzman et al. 2019):** keep the smallest set of tokens whose cumulative probability ≥ `p` (e.g. `p = 0.9–0.95`), then renormalize and sample. Adapts the candidate-set size to the distribution's shape (sharp distributions → few candidates; flat → many). Requires a sort or partial sort of the vocab, so it's slightly more expensive than top-k.

These compose with penalties:

- **presence_penalty / frequency_penalty:** subtract a constant (presence) or count-scaled value (frequency) from logits of already-generated tokens to reduce repetition (OpenAI-style).
- **repetition_penalty:** multiplicatively dampen logits of seen tokens (CTRL-style).

All of these are per-sequence parameters, which is why the sampler must support a *batch* of sequences each with different sampling params — vLLM's `SamplingParams` and SGLang's sampling info carry per-request settings, and the sampling kernel branches/masks accordingly.

### 8.4 Beam search

Maintain `b` partial sequences (beams). At each step, expand every beam by every vocabulary token, score the `b·V` candidates by cumulative log-probability, and keep the top `b`. Memory cost: `b` copies of the KV cache (after beams diverge from the shared prefix). Beam search is **rarely used in modern LLM serving**: it is `b×` more expensive, tends to produce bland/repetitive text for open-ended generation, and complicates batching. It survives in some constrained tasks (translation, structured extraction). vLLM models it via a `SequenceGroup` of `b` sequences sharing prompt KV through copy-on-write (File 03, File 04).

### 8.5 The sampling kernel

In practice the sampler is a fused GPU kernel: apply penalties (gather seen-token logits, subtract), apply temperature, compute softmax, apply top-k (partial sort/threshold), apply top-p (cumulative sum over sorted probs), draw a sample via the Gumbel-max trick or inverse-CDF with a per-sequence RNG seed. Vectorizing this across a batch of hundreds of sequences with heterogeneous params is non-trivial; it is a measurable fraction of decode-step time for small models where the vocab is large relative to the model (e.g. 128K-token vocabularies).

---

## 9. Speculative and Tree-Based Decoding (Foundations)

Speculative decoding attacks the fundamental decode inefficiency: each step reads all weights to produce *one* token. If we could verify *several* candidate tokens in one weight read, we'd amortize the memory-bound cost. File 12 is the full treatment; here are the foundations every inference engineer needs.

### 9.1 The core algorithm (Leviathan et al. 2023, arXiv 2211.17192)

A small, cheap **draft model** `q` autoregressively proposes `K` tokens. The large **target model** `p` then verifies all `K` proposed tokens in a *single* forward pass (the K positions are processed in parallel, like a mini-prefill). For each position, **rejection sampling** accepts the draft token `x` with probability `min(1, p(x)/q(x))`; on the first rejection, a corrected token is sampled from the adjusted distribution `(p − q)_+` and verification stops. Crucially, the accepted sequence is **distributed exactly as if sampled from the target model** — speculative decoding is *lossless* with respect to the target's output distribution.

### 9.2 The speedup model

Let `β` be the expected per-token acceptance rate (`β = Σ_x min(p(x), q(x))`, the overlap of the two distributions) and `c` the draft-to-target cost ratio per token. The expected number of tokens produced per target call is `≈ (1 − β^{K+1})/(1 − β)`, and the speedup over plain decoding is roughly:

```
speedup ≈  E[accepted tokens + 1]  /  (1 + K·c)
```

For `β = 0.7`, `K = 4`, `c = 0.1`: numerator ≈ 3.8, denominator ≈ 1.4 → **~2.7×**. Speedup rises with acceptance rate (so draft and target should agree often → use a same-family, same-tokenizer draft, often run greedy) and falls if the draft is too expensive (so the draft must be far smaller, ~10×).

### 9.3 Tree/feature variants

- **Token-tree speculation:** instead of a single linear draft of `K` tokens, propose a *tree* of candidates; the target verifies the whole tree in one pass using a **tree attention mask**. Higher expected acceptance for the same target compute.
- **Medusa** (Cai et al. 2024, arXiv 2401.10774): adds several lightweight prediction *heads* to the target model itself, each predicting a token at a future offset; verify with tree attention. No separate draft model (no second set of weights to hold) but lower acceptance than a good draft model.
- **EAGLE** (arXiv 2401.15077): a feature-level draft that consumes the target's *hidden states* (not just tokens), achieving higher acceptance; **EAGLE-2** builds the draft tree dynamically based on per-node acceptance probability. 3–4× on code is achievable.

vLLM (`vllm/spec_decode/`) and SGLang both implement these; the engine challenges are tree-attention kernels, overlapping draft and target on CUDA streams, and rejection sampling at batch scale (File 12).

---

## 10. Constrained / Structured Decoding (Foundations)

Production LLM use (function calling, JSON APIs, tool use, agents) demands outputs that conform to a schema. **Constrained decoding** guarantees this by masking, at every step, all tokens that would violate the grammar — set their logits to `−∞` before sampling.

### 10.1 The naive approach and its cost

Maintain a finite-state machine (FSM) representing the grammar. At each step, the current FSM state defines a set of *allowed* next tokens; mask all others. The problem is **scale**: vocabularies are 32K–256K tokens, so naively computing "which of the 128K tokens are valid in this state" every step is expensive — 5–15% of decode latency in early implementations.

### 10.2 FSM/PDA-based engines: Outlines and XGrammar

- **Outlines** compiles regexes/JSON-schemas to an FSM and precomputes, per FSM state, the *allowed-token bitmask* over the vocabulary, so runtime is a cheap masked lookup.
- **XGrammar** (Zheng et al., arXiv 2411.15100), SGLang's engine, compiles context-free grammars (JSON schema → EBNF → LL(1) tables → **push-down automaton**), precomputes per-state token masks, and caches masks for recently seen states. It applies the mask with a CUDA kernel across the batch and achieves **<1% decode overhead** — effectively free constrained decoding. It handles nested/recursive structures (true CFGs) that a plain FSM cannot. Full coverage in File 08.

The inference-engine integration is a per-sequence "grammar matcher" object advanced one token at a time, and a batched logit-masking kernel; multiple requests with *different* grammars can share a batch.

---

## 11. Quantization Fundamentals for Inference

Quantization reduces the bits per weight (and optionally per activation and per KV element), shrinking memory and — when the hardware supports low-precision matmul — increasing throughput. This section establishes the schemes; Files 06 and 10 cover their kernels.

### 11.1 Post-Training Quantization (PTQ) vs QAT

**PTQ** quantizes an already-trained model with at most a small calibration set — no retraining. This is what inference engines use (you serve checkpoints you didn't train). Quantization-Aware Training (QAT) bakes quantization into training for better low-bit accuracy but is a training concern, out of scope here.

### 11.2 Weight-only quantization (W4A16): GPTQ and AWQ

The model is stored with **4-bit weights** but **16-bit activations**; weights are dequantized to FP16 just before the matmul, which still runs in FP16. This cuts weight memory ~4× (the dominant memory term for large models) with no activation-quantization noise.

- **GPTQ** (Frantar et al., arXiv 2210.17323): layer-wise quantization using approximate second-order (Hessian) information — a practical descendant of Optimal Brain Quantization (OBQ). It greedily quantizes columns and updates the remaining weights to compensate for the error, minimizing the layer's output MSE. Achieves 4-bit with small perplexity loss. The **Marlin** kernel (Frantar et al. 2024) makes GPTQ W4A16 matmul nearly as fast as FP16 on A100/H100 by interleaving dequantization with the matmul.
- **AWQ** (Lin et al., arXiv 2306.00978): observes that a small fraction of weight channels (those multiplied by high-magnitude activations) are "salient." It scales those channels up before quantization (and compensates in the activation), protecting them from quantization error. Often beats GPTQ at the same bit-width and is simpler (no Hessian). vLLM supports both via `--quantization gptq` / `awq`.

### 11.3 Weight+activation quantization (W8A8)

Quantizing **both** weights and activations to INT8 enables **INT8 tensor-core matmul** (~2× the FP16 TFLOP/s on A100), benefiting compute-bound prefill, not just memory. The challenge is activation **outliers**: a few channels have huge magnitudes that, if quantized naively, destroy accuracy.

- **LLM.int8()** (Dettmers et al. 2022): mixed-precision decomposition — keep the outlier channels in FP16, quantize the rest to INT8. Correct but the FP16 path limits speedup.
- **SmoothQuant** (Xiao et al., arXiv 2211.10438): mathematically *migrate* the quantization difficulty from activations to weights by scaling: `(X·diag(s)^{-1})·(diag(s)·W)`. Choose `s` so activations and weights are both quantization-friendly. Enables full INT8 matmul. Requires calibration to choose `s`; vLLM exposes W8A8 paths.

### 11.4 FP8 quantization

H100/Ada/Blackwell tensor cores natively support **FP8** in two formats: **E4M3** (4 exponent, 3 mantissa bits — more precision, smaller range, used for weights/activations) and **E5M2** (5 exponent, 2 mantissa — more range, used for gradients/KV). H100 FP8 peak is ~1979 TFLOP/s vs ~989 BF16 → 2×. FP8 inference uses per-tensor or per-channel scales, static (calibrated) or dynamic (computed online). **DeepSeek-V3 trains and serves in FP8 end-to-end.** FP8 has lower conversion overhead than INT8 on Hopper (no integer↔float accumulation gymnastics) and is the default low-precision path on H100+ in vLLM (`--quantization fp8`) and SGLang. Typical result: LLaMA-3 70B FP8 ≈ 1.8× BF16 throughput on H100.

### 11.5 GGUF / llama.cpp k-quants

For completeness (and ecosystem awareness): llama.cpp's **GGUF** format defines quantization types `Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q8_0`, etc. The **k-quants** use a *superblock* structure — a block of weights shares a coarse scale, sub-blocks share finer scales — and an **importance matrix** to allocate bits where they matter. These target CPU/edge inference and are not used directly by vLLM/SGLang (GPU-focused), but vLLM *can* load GGUF files by dequantizing to FP16 for GPU execution, useful for community checkpoints only distributed as GGUF.

### 11.6 Choosing a scheme (inference-engineer heuristics)

- **Memory-bound, large model, decode-heavy:** W4A16 (AWQ/GPTQ) or FP8 weights — shrink the dominant weight read.
- **Compute-bound, prefill-heavy, H100+:** FP8 W8A8 — get the 2× tensor-core throughput.
- **KV-cache-bound (long context, high concurrency):** FP8 KV cache (§7) on top of weight quantization.
- **Always validate** perplexity / task accuracy on *your* workload; quantization error is data-dependent and a benchmark win can hide a tail-quality regression.

---

## 12. Synthesis: The Inference-Optimization Lens

Every component above maps to a lever on the roofline (File 01 §2):

| Component | Lever | Effect |
|---|---|---|
| GQA / MQA / MLA | ↓ KV bytes/token | more concurrency, faster decode |
| Sliding window | ↓ KV cache length | bounded memory for long streams |
| FlashAttention | ↓ HBM traffic | faster prefill & decode |
| SwiGLU/MoE fusion | ↓ HBM round-trips | faster FFN |
| RoPE/YaRN | enable long context | larger usable context window |
| FP8/INT8 weights | ↓ weight bytes, ↑ TFLOP/s | faster decode & prefill |
| KV quantization | ↓ KV bytes | more concurrency |
| Speculative decoding | amortize weight reads | more tokens per weight load |
| Constrained decoding | correctness | structured output at ~0 cost |

The systems files that follow are, in effect, the engineering required to realize these levers at production scale: PagedAttention (File 03) makes the KV cache levers actually usable; the scheduler (File 04) turns the batching insight into throughput; distributed inference (File 05) extends all of this across GPUs; kernels and hardware (File 10) make each operation hit the roofline. Master the architecture here, and the systems become legible.

---

## 13. Normalization Layers (RMSNorm, LayerNorm) for Inference

Normalization is easy to overlook because it has few parameters and few FLOPs, but it is a **bandwidth-bound** operation that runs twice per transformer block (pre-attention and pre-FFN, plus a final norm), and it is a prime fusion target. Getting it wrong — numerically or in placement — corrupts the whole forward pass.

### 13.1 LayerNorm vs RMSNorm

**LayerNorm** (Ba et al. 2016) normalizes each token vector to zero mean and unit variance, then applies a learned scale `γ` and shift `β`:

```
LayerNorm(x) = γ ⊙ (x − μ) / √(σ² + ε) + β
μ = mean(x),  σ² = var(x)   over the d_model dimension
```

**RMSNorm** (Zhang & Sennrich 2019) drops the mean-centering and the bias, normalizing only by the root-mean-square:

```
RMSNorm(x) = γ ⊙ x / √( (1/d)·Σ x_i² + ε )
```

RMSNorm is the modern default (LLaMA, Mistral, Qwen, Gemma, DeepSeek) because it is cheaper (no mean, no subtraction, no bias) and empirically matches LayerNorm quality. For an inference kernel, RMSNorm is a single reduction (sum of squares) over `d_model`, a reciprocal-sqrt, and a scaled multiply.

### 13.2 Why normalization is bandwidth-bound, and the fusion opportunity

RMSNorm reads `x` (`d_model` elements), does `O(d_model)` cheap arithmetic, and writes `d_model` elements. The arithmetic intensity is ~O(1) — it's memory-bound. A naive implementation issues a separate kernel that reads `x` from HBM and writes the normalized result back, only for the *next* kernel (the QKV or gate/up projection) to read it again. vLLM and SGLang **fuse** RMSNorm with the subsequent linear layer — and frequently fuse the **residual add** into the same kernel — so the normalized activations live in registers/SMEM and never round-trip through HBM. A Triton RMSNorm kernel (File 10 §RMSNorm Triton kernel) tiles along `d_model`, computes the partial sum-of-squares with a warp-level `tl.sum` reduction, and applies the scale, optionally writing back both the normalized value and the residual stream.

### 13.3 Numerical precision of normalization

The sum-of-squares reduction should be accumulated in **FP32** even when activations are BF16/FP16, because squaring small BF16 values and summing thousands of them loses precision catastrophically in low precision. Inference kernels universally up-cast the reduction to FP32 and down-cast only the final result. The `ε` (e.g. `1e-5` or `1e-6`) prevents division by zero for all-zero vectors and must match the value the model was trained with — a mismatched `ε` is a subtle source of quality drift.

### 13.4 Pre-norm vs post-norm and QK-norm

Modern decoders are **pre-norm**: normalization is applied to the *input* of each sublayer, and the sublayer output is added to the residual stream (`x = x + Attn(Norm(x))`). Pre-norm stabilizes training of deep stacks. A few recent models add **QK-normalization** (normalizing queries and keys before the dot product) to control attention-logit magnitude at long context; the inference engine must apply it inside the attention path. The placement of norms determines where in the fused-kernel pipeline they execute; getting the order wrong (post-norm vs pre-norm) silently produces garbage, so engines key the norm placement off the model architecture registry (File 06).

---

## 14. Residual Stream, Embeddings, and the LM Head

### 14.1 The residual stream

Every sublayer reads from and writes to a shared **residual stream** of width `d_model`: `x ← x + Sublayer(Norm(x))`. For inference this means the residual add is one of the most frequent elementwise ops in the model (`2 · num_layers` times). Like normalization it is bandwidth-bound and is fused into adjacent kernels (the down-projection of the FFN or the output projection of attention writes its result added to the residual in one pass). The residual stream is also why activation memory at long context is large: it is `batch · seq_len · d_model` per layer, which under tensor parallelism is *replicated* on every TP rank (motivating sequence parallelism, File 05 §SP).

### 14.2 Token embedding

The input token IDs index into an embedding table `E ∈ ℝ^{vocab × d_model}`. This is a gather, not a matmul — each token fetches one row. For a 128K-token vocabulary and `d_model = 8192` in BF16, the embedding table alone is `128000 · 8192 · 2 ≈ 2.1 GB`. The gather is cheap compute but the table consumes meaningful memory and, under tensor parallelism, is typically **vocab-parallel** (each rank holds a slice of the vocabulary).

### 14.3 The LM head and weight tying

The final hidden state is projected to vocabulary logits by the **LM head** `W_lm ∈ ℝ^{d_model × vocab}`. This is a large GEMM: for `d_model = 8192`, `vocab = 128000`, it is a `[*, 8192] × [8192, 128000]` matmul producing logits for every position. During **prefill** you only need logits for the *last* token (to start generation), so a key optimization is to compute the LM head **only for the final position** of each prefilling sequence rather than all positions — this saves a `seq_len×` factor of LM-head compute. During decode you need logits for the one new token per sequence. Many models **tie** the LM head to the embedding table (`W_lm = Eᵀ`), halving that parameter memory; the inference engine must honor `tie_word_embeddings` from the config or it will load duplicate weights or, worse, mismatched ones.

### 14.4 Vocabulary-parallel logits and the sampling reduction

Under tensor parallelism the LM head is split across ranks (each computes logits for a slice of the vocabulary). Sampling — especially argmax/top-k — then requires a cross-rank reduction to find the global max or top-k over the full vocabulary. vLLM and SGLang implement vocab-parallel logit gathering followed by an all-gather or a distributed top-k so the sampler sees the complete distribution. This is a small but latency-sensitive collective on the decode critical path.

---

## 15. Worked Examples: Putting the Cost Models Together

Concrete arithmetic builds the intuition that turns these formulas into engineering judgment.

### 15.1 Prefill FLOPs for LLaMA-3 70B on a 2,048-token prompt

A transformer's forward pass costs ≈ `2 · P` FLOPs per token (the factor of 2 for multiply+add over every parameter), ignoring attention's quadratic term for moderate lengths. For `P = 70×10⁹` and `2048` tokens:

```
prefill FLOPs ≈ 2 · P · seq_len = 2 · 70e9 · 2048 ≈ 2.87 × 10¹⁴ FLOPs = 287 TFLOP
On A100 (312 TFLOP/s peak):  287 / 312 ≈ 0.92 s   at 100% utilization
Realistically ~83% util →  ~1.1 s
```

Add the attention term: at `seq_len = 2048`, `d = 128`, 64 heads, 80 layers, the `O(n²d)` attention compute is `≈ 2 · 80 · 64 · 2048² · 128 ≈ 5.5 × 10¹³ ≈ 55 TFLOP` — a ~19% addition. So prefill is dominated by the dense matmuls and is firmly compute-bound, exactly as the prefill–decode duality predicts (File 01 §3).

### 15.2 Decode latency for the same model, batch 32

Decode reads all weights once per step (`2·P·bytes` for the weight matmuls, treating activations as negligible) regardless of batch (the weights are shared across the batch):

```
weight bytes (BF16) = 70e9 · 2 = 140 GB
At A100 2 TB/s:  140 GB / 2 TB/s = 70 ms   just to stream weights once
```

Add per-sequence KV reads: at batch 32 and ~1,000 tokens each, KV reads ≈ `32 · 327 MB ≈ 10.5 GB → ~5 ms`. So a decode step is ~75 ms, producing 32 tokens (one per sequence) → ~2.3 ms/token/sequence amortized, vs ~70 ms/token if you ran batch 1. **This 30× amortization is the entire economic case for continuous batching** (File 04): the weight read is a fixed cost paid once per step and split across the batch.

### 15.3 Break-even batch revisited with KV pressure

§15.2 shows decode throughput improves with batch — but each added sequence costs `~512 MB–32 GB` of KV cache (depending on context). The achievable batch is therefore set by **KV memory**, not by the ridge point, for large models at long context. This is the crux: PagedAttention (File 03) exists to push the achievable batch as close as possible to the memory limit by eliminating fragmentation, and quantized KV (§7) and GQA/MLA (§2) exist to lower the per-sequence KV cost so the batch can grow further before memory runs out.

### 15.4 MoE all-to-all cost (DeepSeek-V3-scale)

For an EP all-to-all with `batch = 1024` tokens, `top_k = 8` effective dispatch, `d_model = 7168`, BF16, over 400 Gbps (50 GB/s) InfiniBand per GPU:

```
dispatch volume per GPU ≈ batch · d_model · bytes (its share) 
                        ≈ 1024 · 7168 · 2 ≈ 14.7 MB
time ≈ 14.7 MB / 50 GB/s ≈ 294 µs  per all-to-all
two all-to-all per MoE layer → ~588 µs/layer of pure communication
```

This is why MoE serving at scale lives or dies on interconnect bandwidth and on **overlapping** communication with computation (DeepSeek's DualPipe; File 15). It also explains why MoE deployments cluster TP within a high-bandwidth NVLink node and reserve EP for the cross-node dimension.

---

## 16. Attention Numerical Stability and Precision Choices

Beyond the max-subtraction in softmax (§1.1, §3.2), inference engineers face several precision decisions in attention:

- **Score accumulation in FP32:** the `QKᵀ` dot products and the softmax denominator are accumulated in FP32 even for BF16 inputs; BF16 has only 8 mantissa bits and would lose accuracy summing over thousands of keys at long context. FlashAttention keeps the running `ℓ` and `m` in FP32.
- **Softmax in the presence of FP8:** FA-3's FP8 path uses "incoherent processing" (a random orthogonal rotation) to spread outliers across channels before quantizing, preventing a few large logits from dominating the FP8 dynamic range.
- **Causal mask and `−∞`:** implemented as a large negative finite value (e.g. the dtype's lowest representable, or `−1e9` in FP32-accumulate) rather than literal `−∞`, to avoid `NaN` from `0·∞` in edge cases.
- **RoPE precision:** the sine/cosine tables are precomputed in FP32 and the rotation applied in FP32 or BF16; low-precision sin/cos at large positions accumulates angular error that, over a 128K context, can rotate keys noticeably — another reason long-context RoPE is delicate.

These choices are invisible in the math but decisive in a kernel: a backend that accumulates softmax in BF16 will pass short-context tests and silently degrade at 32K+ tokens.

---

## 17. How Architecture Choices Propagate to the Serving Stack

A final synthesis tying §§1–16 to the systems files, because the whole point of this reference is to make the downstream engineering predictable:

- **num_kv_heads (GQA/MQA/MLA)** → sets per-token KV bytes → sets achievable batch (File 03 block math) and TP divisibility constraints (File 05).
- **Sliding window** → block manager must free aged blocks; changes the prefix-cache semantics (Files 03, 08).
- **RoPE + scaling** → `--max-model-len` and `rope_scaling` must be set consistently; affects long-context memory (File 13).
- **MoE** → FusedMoE kernel + expert parallelism + all-to-all interconnect planning (Files 06, 05, 15).
- **SwiGLU/RMSNorm/residual** → fusion opportunities that determine decode kernel efficiency (File 10).
- **Vocabulary size** → embedding/LM-head memory, vocab-parallel logits, sampling-reduction cost (Files 05, 06).
- **Attention variant + precision** → which attention backend (FlashAttention / FlashInfer / Triton) and which KV dtype are valid (Files 09, 10).

An inference engineer who can read a model's `config.json` and immediately predict its KV cache cost, its TP constraints, its fusion opportunities, and its long-context behavior has internalized this file. The remaining files show how vLLM and SGLang turn those predictions into running systems.

---

## 18. Tokenization and Detokenization in the Serving Loop

The transformer operates on integer token IDs, but users send and receive text. Tokenization (text → IDs) and detokenization (IDs → text) bracket every request and, while not GPU work, sit on the latency-critical path and have subtle correctness issues that bite inference engineers.

### 18.1 BPE, byte-level BPE, and SentencePiece

Modern LLMs use subword tokenization. **Byte-Pair Encoding (BPE)** starts from characters (or bytes) and greedily merges the most frequent adjacent pair, building a vocabulary of subwords; the learned *merge rules* are applied at inference to segment new text. **Byte-level BPE** (GPT-2 onward) operates on UTF-8 bytes so that *any* string is representable with no out-of-vocabulary failures — important for code, emoji, and multilingual text. **SentencePiece** (LLaMA, many others) is a library implementing BPE/unigram models that treats the input as a raw stream (including whitespace as a meta-symbol `▁`), making tokenization reversible and language-agnostic.

For the inference engine the practical consequences are: (1) tokenization is CPU work done in a tokenizer process/thread, ideally **overlapped** with GPU execution of other requests (SGLang runs it in the `TokenizerManager` process, File 09); (2) the vocabulary size (32K for LLaMA-2, 128K for LLaMA-3, up to 256K for some multilingual models) sets the embedding/LM-head memory and the sampling-reduction cost (§14); (3) special tokens (BOS, EOS, system/role markers, tool-call delimiters) must be inserted exactly as the chat template specifies.

### 18.2 Streaming detokenization is not just `decode(ids)`

The naive approach — detokenize the full ID list every step and diff — is both O(n²) over a generation and *incorrect* for subword and byte-level tokenizers. A single character (or a multi-byte UTF-8 sequence, like an emoji or a CJK character) can span multiple tokens; decoding tokens one at a time can emit replacement characters (`�`) or split a grapheme. Production engines maintain **incremental detokenization state**: they buffer the trailing tokens, decode the buffer, and only emit the *prefix* of the decoded text that is stable (cannot change when more tokens arrive). This is why vLLM and SGLang have a dedicated detokenizer with a "read offset" and "prefix offset" that advance only when bytes are unambiguously complete. Getting this wrong manifests as flickering or corrupted streamed output, especially for non-Latin scripts.

### 18.3 Stop conditions

Generation halts on: the EOS token, reaching `max_tokens`, or matching a user **stop string**. Stop strings are tricky because they are defined in *text* but generation happens in *tokens* — a stop string may not align to token boundaries, so the engine must detokenize incrementally and check the decoded suffix against each stop string after every step, then truncate the output precisely at the stop string (not at the token boundary). `stop_token_ids` is the cheaper token-level variant. These checks run on CPU per step per sequence and must be efficient at batch scale.

---

## 19. Position Handling Across Prefill, Decode, and Caching

Positions seem trivial until you have a paged, prefix-cached, chunked-prefill system, at which point correct **position ID** assignment becomes a frequent source of bugs.

- **Prefill:** positions `0, 1, …, n−1` for an `n`-token prompt. RoPE rotates each Q/K by its position angle (§5).
- **Decode:** the new token's position is `context_len` (the number of tokens already in the KV cache for that sequence), monotonically increasing each step.
- **Prefix caching:** when a request reuses a cached prefix of length `p` (Files 03, 08), the *new* tokens must be assigned positions starting at `p`, not 0 — the cached K/V were already rotated for positions `0..p−1`, so re-rotating or mis-positioning the suffix corrupts attention. The engine tracks a per-sequence `num_computed_tokens` to assign suffix positions correctly.
- **Chunked prefill:** when a long prompt is split into chunks (File 04), each chunk's tokens get positions continuing from the previous chunk's end. The KV from earlier chunks stays in cache; the attention for a later chunk attends over all earlier chunks plus itself.
- **Sliding window:** positions keep increasing even as old KV blocks are freed; RoPE uses the true absolute position, but attention only reaches back `w` tokens.

The `slot_mapping` and `positions` tensors that vLLM's `ModelRunner` builds each step (File 06) encode exactly this, mapping each token in the batch to its physical KV slot and its position angle. A mismatch between `positions` and the cached state is the classic "works for fresh requests, breaks for cache hits" bug.

---

## 20. Packed (Variable-Length) Attention and Batch Construction

A continuous-batching engine runs, in a single step, a batch of sequences with *different* lengths and a mix of prefill and decode work. Two layout strategies exist:

- **Padded batch:** pad every sequence to the batch's max length and mask the padding. Simple but wasteful — compute and memory scale with `batch · max_len` even if most sequences are short. Unacceptable for the skewed length distributions of real traffic (File 11).
- **Packed / ragged (varlen) layout:** concatenate all sequences end-to-end into one long buffer and pass a **cumulative-length index** (`cu_seqlens` / `indptr`) marking sequence boundaries. The attention kernel uses the index to confine each sequence's attention to its own span. FlashAttention's varlen API and FlashInfer's `indptr`-based wrappers (Files 09, 10) support this natively, eliminating padding waste.

Building this packed batch every step — gathering token IDs, positions, slot mappings, block tables, and the `cu_seqlens` index for a heterogeneous prefill+decode batch — is the job of the `ModelRunner` (vLLM, File 06) / `ModelWorkerBatch` builder (SGLang, File 09), and doing it with minimal CPU overhead and zero-copy transfer to the GPU is a real engineering focus (SGLang uses shared memory; File 09 §shared memory).

---

## 21. Mixture of Experts: Routing Mathematics in Depth

Returning to MoE (§4.2) with the detail an inference engineer needs to reason about latency and load.

### 21.1 The router and top-k selection

```
g = x · W_router                      # router logits, shape [E]
(values, idx) = TopK(g, k)            # the k selected experts and their logits
w = softmax(values)                   # normalized gates over the selected experts only
y = Σ_{j=1..k}  w_j · FFN_{idx_j}(x)  # weighted combination
```

Note the softmax is over **only the selected** `k` experts (not all `E`) in most modern designs (Mixtral, DeepSeek), which keeps the gate weights well-scaled. Some models add **shared experts** that every token uses unconditionally (DeepSeek-MoE) plus routed experts on top.

### 21.2 Capacity factor and token dropping

In training, each expert has a fixed **capacity** `C = capacity_factor · (tokens / E) · k`. If more tokens route to an expert than its capacity, the overflow tokens are **dropped** (their FFN output is zero, only the residual passes through). Inference can either (a) honor a capacity factor to bound per-expert batch size and memory, or (b) run **dropless** (process all routed tokens, variable per-expert batch). vLLM's `FusedMoE` is typically dropless at inference (you want every token computed for quality), which means per-expert batch sizes vary with the data — a load-balancing and kernel-padding challenge.

### 21.3 The grouped-GEMM problem

After routing, tokens are grouped by their selected expert, producing `E` variable-size groups. The compute is then a **grouped GEMM**: for each expert `e`, multiply its `n_e` assigned token vectors by that expert's weight matrices. The kernel must (1) gather tokens per expert (a scatter by `idx`), (2) pad each group to a tensor-core-friendly multiple (e.g. 16) so the GEMM is efficient, (3) run the per-expert matmuls (possibly as one batched/grouped kernel), and (4) scatter the weighted results back to token positions. vLLM stacks expert weights as `[E, d_ff, d_model]` and `[E, d_model, d_ff]` tensors and uses a Triton grouped-matmul (`fused_moe`) that iterates experts; FP8/GPTQ-quantized experts are supported by dequantizing per group inside the kernel. The padding to a multiple of 16 wastes some compute on imbalanced routing — another reason load balance matters at inference even though there's no balancing *loss* at serve time.

### 21.4 Latency model for an MoE layer

Single-GPU (all experts local), no EP:

```
MoE layer time ≈ router_matmul + gather/scatter + Σ_e (padded grouped GEMM) 
```

The grouped GEMM total FLOPs ≈ `2 · batch · k · (FFN params per expert)` — i.e. you compute `k` experts' worth of FFN per token, far less than dense (`E` experts' worth) but more than a single dense FFN. Distributed (EP), add the two all-to-all costs from §15.4. The decode-time MoE is still memory-bound on reading the *selected* experts' weights from HBM, and because the selection is data-dependent, the set of weight pages touched varies step to step — defeating some caching assumptions and making MoE decode latency noisier than dense decode.

---

## 22. Multi-Token Prediction (MTP) and Inference-Aware Architecture

DeepSeek-V3 introduced **Multi-Token Prediction**: auxiliary heads trained to predict several future tokens, not just the next one. At inference these MTP heads double as a built-in **draft model** for speculative decoding (File 12) — the model proposes multiple tokens from its own MTP heads and verifies them in one pass, with no separate draft network to host. This is a clean example of **inference-aware architecture**: the model was designed with serving cost as a first-class constraint, like GQA (reduce KV cache), MLA (reduce KV cache further), and quantization-friendly weight distributions. The trend — covered in File 20 — is that inference engineers increasingly influence architecture: attention-head counts chosen for TP divisibility, KV schemes chosen for cache size, draft heads built in for speculation, and weight statistics shaped for low-bit quantization. Reading a frontier model's design is, increasingly, reading a set of decisions about how it will be *served*.

---

## 23. Reference: Quick Cost Formulas

A consolidated cheat sheet for back-of-envelope inference reasoning:

```
Weight memory          ≈ P · bytes_per_param
KV cache per token     = 2 · num_layers · num_kv_heads · head_dim · bytes
Prefill FLOPs          ≈ 2 · P · seq_len  (+ O(layers·heads·seq²·head_dim) attention)
Decode weight bytes/step ≈ P · bytes        (shared across the batch)
Decode KV bytes/step   ≈ batch · (KV per token) · context_len / num_layers · num_layers
Arithmetic intensity (decode) ≈ batch · 2 / bytes_per_param  ≈ batch (BF16)
Ridge point (A100 BF16) ≈ 156 FLOP/byte ; (H100 BF16) ≈ 295 ; (H100 FP8) ≈ 590
MoE all-to-all per layer ≈ 2 · (batch · d_model · bytes / interconnect_BW)
```

With these and a model's `config.json`, you can predict memory budget, achievable batch, prefill/decode latency, and the dominant bottleneck before writing a line of deployment config — which is the entire purpose of treating the transformer as a *cost model*, not just an architecture.

---

## 24. Activation Functions in Detail

The activation in the FFN is a small but non-trivial part of the inference kernel, and its exact form must match training.

- **ReLU** `max(0, x)`: cheap, but dead-neuron issues; rare in modern LLMs.
- **GELU** `x · Φ(x)` where `Φ` is the standard normal CDF: smooth, used in GPT-2/3, BERT, PaLM (GeGLU). Often approximated as `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` — the **tanh approximation** — which is what most kernels implement; using the exact `erf`-based GELU when the model trained with the tanh approximation (or vice versa) causes a small but real distribution shift.
- **SiLU / Swish** `x · σ(x)` where `σ` is the sigmoid: used in LLaMA/Mistral SwiGLU. Smooth, non-monotonic, self-gating.
- **Gated variants (SwiGLU, GeGLU):** as in §4.1, the FFN computes a gate projection and an up projection, applies the activation to the gate, multiplies elementwise, and down-projects. The fusion target is `act(gate) ⊙ up` (File 10): one kernel reads both projections (or computes them), applies the activation, multiplies, and writes one output, avoiding two extra HBM round-trips.

Numerically, `σ` and `tanh` are computed with fast device intrinsics; the precision is adequate in BF16 for activations (unlike the normalization reduction, which needs FP32). The activation is bandwidth-bound and essentially free when fused, but a measurable cost if run as a standalone kernel — which is why no serious engine runs it standalone.

---

## 25. A Complete Decode-Step Walkthrough

To cement how the components compose, trace a *single decode step* for one sequence in a batch, on a GQA model like LLaMA-3 70B (pre-norm, RoPE, SwiGLU, 80 layers). Assume the sequence already has `c` tokens cached.

1. **Embedding lookup.** The new token ID gathers one row from `E` → hidden vector `h ∈ ℝ^{d_model}` (8192). Cheap gather from HBM.
2. **For each of 80 layers:**
   a. **RMSNorm** of `h` (FP32 reduction over 8192 dims) → `h_norm`. Fused with the next matmul.
   b. **QKV projection.** `h_norm` × the (column-parallel under TP) QKV weights → `q` (for all query heads), `k`, `v` (for the 8 KV heads). **RoPE** is fused in: `q` and `k` are rotated by the angle for position `c`.
   c. **Write K/V to the cache.** The new `k`, `v` are written into the sequence's KV cache at the physical slot for logical position `c` (via `slot_mapping`; paged, File 03).
   d. **Attention.** The single query `q` attends over all `c+1` cached keys/values for this sequence (paged, varlen kernel; FlashInfer decode wrapper, File 10) → attention output `o`. Memory-bound: reads `c+1` K and V vectors per layer.
   e. **Output projection** (row-parallel under TP) of `o`, **AllReduce** across TP ranks, **residual add** into `h` (fused). Now `h ← h + Attn(Norm(h))`.
   f. **RMSNorm**, **gate/up projection** (column-parallel), **SiLU⊙multiply** (fused), **down projection** (row-parallel), **AllReduce**, **residual add**. Now `h ← h + FFN(Norm(h))`.
3. **Final RMSNorm** of `h`.
4. **LM head** projection (vocab-parallel) → logits over the 128K vocabulary; **gather/all-gather** across TP ranks for the full distribution.
5. **Sampling.** Apply penalties, temperature, top-k/top-p; draw the next token (per-sequence RNG). 
6. **Detokenize incrementally**, check stop conditions, stream the token to the client.

Now multiply: this happens for *every sequence in the batch in parallel* (the matmuls are batched, the attention is varlen-batched), and the weight reads in steps 2b/2e/2f are **shared across the entire batch** — the amortization that makes batched decode efficient. The two AllReduces per layer (160 total for 80 layers) under TP are the communication on the decode critical path (File 05). The KV reads in 2d are the per-sequence memory cost that grows with context (File 13). Every optimization in this database targets one of these labeled steps.

---

## 26. KV Cache Comparison Across Real Models

A worked comparison makes the architectural choices concrete. KV cache per token = `2 · L · num_kv_heads · head_dim · bytes` (BF16, `bytes = 2`):

| Model | L | kv_heads | head_dim | KV/token | KV @ 8K ctx | Note |
|---|---|---|---|---|---|---|
| GPT-3 175B (MHA) | 96 | 96 | 128 | ~4.7 MB | ~38 GB | MHA → huge cache |
| LLaMA-2 7B (MHA) | 32 | 32 | 128 | 512 KB | 4 GB | MHA |
| LLaMA-2 70B (GQA-8) | 80 | 8 | 128 | 320 KB | 2.5 GB | GQA 8× saving |
| LLaMA-3 8B (GQA-8) | 32 | 8 | 128 | 128 KB | 1 GB | |
| LLaMA-3 70B (GQA-8) | 80 | 8 | 128 | 320 KB | 2.5 GB | |
| Mistral 7B (GQA-8, SWA) | 32 | 8 | 128 | 128 KB | capped @ w | window bounds it |
| Mixtral 8×7B (GQA-8) | 32 | 8 | 128 | 128 KB | 1 GB | MoE weights huge, KV modest |
| DeepSeek-V2 (MLA) | 60 | latent 512 | — | ~70 KB-equiv | ~0.55 GB | MLA ~64× vs MHA |

Two lessons jump out. First, **GQA's 8× reduction is the difference between a 70B model being servable at long context or not** — compare LLaMA-2 70B's 2.5 GB @ 8K (GQA) to what MHA would cost (20 GB @ 8K). Second, **MLA pushes further still**, which is why DeepSeek can offer very long context economically. For an MoE model like Mixtral, note the *weights* dominate memory (~94 GB at BF16) while the KV cache is modest — the opposite balance from a dense long-context deployment, and a reminder that "what dominates memory" depends entirely on the architecture and the workload.

### 26.1 Implication for achievable concurrency

On an 80 GB H100 serving LLaMA-3 70B (GQA), after ~140 GB of BF16 weights — wait, 70B BF16 is 140 GB, which exceeds one H100, so this model *requires* TP≥2 or quantization. Under TP=4 (4× H100), weights are ~35 GB/GPU, leaving ~40 GB/GPU for KV cache → at 320 KB/token, ~131K tokens of KV per GPU, i.e. roughly 32 requests at 4K context or 1 request at ~128K. Under FP8 weights (70 GB total, ~17.5 GB/GPU at TP=4), KV budget rises to ~57 GB/GPU → ~178K tokens. This is the kind of capacity arithmetic — combining weight memory, quantization, TP degree, and KV-per-token — that File 11 turns into a tuning methodology and File 05 into a topology decision.

---

## 27. Closing: The Architecture as Contract

Everything in this file is, from the serving engine's perspective, a **contract** declared by the model's configuration: the attention variant fixes the KV cost and TP constraints; the positional scheme fixes the context behavior and the RoPE kernel; the FFN/MoE structure fixes the FLOP profile and the fusion plan; the normalization and activation fix the kernel pipeline; the vocabulary fixes the embedding/head memory and sampling cost; the precision fixes which kernels are legal. vLLM and SGLang read this contract from `config.json` and the model registry (File 06, File 09) and instantiate the corresponding kernels, memory layout, and parallelism plan. An inference engineer who has internalized the contract can look at any new model and immediately answer the three questions that matter operationally: *How much memory does it need? How fast will it decode? Where is the bottleneck?* The rest of this database is the machinery that answers those questions in code.

---

## 28. Appendix A: How GQA Is Actually Implemented in the Kernel

Section 2.3 gave the math of grouped-query attention; the kernel realization has two strategies with different efficiency.

**Strategy 1 — KV-head replication.** The simplest correct implementation replicates each KV head `h/G` times so the attention sees a "virtual MHA" with `h` KV heads. This lets you reuse an unmodified MHA kernel, but it *materializes* the replicated K/V in registers/SMEM (not in the cache — the cache still stores only `G` heads) and wastes the bandwidth saving on the compute side: each query group re-reads the same KV head it shares. It is used as a fallback.

**Strategy 2 — Grouped attention kernel.** The efficient implementation, used by FlashInfer's GQA decode wrapper and FA-2/3's GQA mode, keeps `G` KV heads and has each *group* of `h/G` query heads attend to its single shared KV head, reading that KV head from HBM **once** and reusing it across the group's queries in SMEM. This preserves both the memory saving (cache stores `G` heads) *and* the bandwidth saving (each KV head read once per step, not `h/G` times). For LLaMA-3 70B with 64 query heads and 8 KV heads, each KV head serves 8 query heads; the grouped kernel reads `8` KV heads' worth of cache per layer per step instead of `64`, which is the entire point of GQA and only realized if the kernel is GQA-aware. This is why "which attention backend" (File 10) matters for GQA models specifically: a non-GQA-aware backend silently throws away the bandwidth win even though the cache is small.

## 29. Appendix B: Attention Sinks and the Softmax Denominator

A subtle empirical phenomenon with real engineering consequences: in trained transformers, the **first few tokens** of a sequence receive a disproportionate share of attention from later tokens, regardless of their semantic content. This "attention sink" arises because softmax forces the attention weights to sum to 1 — the model needs somewhere to "dump" attention mass when no past token is strongly relevant, and it learns to park that mass on the first tokens (often the BOS token). 

The engineering payoff (Xiao et al., StreamingLLM, arXiv 2309.17453; File 13): if you want to stream generation over an unbounded sequence with a *bounded* KV cache, you cannot simply keep a sliding window of the most recent `W` tokens — dropping the attention-sink tokens destroys quality, because the model's attention distribution collapses without its accustomed sink. The fix is to **retain the first few tokens (the sinks) plus a recent window**: KV cache = `{sink tokens} ∪ {recent W tokens}`. This enables fixed-memory streaming at arbitrary length with modest quality loss, and is exposed in vLLM via sliding-window configuration with sink retention. Related: some architectures add a learned "attention sink" / off-by-one softmax (a `+1` in the denominator) so the model has an explicit place to dump mass without corrupting a real token's representation.

## 30. Appendix C: Prefill Attention vs Decode Attention in the Kernel

The two phases use *different kernel code paths* even for the same model, because their shapes are opposite:

- **Prefill attention** processes `n` queries against `n` keys (causal), a `[n×d]×[d×n]` GEMM producing an `[n×n]` (lower-triangular) score region. It is compute-bound and uses the FlashAttention *prefill* kernel: tile both Q and K/V, exploit the causal structure to skip the upper triangle, accumulate with online softmax. With prefix caching, the `n` new queries may also attend to `p` cached keys (the prefix), so the kernel does a `[n × (p+n)]` causal-with-prefix attention. With chunked prefill, `n` is a chunk and the keys include all earlier chunks.
- **Decode attention** processes `1` query against `c+1` keys, a GEMV against the cache. It is memory-bound and uses the FlashAttention/FlashInfer *decode* kernel: no Q tiling needed (one query), parallelize over the KV length, online-softmax-reduce across KV blocks. The paged variant gathers KV from non-contiguous physical blocks via the block table (File 03).

A serving step that mixes prefill and decode sequences (continuous batching, File 04) either runs two kernels (one per phase) or a unified varlen kernel that handles both via the `cu_seqlens`/`indptr` index, with query-length 1 for decode rows and query-length `n` for prefill rows. FlashInfer's unified wrappers do exactly this, which is why they are the default for mixed batches in both engines.

## 31. Appendix D: Cross-Attention and Encoder-Decoder Notes

Decoder-only models (the focus of this database) use only self-attention. Encoder-decoder models (T5, Whisper, some multimodal stacks; File 14) additionally use **cross-attention**, where decoder queries attend to *encoder* keys/values. For inference this means an additional, *static* KV cache: the encoder output is computed once and its K/V are cached for the entire decode, never growing. This is actually friendlier than self-attention's growing cache — the cross-attention KV is fixed-size — but it adds a second attention per layer and a separate cache to manage. Whisper-style audio models and Idefics-style cross-attention VLMs exercise this path; vLLM and SGLang support it for the relevant model families, treating the encoder KV as a read-only cache distinct from the autoregressive self-attention cache. The vast majority of production LLM serving, however, is decoder-only self-attention, which is why the rest of this database concentrates there.

## 32. Appendix E: Foundational Papers Referenced

For the inference engineer who wants primary sources, the architecture lineage in this file rests on:

- *Attention Is All You Need* — Vaswani et al. 2017 (the Transformer, MHA).
- *Fast Transformer Decoding: One Write-Head Is All You Need* — Shazeer 2019 (MQA).
- *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — Ainslie et al. 2023, arXiv 2305.13245.
- *DeepSeek-V2 / DeepSeek-V3 Technical Reports* — MLA, MTP, FP8, EP128.
- *RoFormer: Enhanced Transformer with Rotary Position Embedding* — Su et al., arXiv 2104.09864 (RoPE).
- *YaRN: Efficient Context Window Extension of Large Language Models* — Peng et al., arXiv 2309.00071.
- *Train Short, Test Long: Attention with Linear Biases* — Press et al. 2022 (ALiBi).
- *FlashAttention* (arXiv 2205.14135), *FlashAttention-2* (arXiv 2307.08691), *FlashAttention-3* (2024) — Dao et al.
- *GLU Variants Improve Transformer* — Shazeer 2020 (SwiGLU/GeGLU).
- *Switch Transformers* — Fedus et al. 2022; *Mixtral of Experts* — Jiang et al. 2024 (MoE).
- *Root Mean Square Layer Normalization* — Zhang & Sennrich 2019 (RMSNorm).
- *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., arXiv 2211.17192.
- *Medusa* (arXiv 2401.10774), *EAGLE* (arXiv 2401.15077) — speculative decoding heads/features.
- *GPTQ* (arXiv 2210.17323), *AWQ* (arXiv 2306.00978), *SmoothQuant* (arXiv 2211.10438), *LLM.int8()* (Dettmers et al. 2022) — quantization.
- *KIVI* (arXiv 2402.02750) — 2-bit/4-bit KV cache.
- *XGrammar* (arXiv 2411.15100) — constrained decoding.
- *Efficient Streaming Language Models with Attention Sinks* — Xiao et al., arXiv 2309.17453.

## 33. Appendix F: Inference-Engineer FAQ

**Q: Why is my decode slow even though GPU utilization shows ~100%?** Because decode is memory-bound: `nvidia-smi` "GPU-Util" reports the fraction of time a kernel is running, not whether the tensor cores are busy. Your kernels are running (reading HBM) but the tensor cores are idle. Increase batch size (File 04) or reduce bytes moved (quantization, GQA/MLA) to actually speed it up.

**Q: I enabled a 4-bit weight quant and prefill got slower. Why?** W4A16 dequantizes weights to FP16 before an FP16 matmul; the matmul FLOPs are unchanged but you add dequant overhead. W4A16 helps *memory-bound decode* (less weight bytes to read) and total memory footprint, not *compute-bound prefill*. For faster prefill, use FP8 W8A8 on H100 to get 2× tensor-core throughput.

**Q: My model outputs degraded after I raised `--max-model-len`.** Almost certainly a RoPE-scaling mismatch (§5.2). The base model was trained/extended for a specific context length with a specific `rope_scaling`; serving beyond that without the matching scaling factor extrapolates RoPE into unseen angles. Set `rope_scaling` to match the checkpoint.

**Q: Two requests share a long system prompt — am I recomputing it each time?** Without prefix caching, yes. Enable automatic prefix caching (vLLM) or rely on RadixAttention (SGLang) to reuse the system-prompt KV across requests (Files 03, 08). Position IDs for the non-cached suffix must continue from the cached prefix length (§19).

**Q: How many concurrent requests can I serve?** Compute weight memory, subtract from (GPU memory × utilization), divide the remainder by (KV-per-token × expected context length). That gives the KV-bound concurrency, which is almost always the binding constraint for large models (§26.1, File 11).

**Q: Why does the LM head sometimes dominate small-model latency?** For a 1–3B model with a 128K–256K vocabulary, the LM-head GEMM (`d_model × vocab`) and the sampling reduction over the vocabulary are large *relative* to the tiny transformer body. The fix is to compute the LM head only for the last prefill position (§14.3), use vocab-parallelism under TP, and a fused top-k/top-p sampling kernel. On small models the Python/launch overhead also matters more, which is why CUDA graphs (File 09) and `torch.compile` help most there.

**Q: My MoE model's decode latency is noisy step-to-step. Bug?** Probably not. MoE routing is data-dependent, so each step touches a different set of expert weight pages and dispatches different per-expert batch sizes; padding to tensor-core multiples and variable cache locality make latency jitter inherent (§21.4). Load imbalance also shifts which expert is the straggler. Smooth it with larger batches (more uniform routing) and expert-parallel placement that balances hot experts.

These questions all reduce to the cost models in §§15 and 23 — which is exactly why those formulas, not memorized config flags, are the durable knowledge an inference engineer carries from one model and one engine to the next. Memorize the cost models in §§15, 23, and 26; the rest is detail you can always look up — and that the remaining files of this database supply in full.







