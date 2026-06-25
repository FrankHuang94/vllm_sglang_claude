# Attention Kernels, CUDA Optimization, and Hardware Backends

> **PRIMARY reference file.** This file descends to the GPU itself: the FlashAttention family (FA-1/2/3), paged-attention kernels (vLLM native and FlashInfer), the H100 (Hopper) and MI300X (CDNA3) architectures for inference, Triton kernel development, multi-GPU attention with tensor parallelism, and CUDA profiling for inference engineers. It is the kernel layer beneath both engines' architectures (Files 03–09). Prerequisites: File 02 (attention math, §3), File 03 (paged KV), File 05 (TP/collectives).

---

## Table of Contents

1. The Attention IO Problem
2. FlashAttention: Tiling and Online Softmax
3. FlashAttention-2
4. FlashAttention-3 (Hopper)
5. Paged Attention Kernels (vLLM native)
6. FlashInfer
7. H100 / Hopper Architecture for Inference
8. AMD ROCm and MI300X
9. Triton Kernel Development
10. Fused Kernels
11. Multi-GPU Attention with Tensor Parallelism
12. CUDA Profiling for Inference

---

## 1. The Attention IO Problem

Attention's naive implementation is bottlenecked not by compute but by **HBM traffic** (File 02 §3.1). For sequence length `n` and head dimension `d`, computing `softmax(QKᵀ)V` naively materializes the `n×n` score matrix `S` and probability matrix `P` in HBM:

```
HBM traffic ≈ Θ(n² + n·d)   for the n×n intermediates
```

At `n = 8192`, `d = 128`, FP16, the `2n²` term (~128 MB) dwarfs the `2n·d` term (~2 MB). So naive attention spends its time writing the giant `S`/`P` matrices to HBM and reading them back — it's memory-bound on HBM, and the tensor cores idle. The FLOPs are `O(n²·d)` but the *bottleneck* is the `O(n²)` memory traffic. This is the problem FlashAttention solves, and understanding it is prerequisite to understanding every modern attention kernel: the goal is to compute the exact same result *without* materializing the `n×n` intermediates in HBM.

---

## 2. FlashAttention: Tiling and Online Softmax

FlashAttention (Dao et al. 2022, arXiv 2205.14135) computes exact attention while keeping the `n×n` intermediates out of HBM, by **tiling** the computation to fit in on-chip SRAM and using **online softmax** to combine tiles incrementally.

### 2.1 Tiling

SRAM (shared memory) is small (~192 KB/SM on A100, ~228 KB on H100) but fast (~10–20 TB/s, an order of magnitude faster than HBM). FlashAttention tiles `Q` into row blocks and `K`/`V` into column blocks sized to fit in SRAM. For each Q tile, it streams through the K/V tiles, computing partial attention in SRAM, and accumulates the output — never writing `S` or `P` to HBM. Only Q, K, V (read once) and the output O (written once) touch HBM, plus small softmax statistics. The `n×n` matrix exists only transiently in SRAM, tile by tile.

### 2.2 Online softmax

The challenge in tiling is the softmax, which normally needs the *full* row of scores (to compute the max and the sum of exponentials) before any output can be produced. FlashAttention uses **online softmax** (the Milakov–Gimelshein running-statistics trick, File 02 §3.2): maintain a running max `m`, running denominator `ℓ`, and running output accumulator `O`. For each new K/V tile producing local scores `S_tile`:

```
m_new = max(m_old, rowmax(S_tile))
correction = exp(m_old − m_new)
ℓ_new = correction · ℓ_old + rowsum(exp(S_tile − m_new))
O_new = correction · O_old + exp(S_tile − m_new) · V_tile
```

When a new tile reveals a larger max, the `correction` factor rescales the previously-accumulated output and denominator, keeping everything numerically stable in a single streaming pass. At the end, `O = O / ℓ`. This is the algorithmic key — it lets attention be computed tile-by-tile (so tiles fit in SRAM) while producing the exact softmax-normalized result, with no full-row materialization.

### 2.3 The IO complexity result

By keeping tiles in SRAM of size `M`, HBM traffic drops from `Θ(n²)` to:

```
FlashAttention HBM traffic ≈ Θ(n²·d / M)
```

For `M ≈ 192 KB`, `d = 128`, this is roughly `n·d` (linear in `n`) rather than `n²` for `n ≫ √M` — eliminating one factor of `n`. The practical result: **2–4× speedup** over naive attention and, crucially, the removal of the `O(n²)` memory blowup, which is what makes long-context prefill (tens of thousands of tokens) feasible at all (File 13). FlashAttention is exact (not an approximation) — it computes bit-equivalent results to naive attention, just with a better IO schedule.

---

## 3. FlashAttention-2

FlashAttention-2 (Dao 2023, arXiv 2307.08691) refines FA-1's work partitioning for better GPU utilization:

- **Parallelize over the query dimension:** FA-1 parallelized over K/V blocks (with the Q in the inner loop); FA-2 parallelizes over Q blocks across thread blocks/warps, which maps better to the GPU's parallelism and reduces synchronization. Each thread block handles a Q tile independently.
- **Reduce non-matmul FLOPs:** the online-softmax rescaling (the `correction` multiplies) are non-matmul operations that don't use the tensor cores. FA-2 reorganizes to minimize these — e.g. deferring the final `/ℓ` normalization to the end rather than rescaling every tile — so more of the work is tensor-core matmul.
- **Better warp-level work distribution:** reduces shared-memory bank conflicts and improves occupancy.

The result: FA-2 reaches **~70–75% of A100 peak FLOP/s** for attention (up from FA-1's ~30–50%), roughly 2× FA-1. It became the standard attention kernel for prefill on A100-class hardware and the basis for paged variants.

---

## 4. FlashAttention-3 (Hopper)

FlashAttention-3 (2024) is specialized for the **H100 (Hopper)** architecture, exploiting hardware features absent on A100:

- **TMA (Tensor Memory Accelerator):** Hopper's hardware engine for asynchronous bulk transfers between HBM and SMEM. FA-3 uses TMA to load tiles asynchronously, freeing the SMs to compute while data is fetched — overlapping memory and compute at the hardware level.
- **WGMMA (Warp Group Matrix Multiply-Accumulate):** Hopper's larger matrix-multiply instruction operating on warp *groups* (4 warps), with bigger tile shapes (e.g. 64×256×16). FA-3 issues WGMMA for the QKᵀ and PV matmuls, using the tensor cores more efficiently.
- **Warp specialization (producer-consumer):** FA-3 splits warps into *producers* (which use TMA to fetch the next tiles into SMEM) and *consumers* (which compute on the current tiles with WGMMA). This producer-consumer pattern overlaps the memory fetch with the matmul — while consumers compute on tile `i`, producers fetch tile `i+1` — hiding HBM latency behind compute (the overlap principle, File 05 §34, at the warp level).
- **FP8 support:** FA-3 adds an FP8 path (with incoherent processing — a random rotation spreading outliers, File 02 §16 — for accuracy) for ~2× throughput over FP16.

The result: FA-3 reaches **up to ~740 TFLOP/s in BF16 on H100** (and higher with FP8), a major step over FA-2's A100 numbers. The asynchronous, warp-specialized design is what extracts Hopper's full attention throughput, and it's why H100 attention is so much faster than A100 — not just more FLOPs, but a kernel designed to keep the tensor cores fed via TMA-driven async pipelining.

---

## 5. Paged Attention Kernels (vLLM Native)

Standard FlashAttention assumes *contiguous* K/V. Inference engines need attention over a *paged, non-contiguous* KV cache (File 03). vLLM's original paged-attention kernel (File 03 §4, §16) gathers K/V from physical blocks via the block table.

- **Inputs:** the query, the paged `key_cache`/`value_cache` (vectorized `x`-layout for coalesced 128-bit loads, File 03 §16.1), the block tables (logical→physical), and context lengths.
- **Decode kernel:** each thread block handles a `(sequence, head)` pair, iterates the sequence's logical blocks, resolves each to a physical block via the block table, loads K/V, computes partial QKᵀ and online-softmax-accumulates (§2.2) — never materializing the full score row.
- **Two-phase (split-K) reduction:** for long contexts at small batch, partition the KV across thread blocks (phase 1: partial outputs + softmax stats per partition; phase 2: log-sum-exp combine), exposing more parallelism to fill the GPU (File 03 §16.3).
- **Weakness:** the scattered physical-block access is less coalesced than a contiguous read, hurting performance for many short sequences (File 03 §16.5). This motivated FlashInfer (§6), which recovers coalescing and adds GQA-aware grouping.

The paged kernel is the marriage of FlashAttention's online-softmax tiling with PagedAttention's block-table indirection — exact attention over a non-contiguous cache. It's correct and was foundational, but FlashInfer now usually supersedes it on performance (File 03 §16.4).

---

## 6. FlashInfer

FlashInfer (Ye et al., arXiv 2501.01005) is a library of optimized attention kernels purpose-built for LLM serving, now the default high-performance backend for both vLLM and SGLang on NVIDIA (File 03 §16.4, File 09 §7).

### 6.1 What it provides

- **Paged-KV attention** with the block-table/page indirection, but with layouts and access patterns tuned to recover the coalescing the vLLM native kernel lost (§5).
- **Variable-length (ragged) batch support** via `indptr` cumulative-length arrays — handles mixed prefill (multi-token queries) + decode (single-token queries) batches without padding (File 02 §20).
- **GQA-optimized grouping:** reads each shared KV head once per query-head group (File 02 §28), preserving GQA's bandwidth saving — a key advantage over GQA-unaware kernels.
- **Specialized wrappers:** `BatchPrefillWithPagedKVCacheWrapper` (prefill, compute-bound) and `BatchDecodeWithPagedKVCacheWrapper` (decode, memory-bound), each optimized for its regime.
- **Plan/run split** (File 09 §18): a planning phase computes the work partitioning (CUDA-graph-compatible), and a run phase executes it.

### 6.2 Why it's faster

FlashInfer combines FlashAttention's tiling/online-softmax with paged-KV gather, better block layouts for coalescing, warp specialization, autotuned tile sizes, and GQA grouping. The net is **1.5–2× over vLLM's native paged kernel** in many regimes (File 03 §16.4), especially the decode path (many short sequences, GQA) where the native kernel's scattered access hurt. It also unifies prefill and decode into one library with a consistent paged-KV, varlen interface, simplifying the engines' attention backends. FlashInfer's adoption by both vLLM and SGLang reflects that purpose-built serving-attention kernels beat both naive FlashAttention (which doesn't do paged KV) and the first-generation paged kernels (which weren't as optimized). It is the current state of the art for production paged attention on NVIDIA.

### 6.3 Workspace and CUDA-graph integration

FlashInfer requires a **workspace buffer** (scratch for the split-K partitioned reductions), pre-allocated by the engine. Its plan/run split makes it CUDA-graph-friendly: the plan (work partitioning for a given batch shape) is baked into the captured graph (File 09 §18), so graph replay skips re-planning. This integration with CUDA graphs (§7, File 09 §8) is part of why FlashInfer + CUDA-graph decode is so fast — the per-step attention has neither launch overhead (graph) nor planning overhead (baked in).

---

## 7. H100 / Hopper Architecture for Inference

Understanding the GPU is essential to understanding the kernels. The H100 SXM5 (Hopper) is the dominant inference GPU of 2023–2025.

### 7.1 Specs

- **80 GB HBM3 at ~3.35 TB/s** — the memory bandwidth that bounds decode (File 01 §2). (H100 NVL and H200 variants have more/faster memory.)
- **989 TFLOP/s BF16** (1979 with FP8) — the compute that bounds prefill.
- **132 SMs**, 4th-gen tensor cores supporting FP8/BF16/FP16/TF32/INT8.
- **67 MB L2 cache.**
- **NVLink 4.0 at 900 GB/s** bidirectional (NVL8/NVSwitch) — the interconnect for TP (File 05 §2.4).

### 7.2 Hopper features that kernels exploit

- **TMA (Tensor Memory Accelerator):** hardware async bulk HBM↔SMEM transfer, freeing SMs to compute during transfers (used by FA-3, §4).
- **WGMMA (warp-group MMA):** larger matrix-multiply instructions over 4-warp groups, higher tensor-core utilization (FA-3, §4).
- **Thread block clusters / distributed shared memory:** SMs can share SMEM within a cluster, enabling larger effective tiles.
- **Transformer Engine:** automatic FP8/FP16 mixed-precision with dynamic scaling, for FP8 inference (File 02 §11.4).
- **The ridge point** (File 01 §2): H100's `989 TFLOP/s ÷ 3.35 TB/s ≈ 295 FLOP/byte` (BF16) means it needs large batches to be compute-bound — its huge compute makes it even more memory-bound at small batch than A100, reinforcing that decode is bandwidth-limited and that reducing bytes moved (GQA, MLA, quantization, FP8 KV) is the key decode optimization.

### 7.3 Why H100 is the inference workhorse

H100's combination of high bandwidth (fast decode), high FP8 compute (fast prefill, efficient quantized serving), NVLink (efficient TP), and the Hopper features that FA-3/FlashInfer exploit make it the standard production inference GPU. The successor generations (H200 with 141 GB/4.8 TB/s, Blackwell B200 with 192 GB/8 TB/s and FP4) push bandwidth and add lower-precision formats, but the architectural lessons — bandwidth bounds decode, async-pipelined kernels extract throughput, low precision (FP8/FP4) doubles compute — carry forward (File 16).

---

## 8. AMD ROCm and MI300X

AMD's MI300X is the leading non-NVIDIA inference GPU, and both vLLM and SGLang support it via ROCm.

### 8.1 MI300X specs

- **192 GB HBM3** (3 stacks × 64 GB) at **~5.3 TB/s** — substantially more capacity and bandwidth than H100's 80 GB/3.35 TB/s.
- **~1307 TFLOP/s FP16** (no sparsity), **~2614 TFLOP/s FP8.**
- **CDNA3 architecture**, 8 GPU dies (chiplets) in one package (MI300A adds CPU dies for a unified APU).

### 8.2 The bandwidth and capacity advantage

MI300X's **5.3 TB/s vs H100's 3.35 TB/s** is a 1.58× bandwidth advantage — and since decode is memory-bandwidth-bound (File 01 §2), decode *should* be up to 1.58× faster. In practice it's ~1.2–1.4× due to memory-access-pattern and kernel-maturity differences. The **192 GB capacity** (2.4× H100) is a bigger deal: it lets a 70B model fit in FP16 on a *single* MI300X (140 GB weights + KV in 192 GB), or larger models on fewer GPUs — reducing the TP degree needed and its communication overhead (File 05 §12). For memory-bound, capacity-hungry inference (large models, long context, high concurrency), MI300X's memory is a genuine advantage.

### 8.3 ROCm software

- **HIP** (Heterogeneous Interface for Portability): compiles CUDA-style code to AMD GPU ISA. Much CUDA code ports to HIP with modest effort, but hand-tuned CUDA kernels (some custom attention, certain quantized matmuls) need HIP porting.
- **RCCL:** the ROCm equivalent of NCCL for collectives (File 05 §3).
- **Triton on ROCm (HIP Triton):** Triton kernels (§9) compile to AMD, so the Triton attention backend (File 09 §7.2) works on MI300X — a major reason Triton matters (portability across vendors).
- **FlashAttention ROCm:** a Triton-based FA port provides FlashAttention on AMD.

vLLM and SGLang both have ROCm builds and AMD CI/CD. Performance is typically **~85–90% of CUDA** in LLM workloads, the gap from kernel maturity (NVIDIA kernels are more tuned) and some features being CUDA-first. The capacity advantage often offsets the per-kernel gap for memory-bound large-model serving. MI300X is the most credible NVIDIA alternative for inference, and its support in both engines (with Triton providing portable kernels) is a significant counterweight to NVIDIA's dominance (File 16, File 20 §competitive dynamics).

---

## 9. Triton Kernel Development

**Triton** (OpenAI's GPU programming language) is a Python-embedded DSL for writing GPU kernels, central to both engines' portability and custom kernels.

### 9.1 The programming model

Triton lets you write tile-based GPU kernels in Python with `@triton.jit`. You specify tile sizes and the per-program-instance logic; Triton handles the low-level details (thread/warp mapping, memory coalescing, some optimization) and compiles to PTX (NVIDIA) or via LLVM to AMD. Key primitives: `tl.load`/`tl.store` (tiled DRAM access), `tl.dot` (tensor-core matmul), `tl.exp`/`tl.sum`/`tl.max` (math/reductions), `tl.atomic_add` (atomics). `@triton.autotune` searches tile-size configurations for the best performance.

### 9.2 Why Triton matters for inference engines

- **Portability:** the same Triton kernel compiles to NVIDIA *and* AMD (HIP Triton), so a Triton attention backend (File 09 §7.2) serves both vendors — far less effort than maintaining separate CUDA and HIP kernels. This is why Triton is the AMD fallback backend.
- **Productivity:** writing a fused kernel (§10) in Triton is far faster than in raw CUDA, letting the engines implement custom kernels (attention variants, fused norm/activation, quantized matmul) quickly.
- **Performance:** Triton kernels reach ~80–90% of hand-tuned CUDA (File 09 §7.2) — not always optimal, but close, and the productivity/portability often justify the small gap. Critical kernels (FlashInfer attention) use hand-tuned CUDA; the long tail of custom kernels uses Triton.

### 9.3 Example kernels

- **RMSNorm** (§10): tile along the hidden dim, load the tile, compute the partial sum-of-squares with a warp reduction (`tl.sum`, accumulated in FP32, File 02 §13.3), normalize and scale, fused with the weight multiply. `BLOCK_SIZE` autotuned.
- **SiLU + multiply** (SwiGLU, §10): load the gate and up projections, compute `silu(gate) * up`, write — one kernel, avoiding HBM round-trips.
- **Quantized matmul:** GPTQ Marlin-style INT4×FP16 with interleaved dequantization (File 06 §3.1).
- **Attention:** the Triton attention backend implements FlashAttention-style tiling/online-softmax in Triton for portability and for variants not yet in FlashInfer.

SGLang and vLLM both ship many Triton kernels; Triton is the practical answer to "write a fast, portable GPU kernel without months of CUDA work," and it's increasingly how new kernels (including for new hardware and new attention variants) are developed.

---

## 10. Fused Kernels

The recurring optimization across the stack: **fuse operations to cut HBM round-trips**, because the bandwidth-bound decode path (File 01 §2) is limited by memory traffic, not compute. Each unfused operation reads its inputs from HBM and writes its outputs back, only for the next operation to read them again — wasteful. Fusion keeps intermediates in registers/SMEM.

- **RMSNorm + QKV projection + RoPE:** normalize, project to Q/K/V, and rotate (RoPE) in a fused sequence, keeping the normalized activations and rotated Q/K in registers rather than round-tripping HBM (File 02 §13.2, §5.1).
- **SiLU ⊙ multiply (SwiGLU):** the gate and up projections feed a fused activation-and-multiply, avoiding two HBM passes (File 02 §24).
- **Residual add fusion:** the residual add is fused into the adjacent matmul's epilogue (File 02 §14.1), and into the all-reduce for TP (File 05 §10.2).
- **Quantized matmul with fused dequant:** dequantize weights inside the GEMM (Marlin, AWQ, FP8) rather than dequantizing to HBM then multiplying (File 06 §3.1) — the dequant overlaps with the matmul.
- **Sampling fusion:** penalties + temperature + top-k/top-p + sample in one kernel (File 02 §8.5).

The cumulative effect is significant: a transformer layer has many bandwidth-bound elementwise/normalization operations (RMSNorm ×2, residual ×2, activation, RoPE) that, unfused, would each cost an HBM round-trip; fused, they cost nearly nothing beyond the matmuls. For decode, where the layer is already memory-bound on weight reads, eliminating these extra round-trips is a direct latency win. Both engines fuse aggressively (vLLM's `csrc`/Triton kernels, SGLang's sgl-kernel, File 06 §17, File 09 §13), and the fusion set is the same because the same bandwidth bottleneck drives both. Fusion is the kernel-level expression of the database's central theme: minimize bytes moved (File 01 §2).

---

## 11. Multi-GPU Attention with Tensor Parallelism

Under tensor parallelism (File 05 §2), attention is split by head, and the kernel layer must handle the sharding and the collective.

### 11.1 The TP attention kernel flow

- **Q/K/V projection (column-parallel):** each GPU computes Q for its subset of query heads and K/V for its subset of KV heads. For GQA with 8 KV heads and TP=8, each GPU has 1 KV head and its share of query heads (File 05 §2.3).
- **Attention (local):** each GPU runs the (paged, FlashInfer) attention for *its* heads independently — no communication, because each head's attention is self-contained. This is why attention parallelizes cleanly under TP (the heads don't mix).
- **Output projection (row-parallel) + AllReduce:** the per-head outputs (partitioned across GPUs) are projected and **AllReduced** to combine the heads (File 05 §2.2). This is the one collective per attention block.

### 11.2 The AllReduce in the kernel path

The output-projection AllReduce is on the decode critical path (File 05 §2.4): for decode (batch of single tokens), it's a small (~16 KB/sequence) but frequent (per layer) latency-bound collective. The kernel-level optimizations (File 05 §3.4, §10.2):

- **Custom all-reduce:** for small tensors, a hand-written kernel using direct NVLink P2P, lower-latency than NCCL.
- **Fused residual add:** the residual add after the output projection is fused into the all-reduce, saving a kernel launch.
- **CUDA-graph capture of the collective:** the AllReduce is captured into the decode CUDA graph (File 05 §30, File 09 §8.3), eliminating its launch overhead.

So the multi-GPU attention kernel path is: local per-head paged attention (FlashInfer) → row-parallel output projection → custom/fused AllReduce (graph-captured). The collective is the only inter-GPU step, and the kernel-level work (custom all-reduce, fusion, graph capture) minimizes its critical-path cost.

### 11.3 GQA under TP at the kernel level

With GQA and high TP, each GPU may hold only 1 KV head serving several query heads. The attention kernel must read that 1 KV head once and reuse it across the query heads (GQA grouping, File 02 §28) — which FlashInfer does. If the KV heads don't divide evenly by the TP degree (e.g. 8 KV heads, TP=16), KV heads are replicated across GPU pairs (File 05 §2.3), and the kernel handles the replicated layout. The interaction of GQA, TP, and the attention kernel is a place where the architecture (TP sharding), the model (GQA head counts), and the kernel (GQA-aware gather) must all align — a mismatch (e.g. a GQA-unaware kernel under TP) silently wastes the bandwidth saving (File 02 §28).

---

## 12. CUDA Profiling for Inference

Optimizing kernels requires measuring them. Two NVIDIA tools are essential.

### 12.1 Nsight Systems (nsys) — system-level

`nsys profile --trace=cuda,nvtx,nccl python serve.py` produces a **timeline** of CPU, CUDA, and NCCL activity. It answers: is the GPU idle between steps (scheduling overhead, File 09 §29)? Are the AllReduces (NCCL) on the critical path? Is there a CPU bottleneck (tokenization, sampling)? Is memory transfer (H2D/D2H) overlapping with compute (File 09 §12)? The timeline view is the first place to look for *architectural* inefficiencies — gaps where the GPU waits for CPU or communication. NVTX range annotations (which both engines add) label the timeline with meaningful phases (prefill, decode, attention, sampling), making the trace readable.

### 12.2 Nsight Compute (ncu) — kernel-level

`ncu --set full python kernel_bench.py` profiles individual kernels against the **roofline** (File 01 §2): achieved FLOP/s vs peak, achieved bandwidth vs peak, memory-access efficiency (L2 hit rate, global load/store efficiency), and warp occupancy. It answers: is this kernel compute-bound or memory-bound? Is it near the roofline or leaving performance on the table? Are memory accesses coalesced (high efficiency) or scattered (low efficiency — the paged-attention concern, File 03 §16.5)? ncu is the tool for *kernel* optimization — diagnosing whether the attention kernel, the GEMM, or the sampling kernel is the bottleneck and why.

### 12.3 What to profile in LLM inference

- **GEMM kernels (QKV/FFN projections):** should be near the compute roofline for large batch (prefill). If not, check tile sizes, quantization-kernel efficiency.
- **Attention kernel:** memory-bound for decode (small batch), compute-bound for prefill (large batch). Check coalescing (paged access) and GQA grouping.
- **AllReduce (TP):** latency-sensitive; check the NCCL trace in nsys for whether it's on the critical path and whether NVLink is used (File 05 §27).
- **RMSNorm/LayerNorm, RoPE, activation:** should be fast and bandwidth-bound; if they show up significantly, they're probably unfused (fix: fuse them, §10).
- **Sampling kernel:** vectorized softmax + top-k; can matter for small models with large vocabularies (File 02 §14.4).

### 12.4 Engine profiling integration

Both engines integrate profiling: vLLM's `--profile` flag and torch.profiler integration with NVTX annotations; SGLang's `--enable-profiling` and `python -m sglang.bench_serving --profile`. These let you profile *while serving* (realistic workload) rather than synthetic microbenchmarks. The workflow: use nsys to find *where* time goes (which phase, GPU idle gaps, communication), then ncu to understand *why* a specific kernel is slow (roofline position, occupancy, coalescing). This top-down approach — system timeline first, then kernel deep-dive — is the disciplined way to optimize, avoiding the trap of micro-optimizing a kernel that isn't the bottleneck.

---

## 13. FlashAttention: A Worked Derivation

To solidify §2, work the online-softmax recurrence on a tiny example. Suppose a query attends to 4 keys with scores `s = [1, 3, 2, 5]` (post-scaling), processed in two tiles of 2: `[1, 3]` then `[2, 5]`. The correct softmax-weighted output is `Σ softmax(s)_i · v_i`. FlashAttention computes this without ever holding all 4 scores at once.

**Tile 1 — scores `[1, 3]`, values `v1, v2`:**
```
m = max(1, 3) = 3
p = [exp(1−3), exp(3−3)] = [0.135, 1.0]
ℓ = 0.135 + 1.0 = 1.135
O = 0.135·v1 + 1.0·v2   (unnormalized accumulator)
```

**Tile 2 — scores `[2, 5]`, values `v3, v4`:**
```
m_new = max(3, max(2,5)) = 5
correction = exp(m_old − m_new) = exp(3−5) = 0.135
# rescale the old accumulator and denominator to the new max
ℓ = correction·1.135 + (exp(2−5) + exp(5−5)) = 0.153 + (0.050 + 1.0) = 1.203
O = correction·O_old + (exp(2−5)·v3 + exp(5−5)·v4)
  = 0.135·(0.135·v1 + 1.0·v2) + (0.050·v3 + 1.0·v4)
```

**Finalize:** `O / ℓ`. Expanding, the weights are `[0.135·0.135, 0.135·1.0, 0.050, 1.0] / 1.203 = [0.0152, 0.1124, 0.0416, 0.831]`, which equals `softmax([1,3,2,5])` exactly. The recurrence produced the exact softmax-weighted output while only ever holding 2 scores at a time — the essence of FlashAttention. The `correction = exp(m_old − m_new)` rescaling is what keeps the accumulator consistent as larger maxima are discovered tile by tile, and accumulating in FP32 (File 02 §16) keeps it numerically stable over thousands of keys. Scaling this to real tiles (e.g. 128 keys per tile, `d=128` value vectors), the same recurrence streams through the KV in SRAM-sized tiles, never materializing the full score row in HBM (§2.3).

### 13.1 Worked IO numbers

For `n = 8192`, `d = 128`, FP16, A100 (SRAM `M ≈ 192 KB`, HBM `2 TB/s`):
- **Naive:** `2n² = 2·8192² ≈ 128 MB` written + read for S/P → at 2 TB/s, ~`128 MB / 2 TB/s × 2 (write+read) ≈ 128 µs` just for the score-matrix traffic, before the value aggregation.
- **FlashAttention:** `~4nd = 4·8192·128·2 ≈ 8 MB` for Q/K/V/O → ~`4 µs` of HBM traffic. The `n×n` never touches HBM.

That's a ~30× reduction in HBM traffic for the score matrix, which is why FlashAttention is 2–4× faster overall (HBM traffic was the bottleneck) and why it scales to long contexts where naive attention's `O(n²)` memory simply wouldn't fit (128 MB at 8K, ~2 GB at 32K, ~32 GB at 128K — impossible to materialize). FlashAttention's linear-in-`n` HBM traffic is what makes 128K+ context prefill feasible (File 13).

---

## 14. The GPU Memory Hierarchy

Kernel performance is governed by the memory hierarchy, and inference kernels are designed around it.

```
Registers:   ~256 KB/SM, ~fastest, per-thread       (kernel working set)
SMEM/L1:     ~228 KB/SM (H100), ~10–20 TB/s          (FlashAttention tiles)
L2 cache:    ~50 MB (H100), shared across SMs        (reused data)
HBM:         80 GB (H100), ~3.35 TB/s                (weights, KV cache)
```

The 3 orders of magnitude bandwidth gap between SRAM (~tens of TB/s aggregate) and HBM (~3 TB/s) is why FlashAttention's tiling (keep data in SRAM) wins, and why fusion (keep intermediates in registers/SMEM rather than HBM) wins (§10). The art of an inference kernel is **maximizing data reuse in the fast tiers**: load a tile from HBM once, reuse it many times in SRAM/registers, write the result once. The arithmetic intensity (FLOP/byte-from-HBM, File 01 §2) measures this reuse — high reuse = high intensity = compute-bound (good, the hardware is busy); low reuse = low intensity = memory-bound (the hardware waits on HBM). Decode is inherently low-reuse (read all weights to produce one token, File 01 §2), which is why it's memory-bound; the kernel can't change that, but it can ensure the reuse that *is* possible (KV in SRAM during attention, weights streamed efficiently) is captured. TMA (§4, §7.2) accelerates the HBM→SMEM transfer specifically to feed the SRAM tiers faster.

---

## 15. Occupancy and the Kernel-Launch Cost

Two GPU concepts that recur in inference kernel tuning:

### 15.1 Occupancy

**Occupancy** is the ratio of active warps to the maximum the SM supports. Higher occupancy lets the GPU hide memory latency (while one warp waits on HBM, another computes). But occupancy is limited by per-thread register usage and per-block SMEM usage — a kernel using lots of registers/SMEM (like a large-tile FlashAttention) runs fewer concurrent warps. There's a sweet spot: enough occupancy to hide latency, but large enough tiles to maximize reuse. FA-2's repartitioning (§3) improved occupancy; ncu (§12.2) reports it. For memory-bound decode kernels, sufficient occupancy to hide HBM latency is key; for compute-bound prefill GEMMs, tensor-core utilization matters more than occupancy per se.

### 15.2 Kernel-launch overhead

Each CUDA kernel launch costs ~5 µs of CPU→GPU overhead. A decode step has ~100+ kernels (per-layer matmuls, attention, norms, AllReduces) → ~500 µs of pure launch overhead, which for a fast decode step (a few ms) is 10–25%. This is what **CUDA graphs** (§7, File 09 §8) eliminate by capturing the whole step as one replayable unit. Launch overhead is a *fixed* cost per kernel, so it hurts most when the per-kernel *work* is small — exactly the small-batch decode regime. This is why CUDA graphs give 2–3× on small-batch decode (launch overhead dominated) and little on large compute-heavy steps (launch overhead negligible). Understanding launch overhead explains why kernel *fusion* (§10) helps doubly: fewer kernels means both less HBM traffic *and* less launch overhead.

---

## 16. Quantized Matmul Kernels: Marlin and FP8

Quantization (File 06 §3) is only fast if the matmul kernel is efficient. Two key kernels:

### 16.1 GPTQ Marlin (W4A16)

The naive W4A16 path dequantizes 4-bit weights to FP16 in HBM, then runs an FP16 GEMM — but the dequant-to-HBM step adds a full HBM round-trip, negating much of the memory saving. **Marlin** (Frantar et al. 2024) fuses dequantization into the GEMM: it loads the 4-bit weights (4× less HBM traffic than FP16), dequantizes them *in registers/SMEM* just before the tensor-core matmul, and never writes FP16 weights to HBM. With careful layout (interleaving the 4-bit values for efficient unpacking) and pipelining (overlapping dequant with matmul), Marlin reaches **near-FP16 GEMM throughput at 4-bit storage** on A100/H100. This is what makes GPTQ/AWQ W4A16 actually fast for decode — the 4× weight-read reduction (the memory-bound bottleneck, File 02 §FAQ) is realized because Marlin doesn't round-trip the dequantized weights through HBM. Without Marlin (the generic dequant-then-GEMM path), W4A16 is much slower.

### 16.2 FP8 GEMM

FP8 matmul (File 02 §11.4) uses the H100's native FP8 tensor cores (~2× BF16 TFLOP/s). The kernel loads FP8 weights and activations (half the bytes of BF16), runs the FP8 tensor-core matmul, and accumulates in higher precision (FP16/FP32) for accuracy. Per-tensor or per-channel scales are applied. FP8 GEMM is simpler than INT8 (no integer↔float conversion gymnastics, File 02 §11.4) and benefits *both* prefill (compute, 2× tensor-core throughput) and decode (memory, half the weight bytes). The FP8 path requires the Transformer Engine or equivalent scaling logic; both engines support it on H100+. FP8 is increasingly the default low-precision path because its kernel is efficient and its accuracy impact is small (<0.5%).

### 16.3 The general principle

Quantization's benefit is only realized if the kernel keeps the low-precision data low-precision through the HBM read and dequantizes only at the last moment (in registers, fused with the matmul). A kernel that dequantizes to HBM first throws away the memory saving. This is why Marlin and the FP8 kernels matter as much as the quantization *algorithm* (GPTQ/AWQ) — the algorithm decides the bits, but the kernel decides whether the bit reduction translates to a speedup. It's another instance of the file's theme: the kernel determines whether an optimization's theoretical benefit is realized in practice.

---

## 17. The MLA Attention Kernel

Multi-head Latent Attention (DeepSeek, File 02 §2.4, File 06 §7, File 15) poses a special kernel challenge because it caches a low-rank latent `c_KV`, not full K/V.

- **Naive (materialize) approach:** up-project the latent to full K/V before a standard attention kernel. This is FlashAttention-compatible (the kernel sees normal K/V) but loses MLA's efficiency — you materialize the full K/V you were trying to avoid, paying the memory bandwidth MLA's latent was meant to save during the attention compute (you still saved on *cache storage*, but not on the attention's K/V read).
- **Native MLA kernel:** attend directly on the latent `c_KV` using the weight-absorbed query (`W_Q_absorbed = W_Q · W_UKᵀ`, File 06 §7), so the attention's effective K is the compact latent. This preserves MLA's bandwidth saving in the attention itself (reading the small latent rather than full K/V) and is the high-performance path. It also handles the decoupled RoPE keys (`K_R`, File 02 §2.4) — concatenated with the latent-derived scores. A native MLA kernel is more complex (non-standard attention shape, the absorbed projection, the decoupled RoPE) and has been a development frontier; FlashInfer and the engines have added MLA support.

The MLA kernel is a case where a model architecture (low-rank KV) requires a *new kernel* to realize its benefit — a generic attention kernel can run MLA only by materializing K/V (losing the attention-time benefit). This is why inference-aware architecture (File 02 §22) and kernel support co-evolve: DeepSeek designed MLA for serving cost, and realizing that cost benefit required the engines and FlashInfer to build native MLA kernels. The DeepSeek-V3-scale serving (File 05 §29.2) depends on efficient MLA attention.

---

## 18. Tree Attention for Speculative Decoding

Tree-based speculative decoding (File 02 §9.3, File 12) proposes a *tree* of candidate tokens and verifies them in one target forward pass — requiring a **tree attention** kernel.

In standard causal attention, token `i` attends to tokens `0..i` (a linear causal mask). In tree speculation, the candidates form a tree (each node a proposed token, paths from root being candidate continuations), and a candidate token should attend only to its *ancestors* in the tree, not to siblings or other branches. This is a **tree attention mask** — a custom mask encoding the tree's ancestor relationships, instead of the linear lower-triangular causal mask. The kernel applies this mask so each candidate attends over exactly its path from the root, letting the target verify all tree paths in one batched forward pass (each path's attention is correct because the mask confines it to its ancestors). FlashInfer and the engines provide tree-attention kernels (a custom-mask extension of the paged attention) for EAGLE/Medusa-style tree speculation (File 12, File 09 §25). The tree mask is the kernel-level enabler of tree speculation's higher acceptance (more candidates verified per target pass than linear speculation), and building it efficiently (the mask must be applied without materializing a dense `n×n` mask) is the kernel challenge.

---

## 19. The Sampling Kernel

After the forward pass produces logits, sampling (File 02 §8, File 06 §10) runs as a GPU kernel — and for small models with large vocabularies it's a measurable fraction of step time.

The sampling kernel, per sequence in the batch, applies: penalties (gather already-generated-token logits, subtract), temperature (divide), top-k (partial sort / threshold to keep the k highest), top-p (cumulative-sum over sorted probabilities, threshold), and the draw (Gumbel-max or inverse-CDF with a per-sequence RNG seed). The challenges: (a) it's over the full vocabulary (32K–256K), so the reductions (max, sum, sort) are large; (b) it must handle a *batch* of sequences each with *different* sampling parameters (File 02 §8.3), so the kernel branches/masks per sequence; (c) under TP, the logits are vocab-parallel, requiring the sampling collective (File 05 §17, §33). Efficient implementation: a fused kernel doing penalties+temperature+top-k+top-p+sample in one pass (avoiding multiple HBM round-trips over the large logit vector, §10), with the top-k done via a partial selection (not a full sort) and top-p via a sorted cumulative sum. For a 1–3B model with a 128K vocabulary, the LM-head GEMM and this sampling kernel are large relative to the tiny transformer body (File 02 §14.4), so optimizing them (and using CUDA graphs to cut their launch overhead) matters most for small models. For large models, sampling is a small fraction and gets little attention.

---

## 20. Roofline Worked Examples for Inference Kernels

Apply the roofline (File 01 §2) to specific kernels to predict and interpret their performance.

### 20.1 Prefill GEMM (compute-bound)

A prefill FFN matmul for LLaMA-3 70B at batch×seqlen = 512×512 = 262144 tokens, one layer's gate projection (`d_model 8192 → d_ff ~28672`): FLOPs ≈ `2 · 262144 · 8192 · 28672 ≈ 1.2 × 10¹⁴`. Bytes ≈ weights (`8192·28672·2 ≈ 470 MB`, read once) + activations (`262144·8192·2 ≈ 4.3 GB` in, similar out). Arithmetic intensity ≈ `1.2e14 / ~9e9 ≈ 13000 FLOP/byte` — far above H100's ridge point (295), so **compute-bound**. Expected time ≈ `1.2e14 / 989e12 ≈ 121 ms` at peak; if ncu shows ~150 ms, that's ~80% of peak FLOP/s — good. If it shows 400 ms, the kernel is inefficient (poor tiling, low tensor-core utilization) — investigate.

### 20.2 Decode GEMM (memory-bound)

The same gate projection in *decode* (batch 32, seqlen 1 → 32 tokens): FLOPs ≈ `2 · 32 · 8192 · 28672 ≈ 1.5 × 10¹⁰`. Bytes ≈ weights (`470 MB`, read once, shared across the 32 tokens) + tiny activations. Arithmetic intensity ≈ `1.5e10 / 4.7e8 ≈ 32 FLOP/byte` — below H100's ridge (295), so **memory-bound**. Expected time ≈ `470 MB / 3.35 TB/s ≈ 140 µs` (weight-read-bound); the FLOPs would take only `1.5e10/989e12 ≈ 15 µs`, so the kernel waits on the weight read. This is the decode regime — the time is the weight read, and more batch (more tokens sharing that read) raises intensity toward the ridge (File 02 §15.2). ncu would show this kernel near the *bandwidth* roofline, not the compute roofline — the signature of memory-bound decode.

### 20.3 Interpreting the roofline position

The roofline position tells you the optimization lever: a compute-bound kernel below the FLOP roofline needs better tiling/tensor-core use (or lower precision — FP8 to raise the roofline); a memory-bound kernel below the bandwidth roofline needs better coalescing or fewer bytes (quantization, GQA). A kernel *at* its roofline is optimal — further speedup requires changing the regime (more batch to become compute-bound, or lower precision to raise the ceiling). This is why ncu's roofline view (§12.2) is the core kernel-optimization tool: it immediately shows whether a kernel is compute- or memory-bound and how far from optimal, directing the fix.

---

## 21. Hopper Mechanics in Depth: TMA, WGMMA, Warp Specialization

FlashAttention-3's speedup (§4) comes from three Hopper features that deserve deeper treatment, because they represent how modern attention kernels are built.

### 21.1 TMA (Tensor Memory Accelerator)

Pre-Hopper, loading a tile from HBM to SMEM used many threads issuing individual loads — consuming registers and instruction slots, and the SM stalled waiting. TMA is a *dedicated hardware engine* that performs a bulk asynchronous copy of a multi-dimensional tile from HBM to SMEM with a single instruction, freeing the SM's threads to compute meanwhile. The kernel issues a TMA load for the next tile, continues computing on the current tile, and the TMA engine fills SMEM in the background. This is hardware-level memory/compute overlap (the §10/File 05 §34 overlap principle, in silicon). TMA is why FA-3 can keep the tensor cores busy — the next K/V tile arrives via TMA while the current one is being multiplied.

### 21.2 WGMMA (Warp-Group Matrix Multiply-Accumulate)

Pre-Hopper tensor-core instructions (`mma`) operated at the warp level on small tiles. WGMMA operates at the *warp-group* level (4 warps = 128 threads) on larger tiles (e.g. M×N×K = 64×256×16), issuing more work per instruction and using the tensor cores more efficiently (less instruction overhead, better data reuse). WGMMA is also *asynchronous* — it can be issued and the warp continues, with results collected later — enabling overlap with TMA loads and softmax computation. FA-3's QKᵀ and PV matmuls use WGMMA, which is a major part of its throughput.

### 21.3 Warp specialization (producer-consumer)

FA-3 divides the thread block's warps into **producers** and **consumers**. Producers issue TMA loads to fetch the next tiles into SMEM; consumers issue WGMMA to compute on the current tiles. They coordinate via SMEM barriers (a producer signals "tile ready," a consumer signals "tile consumed, reuse the buffer"). This producer-consumer pattern, with a software pipeline of (typically) 2+ stages, overlaps the memory fetch (producers + TMA) with the matmul (consumers + WGMMA), so neither waits on the other — the kernel approaches being limited by the *slower* of memory and compute rather than their sum. This is the same overlap idea as the engine architecture's CPU/GPU overlap (File 09 §10) and EP's communication/compute overlap (File 05 §21), here at the warp level inside one kernel. Warp specialization is the defining technique of the latest attention kernels, and it's why FA-3 extracts far more of Hopper's throughput than a naive port of FA-2.

---

## 22. The Custom All-Reduce Kernel

vLLM's (and SGLang's) custom all-reduce for small TP decode AllReduces (File 05 §3.4, §10.2, §11.2) is worth a kernel-level look. NCCL's ring AllReduce (File 05 §15) is bandwidth-optimal but has per-step launch and synchronization overhead that dominates for tiny (16 KB) messages. The custom all-reduce instead uses **direct peer-to-peer GPU memory access over NVLink**: each GPU can read another GPU's memory directly (NVLink P2P), so a small AllReduce is implemented as each GPU reading the others' partial results and summing them locally — a few P2P reads and a local reduction, in one lightweight kernel, with lower latency than NCCL's multi-step ring for small messages. It can also **fuse the residual add** into this kernel (the AllReduce result is added to the residual stream in the same kernel, File 05 §10.2), saving a separate launch. And it's **CUDA-graph-capturable** (File 09 §8.3), so in the captured decode graph the AllReduce has no launch overhead. The custom all-reduce is a targeted optimization for the specific regime (small, frequent, latency-bound decode AllReduces) where NCCL's generality costs more than a specialized kernel — a recurring pattern where a hand-written kernel beats a general library for a specific shape (like Marlin beating generic GEMM for W4A16, §16.1).

---

## 23. Kernel Autotuning

Many inference kernels are **autotuned**: the optimal tile sizes, number of warps, pipeline stages, and other parameters depend on the GPU, the problem shape (batch, sequence length, head dim), and the dtype, and the best configuration isn't known a priori. Triton's `@triton.autotune` (§9.1) searches a space of configurations (benchmarking each) and caches the best per shape. FlashInfer and CUTLASS-based kernels similarly tune or select kernels per shape. This is why warmup (File 07 §19, File 09 §17) includes autotuning — the first forward passes trigger the configuration search so steady-state runs use the best kernel. Autotuning matters because a kernel tuned for one shape (e.g. large prefill) can be far from optimal for another (small decode), and inference sees both regimes (prefill and decode, varying batch). The engines tune for the shapes they'll actually run (the captured CUDA-graph batch sizes, the typical context lengths). Autotuning is the practical reality behind "the kernel reaches 80% of peak" — that 80% is the *tuned* configuration; an untuned kernel might reach 40%. It's also why a new GPU or a new model shape can need re-tuning to hit peak performance, and why kernel libraries ship with tuned configurations for common shapes.

---

## 24. The Split-K Decode Reduction, Quantified

The two-phase split-K decode kernel (§5, File 03 §16.3) addresses a specific problem: at small batch but long context, assigning one thread block per (sequence, head) leaves most SMs idle. Quantify: batch 4, 8 query heads (after TP), on an H100 with 132 SMs → only `4 × 8 = 32` thread blocks, using 32 of 132 SMs (~24% of the GPU). The other ~76% sits idle while those 32 blocks each grind through a long context's KV sequentially.

Split-K fixes this by partitioning each sequence's KV (say 50K tokens) into, e.g., 8 partitions, launching `32 × 8 = 256` thread blocks — now exceeding the 132 SMs, fully occupying the GPU. Each block computes partial attention + softmax stats for its KV partition (phase 1); a reduction combines the 8 partials per (sequence, head) via log-sum-exp (phase 2, the same online-softmax combination, §2.2). The result: the long-context decode that would have used 24% of the GPU now uses 100%, a ~4× speedup for this small-batch-long-context regime. The number of partitions is chosen at launch based on context length and SM count to just fill the GPU. This split-K technique (analogous to split-K in GEMM) is essential for the increasingly common long-context, low-concurrency decode (a single user with a huge context, File 13) — without it, long-context decode would badly underutilize large GPUs. FlashInfer generalizes this with its plan phase choosing the partitioning (§6.3).

---

## 25. The MoE Grouped-GEMM Kernel

MoE models (File 02 §4.2, §21; File 06 §6) need a **grouped GEMM** kernel: after routing, tokens are grouped by their selected expert, and each expert's FFN multiplies its variable-size group of tokens. The kernel challenges:

- **Variable group sizes:** routing is data-dependent, so each expert gets a different number of tokens. The kernel must handle the ragged grouping efficiently — gathering each expert's tokens (a scatter by routing index), running its matmul, and scattering the results back.
- **Tensor-core padding:** each group is padded to a multiple of the tensor-core tile (e.g. 16) so the GEMM is efficient; imbalanced routing wastes compute on padding (File 02 §21.3).
- **Quantized experts:** for FP8/GPTQ expert weights, dequantization is fused per group (like Marlin, §16.1).
- **Stacked weights:** expert weights are stored as `[num_experts, d_ff, d_model]` tensors, and the kernel indexes the right expert's slice per group.

vLLM's `fused_moe` Triton kernel (File 06 §6) and SGLang's equivalent implement this — iterating over experts, running each group's (quantized) matmul, with the gather/scatter. The grouped GEMM is the compute core of MoE inference; its efficiency (handling the raggedness, minimizing padding waste, fusing dequant) determines MoE decode/prefill speed. For distributed MoE (EP, File 05 §5), this is preceded by the dispatch all-to-all and followed by the combine — the grouped GEMM is the local-compute part between the two collectives. The kernel's data-dependent grouping is also why MoE step time is noisier than dense (File 02 §FAQ) — the group sizes (and thus the padding waste and the work distribution) vary per step with the routing.

---

## 26. A Worked Per-Step Kernel Breakdown

To connect the kernels to a real step, break down the kernels in one decode step for a dense GQA model (LLaMA-3 70B class) under TP, per layer:

1. **RMSNorm** (fused with QKV) — bandwidth-bound, tiny, fused (§10).
2. **QKV projection GEMM** (column-parallel) — memory-bound at decode (weight read), with RoPE fused in.
3. **Write K/V to paged cache** — a scatter to the physical slots (`slot_mapping`, File 03 §3.2).
4. **Paged attention** (FlashInfer decode wrapper) — memory-bound, reads the paged KV, GQA-grouped, possibly split-K for long context (§24).
5. **Output projection GEMM** (row-parallel) — memory-bound, weight read.
6. **AllReduce** (custom, fused residual) — latency-bound collective (§22).
7. **RMSNorm** (fused with gate/up) — bandwidth-bound, fused.
8. **Gate/up projection GEMM** (column-parallel) — memory-bound, weight read.
9. **SiLU ⊙ multiply** — bandwidth-bound, fused (§10).
10. **Down projection GEMM** (row-parallel) — memory-bound, weight read.
11. **AllReduce** (custom, fused residual) — latency-bound collective.

× 80 layers, then the final RMSNorm, LM-head GEMM, and sampling kernel. Without CUDA graphs, that's `~11 × 80 + a few ≈ ~900 kernel launches`, costing `~900 × 5 µs ≈ 4.5 ms` of pure launch overhead — which CUDA graphs (§15.2, §7) eliminate by replaying the whole step as one unit. The GEMMs (steps 2, 5, 8, 10) are weight-read-memory-bound at decode (§20.2), so their time is the weight read; the attention (step 4) is KV-read-memory-bound; the norms/activations (1, 7, 9) are fused and near-free; the AllReduces (6, 11) are latency-bound collectives minimized by the custom kernel. The whole step's time ≈ the weight reads (all GEMMs, ~`70 GB/3.35 TB/s ≈ 21 ms` for the full model in BF16, or ~half that in FP8) + KV reads + the AllReduce latencies, with launch overhead removed by graphs. This breakdown shows where every microsecond goes and which kernel-level optimization (fusion, FlashInfer, custom all-reduce, CUDA graphs, quantization) targets which part — the kernel layer's contribution to the step latency.

---

## 27. Attention Backend Comparison

Summarizing the attention kernels covered, as the engines' selectable backends (File 06 §13, File 09 §7):

| Backend | Paged KV | Varlen | GQA-opt | Hardware | Speed | Use |
|---|---|---|---|---|---|---|
| FlashInfer | yes | yes | yes | NVIDIA (Hopper+) | fastest | production NVIDIA default |
| FlashAttention (2/3) | limited/varlen | yes | yes | NVIDIA (A100/H100) | fast | prefill, standard path |
| vLLM native paged | yes | yes | partial | NVIDIA | moderate | fallback |
| Triton | yes | yes | yes | NVIDIA + AMD | ~80–90% of FlashInfer | AMD/portable/fallback |
| Torch (reference) | yes | yes | n/a | any | slow | debugging |

The hierarchy: FlashInfer for production NVIDIA (purpose-built paged-KV serving attention, §6), FlashAttention for the standard prefill path, Triton for AMD and portability (§9), the native paged kernel as a fallback, and the torch reference for correctness. The backend choice (File 06 §13, File 09 §7) must match the hardware and the model (MLA needs special handling, §17), and the KV page size must match the configured block size (File 03 §13). For most NVIDIA deployments, FlashInfer is the answer; for AMD, Triton; the others are situational. The backend is selectable because no single kernel is optimal everywhere — the engines pick a good default and expose overrides for tuning (File 11).

---

## 28. Blackwell, FP4, and the Hardware Trajectory

The hardware trajectory (detailed in File 16) shifts the kernel landscape:

- **H200:** H100 compute with 141 GB HBM3e at 4.8 TB/s — 1.43× H100 bandwidth → ~1.43× decode throughput (decode is bandwidth-bound), no kernel changes needed (the same kernels run faster on faster memory).
- **Blackwell (B200):** 192 GB HBM3e at ~8 TB/s, ~4500 TFLOP/s FP4, NVLink 5.0 at 1.8 TB/s. Adds **FP4** (E2M1) — 4-bit floating point for inference with minor quality loss, doubling compute over FP8. New kernels (FP4 GEMM, FP4 attention) and the 2nd-gen Transformer Engine exploit it. FP4 inference is the next step in the precision-reduction trend (BF16 → FP8 → FP4), each doubling compute and halving weight bytes, requiring matching kernels to realize the benefit (§16.3's principle).
- **The trend:** more bandwidth (faster decode, same kernels), lower precision (FP4/FP6, more compute, new kernels), larger/faster NVLink (TP scales further). The kernel work tracks the hardware — each new precision needs a kernel that keeps the low-precision data low-precision through the HBM read and dequantizes/computes at the last moment (§16.3). The architectural lessons (bandwidth bounds decode, async-pipelined kernels via TMA/WGMMA, low precision doubles compute) carry forward; the numbers and the precision formats change. An inference engineer tracks both the hardware (File 16) and the kernels (this file) because the optimal configuration shifts with each generation — but the roofline reasoning (File 01 §2) for *why* a configuration is optimal stays constant.

---

## 29. Varlen Batch Kernels: Packing and the `indptr` Layout

The continuous-batching engines (Files 04, 09) feed the attention kernel a *ragged* batch — sequences of different lengths, mixing multi-token prefill chunks and single-token decodes (File 02 §20). The kernel must process this without padding waste, which is what the **varlen** (variable-length) kernel APIs provide.

Instead of a padded `[batch, max_len, ...]` tensor (wasteful when lengths vary), the varlen layout concatenates all sequences end-to-end into one flat buffer and supplies a **cumulative-length index** (`cu_seqlens` in FlashAttention, `indptr` in FlashInfer) marking each sequence's boundaries: `indptr = [0, len_0, len_0+len_1, ...]`. The kernel uses this index to confine each sequence's attention to its own span — sequence `i` occupies the flat buffer from `indptr[i]` to `indptr[i+1]`, and its query attends only over its own KV (resolved via its block table). For a decode sequence, the query length is 1; for a prefill chunk, it's the chunk size — the same kernel handles both via the per-sequence query-length the index encodes. This is how one kernel invocation serves a mixed prefill+decode batch (File 02 §20, File 09 §7.1) with zero padding waste. The packing is built by the model runner / batch constructor each step (File 04 §38, File 06 §2), and the varlen kernel consuming it is what makes the heterogeneous continuous batch efficient. Without varlen support, the engine would pad to the max length (wasting compute on padding) or run separate kernels per sequence (losing batching) — varlen is the kernel feature that makes efficient continuous batching possible, which is why FlashInfer's varlen wrappers are central to both engines.

---

## 30. Memory Coalescing and Access Patterns

A recurring kernel-performance factor is **coalescing**: when the threads of a warp (32 threads) access *consecutive* memory addresses, the hardware combines them into one (or few) wide memory transactions — efficient. When they access *scattered* addresses, each becomes a separate transaction — inefficient, wasting bandwidth (the bottleneck for memory-bound kernels). The vLLM paged-attention `x`-layout (File 03 §16.1) exists precisely to make the KV-cache reads coalesced: by laying out the inner dimension as a 16-byte-aligned vector, each thread issues a 128-bit load and the warp's loads coalesce. The paged kernel's weakness (File 03 §16.5) is that consecutive *logical* blocks can map to scattered *physical* blocks, so reads across block boundaries lose coalescing — which FlashInfer's better layouts and the GQA grouping partly recover. For GEMMs, coalescing is handled by the tiling (the tile loads are coalesced by construction). ncu (§12.2) reports "global load/store efficiency" — the fraction of fetched bytes actually used — which is the coalescing metric; low efficiency means scattered access wasting bandwidth. For memory-bound decode kernels (where bandwidth is the bottleneck), coalescing directly determines performance: a kernel reading the same bytes with poor coalescing achieves a fraction of peak bandwidth. This is why KV-cache layout (File 03 §16.1) and attention-kernel access patterns are engineered for coalescing — on the memory-bound decode path, wasted bandwidth from poor coalescing is wasted time.

---

## 31. The LM-Head Kernel and the Vocabulary Reduction

The final LM-head projection (File 02 §14.3) is a large GEMM (`d_model × vocab`, e.g. `8192 × 128000`) that produces logits over the vocabulary, followed by sampling (§19). Kernel considerations:

- **Prefill optimization:** during prefill, logits are needed only for the *last* token of each sequence (to start generation), so the LM-head should be computed only for the final position, not all positions — saving a `seq_len×` factor (File 02 §14.3). The kernel/runner selects the last-token hidden states before the LM-head GEMM.
- **Vocab-parallel under TP:** the LM head is split by vocabulary across ranks (File 05 §17), so each rank computes logits for its vocab slice, and sampling requires the cross-rank reduction (the sampling collective, File 05 §33). For top-k sampling, each rank computes its local top-k and the ranks merge — far cheaper than all-gathering the full 128K-wide logits (File 05 §33's worked numbers).
- **Memory:** for a 128K vocabulary, the logit tensor is large (`batch × 128000 × 2 bytes`), and for a small model it's large *relative* to the model (File 02 §14.4), making the LM-head GEMM and the sampling reduction a meaningful fraction of step time for small models — hence optimizing them (fused sampling, distributed top-k, CUDA graphs) matters most there.

The LM-head + sampling is the step's tail, and for large models it's a small fraction; for small models with large vocabularies it can be significant, which is why the engines optimize it (last-token-only prefill, vocab-parallel distributed top-k, fused sampling kernel) — another case where the optimal kernel work depends on the model size and vocabulary (small-model serving has different bottlenecks than large-model serving).

---

## 32. The SMEM / Register / Occupancy Trade-off, Worked

FlashAttention's tile sizes are a concrete instance of the occupancy trade-off (§15.1). Larger tiles mean more data reuse (each loaded tile is multiplied more times before eviction — higher arithmetic intensity) but consume more SMEM and registers, reducing how many warps/blocks run concurrently on an SM (lower occupancy, less latency hiding). Smaller tiles mean higher occupancy (more latency hiding) but less reuse (more HBM traffic). The optimum balances these.

Worked: an H100 SM has ~228 KB SMEM. A FlashAttention tile of Q `[128 × 128]` + K `[128 × 128]` + V `[128 × 128]` in FP16 is `3 × 128 × 128 × 2 ≈ 98 KB` plus softmax statistics and double-buffering for the producer-consumer pipeline (§21.3) — pushing toward the SMEM limit, allowing perhaps one or two blocks per SM. That's *low* occupancy, but acceptable for FA-3 because the warp-specialized pipeline (TMA + WGMMA overlap, §21) hides latency without needing many resident warps — the pipeline itself provides the latency hiding that occupancy would otherwise provide. This is a subtle point: FA-3 trades occupancy for larger tiles and explicit software pipelining, and it wins because the explicit pipeline hides latency more effectively than relying on high occupancy. Pre-Hopper kernels (FA-2) leaned more on occupancy. The shift reflects Hopper's hardware (TMA, async WGMMA) enabling the explicit-pipeline approach. For the inference engineer, the lesson is that "maximize occupancy" is not always the goal — for kernels with explicit software pipelines, larger tiles with lower occupancy can win, and ncu's occupancy metric (§12.2) must be interpreted alongside the kernel's design (does it rely on occupancy or on explicit pipelining for latency hiding?).

---

## 33. Key Takeaways

1. **Attention's bottleneck is HBM traffic, not FLOPs** (§1). FlashAttention (§2) computes exact attention while keeping the `n×n` intermediates in SRAM via tiling + online softmax, cutting HBM traffic from `O(n²)` to `O(n²/M)` — 2–4× faster and enabling long context.
2. **FA-2 improved work partitioning** (~70–75% of A100 peak); **FA-3 exploits Hopper** (TMA, WGMMA, warp specialization) for ~740 TFLOP/s on H100 (§§3–4, §21) — the async producer-consumer pipeline is the defining modern technique.
3. **Paged attention** (§5) marries online-softmax tiling with block-table indirection; **FlashInfer** (§6) is the optimized, GQA-aware, varlen, paged backend that's now the NVIDIA default for both engines.
4. **The hardware** (H100, §7; MI300X, §8) sets the rooflines: bandwidth bounds decode, compute bounds prefill, and the kernels are designed around the memory hierarchy (§14) — keep data in fast tiers, fuse to avoid HBM round-trips (§10).
5. **Triton** (§9) provides portable, productive kernels (NVIDIA + AMD), used for the long tail of custom and fallback kernels; hand-tuned CUDA (FlashInfer, Marlin) for the critical paths.
6. **Quantized matmul kernels** (Marlin §16.1, FP8 §16.2) realize quantization's benefit by keeping data low-precision through the HBM read and fusing dequant — the algorithm decides the bits, the kernel decides the speedup.
7. **CUDA graphs** (§7, §15.2, §26) eliminate the ~5 µs/kernel launch overhead (significant for ~900-kernel decode steps), and **custom all-reduce** (§22) minimizes the latency-bound TP collectives — both critical for fast decode.
8. **Profiling** (§12): nsys for system-level (GPU idle, communication), ncu for kernel-level (roofline position, coalescing, occupancy). Profile top-down: find where time goes, then why a kernel is slow.

The kernel layer is where the roofline (File 01 §2) is either hit or missed — where the theoretical memory/compute limits become actual performance. Every optimization in this database ultimately depends on a kernel realizing it: PagedAttention needs the paged kernel, quantization needs Marlin/FP8 GEMMs, GQA needs GQA-aware attention, TP needs the custom all-reduce, and the whole decode step needs CUDA graphs. The architecture (File 09) keeps the kernels fed; the kernels (this file) make each operation fast; together they determine whether a deployment hits its roofline. File 11 unifies the algorithmic, architectural, and kernel layers into a performance-tuning methodology — measuring where a real deployment sits relative to its rooflines and which layer's optimization will move it.

---

## 34. Appendix: Source Pointers

- **Papers:** FlashAttention (arXiv 2205.14135), FA-2 (arXiv 2307.08691), FA-3 (2024); FlashInfer (arXiv 2501.01005); Marlin (Frantar et al. 2024). Hopper/Ada/Blackwell architecture whitepapers (NVIDIA); CDNA3/MI300 (AMD).
- **Code:** vLLM `csrc/` (CUDA kernels: paged attention, custom all-reduce, quantized matmul) and `vllm/attention/backends/` (backend selection); SGLang `sgl-kernel/` and `sglang/srt/layers/attention/`; FlashInfer (`flashinfer-ai/flashinfer`); Triton (`triton-lang/triton`); CUTLASS (NVIDIA, the GEMM template library underlying many kernels).
- **Tools:** Nsight Systems (`nsys`), Nsight Compute (`ncu`), the engines' `--profile`/`--enable-profiling` flags, `nvidia-smi topo -m` (interconnect topology), `NCCL_DEBUG=INFO` (collective diagnostics).

To trace a kernel-level optimization: start with nsys to find GPU idle gaps or slow phases → identify the bottleneck kernel → ncu to see its roofline position (compute- or memory-bound, how far from peak) and coalescing/occupancy → apply the matching fix (fusion for HBM traffic, better tiling for compute, coalescing for bandwidth, CUDA graphs for launch overhead, quantization for precision/bytes) → re-measure. This measurement-driven loop, grounded in the roofline, is how kernel performance is actually improved — and it's the kernel-level instance of the performance methodology File 11 develops for the whole system.

---

## 35. The Chunked-Prefill Attention Kernel

Chunked prefill (File 04 §7) splits a long prompt into chunks processed across steps, which shapes the attention kernel: when processing chunk `i`, the new chunk's queries must attend over *all previously-cached chunks* (`0..i-1`) plus the new chunk itself (causally). This is a **prefix-attention** kernel — `chunk_size` queries attending over `(cached_prefix + chunk_size)` keys, where the cached prefix is read-only (already in the paged cache) and the new chunk's K/V are freshly written. The kernel:

- Reads the cached prefix K/V (read-only, possibly shared prefix-cache blocks, File 03 §6) via the block table.
- Reads/writes the new chunk's K/V.
- Computes causal attention: the new queries attend over the full prefix (no causal restriction — all prefix tokens precede them) plus their causal portion of the new chunk.

This is a generalization of the standard prefill kernel (where all queries are new and there's no cached prefix) and the decode kernel (where there's one query and a long cached prefix). FlashInfer's prefill wrapper (§6, File 09 §18) handles the prefix-plus-chunk causal attention via the varlen/`indptr` layout (§29), with the cached prefix and new chunk distinguished by the block table and position offsets (File 02 §19). The cumulative KV read grows with each chunk (chunk `i` reads `i × chunk_size` prefix tokens), which is the source of chunked prefill's slight prefill-throughput cost (File 04 §7.4) — the later chunks re-read more accumulated KV. The kernel correctly handling the prefix-plus-chunk attention is what makes chunked prefill possible at the kernel level; without a kernel that attends over cached-prefix-plus-new-chunk, you couldn't interleave bounded prefill chunks with decode (File 04 §7).

---

## 36. A100 vs H100 vs MI300X: Kernel Performance Compared

Concretely comparing the GPUs through the kernel lens (detail in File 16):

- **A100 SXM (80 GB, 2 TB/s, 312 TFLOP/s BF16):** FA-2 reaches ~70–75% of peak for attention; decode is bandwidth-bound at 2 TB/s. No FP8 tensor cores (INT8 for W8A8). The prior-generation workhorse, still widely deployed.
- **H100 SXM (80 GB, 3.35 TB/s, 989 TFLOP/s BF16, 1979 FP8):** FA-3 reaches ~740 TFLOP/s via TMA/WGMMA/warp-specialization (§21); decode 1.67× faster than A100 (bandwidth ratio); FP8 doubles prefill compute and halves weight bytes. The current inference standard.
- **H200 (141 GB, 4.8 TB/s, same compute as H100):** ~1.43× H100 decode throughput from bandwidth; the same kernels run faster. More capacity for KV / larger models.
- **MI300X (192 GB, 5.3 TB/s, 1307 TFLOP/s FP16, 2614 FP8):** highest bandwidth → fastest decode in principle (1.58× H100), realized at ~1.2–1.4× due to kernel maturity (§8.2); huge 192 GB capacity fits 70B in FP16 on one GPU. Triton/HIP kernels at ~85–90% of CUDA maturity.
- **B200 (Blackwell, 192 GB, ~8 TB/s, ~4500 TFLOP/s FP4):** next-gen — FP4 doubles compute over FP8, ~8 TB/s nearly doubles H100 decode bandwidth, NVLink 5.0 (1.8 TB/s) scales TP further. New FP4 kernels required (§28).

The pattern across generations: **bandwidth grows** (faster decode, the bandwidth-bound phase, with the *same* kernels), **precision drops** (BF16→FP8→FP4, doubling compute and halving bytes each step, requiring *new* kernels), and **interconnect grows** (NVLink scaling TP). The kernels evolve to exploit each generation's features (FA-2 for A100, FA-3 for Hopper, FP4 kernels for Blackwell), but the roofline reasoning for *why* decode is bandwidth-bound and *why* lower precision helps stays constant (File 01 §2). An inference engineer choosing hardware reasons about the workload's bottleneck (decode-heavy → prioritize bandwidth/capacity, e.g. H200/MI300X; prefill/compute-heavy → prioritize FLOP/s and low precision, e.g. H100/B200 FP8/FP4) and whether the kernels for that hardware are mature (NVIDIA most mature, AMD ~85–90%, newest generations needing kernel updates). This hardware-kernel co-reasoning is developed further in File 16.

---

## 37. Why Two Kernel Regimes: Prefill vs Decode at the Kernel Level

A unifying observation across this file: attention (and the whole forward pass) has two fundamentally different kernel regimes that require different optimization, mirroring the prefill/decode duality (File 01 §3).

**Prefill kernels** process many tokens (the prompt or a chunk) against many keys — the operations are large GEMMs (matrix-matrix), compute-bound, with high arithmetic intensity. The optimization target is **tensor-core utilization**: large tiles, WGMMA, FP8 to raise the compute roofline, keeping the tensor cores saturated. The FA-3 prefill kernel, the prefill GEMMs, and the chunked-prefill prefix attention (§35) are all in this regime — make the matmuls big and efficient.

**Decode kernels** process one token per sequence against the cached KV — the operations are GEMVs (matrix-vector), memory-bound, with arithmetic intensity ≈ 1 (File 02 §1.3). The optimization target is **bandwidth and reuse**: coalesced access (§30), GQA grouping (read each KV head once), reduce bytes (quantization, FP8 KV), split-K to fill the GPU at small batch (§24), and CUDA graphs to eliminate launch overhead (§15.2). The FlashInfer decode wrapper, the paged decode kernel, and the custom all-reduce are all in this regime — minimize and accelerate the memory traffic.

These two regimes need different kernels (which is why FlashInfer has separate prefill and decode wrappers, §6), different tile strategies (large compute tiles vs bandwidth-optimized gather), and different optimizations (tensor-core utilization vs bandwidth/launch-overhead). A serving step mixing prefill and decode (continuous batching, §29) runs both regimes — often via a unified varlen kernel that handles each sequence's query length appropriately, or via separate kernels. Understanding which regime a kernel is in (from its arithmetic intensity / roofline position, §20) immediately tells you which optimizations apply — the single most useful kernel-level diagnostic. A prefill kernel below the compute roofline needs better tiling/precision; a decode kernel below the bandwidth roofline needs better coalescing/fewer bytes. Misapplying — trying to "add more FLOPs efficiency" to a memory-bound decode kernel — wastes effort; the regime dictates the lever.

---

## 38. Kernel Selection Decision Guide

Tying the file's kernels into practical selection (with File 06 §13, File 09 §7):

- **Attention backend:** FlashInfer for production NVIDIA (paged, varlen, GQA-aware, fastest); FlashAttention for standard prefill; Triton for AMD ROCm or as a portable fallback; native paged as last-resort fallback; torch for debugging. Match the page size to the block size.
- **MLA models (DeepSeek):** need MLA-aware attention (native MLA kernel for full benefit, or materialize-K/V fallback, §17).
- **Speculative decoding:** need tree-attention kernels for tree speculation (§18).
- **Quantization:** Marlin for GPTQ/AWQ W4A16 (§16.1), FP8 GEMM for W8A8/weight-FP8 on H100+ (§16.2) — both fuse dequant to realize the benefit.
- **Always:** enable CUDA graphs for decode (§15.2), use fused kernels for norm/activation/RoPE/residual (§10), and the custom all-reduce for TP (§22).

The engines pick good defaults (FlashInfer + CUDA graphs + fused kernels + custom all-reduce on NVIDIA), so kernel selection is mostly automatic — the inference engineer's role is to verify the fast path is engaged (FlashInfer active, CUDA graphs not falling back to eager, NVLink used for collectives) and to profile (§12) when performance is below expectation, mapping the bottleneck to the responsible kernel and its regime (§37).

---

## 39. Closing

The kernel layer is where the roofline becomes reality. FlashAttention's tiling and online softmax (§§2, 13) turned attention from `O(n²)`-memory-bound to linear, enabling long context; FA-2 and FA-3 (§§3–4, §21) extracted the GPU's compute through better partitioning and Hopper's async pipelining; FlashInfer (§6) specialized this for paged-KV serving; Marlin and FP8 kernels (§16) realized quantization's promise; fusion (§10) eliminated bandwidth-wasting round-trips; CUDA graphs (§15) and the custom all-reduce (§22) removed launch and collective overhead from the decode critical path. The two regimes — compute-bound prefill, memory-bound decode (§37) — require different kernels and optimizations, and the roofline (§20) tells you which applies. Beneath every algorithm (PagedAttention, RadixAttention, speculative decoding, quantization, MoE) and every architecture (Files 03–09) is a kernel that either realizes its benefit or squanders it — Marlin makes W4A16 fast or the generic path makes it slow; FlashInfer makes paged attention fast or the naive kernel makes it scattered; CUDA graphs make decode fast or launch overhead dominates. This is why the kernel layer, though the least visible, is decisive: it's the layer where bytes are actually moved and FLOPs actually computed, where the roofline limits are met or missed. With the kernels mapped, File 11 unifies the three layers — algorithms (the optimizations), architecture (keeping the GPU fed), and kernels (hitting the roofline) — into a measurement-driven performance and tuning methodology, turning the understanding built across Files 01–10 into the practical discipline of making a real deployment fast.

---

## 40. Appendix: Causal-Mask Optimization in the Prefill Kernel

A kernel-level detail with real performance impact: in causal prefill attention, token `i` attends only to tokens `0..i`, so the score matrix is lower-triangular — the upper triangle is masked out (File 02 §1.2). A naive kernel computes the full `n×n` scores then masks the upper half, wasting ~half the compute on entries that become zero. An optimized FlashAttention prefill kernel **skips** the tiles that are entirely in the upper triangle (where all entries would be masked): for a Q tile of rows `[r0, r1]` and a K tile of columns `[c0, c1]`, if `c0 > r1` (the K tile is entirely "in the future" of the Q tile), the kernel skips it — no computation. Only tiles on or below the diagonal are processed, with the diagonal tiles applying the partial causal mask within them. This roughly **halves** the prefill attention compute (skipping the upper triangle), a significant saving since prefill is compute-bound and attention is a meaningful fraction of it. FlashAttention/FlashInfer implement this tile-skipping; it's why the realized prefill attention FLOPs are about half the dense `n²` (only the lower triangle is computed). For the chunked-prefill prefix attention (§35), the cached-prefix portion is *not* causally masked (all prefix tokens precede the new queries), so those tiles are fully computed, while the new-chunk portion is causal (tile-skipped) — the kernel handles the mixed mask correctly. This causal-skip optimization, combined with the IO reduction (§13.1), is why FlashAttention prefill is both compute-efficient (skip the upper triangle) and memory-efficient (no `n×n` materialization) — two distinct wins from the tiled formulation. It's a small example of how kernel-level awareness of the problem structure (the causal mask) translates directly to a ~2× compute saving that a structure-unaware kernel leaves on the table.

The broader lesson, closing the file: kernels that *understand the structure* of the computation — the causal mask (skip the upper triangle), GQA (read each KV head once), the sparsity of quantized weights (fuse dequant), the tree structure of speculation (tree mask), the raggedness of the batch (varlen) — beat structure-unaware kernels by large factors. The structure is in the algorithm and the model; realizing the benefit requires a kernel that exploits it. This is the deepest reason the kernel layer matters and the reason it co-evolves with the algorithms and architectures above it: each new structural idea (paged KV, low-rank MLA, tree speculation, MoE routing) needs a kernel that understands and exploits that structure to realize its theoretical benefit. The kernel is where structure becomes speed.

This principle also explains the field's trajectory: as new structural ideas emerge (the next attention variant, the next quantization format, the next speculative scheme), the kernel layer must follow, and there is often a lag between an algorithm's publication and a kernel that realizes its benefit at the roofline. MLA's native kernel (§17), FP4 GEMMs (§28), and tree-attention (§18) are all examples where the structural idea preceded the optimized kernel, and the engines' performance on those features improved as the kernels matured. For the inference engineer, this means that a new model or technique may underperform initially not because the idea is flawed but because its kernel is immature — and that performance will improve as the kernel catches up, often dramatically (as FlashInfer did for paged attention, or Marlin for W4A16). Tracking the kernel maturity for a given feature on given hardware is therefore part of capacity and performance planning: the same model can serve very differently depending on whether its attention variant, quantization, and the hardware's precision formats have mature, roofline-hitting kernels. The kernel layer is not static — it is where much of the field's ongoing performance improvement happens, generation after generation, structural idea after structural idea, each new kernel closing the gap between an optimization's theoretical promise and its realized speed. In that sense the kernel layer is the field's perpetual frontier — the algorithms and architectures set the agenda, and the kernels, generation after generation, deliver on it, which is why an inference engineer who can read a roofline, profile a kernel, and reason about coalescing, occupancy, and fusion holds a skill that stays valuable across every hardware generation and every new model architecture, long after any specific kernel is superseded. Those reasoning skills — not memorized kernel names — are the durable takeaway of this file, just as the cost models, not memorized flags, were the takeaway of the architecture and scheduling files: the specifics churn, the principles endure, and the engineer who holds the principles can re-derive the specifics for whatever comes next. That is the through-line of this entire database, and the kernel layer — closest to the silicon, furthest from the abstractions — is where it is most concretely true — every microsecond traceable to a kernel, every kernel to a roofline, every roofline to the immutable physics of memory bandwidth and arithmetic intensity that this database began with in File 01. The kernels change; the physics does not — and that is precisely why the roofline remains the inference engineer's most trustworthy guide across every kernel, every architecture, and every hardware generation this database describes — and across whatever new ones the field invents next, since the physics it encodes does not change even as the silicon and the software around it relentlessly do.









