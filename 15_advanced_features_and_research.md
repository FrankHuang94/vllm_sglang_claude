# Advanced Inference Techniques — Disaggregation, MoE Optimization, and Research Frontiers

> **PRIMARY reference file.** This file covers the research frontier and advanced production techniques: prefill–decode disaggregation (Mooncake, DistServe), MoE inference optimization (routing, expert parallelism, DeepSeek-V3), continuous-batching enhancements (Sarathi, FastServe), KV cache compression and reuse, MLA in depth, and inference for reasoning models. Prerequisites: File 04 (scheduling, PD disaggregation intro), File 05 (EP), File 02 §2.4 (MLA), File 03 §21 (KV transfer), File 12 (speculative decoding).

---

## Table of Contents

1. Prefill–Decode Disaggregation — Production Systems
2. The KV Transfer Problem
3. Disaggregation Economics and Heterogeneous Hardware
4. Mooncake and DistServe
5. MoE Inference Optimization
6. DeepSeek-V3 Inference Deep Dive
7. Continuous Batching Enhancements (Sarathi, FastServe)
8. Response Length Prediction
9. KV Cache Compression and Cross-Request Reuse
10. MLA (Multi-head Latent Attention) Deep Dive
11. Inference for Reasoning Models
12. The Research Frontier

---

## 1. Prefill–Decode Disaggregation — Production Systems

Prefill–decode disaggregation (introduced File 04 §10) separates the two phases onto distinct instances, motivated by their opposite characteristics: prefill is compute-bound (matrix-matrix), decode is memory-bandwidth-bound (matrix-vector) — File 01 §3.

### 1.1 The architecture

```
request → load balancer → PREFILL instance (compute prompt KV, emit first token)
                              │  transfer KV cache over interconnect
                              ▼
                          DECODE instance (autoregressive generation, stream output)
```

A new request's lifecycle: HTTP → load balancer → prefill instance (blocks until the prompt is processed and the first token emitted) → KV cache transferred to a decode instance → decode instance generates and streams the remaining tokens. The prefill pool and decode pool are separately sized, separately optimized, and independently scalable.

### 1.2 Why separate the phases

Three problems with co-locating prefill and decode in one engine (File 04 §10.1):
- **Head-of-line blocking:** a long prefill delays decodes in the same batch, spiking TPOT (partly fixed by chunked prefill, File 04 §7, but disaggregation eliminates it entirely — decodes never share a batch with prefills).
- **Hardware mismatch:** the ideal prefill GPU (high FLOP/s) differs from the ideal decode GPU (high HBM bandwidth/capacity) — §3.
- **Independent scaling:** the prefill:decode workload ratio varies (chat is decode-heavy, summarization prefill-heavy); disaggregation lets each pool scale to match (§4, DistServe).

Disaggregation is the architectural response to the prefill/decode duality (File 01 §3) — instead of mixing the two regimes in one engine (and managing their interference with chunked prefill and scheduling), physically separate them so each runs in its optimal configuration on its optimal hardware.

---

## 2. The KV Transfer Problem

The crux of disaggregation is transferring the prompt's KV cache from the prefill instance to the decode instance (File 03 §21).

### 2.1 Transfer mechanisms and bandwidth

- **NVLink** (intra-node): ~600–900 GB/s — fastest, for prefill and decode on the same node.
- **PCIe P2P** (between GPUs): ~64 GB/s.
- **RDMA over InfiniBand** (inter-node): ~50 GB/s at 400 Gbps — for cross-node disaggregation, the typical large-scale case.
- **TCP** (baseline): slow, avoid.

### 2.2 The amortization condition

The transfer is worthwhile when its time is small relative to the prefill compute it offloads. Worked (File 04 §10.3): LLaMA-3 70B, 1,000-token prompt → KV ≈ `1000 × 320 KB = 320 MB`. Over 50 GB/s RDMA: ~6.4 ms transfer. The prefill compute for 1,000 tokens is ~1 s (File 01 §15.1 scaled). So transfer (6.4 ms) ≪ prefill (1 s) — easily amortized. The condition: `KV_transfer_time ≪ prefill_compute_time`, which holds for typical prompts because the KV (linear in tokens) transfers fast relative to the prefill (which is heavy compute). For very short prompts (little prefill to amortize) or very slow interconnects, disaggregation's overhead may not pay off — but for typical prompts on RDMA/NVLink, it does.

### 2.3 KV transfer and the block abstraction

The transfer is clean because of PagedAttention's block abstraction (File 03 §21, §28): KV is already chunked into transferable, independently-addressable blocks, so "transfer a sequence's KV" is transferring its list of blocks and rebuilding the block table on the decode side. This is the same movable-block design that enables CPU swap (File 03 §8) and external KV stores (File 03 §28) — disaggregation is KV transfer over the network rather than to local CPU. vLLM's `--kv-transfer-config` and connector framework (with backends like LMCache) make the transfer mechanism pluggable (File 04 §10.5).

---

## 3. Disaggregation Economics and Heterogeneous Hardware

The deepest motivation for disaggregation is **cost**, via heterogeneous hardware (File 05 §36).

### 3.1 Different hardware for different phases

- **Prefill (compute-bound):** wants high FLOP/s. H100/B200 with FP8 (high compute) are ideal. Large batches of long prompts maximize the compute utilization.
- **Decode (memory-bandwidth-bound):** wants high HBM bandwidth and capacity, not necessarily high FLOP/s. High-bandwidth GPUs (MI300X's 5.3 TB/s, H200's 4.8 TB/s) or *more numerous cheaper* GPUs (A10G, L40S) can serve decode cost-effectively — the decode doesn't need the expensive compute.

### 3.2 The cost reduction

By running prefill on expensive compute-optimized GPUs (only for the prefill phase) and decode on cheaper bandwidth-optimized (or more numerous cheaper) GPUs (for the long decode phase that dominates token generation), disaggregation can cut cost **2–3× at equal throughput** (File 04 §10.3, File 05 §36). The expensive compute GPUs are used only where compute matters (prefill); the decode runs on hardware matched to its bandwidth-bound nature. This is the economic case: not just avoiding interference, but spending hardware dollars where each phase needs them.

### 3.3 Independent scaling and the P:D ratio

The prefill and decode pools scale independently to match the workload's prefill:decode ratio (File 04 §30). DistServe's analysis (§4) sets the optimal ratio: ~1:1 for prompt-heavy (summarization), heavily decode-weighted (1:4 to 1:8) for output-heavy (chat). Mis-provisioning (equal pools for a decode-heavy workload) wastes the prefill pool. The ratio is workload-specific and derived from the relative prefill and decode instance-time per request (File 04 §30's worked example).

---

## 4. Mooncake and DistServe

### 4.1 DistServe (arXiv 2401.09670)

DistServe provides the *analytical model* for disaggregation. It treats prefill and decode as separate queuing systems (M/M/1-like queues), and optimizes the prefill:decode worker allocation to maximize **goodput** (requests meeting SLO, File 11 §2). Key results: disaggregation eliminates the prefill-decode interference (no head-of-line blocking), and the optimal P:D ratio depends on the workload's prefill-to-decode time ratio. DistServe quantifies when disaggregation beats co-location (when the interference cost exceeds the transfer cost) and how to provision the pools. It's the theoretical foundation that justifies and sizes disaggregated deployments.

### 4.2 Mooncake (Kimi.ai)

Mooncake is a *production* disaggregation system (deployed for Kimi.ai's serving). Its features:
- **P2P KV transfer over RDMA** — the prefill instance's KV is transferred to the decode instance via RDMA (§2).
- **Global KV cache manager** — tracks which worker holds which sequence's KV, enabling KV migration between workers and a distributed view of the cache.
- **Prefix-cache-aware routing** — requests sharing a prefix route to the same prefill worker, concentrating that prefix's cache hits (File 04 §10.4, File 08 §35). This makes prefix caching effective in the disaggregated setting (each prefill worker has its own prefix cache).
- **KV-cache-centric design** — Mooncake treats the KV cache as the central resource (File 01 §4), with a dedicated KV cache pool and transfer infrastructure, reflecting that KV management is the heart of serving.

Mooncake demonstrates production disaggregation at scale, validating DistServe's analysis and showing the engineering (RDMA transfer, global KV management, prefix-aware routing) required to make it work. It's the reference production disaggregation system, as DeepSeek's code is the reference EP128 MoE system (§6).

### 4.3 vLLM and SGLang disaggregation

Both engines have moved toward production disaggregation (File 04 §10.5, File 09 §35). vLLM's `--kv-transfer-config` + connector framework (LMCache integration); SGLang's disaggregation support. The multi-process architectures (File 09 §35) ease this — the process boundaries generalize to the prefill/decode boundary. Production-grade disaggregation has been a major roadmap item for both, maturing through 2024–2025.

---

## 5. MoE Inference Optimization

Mixture-of-Experts inference (File 02 §4.2, §21; File 05 §5) has distinct optimization challenges, since MoE decouples total parameters (capacity, all in memory) from active parameters (compute per token).

### 5.1 Expert routing mechanics

```
router_logits = x @ W_router        # [num_experts]
(values, idx) = TopK(router_logits, k)    # select top-k experts
gate = softmax(values)              # normalize over selected
y = Σ_{e ∈ idx} gate_e · FFN_e(x)   # weighted combination
```

The router is a small linear layer; the top-k selection and gating are cheap; the expense is the `k` expert FFNs per token (File 02 §21.1). Some models (DeepSeek) add *shared experts* (always active) plus routed experts.

### 5.2 The grouped-GEMM and load imbalance

After routing, tokens are grouped by expert and processed via a **grouped GEMM** (File 02 §21.3, File 06 §6, File 10 §25): gather tokens per expert, pad to tensor-core multiples, run per-expert matmuls, scatter results. The data-dependent grouping causes **load imbalance** (File 02 §21.2): hot experts get more tokens (straggler, possible OOM), cold experts waste capacity. Padding to multiples wastes compute on imbalanced groups. This makes MoE step time noisier than dense (File 02 §FAQ). Mitigations: capacity factors (drop/defer overflow — quality cost), expert replication (hot experts on multiple GPUs), or relying on the model's training-time balancing (DeepSeek's auxiliary-loss-free balancing, §6).

### 5.3 Expert parallelism communication

Distributed MoE uses expert parallelism (File 05 §5): experts spread across GPUs, with two all-to-all per MoE layer (dispatch tokens to expert GPUs, combine results back). The all-to-all is bandwidth-bound (File 05 §5.2) — `batch · top_k · d_model · bytes` per dispatch. For large MoE (DeepSeek-V3, §6), this communication dominates unless overlapped with compute (DualPipe, File 05 §21). EP is the axis that spans nodes (fat InfiniBand) while attention/dense layers use TP within nodes (File 05 §8).

### 5.4 The memory challenge

MoE's defining inference challenge: *all* experts must be in VRAM even though only `k` activate per token (File 02 §4.2). Mixtral-8×7B has ~47B params in memory, ~13B active — you pay the full memory for the sparse compute. DeepSeek-V3 has 671B total, 37B active — 671B of weights in memory across the cluster. So MoE serving is memory-capacity-bound on the expert weights (needing many GPUs to hold them), while the per-token compute is modest (only k experts). This inverts the usual balance: dense models are often KV-memory-bound (File 01 §4), MoE models are weight-memory-bound (the experts dominate). Quantization (FP8 experts) and expert parallelism (spread experts across GPUs) address the weight memory.

---

## 6. DeepSeek-V3 Inference Deep Dive

DeepSeek-V3 is the reference large-MoE inference system, combining nearly every advanced technique. Specs: 671B total / 37B active, 256 routed experts + shared experts, top-8 routing, MLA attention (§10), FP8 throughout, MTP heads (File 02 §22).

### 6.1 The inference stack

- **MLA attention** (§10): the low-rank latent KV gives ~64× KV reduction vs MHA (File 02 §26), making DeepSeek's long context economical — the attention KV is tiny relative to a MHA model.
- **EP128** (File 05 §5.4, §29.2): 256 experts across 128 GPUs (2/GPU), top-8 dispatch via all-to-all over InfiniBand+NVLink.
- **DualPipe** (File 05 §21): overlaps the EP all-to-all with expert computation, hiding the communication so the MoE layer approaches compute-bound rather than communication-bound.
- **FP8 throughout** (File 02 §11.4): halves weight and activation bytes (easing the 671B weight memory and the all-to-all volume) and uses FP8 tensor cores. DeepSeek trains *and* serves in FP8.
- **MTP heads** (File 02 §22): built-in multi-token prediction heads serve as the speculative-decoding draft (File 12 §8), amortizing decode without a separate model.
- **Auxiliary-loss-free load balancing:** trained so routing is naturally balanced (§5.2), avoiding inference-time load imbalance.

### 6.2 Why it's the reference

DeepSeek-V3 demonstrates production large-MoE serving with every lever: MLA (KV reduction), EP128 (expert distribution), DualPipe (communication overlap), FP8 (precision), MTP (speculation), balanced routing. Its open-source inference code is the practical reference for EP128 with MLA and DualPipe (File 05 §29.2). It validates that trillion-scale-total MoE (with modest active params) can be served economically by combining the techniques — the cumulative payoff of the field's advances (architecture, parallelism, communication overlap, quantization, speculation) in one system. For an inference engineer, DeepSeek-V3 is the case study showing how the pieces (Files 02, 05, 12, this file) compose into a frontier deployment.

### 6.3 The topology

DeepSeek-V3 serving (File 05 §29.2): TP/DP-attention for the MLA attention (cheap, so DP-attention is efficient, File 09 §11), EP128 for the experts (across nodes, fat IB), DualPipe overlap, FP8 weights/activations/KV. The attention is cheap (MLA's tiny KV), so the design focuses on the experts (EP, all-to-all overlap) — the opposite balance from a dense model (where attention/FFN dominate). This is the MoE serving profile: experts dominate (memory and the all-to-all), attention is cheap (especially with MLA), and the engineering centers on the expert distribution and communication.

---

## 7. Continuous Batching Enhancements

Beyond basic continuous batching (File 04 §3), research has refined scheduling.

### 7.1 Sarathi and Sarathi-Serve

**Sarathi** (Agrawal et al., arXiv 2308.16369) introduced **chunked prefill + stall-free batching** (File 04 §17): bound each prefill chunk so decodes never stall, eliminating the prefill-induced TPOT spikes. **Sarathi-Serve** (arXiv 2403.02310) productionizes this with SLO awareness, reporting 2.6× P99 TTFT improvement vs an Orca baseline. The key insight: a stall-free schedule (every step makes decode progress) requires bounding prefill, which chunked prefill provides (File 04 §17.2). This is now standard (chunked prefill default-on, File 04 §7.5).

### 7.2 ORCA (recap)

ORCA (Yu et al., OSDI 2022, File 04 §25.1) introduced iteration-level scheduling and selective batching — the foundation of continuous batching. vLLM evolved from ORCA, adding PagedAttention memory and (via Sarathi) chunked prefill.

### 7.3 FastServe

FastServe (Wu et al., arXiv 2305.05920, File 04 §25.2) introduced preemptive size-aware scheduling (MLFQ approximating SRPT), reducing average job completion time ~1.4× by prioritizing short requests via iteration-level preemption (which PagedAttention makes cheap). It addresses head-of-line blocking at the request level (short requests behind long ones).

### 7.4 The cumulative scheduling stack

The scheduling research stacks (File 04 §25.3): static → iteration-level (ORCA) → memory-managed (PagedAttention) → stall-free (Sarathi) → size-aware (FastServe). Each addressed a specific failure of the previous. Modern engines integrate iteration-level scheduling + PagedAttention + chunked prefill, with size-aware policies (priority, length prediction §8) as options. The frontier continues — better length prediction, smarter preemption, disaggregation-aware scheduling — but the core (continuous batching + chunked prefill + paged memory) is settled.

---

## 8. Response Length Prediction

Many scheduling improvements (SJF/SRTF, File 04 §8.3; disaggregation P:D sizing, §3.3) need to know a request's output length, which is unknown at arrival. **Length prediction** estimates it:

- **A small model** (Bayesian regression, regression forest, or a tiny neural net) predicts the output length from the prompt — trained on request history (prompt → observed output length).
- **Online learning:** estimate length distributions from observed requests (per endpoint, per user), updating as traffic flows.
- **User hints:** `max_tokens` as an upper bound proxy.

Length prediction enables SRTF-style scheduling (run the shortest-remaining-time request first, reducing average completion time, File 04 §25.2) and better disaggregation provisioning. The prediction error degrades the policy gracefully toward FCFS (a wrong prediction just means suboptimal ordering, not incorrectness). Work by Zhong et al. and others explores length prediction for disaggregated and SRTF scheduling. It's an active area because even approximate length information enables meaningful scheduling improvements (~1.4× lower JCT, File 04 §25.2) — the value is in the *relative* ordering (short vs long), which even a noisy predictor captures. As reasoning models (highly variable output length, §11) grow, length prediction matters more (the variance is huge — 100 to 10,000 tokens).

---

## 9. KV Cache Compression and Cross-Request Reuse

Beyond prefix caching (File 03 §6, File 08), research explores broader KV reuse and compression.

### 9.1 Cross-request reuse beyond prefixes

Prefix caching reuses KV for *exact* shared prefixes. Frontier ideas:
- **RAG document reuse:** the same retrieved document appears in many queries — cache its KV (File 13 §12), reused across queries (with order canonicalization, File 08 §34).
- **Semantic/approximate reuse:** KVSharer (arXiv 2410.xxxxx) and similar explore sharing KV across *semantically similar* (not identical) prefixes — requires a KV-similarity metric and tolerance for approximation. Research-stage.

### 9.2 KV cache merging

For beam search (File 04 §12.3), beams that *reconverge* (produce the same tokens) can have their KV merged (deduplicated via reference counting) — saving memory when beams converge. Niche (beam search is uncommon, File 02 §8.4) but illustrates KV deduplication.

### 9.3 Dynamic sparse KV (eviction)

H2O, SnapKV (File 13 §5): prune low-attention KV entries during decoding, trading quality for memory. The eviction policy (cumulative attention score, recency) determines the quality/memory trade. These reduce KV at a quality cost — distinct from prefix caching (lossless reuse) and quantization (lossy compression of all KV).

### 9.4 Cross-layer KV sharing (CLA)

**CLA (Cross-Layer Attention, arXiv 2405.12981):** share KV between adjacent transformer layers (odd layers reuse even layers' KV), reducing KV cache ~2× at modest quality cost. This is an *architectural* KV reduction (like GQA/MLA, File 02 §2) — a learned sharing pattern across layers. Not yet in production inference systems but a promising direction (it composes with GQA/MLA — share across heads *and* layers). It extends the KV-reduction theme (File 13) to the layer dimension: GQA shares across heads, MLA compresses to a latent, CLA shares across layers — each attacking the `2 · num_layers · num_kv_heads · head_dim` KV formula (File 01 §4) from a different axis.

### 9.5 KV quantization (recap)

FP8/INT8 KV (File 02 §7, File 13 §7) compress all KV uniformly — `--kv-cache-dtype fp8_e5m2`, <0.5% quality, 2× reduction. The production-standard KV compression, composing with the architectural reductions (GQA/MLA) and prefix caching.

---

## 10. MLA (Multi-head Latent Attention) Deep Dive

MLA (DeepSeek-V2/V3, File 02 §2.4, File 06 §7) is the most aggressive KV reduction, worth a detailed treatment.

### 10.1 The architecture

MLA caches a low-rank **latent** vector per token instead of full K/V:
```
c_KV = x @ W_DKV          # down-project: d_model → d_c  (d_c ≪ num_heads · d_head)
K_C  = c_KV @ W_UK        # up-project to keys
V_C  = c_KV @ W_UV        # up-project to values
# Cache stores c_KV (size d_c), not K_C/V_C
```
For DeepSeek-V2: `d_c = 512`, vs MHA's `num_heads · d_head · 2 = 128 · 128 · 2 = 32768` per token → **~64× reduction** (File 02 §2.4). The cache holds the compact latent; K/V are reconstructed (or the projection absorbed, §10.3) at attention time.

### 10.2 RoPE decoupling

RoPE (File 02 §5.1) is position-dependent and cannot be absorbed into the static `W_UK`/`W_UV` up-projections (the rotation depends on position, the projection doesn't). MLA's solution: **decoupled RoPE** — a small separate set of RoPE-carrying key dimensions `K_R` (with reduced dimension `d_R`), computed with traditional per-layer RoPE, concatenated with the latent-derived `K_C` for attention. So the cache holds `c_KV` (the latent, no RoPE) + `K_R` (the small RoPE keys). This adds a little overhead but enables RoPE position encoding with the low-rank latent — the key trick that makes MLA work with RoPE.

### 10.3 Weight absorption at inference

A crucial inference optimization: the up-projections `W_UK`, `W_UV` can be **absorbed** into the query projection at inference time. Since attention computes `q · K_Cᵀ = q · (c_KV · W_UK)ᵀ = (q · W_UKᵀ) · c_KVᵀ`, you can precompute `W_Q_absorbed = W_Q · W_UKᵀ` and attend directly between the absorbed query and the latent `c_KV` — eliminating the explicit up-projection (one matmul) per attention layer (File 06 §7). This makes MLA inference efficient: attend on the compact latent with the absorbed query, no need to materialize full K/V. The absorption happens at model load.

### 10.4 The kernel challenge

MLA needs a kernel that attends on the latent (File 10 §17): either a *native MLA kernel* (attends directly on `c_KV` with the absorbed query, preserving the bandwidth benefit) or a *materialize* approach (up-project to full K/V, then standard attention — loses the attention-time bandwidth benefit, keeping only the cache-storage benefit). The native MLA kernel is the high-performance path and was a development frontier; vLLM (`DeepseekV2Attention`, File 06 §7) and FlashInfer added MLA support. The decoupled RoPE (§10.2) must be handled (concatenate `K_R` scores).

### 10.5 Why MLA matters

MLA's ~64× KV reduction makes long context economical (File 02 §26, File 13 §6) — DeepSeek can serve very long context with a tiny KV cache, where a MHA model would need 64× the memory. It's the most aggressive point on the KV-reduction spectrum (MHA → GQA 8× → MLA 64×, File 02 §2.6). MLA is the clearest example of **inference-aware architecture** (File 02 §22): designed by the DeepSeek team with serving cost (the KV cache) as the primary constraint, requiring engine-side support (latent caching, weight absorption, decoupled RoPE, native kernel) to realize. It exemplifies the co-design of model architecture and inference engine — the architecture chosen for serving efficiency, the engine built to exploit it (File 06 §7, File 10 §17).

---

## 11. Inference for Reasoning Models

Reasoning models (o1, DeepSeek-R1, QwQ-style) that produce long chains of thought are a growing, distinctive inference workload (File 11 §15).

### 11.1 The long-CoT profile

Reasoning models generate **1,000–10,000+ output tokens** (the "thinking") vs 100–500 for standard chat. This makes them **decode-dominated** (long outputs, File 11 §15): TPOT is critical (at 4,000 tokens, even 50 ms/token = 200 s), and the KV cache grows very large *within a single request* (10K tokens ≈ 3.2 GB KV for LLaMA-3 70B, File 11 §15). So reasoning inference is decode-bound with growing per-request KV — a profile demanding speculative decoding (decode amortization, File 12), FP8 KV (the growing KV), and KV-aware concurrency sizing (File 11 §15).

### 11.2 Budget forcing

Some systems limit the thinking length — a `max_thinking_tokens` budget. The trade-off: more thinking = better quality (more reasoning) but higher latency/cost; budget-forced = faster/cheaper but potentially lower quality. The inference engineer (or the application) sets this trade-off. Budget forcing is a reasoning-specific control balancing quality against the long-decode cost.

### 11.3 Parallel reasoning and best-of-N

- **Parallel reasoning (MCTS-style):** generate multiple reasoning paths, score them, select the best. Requires N parallel generations (or tree/speculative decoding). SGLang's `fork` (File 08 §7) is a natural fit — fork N reasoning branches sharing the prompt. Memory: N parallel chains = N × KV (for N=8 and 5,000-token reasoning: 8 × 1.6 GB = 12.8 GB per user, File 02 — substantial).
- **Best-of-N:** generate N complete responses, score each (length-normalized logprob or a reward model), return the best. Simple but N× compute. N=8 for math gives ~2–5% quality improvement at 8× cost. vLLM's `n=8` parallel sampling shares the prompt prefix via CoW (File 03 §5), so only the divergent generations cost extra.

### 11.4 The reasoning serving challenge

Reasoning models stress the decode path (long outputs → high decode load, growing KV) and benefit most from decode optimizations (speculation, FP8 KV, File 11 §15). The parallel-reasoning patterns (fork, best-of-N) multiply the KV (N chains) — RadixAttention/CoW share the common prefix (File 08 §7) but the divergent reasoning is N×. As reasoning models proliferate (File 20 §future), their decode-heavy, KV-growing, sometimes-parallel profile becomes a major workload — and serving them economically (speculation, FP8 KV, prefix-shared parallel reasoning, KV-aware sizing) is an increasingly important specialization. Reasoning is, in a sense, the extreme of the decode-bound regime, making decode optimizations (which were already valuable) essential.

---

## 12. The Research Frontier

Where inference research is heading:

### 12.1 Disaggregation at hyperscale

Disaggregation (§§1–4) is moving from research to production at every major provider. The frontier: dynamic P:D rebalancing (adjust the pool sizes to live traffic), KV-cache-centric architectures (Mooncake's global KV manager, §4.2), and heterogeneous hardware (prefill on compute GPUs, decode on bandwidth GPUs, §3). Disaggregation reorganizes serving around the prefill/decode duality (File 01 §3) and the KV cache (File 01 §4) as first-class concerns.

### 12.2 Hardware specialization

Beyond GPUs, specialized inference hardware (File 16): Groq's LPU (deterministic, SRAM-only, latency-optimized), Cerebras (wafer-scale, fast small-batch), Graphcore, Tenstorrent. These target inference specifically (vs GPUs' general-purpose compute), trading flexibility for efficiency on the inference workload. The trend: as inference dominates cost (File 01 §8), specialized inference accelerators proliferate, each with its own software stack — challenging the GPU/CUDA dominance.

### 12.3 Model compression for edge

Sub-1B models for edge/on-device inference (phones, laptops) — MLC-LLM, llama.cpp (File 17), quantization to 4-bit and below. The frontier: capable small models (distillation, better architectures) that run on-device, complementing cloud serving. Edge inference has different constraints (battery, memory, no datacenter) and a different stack (File 17).

### 12.4 Long context as commodity

128K context is becoming standard; 1M+ is the frontier (File 13). The techniques (GQA/MLA, FP8 KV, sequence parallelism, eviction) are bringing long context to commodity status. The frontier: efficient 1M+ context (ring attention, KV compression, hierarchical caching), making long-document and long-conversation workloads routine.

### 12.5 Agent-oriented inference

The biggest frontier (File 20 §future): agents that make many LLM calls (multi-step tool use, planning, reasoning) require structured output + long context + low latency *simultaneously*. This is exactly the multi-call, structured workload SGLang targets (RadixAttention + XGrammar + program-aware execution, File 08). Agent workloads stress prefix caching (shared context across steps), structured output (tool calls), long context (accumulated state), and low latency (interactive) — the union of the database's techniques. As LLM applications shift from single-shot completion to agentic, multi-step interaction, the inference stack must serve this efficiently — and the techniques (RadixAttention for the shared context, XGrammar for the tool calls, speculation for the decode, disaggregation for the cost) are the toolkit. Agent-oriented inference is where the field is heading, and it exercises everything in this database at once.

### 12.6 Inference-aware architecture co-design

The trend (File 02 §22, File 20 §co-design): models increasingly designed with serving cost as a constraint — GQA (KV reduction), MLA (further KV reduction), MTP (built-in speculation), quantization-friendly weights, attention-head counts divisible by TP degrees. Inference engineers increasingly influence architecture, and architects design for serveability. This co-design (the model and the engine evolving together) is a structural trend — the frontier models (DeepSeek's MLA+MTP+FP8) are designed *for* efficient serving, not just for quality.

---

## 13. DualPipe and Communication Overlap, Deeper

DeepSeek's DualPipe (§6, File 05 §21) is the reference for hiding MoE communication, worth detailed treatment because communication overlap is central to large-MoE serving.

The MoE layer's all-to-all (dispatch tokens to expert GPUs, combine results, File 05 §5.1) is bandwidth-bound and, for large EP (EP128), can dominate the layer's latency. DualPipe overlaps this communication with computation by pipelining across token chunks: while the dispatch all-to-all for chunk `i` is in flight (moving tokens over the network), the GPU computes the expert FFN for chunk `i-1` (whose tokens already arrived); meanwhile the combine all-to-all for chunk `i-2` completes. By keeping the network busy (all-to-all) and the compute busy (expert GEMM) simultaneously on different chunks, the wall-clock MoE-layer time approaches `max(communication, computation)` rather than `communication + computation` — effectively hiding the all-to-all behind the expert compute.

The requirement (File 05 §21): the interconnect bandwidth must be sufficient to complete each chunk's communication within the time the GPU spends computing the previous chunk's experts. If communication is *slower* than computation (insufficient bandwidth), overlap hides only part of it and MoE becomes communication-bound. This is why large-MoE serving needs fat interconnect (InfiniBand) *and* the overlap (DualPipe) — the bandwidth provides the capacity, the overlap hides it behind compute. DeepSeek-V3's EP128 with DualPipe is the proof that trillion-total-param MoE can be served with the communication hidden, the MoE layer running at near-compute-bound speed. The technique generalizes (File 05 §34): use separate CUDA streams to overlap the communication (NCCL all-to-all) with the compute (expert GEMM), the same overlap principle applied throughout the stack. DualPipe is the MoE-specific, large-scale instance of communication-computation overlap — the answer to "the all-to-all is on the critical path."

---

## 14. Disaggregation Cost Analysis, Worked

Quantify disaggregation's cost benefit (§3) for a decode-heavy chat workload (avg 500-token prompt, 500-token output).

**Co-located baseline (one engine, H100s):** the H100s do both prefill (compute-bound, uses the FP8 compute) and decode (memory-bound, uses the bandwidth). But decode dominates the time (500 output tokens × decode-step-time vs the one-shot prefill), so the expensive H100 compute (its FP8 tensor cores) sits mostly idle during the long decode — you're paying for compute you don't use during decode.

**Disaggregated:** prefill on H100s (use the FP8 compute for the compute-bound prefill), decode on cheaper high-bandwidth GPUs (e.g. more numerous L40S/A10G, or MI300X). The decode runs on hardware matched to its bandwidth-bound nature, not on expensive idle-compute H100s. If decode is 90% of the per-request time (long output) and the decode hardware is, say, half the $/throughput of H100 for the bandwidth-bound decode, the blended cost drops significantly — the 2–3× cost reduction (§3.2). The prefill H100s are used efficiently (compute-bound prefill keeps them busy); the decode GPUs are matched to decode (bandwidth, not idle compute).

The condition (§2.2): the KV transfer (prefill→decode) must be cheap relative to the work — true for typical prompts on RDMA/NVLink. The complexity cost: managing two pools, the transfer infrastructure, the routing (Mooncake's global KV manager). For large-scale, decode-heavy serving, the cost reduction justifies the complexity; for small-scale or prefill-heavy, co-location may be simpler and adequate. Disaggregation is a large-scale cost optimization, where the 2–3× savings on a large GPU bill justifies the engineering — which is why hyperscalers and large providers adopt it (§12.1, File 20).

---

## 15. MLA KV Reduction, Worked

Quantify MLA's (§10) memory benefit for a DeepSeek-V2-class model. MHA-equivalent KV per token per layer would be `2 (K,V) · num_heads · d_head · bytes`. With, say, 128 heads × 128 d_head, BF16: `2 · 128 · 128 · 2 = 65,536 B = 64 KB/token/layer`. MLA caches the latent `c_KV` (d_c = 512) + decoupled RoPE keys (`d_R`, small, say 64): `(512 + 64) · 2 ≈ 1,152 B ≈ 1.1 KB/token/layer` — a **~57× reduction** per layer. Across 60 layers, MHA-equivalent would be `64 KB · 60 = 3.84 MB/token`; MLA is `1.1 KB · 60 ≈ 66 KB/token` — the ~64× reduction (File 02 §26).

The consequence: at 128K context, an MHA-equivalent model would need `131072 · 3.84 MB ≈ 503 GB` of KV for one request (impossible); MLA needs `131072 · 66 KB ≈ 8.6 GB` — fitting comfortably. This is why DeepSeek can serve very long context economically — MLA shrinks the KV from prohibitive to manageable. The cost is the up-projection (absorbed at inference, §10.3) and the decoupled RoPE (§10.2). MLA is the extreme of KV reduction, and its worked numbers show why it's transformative for long-context MoE serving: the KV, normally the binding constraint (File 01 §4), becomes a minor cost. Combined with FP8 KV (another 2×), MLA brings even 1M-context KV into single-GPU range — the architectural enabler of DeepSeek's long-context capability.

---

## 16. EP All-to-All Overlap, Worked

Quantify the DualPipe overlap (§13) for DeepSeek-V3-scale EP. Per MoE layer, the dispatch all-to-all moves `batch · top_k · d_model · bytes`; for batch 1024, top-8 dispatch, d_model 7168, FP8 (1 byte): `1024 · 8 · 7168 · 1 ≈ 59 MB` total dispatch (File 05 §29.2), spread across 128 GPUs and overlapped. Per-GPU all-to-all time ≈ (its share) / bandwidth; at ~50 GB/s effective per GPU, the communication per all-to-all is hundreds of µs (File 02 §15.4: ~294 µs for a similar config), ~588 µs per layer (two all-to-all).

The expert compute per layer: each token runs 8 experts' FFN (top-8); for 1024 tokens × 8 × the FFN FLOPs, on the GPUs' FP8 compute, also hundreds of µs. **If the compute time ≳ the communication time, DualPipe hides the communication** (compute chunk `i-1` while dispatching chunk `i`), so the layer runs at ~the compute time. If communication > compute (insufficient bandwidth), the layer is communication-bound and DualPipe hides only part. The design balances chunk sizes and pipeline depth so compute and communication overlap maximally. DeepSeek's reported result: with DualPipe, the all-to-all is almost entirely hidden, so EP128 runs at near-compute-bound speed despite the massive communication. Without overlap, the ~588 µs/layer of communication × many layers would dominate; with it, the MoE serving is efficient. This worked analysis shows the overlap's necessity at scale — the communication is large (59 MB/layer), and only hiding it behind compute keeps EP128 viable.

---

## 17. Reasoning Model Serving, Worked

Quantify the reasoning serving challenge (§11) for a 70B reasoning model generating 8,000-token chains.

- **KV growth:** an 8,000-token reasoning chain accumulates `8000 × 320 KB (BF16) ≈ 2.56 GB` KV per request (or 1.28 GB FP8). For 32 concurrent reasoning requests: 32 × 2.56 GB = 82 GB — exceeding one GPU. So reasoning is KV-memory-bound on the *growing* per-request KV (the chains get long), and concurrency is limited (File 11 §15).
- **Decode load:** each request generates 8,000 tokens — 8,000 decode steps per request. At 32 concurrent, that's a sustained heavy decode load (the requests occupy decode slots for a long time, slow turnover). Decode throughput is the bottleneck.
- **The levers (File 11 §15):** speculative decoding (amortize the 8,000 decode steps — a ~2× decode speedup directly halves the generation time, File 12); FP8 KV (halve the 2.56 GB → fit more concurrent chains); KV-aware concurrency (size for the *peak* per-request KV at full chain length, or risk OOM mid-generation). 
- **Parallel reasoning:** if using best-of-N or MCTS (§11.3), N chains per user multiply the KV (N × 2.56 GB) — RadixAttention shares the prompt prefix (File 08 §7) but the N divergent chains are N×. For N=4: ~10 GB per user request.

Reasoning serving is thus the extreme decode-bound, KV-growing regime — speculation and FP8 KV are essential, concurrency is low (KV-bound by long chains), and parallel reasoning multiplies the cost. The worked numbers show why reasoning models are expensive to serve (long decode, large growing KV) and why the decode optimizations (speculation especially) are the key cost levers (File 11 §16, §38). As reasoning models become prevalent (§12.5, File 20), this serving profile — and its optimization — becomes increasingly important.

---

## 18. Speculative Decoding for MoE

Speculative decoding (File 12) for MoE models (§5) has specific considerations (File 12 §8):

- **The draft for a MoE target:** a small dense model approximating the MoE, or (DeepSeek's approach) built-in MTP heads (File 02 §22, §6.1) that integrate speculation into the MoE architecture. MTP avoids a separate draft and "knows" the MoE structure.
- **Acceptance variability:** MoE's data-dependent routing (different tokens activate different experts) can make the acceptance rate vary by the expert combination — the draft's approximation quality may depend on which experts a token would use.
- **Verification cost:** verifying K+1 positions through a MoE target involves the routing and (under EP) the all-to-all per position (File 05 §5) — so speculation interacts with the MoE communication. The amortization (one weight-read region for multiple tokens) still helps, but the per-position MoE communication adds cost.
- **Research:** SpecMoE and related work explore per-expert drafts and MoE-aware speculation. The principle (amortize the memory-bound target cost across verified tokens, File 12 §1) holds, but MoE's routing and EP communication complicate the cost model.

DeepSeek-V3's MTP (§6.1) is the production answer — built-in speculation that integrates with the MoE serving (EP, DualPipe, FP8), giving decode amortization without a separate draft model. For MoE serving, MTP-style built-in speculation is the clean approach, avoiding the separate-draft complications while providing the decode speedup that reasoning/long-output MoE workloads need.

---

## 19. Cross-Request KV Reuse, Worked

Quantify the broader KV reuse (§9) for a RAG workload. A RAG system retrieves documents per query; popular documents recur across many queries.

- **Document KV reuse (§9.1, File 13 §12):** a 4,000-token document, retrieved for 50 queries over an hour. Without reuse: 50 × 4,000 = 200,000 token-prefills for the document. With KV reuse (cache the document's KV, reuse across queries): 4,000 once + 50 × (query only) — a ~50× reduction in document prefill. The document is a large, recurring prefix — exactly what prefix caching (RadixAttention, File 08) captures, *if* the document KV stays cached between the queries (KV pool sized for the hot-document working set) and the document order is canonical (File 08 §34).
- **Hierarchical caching (File 03 §28, File 13 §3.2):** if there are more hot documents than fit in GPU KV, tier the document KV to CPU/SSD — a document hit swaps in (PCIe) rather than recomputing (prefill). The swap-vs-recompute trade (File 03 §9): for a 4,000-token document, swapping in ~1.3 GB KV over PCIe (~20 ms) vs recomputing the 4,000-token prefill (~hundreds of ms) — swapping wins for large documents, making hierarchical KV caching valuable for large-document-corpus RAG.
- **Semantic reuse (§9.1, research):** if two queries retrieve *similar* (not identical) documents, approximate KV reuse (KVSharer) could share — but this is research-stage and lossy (the KV is for different tokens). Exact reuse (same document) is the production lever; semantic reuse is the frontier.

This worked RAG case shows cross-request KV reuse beyond simple prefix caching: recurring documents (not just system prompts) are large reusable prefixes, and caching them (in GPU or tiered to CPU/SSD) is a major lever for document-heavy RAG. It connects the KV-reuse research (§9) with the practical RAG workload (File 13 §12, File 14 §6), showing the frontier (hierarchical/semantic KV reuse) extends the production technique (prefix caching) to broader reuse patterns.

---

## 20. Agent-Oriented Inference, Worked

Agents (§12.5, File 20 §future) are the frontier workload, exercising every technique. Trace a multi-step agent:

An agent answering a complex query makes, say, 10 LLM calls: a planning step, several tool-use steps (each: decide a tool call, get the result, incorporate it), and a synthesis step. Each call shares the accumulating context (the system prompt, the task, the prior steps' outputs).

- **Prefix caching is essential:** each step's prompt extends the prior context. RadixAttention (File 08) caches the accumulating context, so each step computes only its new tokens — without it, each of the 10 calls would recompute the growing context (the 10th call's context includes all 9 prior steps). For a 10-step agent with a context growing to ~5,000 tokens, prefix caching turns `O(10 × growing context)` into `O(10 × new tokens)` — a large saving (File 08 §17.1's extraction example generalizes).
- **Structured output is essential:** the tool-call steps emit structured calls (function name + JSON args). XGrammar (File 08 §11) guarantees valid tool calls at near-zero overhead — without it, malformed tool calls would break the agent (File 07 §12).
- **Low latency matters:** the agent's 10 sequential calls compound latency — each call's TTFT+generation adds up, so the user waits for all 10. Fast per-call latency (low TTFT via prefix caching, low TPOT via the decode optimizations) is critical for interactive agents.
- **Long context accumulates:** the agent's context grows with each step (prior outputs, tool results) — a long-context concern (File 13) for long agent runs.

So agent inference needs prefix caching (shared growing context) + structured output (tool calls) + low latency (sequential calls) + long context (accumulated state) *simultaneously* — the union of the database's techniques. This is exactly what SGLang's co-design (RadixAttention + XGrammar + program-aware execution, File 08) targets, and why agent-oriented inference is SGLang's sweet spot (File 08 §36). As agents become the dominant LLM application pattern (§12.5, File 20 §future), serving them efficiently — exploiting the shared context (prefix caching), the structured output (XGrammar), and the decode (speculation) — is the frontier challenge, and the techniques in this database compose to meet it. The agent workload is the integration test of the entire inference toolkit.

---

## 21. Composing the Advanced Techniques

The advanced techniques compose into frontier deployments (DeepSeek-V3, §6, is the exemplar):

- **MLA** (§10) shrinks the KV → long context economical, attention cheap.
- **EP + DualPipe** (§§5–6, §13, §16) distributes and overlaps the experts → trillion-total-param MoE viable.
- **FP8** (File 02 §11.4) shrinks weights/activations/KV → memory and communication eased.
- **MTP/speculation** (§§6, 18, File 12) amortizes decode → faster long-output generation.
- **Disaggregation** (§§1–4) separates prefill/decode onto ideal hardware → cost reduction.
- **Prefix caching** (File 08) reuses shared context → multi-call/agent efficiency.
- **Chunked prefill** (§7, File 04) → stall-free mixed-traffic latency.

Each addresses a different bottleneck (MLA → KV memory; EP → expert distribution; DualPipe → communication; FP8 → bytes; speculation → decode amortization; disaggregation → hardware matching; prefix caching → redundant compute; chunked prefill → latency interference), and they *stack* — DeepSeek-V3 uses MLA + EP128 + DualPipe + FP8 + MTP together (§6), each lever multiplying the others' benefit. This composition is the state of the art: a frontier model and serving stack designed together (inference-aware architecture, §12.6) with every technique applied. For the inference engineer, understanding how they compose (which addresses which bottleneck, how they stack) is what enables building or operating a frontier deployment — and it's the culmination of the database's content, the advanced techniques integrated into a coherent whole. The composition also shows the field's trajectory: each technique was a research advance (PagedAttention, RadixAttention, MLA, FA-3, speculation, disaggregation, EP, chunked prefill), and the frontier is their integration — the cumulative progress that drove inference cost down an order of magnitude (File 01 §8) and continues.

---

## 22. Hardware Specialization in Depth

The GPU's dominance in inference (File 16) faces challenges from specialized accelerators (§12.2), worth examining because they represent a frontier bet that inference deserves purpose-built silicon.

- **Groq LPU:** deterministic execution, SRAM-only (no DRAM), ~750 GB/s on-chip bandwidth, latency-optimized. Achieves very high single-stream token rates (200K+ tokens/sec for small models) by keeping everything in fast SRAM — but limited by SRAM capacity (small models, or many chips for large models). Latency-optimized, not throughput-optimized; proprietary compiler. The bet: inference (especially low-latency) benefits from deterministic, SRAM-centric hardware over GPUs' DRAM-centric design.
- **Cerebras:** wafer-scale engine (a whole wafer as one chip), enormous on-chip memory and bandwidth, fast for small-batch (the wafer's bandwidth serves few sequences very fast). The bet: wafer-scale integration eliminates the chip-to-chip communication bottleneck.
- **Graphcore IPU, Tenstorrent:** other architectures targeting AI inference with different memory/compute balances.

These accelerators target the *inference* workload specifically — its memory-bandwidth-bound decode (File 01 §2), its latency sensitivity — rather than GPUs' general-purpose compute. The trade-off: efficiency on the inference workload vs the GPU/CUDA ecosystem's flexibility and software maturity (vLLM/SGLang run on GPUs; specialized hardware needs its own stack). As inference dominates cost (File 01 §8), the economic incentive for specialized inference silicon grows, but the software-ecosystem moat (CUDA, the engines) is strong. The frontier: whether specialized accelerators can overcome the ecosystem advantage by being sufficiently more efficient — a bet several companies are making (File 20 §competitive dynamics). For the inference engineer, this means the hardware landscape may diversify beyond NVIDIA/AMD GPUs, each accelerator with its own stack and trade-offs (latency vs throughput, capacity vs bandwidth).

---

## 23. Edge and On-Device Inference

The opposite frontier from hyperscale: inference on phones, laptops, and embedded devices (§12.3, File 16, File 17).

- **Small capable models:** sub-1B to ~3B models (distilled, efficient architectures) that run on-device — the frontier is making them capable enough for real tasks. Distillation (train a small model to mimic a large one), better architectures, and aggressive quantization (4-bit and below) enable this.
- **The edge stack:** MLC-LLM (compile to any device, File 17), llama.cpp (CPU/Metal/edge, File 17), Apple MLX (Apple Silicon). Different from server engines (vLLM/SGLang) — edge prioritizes single-user latency, memory frugality, and battery, not multi-tenant throughput.
- **The constraints:** edge has no datacenter (battery, thermal, limited memory), so the techniques differ — extreme quantization (GGUF k-quants, File 02 §11.5), no continuous batching (single user), unified memory (Apple Silicon, File 16). Edge inference is a distinct regime with its own stack and trade-offs (File 17).
- **The motivation:** privacy (data stays on-device), latency (no network round-trip), cost (no cloud GPU), offline capability. As small models improve, on-device inference for many tasks (autocomplete, summarization, local assistants) becomes viable, complementing cloud serving (large models for hard tasks).

Edge inference is the frontier of model compression and efficient small-model serving — a different discipline from the server-class serving this database focuses on (vLLM/SGLang), but a growing one as small models improve and on-device LLM features proliferate. The two regimes (cloud server, edge device) coexist: cloud for large models and high throughput, edge for privacy/latency/offline with small models (File 17's framework comparison covers the edge stack).

---

## 24. Continuous Batching Enhancements, Deeper

Beyond Sarathi and FastServe (§7), the scheduling frontier continues:

- **Disaggregation-aware scheduling:** in a disaggregated setup (§§1–4), the prefill and decode schedulers coordinate — the prefill scheduler optimizes prefill batch throughput, the decode scheduler optimizes decode latency, and the routing (with prefix awareness, Mooncake §4.2) connects them. This splits the single scheduler (File 04) into coordinated prefill and decode schedulers.
- **Length-prediction-driven scheduling:** with response-length prediction (§8), SRTF-style scheduling (File 04 §8.3) prioritizes short requests, reducing average completion time. As prediction improves, this becomes more effective.
- **SLO-aware scheduling:** scheduling to *meet SLOs* (not just maximize throughput) — Sarathi-Serve's direction (§7.1), bounding per-step work to meet the TPOT SLO, prioritizing requests near their latency deadline. Goodput-maximizing scheduling (File 11 §2) is the frontier — explicitly optimizing for SLO-meeting rate.
- **Fairness and multi-tenancy:** weighted fair queuing across tenants (File 04 §37), at the router or scheduler, for multi-tenant serving. As shared serving grows (SaaS, File 20), fair scheduling matters more.
- **Hardware-aware scheduling:** in heterogeneous deployments (different GPUs, §3), scheduling that routes requests to the appropriate hardware (compute-heavy prefill to compute GPUs, etc.).

The scheduling frontier is increasingly about *coordination* (disaggregated, multi-tenant, heterogeneous) and *SLO-awareness* (goodput, length prediction, deadline-aware) — moving beyond the single-engine continuous batching to cluster-level, SLO-driven, fairness-aware scheduling. The core (continuous batching + chunked prefill) is settled; the frontier is orchestrating it across pools, tenants, and hardware to maximize cluster-wide goodput under SLOs and fairness (File 04 §37, File 11 §2, File 19).

---

## 25. Inference-Aware Architecture Co-Design, Deeper

The co-design trend (§12.6, File 02 §22, File 20 §co-design) — models designed for serveability — is a structural shift worth examining.

Historically, model architecture was chosen for *quality* (and trainability), and inference engineers served whatever was trained. Increasingly, *serving cost* is a first-class architecture constraint, because inference dominates lifetime cost (File 01 §8). Examples of inference-driven architecture choices:

- **GQA** (File 02 §2.3): adopted across modern models specifically because it cuts the KV cache 8× with negligible quality loss — an inference-driven choice (the KV cache is an inference concern, not a training one).
- **MLA** (§10): DeepSeek designed MLA with the KV cache as the primary constraint — a ~64× KV reduction enabling economical long context. The architecture exists *because* of serving cost.
- **MTP** (File 02 §22): built-in multi-token prediction heads serve as the speculative-decoding draft — the architecture includes the speculation mechanism.
- **Quantization-friendly design:** training in FP8 (DeepSeek), or with weight distributions amenable to low-bit quantization — designing for the serving precision.
- **TP-divisible head counts** (File 05 §2.3): choosing attention-head counts divisible by common TP degrees, so the model parallelizes cleanly.

The result: frontier models (DeepSeek's MLA+MTP+FP8, the GQA-everywhere norm) are *designed for efficient serving*, not just quality. Inference engineers increasingly influence architecture (showing the KV-cache benefit of GQA, the cost of MHA), and architects design for serveability. This co-design is a feedback loop: serving constraints shape architecture, architecture shapes serving systems (engines add MLA support, MTP speculation), and the cycle continues. For the inference engineer, this means understanding architecture (File 02) is increasingly about understanding *serving* implications, and the most servable models are those designed with inference in mind. The co-design trend is why this database emphasizes the architecture-as-cost-model lens (File 02 §27) — the architecture *is* a set of serving decisions, increasingly made deliberately.

---

## 26. The 1M+ Context Frontier

Extending context to 1M+ tokens (§12.4, File 13) is a frontier combining several techniques:

- **MLA** (§10): the ~64× KV reduction makes 1M-context KV manageable (File 13 §10's budget — MLA brings 1M context into single-GPU-ish range).
- **Sequence parallelism** (File 05 §6, File 13 §11): ring/Ulysses attention spreads the 1M-token sequence across GPUs when it exceeds one GPU's memory/compute.
- **KV compression/eviction** (§9, File 13 §5): FP8 KV, H2O/SnapKV eviction reduce the 1M-token KV further.
- **Hierarchical KV caching** (File 03 §28, File 13 §3.2): tier the 1M-token KV across GPU/CPU/SSD.
- **Efficient long-context attention kernels** (File 10 §24): split-K decode fills the GPU for the low-batch-long-context regime; FlashAttention's linear memory enables the 1M prefill.

The combination is what makes 1M+ context feasible: shrink the KV (MLA, FP8, eviction), spread it (SP, hierarchical), and use efficient kernels (split-K, FlashAttention). The challenges are both memory (1M tokens of KV is enormous, even with reduction) and the `O(n²)` prefill attention (1M-token prefill is huge compute — chunked prefill and efficient kernels essential) and the per-step KV read at decode (1M-token KV read per step, File 13 §15 — ~hundreds of ms even with reduction). 1M+ context is the extreme of the long-context regime (File 13), pushing every KV-reduction and long-context technique to its limit. As applications demand longer context (whole codebases, long documents, extended conversations), 1M+ becomes a frontier target — and serving it economically requires the full long-context toolkit applied at scale (MLA + FP8 KV + SP + hierarchical caching + efficient kernels). The trajectory (File 13 §16) suggests 1M+ will follow 128K from frontier to commodity as the techniques mature and hardware (bigger HBM) grows.

---

## 27. A Frontier Deployment, Synthesized

Putting the advanced techniques together (§21), sketch a frontier deployment: serving DeepSeek-V3-class MoE for an agentic, long-context, reasoning workload at scale.

- **Model:** MLA attention (§10, tiny KV) + 256 experts (top-8) + FP8 + MTP heads.
- **Topology:** TP/DP-attention for the cheap MLA attention (File 09 §11), EP128 for the experts across nodes (File 05 §5.4), DualPipe overlapping the all-to-all (§13, §16).
- **Memory:** FP8 weights (ease the 671B), FP8 KV (ease the growing reasoning KV), MLA (tiny KV anyway).
- **Decode:** MTP-based speculation (§6.1, §18) amortizes the long reasoning decode (File 12).
- **Disaggregation** (§§1–4): prefill on compute GPUs, decode on bandwidth GPUs, KV transferred via RDMA — cost reduction for the decode-heavy reasoning workload.
- **Prefix caching** (RadixAttention, File 08): the agent's shared growing context reused across steps (§20).
- **Structured output** (XGrammar, File 08 §11): the agent's tool calls.
- **Chunked prefill** (§7): stall-free latency for the mixed agent traffic.
- **Cluster:** prefix-cache-aware routing (Mooncake §4.2, File 08 §35), SLO-aware scheduling (§24), proactive autoscaling (File 11 §10).

This deployment composes nearly every technique in the database — MLA, EP, DualPipe, FP8, MTP/speculation, disaggregation, prefix caching, XGrammar, chunked prefill, cache-aware routing — each addressing a specific bottleneck, stacking into a system that serves a trillion-total-param MoE for a demanding agentic-reasoning workload economically. It's the frontier integration, and it shows where the field is: the individual techniques (the research advances) composed into production systems for the emerging workloads (agents, reasoning, long context). An inference engineer building or operating such a system needs the whole database — the architecture (File 02), the memory (File 03), the scheduling (File 04), the distribution (File 05), the kernels (File 10), the SGLang techniques (Files 08–09), speculation (File 12), long context (File 13), and these advanced techniques — integrated. This synthesis is the destination of the database: not the techniques in isolation, but their composition into frontier deployments for the workloads the field is moving toward.

---

## 28. Cross-Layer Attention and Architectural KV Reduction Frontiers

Beyond GQA and MLA (File 02 §2), the architectural KV-reduction frontier continues:

- **Cross-Layer Attention (CLA, arXiv 2405.12981, §9.4):** share KV between adjacent layers (e.g. odd layers reuse even layers' KV), reducing the KV ~2× by halving the effective `num_layers` in the KV formula (File 01 §4). It composes with GQA/MLA (share across heads/compress to latent *and* across layers). A learned sharing pattern (trained to share specific layers' KV), at modest quality cost. Not yet in production but promising.
- **Layer-wise KV reduction:** other schemes share or compress KV across layers, or use fewer KV-cache layers (some layers attend only locally, not needing full KV).
- **The KV formula attack surface:** the KV cache is `2 · num_layers · num_kv_heads · head_dim · bytes` (File 01 §4). Each factor is a reduction target: `num_kv_heads` (GQA, MQA), the whole `num_heads · head_dim` (MLA's latent), `num_layers` (CLA, cross-layer sharing), `bytes` (quantization), and the sequence length (eviction, windows). The architectural frontier attacks each factor, and the techniques compose — a model with GQA + MLA-style compression + CLA + FP8 KV would multiply the reductions. The trend (File 02 §22, §25) is models designed to minimize this formula, since the KV cache is the inference bottleneck (File 01 §4).

This is the architectural complement to the systems techniques (paging, prefix caching, offloading): reduce the KV at the *architecture* level (fewer heads, latent compression, cross-layer sharing, lower precision) so there's less KV to manage. The two work together — a KV-efficient architecture (GQA/MLA/CLA) plus efficient KV management (paging, caching, quantization) — to make the KV cache, the central resource (File 01 §4), as small and well-managed as possible. The architectural frontier (MLA, CLA, and beyond) is where the biggest KV reductions come from, since they attack the formula directly rather than managing a large KV efficiently.

---

## 29. The KV Cache as the Unifying Theme

Stepping back across the database, the **KV cache** is the unifying thread, and the advanced techniques are largely about managing it (File 01 §4):

- **Reduce its size:** GQA/MQA (fewer heads), MLA (latent compression, §10), CLA (cross-layer, §28), quantization (FP8/INT8 KV, §9.5) — architectural and precision reductions.
- **Reuse it:** prefix caching (RadixAttention/APC, File 08, File 03 §6), cross-request reuse (RAG documents, §9.1, §19), CoW (parallel sampling, File 03 §5).
- **Place it well:** paging (File 03), CPU/SSD offloading (File 03 §8, §28), hierarchical caching (§19, File 13 §3.2), disaggregation transfer (§2).
- **Evict it:** H2O/SnapKV (§9.3, File 13 §5), sliding windows + sinks (File 02 §29, File 13 §2).
- **Spread it:** TP head-sharding (File 05 §9), sequence parallelism (File 05 §6, §26).

Nearly every advanced technique touches the KV cache because the KV cache is the binding resource of LLM inference (File 01 §4) — it dominates memory (capping concurrency), its read dominates decode bandwidth (capping decode speed), and its reuse/transfer enables multi-call and disaggregated serving. The field's progress is, in large part, the progressive mastery of the KV cache: from naive contiguous allocation (pre-2023, <40% utilization) → PagedAttention (>95% utilization) → RadixAttention (cross-request reuse) → MLA (architectural compression) → disaggregated/hierarchical KV (placement) → eviction/compression (reduction). The KV cache, introduced as "the central resource" in File 01 §4, is the protagonist of the database — and the advanced techniques in this file are the latest chapters in its management. An inference engineer who deeply understands the KV cache — its size, its growth, its reuse, its placement, its reduction — understands the heart of LLM inference, because nearly every optimization is, at bottom, about managing this one resource better.

---

## 30. What's Settled vs the Frontier

Distinguishing the settled techniques (production-standard) from the frontier (research/emerging):

**Settled (production-standard, Files 03–11):**
- PagedAttention / paged KV (File 03).
- Continuous batching + chunked prefill (File 04, §7).
- Tensor parallelism (File 05).
- FlashAttention/FlashInfer (File 10).
- Quantization (FP8, W4A16) (File 06).
- Prefix caching (APC/RadixAttention) (File 08).
- CUDA graphs (File 09).
- GQA (architectural) (File 02).
- Speculative decoding (File 12).

**Frontier (emerging/maturing, this file):**
- Prefill-decode disaggregation (§§1–4) — maturing to production.
- Large-scale MoE (EP128 + DualPipe) (§§5–6) — DeepSeek production, broader adoption emerging.
- MLA (§10) — DeepSeek production, spreading.
- Heterogeneous-hardware disaggregation (§3) — emerging.
- Cross-layer attention, semantic KV reuse (§§9, 28) — research.
- Hierarchical KV caching, external KV stores (§19, File 03 §28) — emerging.
- Specialized inference hardware (§22) — emerging, ecosystem-gated.
- Agent-oriented and reasoning-optimized serving (§§11, 20) — rapidly growing.

The settled techniques are the foundation every production engine implements; the frontier is where the next gains come from. The boundary moves — disaggregation, MLA, and large-MoE serving are moving from frontier to settled as they mature and adoption spreads. For the inference engineer, knowing the settled techniques is essential (they're in every deployment); knowing the frontier is forward-looking (where the field and the next cost reductions are heading). This file covers the frontier, building on the settled foundation of Files 03–11. The trajectory is clear: today's frontier (disaggregation, MLA, large MoE, agent serving) becomes tomorrow's standard, as PagedAttention and continuous batching (yesterday's frontier) became today's standard. The field advances by integrating the frontier into the foundation, and this file's techniques are the current edge of that ongoing integration.

---

## 31. Disaggregation Routing and Reliability

Production disaggregation (§§1–4) adds routing and reliability concerns beyond co-located serving:

- **Routing:** the load balancer routes new requests to prefill instances (prefix-cache-aware, Mooncake §4.2, so prefix-sharing requests hit the same prefill worker's cache) and assigns each request's decode to a decode instance. The routing must balance load across both pools while maintaining prefix-cache locality (File 08 §35) — a two-pool routing problem.
- **KV transfer reliability:** the KV transfer (§2) can fail (network issue) — the system must handle a failed transfer (retry, or fall back to recomputing the prefill on the decode instance). The global KV manager (Mooncake §4.2) tracks KV location for recovery.
- **Pool failures:** a prefill instance failure loses in-flight prefills (re-route to another prefill instance); a decode instance failure loses in-flight decodes (the requests' KV may be recoverable from the global manager, or must restart). The pools have independent failure domains, and the system routes around failures (File 19).
- **Pool imbalance:** if the prefill pool saturates (prefill-heavy burst) while decode is idle (or vice versa), the fixed P:D ratio (§3.3) is suboptimal — dynamic rebalancing (move instances between pools) or over-provisioning the bottleneck pool addresses this. DistServe's analysis (§4.1) sizes the pools; dynamic rebalancing (§12.1) adapts to live traffic.

Disaggregation's operational complexity (two pools, transfer, routing, rebalancing) is the cost of its benefits (no interference, hardware matching, independent scaling, §§1–3). For large-scale, decode-heavy serving, the benefits justify the complexity; the engineering (Mooncake's global KV manager, prefix-aware routing, dynamic rebalancing) is what makes production disaggregation work. This is why disaggregation is a hyperscale/large-provider technique (§14, File 20) — the complexity pays off at scale, where the 2–3× cost reduction on a large GPU fleet is substantial. Smaller deployments often use co-located serving (simpler) with chunked prefill (File 04 §7) to manage the prefill-decode interference, reserving disaggregation for when the scale justifies its complexity.

---

## 32. Key Takeaways

1. **Prefill-decode disaggregation** (§§1–4) separates the compute-bound and bandwidth-bound phases onto ideal, independently-scaled, heterogeneous hardware — eliminating interference and cutting cost 2–3× at scale (DistServe analysis, Mooncake production).
2. **The KV transfer** (§2) is cheap relative to prefill (amortizable) thanks to the block abstraction; it's the same movable-block design as swap and external stores (File 03).
3. **Large-MoE inference** (§§5–6) is weight-memory-bound (all experts in memory), needing expert parallelism (distribute experts) + DualPipe (overlap the all-to-all, §13, §16) + FP8 — DeepSeek-V3's EP128 is the reference.
4. **MLA** (§10) achieves ~64× KV reduction via low-rank latent caching + weight absorption + decoupled RoPE — the most aggressive KV reduction, enabling economical long-context MoE.
5. **Scheduling enhancements** (§7): Sarathi's stall-free chunked prefill (now standard), FastServe's size-aware preemption, and the frontier of SLO-aware, disaggregation-aware, length-prediction-driven scheduling.
6. **KV reuse and reduction** (§§9, 28): prefix caching, cross-request/document reuse, eviction (H2O), cross-layer sharing (CLA), quantization — all managing the KV cache, the central resource (§29).
7. **Reasoning models** (§11) are decode-dominated with growing KV — speculation and FP8 KV are essential; the emerging dominant workload.
8. **The frontier** (§§12, 22–26, 30): disaggregation at hyperscale, hardware specialization, edge inference, 1M+ context, agent-oriented serving, and inference-aware architecture co-design — where the field is heading.
9. **The techniques compose** (§21, §27): frontier deployments (DeepSeek-V3-class) stack MLA + EP + DualPipe + FP8 + MTP + disaggregation + prefix caching + chunked prefill, each addressing a bottleneck, multiplying the benefit.

The advanced techniques are the current edge of LLM inference, building on the settled foundation (Files 03–11) and pointing toward the workloads the field is moving to (agents, reasoning, long context, large MoE). They share the database's themes: managing the KV cache (§29), exploiting the memory-bound nature of decode (speculation, quantization), overlapping communication with compute (DualPipe), and composing techniques into integrated systems. The KV cache remains the protagonist; the prefill/decode duality remains the organizing tension; the roofline remains the analytical guide. The frontier is these enduring principles applied to ever-larger models (trillion-param MoE), longer context (1M+), and more complex workloads (agents) — the same physics (memory bandwidth, File 01 §2) and the same central resource (KV cache, File 01 §4), engineered ever more cleverly.

---

## 33. Appendix: Source Pointers

- **Disaggregation:** DistServe (arXiv 2401.09670), Mooncake (Kimi.ai); vLLM `--kv-transfer-config` + connectors (LMCache); SGLang disaggregation.
- **MoE/MLA:** DeepSeek-V2/V3 technical reports (MLA, EP128, DualPipe, MTP, FP8); vLLM `vllm/model_executor/models/deepseek_v2.py`; the DeepSeek open-source inference code.
- **Scheduling:** Sarathi (arXiv 2308.16369), Sarathi-Serve (arXiv 2403.02310), FastServe (arXiv 2305.05920), ORCA (OSDI 2022).
- **KV reuse/reduction:** H2O (arXiv 2306.14048), SnapKV (arXiv 2404.14469), CLA (arXiv 2405.12981), KIVI (arXiv 2402.02750); prefix caching (Files 03, 08).
- **Speculation for MoE:** MTP (DeepSeek-V3), SpecMoE and related.
- **Reasoning:** o1/R1/QwQ-style models; parallel reasoning, best-of-N, budget forcing (§11).

These primary sources document the frontier techniques; the engines' codebases (vLLM, SGLang, DeepSeek's inference code) show their production implementation. The trajectory from these papers to production engines (PagedAttention SOSP 2023 → vLLM; RadixAttention → SGLang; MLA → DeepSeek serving; Sarathi → chunked prefill default) illustrates how research advances become standard techniques — the integration of frontier into foundation (§30) that drives the field's continued progress and cost reduction (File 01 §8). This file's frontier techniques are the current iteration of that process, and tracking them (through the papers and the engines' development) is how an inference engineer stays at the edge of a rapidly advancing field.

---

## 34. Closing

The advanced techniques and research frontier extend the foundation (Files 01–11) toward the field's edge: disaggregation reorganizing serving around the prefill/decode duality and the KV cache; large-MoE inference (EP, DualPipe, MLA) serving trillion-total-param models; scheduling enhancements (Sarathi, SLO-aware) maximizing goodput; KV reuse/reduction/compression managing the central resource; and the emerging workloads (reasoning, agents, 1M context) driving new optimization. The unifying themes persist — the KV cache as protagonist (§29), the prefill/decode duality as the organizing tension, the roofline as the guide, communication-computation overlap as the recurring technique, and the composition of techniques into integrated frontier systems (§21, §27). For the inference engineer, this file maps where the field is heading and how the advanced techniques compose into the frontier deployments serving the workloads of the near future. The settled foundation (Files 03–11) is what you must know to operate today's systems; the frontier (this file) is what you must track to build tomorrow's. Together they span the full discipline of LLM inference engineering — from the roofline physics (File 01) through the foundational systems (vLLM, SGLang) to the research edge — and the remaining files (16–20) cover the hardware, framework comparison, LoRA, operations, and business context that complete the practical picture. The frontier is always moving; the principles (memory bandwidth, the KV cache, the prefill/decode duality, overlap, composition) endure, and they are what let an engineer follow — and contribute to — the field's relentless advance.

---

## 35. Worked MoE Memory Budget

Quantify the MoE weight-memory challenge (§5.4) for DeepSeek-V3-class serving. 671B total parameters, FP8 (1 byte): `671 GB` of weights — far exceeding any single GPU (80–192 GB). This *requires* distribution: with EP128 (128 GPUs), `671 GB / 128 ≈ 5.2 GB` of expert weights per GPU (plus the shared/attention weights replicated or TP-sharded). So the 671B model is feasible only because EP spreads the experts across 128 GPUs — each holds ~2 experts (§6.1). The active parameters (37B) are what compute per token, but all 671B must be *resident* (the memory challenge, §5.4).

Contrast a dense 70B (FP8, 70 GB): fits on 1–2 GPUs. The MoE's 671B total needs ~10× the GPUs just to hold the weights, despite only 37B active (similar active compute to a 37B dense model). This is the MoE trade-off: more capacity (671B total → better quality) for more memory (all experts resident → more GPUs) at modest compute (37B active → fast per token). The economics (File 20): MoE gives high quality (large total params) at low per-token compute (few active), but high memory cost (all params resident). EP distributes the memory; FP8 halves it; MLA makes the attention KV tiny so the experts dominate. The worked budget shows why large-MoE serving is fundamentally a *distributed memory* problem (hold 671B across many GPUs) with *modest compute* (37B active) and *heavy communication* (EP all-to-all) — a distinct profile from dense serving (where weights fit on few GPUs and KV often binds). This profile dictates the MoE serving design (EP for memory, DualPipe for communication, MLA/FP8 for the rest, §6) — all following from the "all experts resident, few active" nature of MoE.

---

## 36. The Continuous-Batching Frontier in Detail

Expanding the scheduling frontier (§24), the cluster-level coordination that the next generation of serving needs:

In a large deployment with disaggregation (§§1–4), multiple replicas (File 05 §7), multi-tenancy (File 04 §37), and heterogeneous hardware (§3), scheduling is no longer one engine's continuous batching but a *cluster orchestration* problem: route requests across prefill pools, decode pools, and replicas (with prefix-cache locality, §31, File 08 §35), maintain per-tenant fairness (File 04 §37), match requests to appropriate hardware (compute-heavy to compute GPUs), meet SLOs (goodput-maximizing, File 11 §2), and rebalance pools to live traffic (§12.1). This is the frontier: the single-engine scheduler (File 04) generalizes to a cluster scheduler coordinating pools, replicas, tenants, and hardware.

The components: a *router* (prefix-cache-aware, KV-aware, tenant-fair, hardware-aware) above the engines (File 05 §7.2, File 08 §35); the engines' continuous-batching schedulers (File 04, File 09) within each; a *global KV manager* (Mooncake §4.2) tracking KV across the cluster; and *autoscaling* (File 11 §10) adjusting pool/replica counts. The frontier research (length prediction §8, SLO-aware scheduling §24, dynamic disaggregation rebalancing §12.1) feeds this cluster orchestration. As serving scales (hyperscale, File 20), the cluster scheduler — not the single-engine scheduler — becomes the locus of optimization, coordinating the whole fleet to maximize cluster-wide goodput per dollar under SLOs and fairness. This is where scheduling is heading: from optimizing one engine's batch to orchestrating a heterogeneous, disaggregated, multi-tenant fleet — a systems problem at the cluster level, building on the per-engine continuous batching (File 04) but operating a layer above it. The principles (continuous batching, prefix-cache locality, goodput, fairness) extend from the engine to the cluster, and the frontier is their cluster-level realization.

---

## 37. Memory-Hierarchy Inference and Offloading Frontiers

For serving models or contexts that exceed GPU memory, the offloading frontier (File 03 §28, File 13 §3) extends the memory hierarchy:

- **FlexGen** (Sheng et al., arXiv 2303.06865): systematic GPU+CPU+SSD offloading for single-GPU inference of large models — schedules weight and KV movement across the hierarchy to run a model that doesn't fit. Very slow (SSD/PCIe bandwidth), for throughput-oriented offline inference (batch processing where latency doesn't matter), not interactive serving. It demonstrated that the memory hierarchy can extend inference beyond GPU capacity at a throughput cost.
- **External KV stores** (LMCache, File 03 §28): KV cached in CPU/SSD/distributed storage, keyed by prefix, surviving across requests and machines — extending prefix caching cluster-wide and across restarts. A frontier for prefix-heavy workloads (RAG, shared prompts) where the prefix working set exceeds GPU KV.
- **Tiered KV** (§19, File 13 §3.2): GPU → CPU → SSD → network hierarchy, hottest KV in HBM, colder tiered down — trading access latency for capacity. Enables holding a large KV working set (many hot documents, long contexts) beyond GPU memory.

The unifying idea: treat memory as a *hierarchy* (HBM → CPU → SSD → network), placing the hottest data (active KV, weights) in the fast tiers and colder data (cached prefixes, swapped KV) in slower tiers — the OS memory-hierarchy principle (File 03 §29) applied to inference. The frontier is making this hierarchy efficient (fast swap-in, smart tiering, prefetching) so the capacity benefit (hold more than GPU memory) doesn't cost too much latency. For offline/throughput workloads (FlexGen) or prefix-heavy workloads (external KV stores), the hierarchy extends inference's reach; for latency-critical serving, the hierarchy must be used carefully (only for cold data, with fast swap-in) to avoid the latency cost. This offloading frontier complements the reduction techniques (§§9, 28): reduce the KV/weights *and* tier what remains across the hierarchy, together extending inference to larger models and contexts than GPU memory alone allows.

---

## 38. Final Synthesis

This file has traversed the advanced and frontier techniques of LLM inference: disaggregation (separating the prefill/decode phases onto ideal hardware), large-MoE serving (EP, DualPipe, MLA for trillion-total-param models), scheduling enhancements (stall-free, SLO-aware, cluster-level), KV reuse/reduction/compression (managing the central resource), and the emerging workloads (reasoning, agents, 1M context) driving the next optimizations. The throughlines are the database's enduring principles: the **KV cache** as the central resource that nearly every technique manages (§29); the **prefill/decode duality** that disaggregation reorganizes around; the **roofline** that explains why decode is memory-bound and how to beat it (reduce bytes, amortize reads, overlap communication); and the **composition** of techniques into integrated frontier systems (§21, §27). The frontier (§30) is the integration of these advances into production — disaggregation, MLA, and large-MoE moving from research to standard, as PagedAttention and continuous batching did before. For the inference engineer, the advanced techniques are the current edge: build on the settled foundation (Files 03–11), track the frontier (this file), and understand how they compose for the workloads the field is moving toward (agents, reasoning, long-context, large MoE). The discipline is the same throughout — reason from the roofline and the KV cache, measure, compose the techniques that address the workload's bottlenecks — applied at the frontier's scale and complexity. The field advances relentlessly, but the principles endure, and mastery of them (the goal of this entire database) is what lets an engineer serve any model, on any hardware, for any workload, at the frontier of what's possible — economically, within SLOs, at scale.

---

## 39. A Note on Pace and Keeping Current

A practical reflection for the inference engineer: this field moves extraordinarily fast. Techniques that were frontier research in 2023 (PagedAttention, speculative decoding) were standard by 2024; 2024's frontier (disaggregation, MLA, EP128, agent serving) is becoming standard in 2025–2026. New attention variants, quantization formats, scheduling schemes, and hardware appear continuously. Keeping current is part of the job — tracking the key venues (the engines' release notes and design docs, the systems conferences OSDI/SOSP/MLSys, arXiv, the model labs' technical reports like DeepSeek's). But the *pace* is precisely why this database emphasizes **principles over specifics**: the specific techniques churn, but the principles (the roofline, the KV cache as central resource, the prefill/decode duality, communication-computation overlap, the goodput objective, the composition of techniques) are stable. An engineer grounded in the principles can read a new technique's paper and immediately understand what bottleneck it addresses, how it composes with existing techniques, and when it applies — re-deriving the specifics from the principles rather than memorizing them. This is the durable capability: not knowing the current best technique (which will change), but understanding the *space* of techniques (defined by the bottlenecks they address) well enough to place any new one and to choose among them for a given workload. The frontier this file describes will be partly obsolete in a year — but the framework for understanding it (bottlenecks, composition, the KV cache, the roofline) will not, and that framework is what the database aims to instill. Read the frontier for *what's possible now*; hold the principles for *understanding whatever comes next*. That combination — current awareness grounded in durable principles — is how an inference engineer stays effective in a field that reinvents its specifics every year while preserving its fundamental physics and abstractions. The frontier moves; the foundation holds — and standing on the foundation is what lets you reach the frontier and, when it shifts, reach the next one. That is the closing lesson not just of this file but of the database: the techniques are many and changing, but the small set of principles beneath them — memory bandwidth, the KV cache, the prefill/decode duality, the roofline, overlap, goodput, composition — is what makes the whole field comprehensible, and what makes an engineer who holds them able to master each new technique as it arrives — which is the lasting value the database aims to deliver, far more than any snapshot of the current frontier, however detailed. The snapshot ages; the understanding compounds. And it is that compounding understanding — accumulated across the architecture, memory, scheduling, distribution, kernels, and now the frontier — that turns the daunting, fast-moving field of LLM inference into a coherent discipline an engineer can confidently practice and extend — and that confidence, grounded in principles and current on the frontier, is the destination of the journey through Files 01 to 15, with the remaining files supplying the hardware, comparison, and operational context that round out the practical picture: File 16 on hardware backends, File 17 comparing the frameworks, File 18 on LoRA serving, File 19 on production operations, and File 20 on the business and ecosystem in which all of this technical work ultimately operates — for inference engineering is, in the end, in service of building economical, reliable, capable LLM systems that deliver value, and the advanced techniques of this file are the means to that practical end, not ends in themselves — a perspective the operational and business files that follow make explicit as they connect the engineering to the deployments, the hardware platforms, and the economics it ultimately serves in real production systems operating at scale every day — the ultimate purpose toward which all the advanced techniques surveyed here, and all the foundational mechanisms before them, are directed.










