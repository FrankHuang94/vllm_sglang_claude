# LLM Inference Engineering — Overview, Problem Space, and Ecosystem

> **Audience:** AI inference engineers who need to reason about LLM serving systems at the level of kernels, schedulers, memory managers, and distributed-systems architecture. This file establishes the problem space and vocabulary that the rest of the database builds on.

---

## 1. Why LLM Inference Is a Distinct Systems Problem

Training and inference of large language models look superficially similar — both run transformer forward passes on GPUs — but they are *fundamentally different systems problems*, and conflating them leads to bad engineering decisions.

Training is a **throughput-bound, compute-bound, statically-shaped** workload. You control the batch size, the sequence length is fixed (or padded to a fixed bucket), every sample runs forward *and* backward, and you optimize a single metric: tokens processed per second per dollar. The arithmetic intensity is high because every parameter is touched by a large batch of tokens, so the matrix multiplies are big enough to saturate the tensor cores. Memory is dominated by optimizer states, gradients, and activations for backprop.

Inference, specifically **autoregressive decoding**, breaks almost every one of those assumptions:

1. **It is memory-bandwidth bound, not compute bound, for most realistic batch sizes.** During decoding the model generates one token at a time. Each decode step reads the *entire model weight matrix* from HBM to multiply against a single token's hidden vector (a matrix–vector product, GEMV). The FLOPs are tiny relative to the bytes moved. The GPU's tensor cores sit mostly idle while the memory subsystem is the bottleneck. This single fact drives the entire design of vLLM and SGLang.

2. **The KV cache memory footprint dominates and grows with every generated token.** Unlike training, where activations are transient, inference must *retain* the key and value projections of every previous token so that future tokens can attend to them. This cache grows linearly with sequence length and must persist for the entire lifetime of a request. For long contexts it dwarfs the model weights.

3. **Requests are variable-length and arrive asynchronously.** Users send prompts of wildly different lengths, request different numbers of output tokens, and arrive at unpredictable times. A static batch — the bread and butter of training — wastes enormous amounts of GPU time waiting for the longest sequence in the batch to finish. This creates a genuine *scheduling* problem, closer to an operating system's process scheduler than to a training loop.

4. **You must satisfy two competing SLOs simultaneously.** Inference serving has to hit *latency* targets (a user waiting on a chat response cares about Time To First Token and inter-token latency) *and* *throughput* targets (the operator cares about tokens/second/GPU because that determines cost). These objectives pull in opposite directions: bigger batches improve throughput but worsen per-request latency. Every scheduler decision is a point on this Pareto frontier.

These four properties — memory-bandwidth boundedness, KV-cache-dominated memory, variable-length asynchronous requests, and dual latency/throughput SLOs — are why "just run `model.generate()` in a loop" is catastrophically inefficient in production, and why dedicated inference engines exist.

---

## 2. The Roofline Model for LLM Inference

The roofline model is the single most useful mental tool for an inference engineer. It plots achievable performance (FLOP/s) against **arithmetic intensity** (FLOPs performed per byte moved from memory). A kernel is either *memory-bound* (left of the ridge point, limited by bandwidth) or *compute-bound* (right of the ridge point, limited by peak FLOP/s).

### 2.1 Arithmetic intensity of a single decode step

Consider generating one token with a dense model of `P` parameters in a format using `b` bytes per parameter (BF16 → `b = 2`). To compute the logits for that one token, the model performs roughly `2P` FLOPs (one multiply and one add per parameter — the factor of 2). To do so it must read all `P` parameters from HBM, costing `P·b` bytes.

```
Arithmetic intensity (decode, batch=1)
  ≈ FLOPs / bytes
  = 2P / (P · b)
  = 2 / b
  = 2 / 2  =  1 FLOP/byte    (for BF16)
```

**One FLOP per byte.** That is extraordinarily low. For comparison, an A100's ridge point (where it transitions from memory-bound to compute-bound) is much higher (see below). Therefore a batch-1 decode step is deep in the memory-bound regime: the GPU spends its time waiting on HBM, and the achievable performance is `bandwidth × arithmetic_intensity`, *not* peak FLOP/s.

### 2.2 The ridge point and the break-even batch size

The way out of the memory-bound regime is **batching**. If you process `B` tokens through the same weights in one step, you still read the weights once (`P·b` bytes) but now do `2·B·P` FLOPs. Arithmetic intensity scales with `B`:

```
Arithmetic intensity (decode, batch=B) ≈ 2B / b  ≈ B   (BF16)
```

You become compute-bound once arithmetic intensity exceeds the hardware ridge point, defined as `peak_FLOP/s ÷ peak_bandwidth`:

```
A100 SXM (BF16):  312 TFLOP/s ÷ 2.0 TB/s  ≈ 156 FLOP/byte
```

So you need roughly **156 tokens "in flight" per step** before an A100 becomes compute-bound on the matrix multiplies. Below that batch size you are leaving FLOPs on the table — the tensor cores are underutilized and adding more concurrent requests is essentially *free* in compute terms (it costs only KV cache memory). This is the quantitative justification for continuous batching: pack as many decode tokens into each step as memory allows, because throughput scales nearly linearly with batch size until you approach the ridge point.

For an H100 the numbers shift:

```
H100 SXM5 (BF16):  989 TFLOP/s ÷ 3.35 TB/s  ≈ 295 FLOP/byte
H100 SXM5 (FP8):  1979 TFLOP/s ÷ 3.35 TB/s  ≈ 590 FLOP/byte
```

The higher ridge point means H100 needs *even larger* batches to be compute-bound — which is fine, because its larger HBM and bandwidth let it hold more concurrent KV cache. The practical lesson: faster tensor cores raise the bar for "enough batching," so memory capacity and bandwidth, not raw FLOP/s, are usually the binding constraint for decode.

### 2.3 Why this matters for every design decision

Almost every optimization in vLLM and SGLang can be understood as either (a) increasing the achievable batch size at fixed memory (PagedAttention, prefix caching, quantized KV cache), or (b) reducing bytes moved per token (GQA, MLA, quantization), or (c) reclassifying work from the memory-bound decode regime into the compute-bound prefill regime where the hardware is efficient (chunked prefill, disaggregation, speculative decoding). Keep the roofline in mind and the rest of this database becomes a catalog of moves on a single chessboard.

---

## 3. The Prefill–Decode Duality

A single LLM request has two phases with opposite performance characteristics.

**Prefill (prompt processing):** All input tokens are processed in parallel in a single forward pass. Because there are many tokens, the per-layer operations are matrix–matrix multiplies (GEMM) with high arithmetic intensity. Prefill is **compute-bound**: it saturates the tensor cores and its latency scales with prompt length. Prefill produces the KV cache for the prompt and the first output token.

**Decode (autoregressive generation):** Output tokens are generated one at a time. Each step processes a single new token per sequence, so the per-layer operations are matrix–vector multiplies (GEMV) with arithmetic intensity ≈ 1. Decode is **memory-bandwidth-bound**: it is limited by the rate at which weights and KV cache can be streamed from HBM, and its latency per token is roughly constant regardless of how many tokens have already been generated (modulo the growing KV cache read).

This duality is the source of most scheduling innovation in the field:

- **Head-of-line blocking:** If a long prefill and several decodes share one batch step, the prefill's heavy compute delays the decodes, spiking inter-token latency for everyone. **Chunked prefill** (File 04) breaks prefill into bounded pieces so decodes can interleave.
- **Hardware mismatch:** The ideal GPU for compute-bound prefill (high FLOP/s, e.g. H100) differs from the ideal GPU for bandwidth-bound decode (high HBM bandwidth and capacity). **Prefill–decode disaggregation** (Files 04, 15) runs them on separate, separately-optimized pools.
- **Batching asymmetry:** Decode benefits enormously from batching (it's memory-bound, so amortizing the weight read across many sequences is pure win); prefill is already compute-bound so batching it has diminishing returns past the ridge point.

---

## 4. KV Cache: The Central Resource

If you remember one formula from this database, make it the KV cache size formula.

```
KV_cache_bytes_per_token = 2 · num_layers · num_kv_heads · head_dim · bytes_per_element
```

The leading `2` is for **K and V** (you cache both the key and value projections). `num_kv_heads` (not `num_query_heads`) is what matters because of Grouped-Query Attention (File 02) — modern models share K/V across query heads to shrink exactly this number.

### 4.1 Worked example: LLaMA-3 70B

LLaMA-3 70B has `num_layers = 80`, `num_kv_heads = 8` (GQA), `head_dim = 128`, BF16 (`bytes = 2`):

```
per_token = 2 · 80 · 8 · 128 · 2  =  327,680 bytes  ≈  320 KB / token
```

At a 100K-token context, a *single request* needs:

```
100,000 · 320 KB  =  32 GB  just for its KV cache
```

That is nearly half of an 80 GB A100/H100 consumed by one long-context request's cache. This is why long context (File 13) is a memory problem first and a compute problem second, and why KV cache reduction techniques (GQA, MLA, quantization, eviction) are among the highest-leverage optimizations available.

### 4.2 Worked example: LLaMA-3 8B

LLaMA-3 8B has `num_layers = 32`, `num_kv_heads = 8`, `head_dim = 128`, BF16:

```
per_token = 2 · 32 · 8 · 128 · 2  =  131,072 bytes  =  128 KB / token
```

At a 4,096-token context: `4096 · 128 KB = 512 MB` per request. After ~16 GB of model weights, an 80 GB GPU has ~64 GB for KV cache → roughly `64 GB / 512 MB ≈ 128` concurrent requests at full context (more in practice, because most requests don't fill the whole context window). The KV cache, not the weights, determines concurrency.

### 4.3 Decode bandwidth cost of the KV cache

Beyond storage, the KV cache must be *read* every decode step (the new query attends to all cached keys/values). For LLaMA-3 70B serving one request at 1,000 tokens of context:

```
KV bytes read per step = 2 · num_kv_heads · head_dim · bytes · context_len · num_layers
                       = 2 · 8 · 128 · 2 · 1000 · 80
                       ≈ 327 MB

At A100 2 TB/s:  327 MB / 2 TB/s  ≈ 164 µs  per decode step  (KV reads alone)
```

This is *on top of* reading the weights. As context grows, KV reads come to dominate decode latency — another reason long-context decode is expensive, and why FlashAttention-style kernels that minimize redundant HBM traffic (File 10) are mandatory.

---

## 5. vLLM: Origin and Significance

**vLLM** originated at UC Berkeley's Sky Computing Lab in 2023, led by Woosuk Kwon and Zhuohan Li with Siyuan Zhuang, Ying Sheng, and collaborators including Ion Stoica. Its foundational contribution is **PagedAttention**, published at SOSP 2023 as *"Efficient Memory Management for Large Language Model Serving with PagedAttention."* The key idea — borrowed directly from operating-system virtual memory — is that the KV cache need not be stored contiguously; it can be partitioned into fixed-size *blocks* mapped through a per-sequence *block table*, eliminating the internal and external fragmentation that crippled earlier serving systems (which pre-allocated worst-case contiguous buffers and achieved <40% memory utilization). PagedAttention is covered exhaustively in File 03.

vLLM was open-sourced in June 2023 and saw extremely rapid adoption: it became the de facto open-source serving engine, was integrated by numerous hosted-inference providers, and was contributed to by NVIDIA, AMD, Intel, AWS, Google, and others. It is now a project under the **LF AI & Data Foundation**, with a large contributor base and an Apache 2.0 license. Its design priorities are broad model support, production robustness, and continuous integration of state-of-the-art techniques (FP8, speculative decoding, chunked prefill, disaggregation).

---

## 6. SGLang: Origin and Significance

**SGLang** ("Structured Generation Language") came out of UC Berkeley and LMSYS in late 2023/2024, led by Lianmin Zheng, Liangsheng Yin, and collaborators. The system paper is *"SGLang: Efficient Execution of Structured Language Model Programs"* (arXiv 2312.07104). SGLang's signature innovation is **RadixAttention** (File 08): automatic, fine-grained KV cache reuse across requests using a radix tree (compressed trie) keyed on token sequences. Where vLLM's prefix caching matches at block granularity, RadixAttention matches at token granularity and naturally handles tree-structured sharing across many concurrent sessions — ideal for multi-turn chat, few-shot prompting, RAG, and tree-of-thought workloads.

SGLang is distinguished by two things beyond RadixAttention. First, it couples a **frontend DSL** (a Python language for expressing multi-call LLM programs with primitives like `gen`, `select`, `fork`, `join`) tightly to its **backend runtime**, enabling whole-program optimizations like structural batching that engine-only systems cannot see. Second, it tends to ship cutting-edge algorithm implementations quickly — RadixAttention, the **XGrammar** constrained-decoding engine (arXiv 2411.15100), data-parallel attention, and aggressive CUDA-graph usage often landed in SGLang ahead of equivalents elsewhere. SGLang is maintained primarily by LMSYS (also responsible for Chatbot Arena and the earlier FastChat), is Apache 2.0 licensed, and is used by several hosted providers.

---

## 7. The Broader Inference Ecosystem

vLLM and SGLang dominate open-source GPU serving, but they live in a crowded ecosystem. Knowing the landscape prevents reinventing wheels and clarifies trade-offs (File 17 compares the major server-class engines in depth).

- **TensorRT-LLM (NVIDIA):** Closed-source, ahead-of-time compiled CUDA kernel library with the best raw FLOP utilization on NVIDIA hardware for supported models. Pays for it with long per-model compilation, limited extensibility, and NVIDIA-only deployment. Often deployed behind Triton Inference Server.
- **TGI — Text Generation Inference (HuggingFace):** Rust HTTP front end + Python backend, deep HuggingFace ecosystem integration, the widest model compatibility, OpenAI-compatible API. Historically trails vLLM on raw throughput.
- **DeepSpeed-MII (Microsoft):** Built on DeepSpeed Inference; "ragged batching," kernel injection, ZeRO-Inference for MoE/large-model offloading. Slower development pace post-2023.
- **MLC-LLM (MLC.AI):** Apache TVM-based compilation to *any* backend — CUDA, ROCm, Metal, Vulkan, WebGPU — enabling deployment on phones, browsers, and desktops. Trades server throughput for unmatched portability.
- **Ollama:** Local-first, llama.cpp backend, dead-simple UX (`ollama run llama3`). Not designed for high-throughput multi-tenant serving (no continuous batching).
- **llama.cpp:** The reference CPU/edge inference engine; GGUF quantization formats (Q4_K_M, Q5_K_M, etc.), Metal/CUDA/Vulkan backends. Foundational for local and embedded inference.
- **LMDeploy (Shanghai AI Lab):** TurboMind C++ backend, strong AWQ W4A16 quantization, continuous batching, paged attention. Fast on NVIDIA, smaller community.
- **Others:** OpenLLM (BentoML, packaging/ops layer), Xinference (multi-model serving), and **Triton Inference Server** as a general orchestration layer that can front any of the above.

A useful taxonomy: **server-class throughput engines** (vLLM, SGLang, TensorRT-LLM, TGI, LMDeploy, DeepSpeed-MII), **portability/compilation stacks** (MLC-LLM), **local/edge runtimes** (llama.cpp, Ollama, MLX), and **orchestration layers** (Triton, OpenLLM). This database focuses on the first category, with vLLM and SGLang as the protagonists.

---

## 8. Market Context: Why Inference Efficiency Is the Whole Game

The economics make the engineering urgency clear.

- **Inference dominates lifetime LLM compute cost.** A model is trained once but served billions of times. In production deployments, inference accounts for an estimated **80–90% of total LLM compute spend**. A 20% throughput improvement in the serving engine therefore translates more or less directly into a 20% reduction in the dominant cost line — which is why hyperscalers and startups alike pour engineering into vLLM/SGLang.
- **The commercial API market** is a layer above these engines. OpenAI, Anthropic, Google, and Mistral run proprietary stacks, while a large tier of providers — **Together AI, Fireworks AI, Anyscale, Modal, Replicate, Baseten, Perplexity** — build their open-weight inference businesses directly on vLLM/SGLang (often forked and customized). **Groq** and **Cerebras** compete on specialized hardware (LPU, wafer-scale) for latency or small-batch throughput. The price of inference is set by how efficiently these engines run on commodity GPUs.
- **Open-source inference as a moat.** For any company building products on open-weight models (LLaMA, Mistral, Qwen, DeepSeek), the inference engine *is* the cost structure. Mastery of vLLM/SGLang internals — exactly the material in this database — is what separates a 2× cost disadvantage from a 2× advantage.
- **The price trajectory** tells the story: the cost per token of comparable model quality has fallen by an order of magnitude in two years, driven roughly equally by better hardware (A100 → H100 → H200 → Blackwell) and better software (PagedAttention, continuous batching, quantization, speculative decoding). The software half of that improvement is precisely what vLLM and SGLang deliver.

---

## 9. How This Database Is Organized

The remaining files drill into each layer of the stack:

- **Foundations:** File 02 (transformer internals for inference — attention variants, MoE, positional encoding, KV layout, sampling, quantization fundamentals).
- **vLLM internals:** File 03 (PagedAttention & memory), File 04 (scheduler & continuous batching), File 05 (distributed inference — TP/PP/EP/SP), File 06 (model execution & quantization), File 07 (serving & APIs).
- **SGLang internals:** File 08 (RadixAttention & XGrammar), File 09 (runtime architecture).
- **Kernels & hardware:** File 10 (FlashAttention, CUDA, hardware backends), File 16 (NVIDIA/AMD/Intel/edge hardware).
- **Performance:** File 11 (benchmarking & tuning), File 12 (speculative decoding), File 13 (long context), File 15 (disaggregation, MoE optimization, research frontiers).
- **Applications & ops:** File 14 (multimodal & embeddings), File 17 (framework comparison), File 18 (LoRA serving), File 19 (production operations), File 20 (business & ecosystem).

Read the roofline section (§2) and the KV cache section (§4) of this file until they are second nature. Everything else is an elaboration of the tension they describe: a memory-bandwidth-bound, KV-cache-hungry, variable-length workload that must be scheduled to satisfy latency and throughput at once.

---

## 10. A Brief Timeline of LLM Inference Systems Research

Understanding where the techniques came from clarifies why the engines are designed as they are.

- **2017–2019 — Transformer and the first efficiency moves.** The Transformer (Vaswani et al. 2017) establishes the architecture. Shazeer's Multi-Query Attention (2019) is the first paper to treat the *KV cache* as the inference bottleneck — a remarkably early recognition of what would become the central resource.
- **2022 — Batching and IO-aware kernels.** **Orca** (Yu et al., OSDI 2022) introduces *iteration-level scheduling* and *selective batching* — the conceptual seed of continuous batching. In parallel, **FlashAttention** (Dao et al.) makes attention IO-aware, removing the `O(n²)` HBM materialization that capped context length. These two ideas — schedule per-iteration, and never materialize the score matrix — are load-bearing for everything after.
- **2023 — PagedAttention and the serving-systems explosion.** vLLM's **PagedAttention** (SOSP 2023) brings OS-style virtual memory to the KV cache, lifting memory utilization from <40% to >95% and roughly doubling throughput. The same year sees GQA (making large-model KV cache tractable), speculative decoding (Leviathan et al.), and a wave of quantization work (GPTQ, AWQ, SmoothQuant). FlashAttention-2 lands.
- **2024 — Structured generation, disaggregation, and MoE at scale.** SGLang's **RadixAttention** generalizes prefix caching to a radix tree across sessions. **Chunked prefill** (Sarathi/Sarathi-Serve) and **prefill–decode disaggregation** (DistServe, Mooncake) attack head-of-line blocking. **XGrammar** makes constrained decoding near-free. **DeepSeek-V2/V3** ship MLA and EP128, proving low-rank KV and 128-way expert parallelism in production. FlashAttention-3 exploits Hopper. FP8 inference becomes mainstream.
- **2025–2026 — Consolidation and hardware diversity.** Disaggregation moves toward production; FlashInfer becomes the default kernel library; AMD MI300X and the NVIDIA Blackwell generation broaden the hardware base; long context (128K+) becomes a baseline expectation rather than a feature. vLLM and SGLang both ship engine rearchitectures (vLLM's V1/EngineCore, SGLang's overlap scheduler) that move scheduling off the critical path.

The throughline: each advance either (a) reduces bytes moved (FlashAttention, GQA, MLA, quantization), (b) increases achievable batch at fixed memory (PagedAttention, prefix caching), or (c) reorganizes *when* and *where* work happens to keep the hardware in its efficient regime (continuous batching, chunked prefill, disaggregation, speculative decoding). The roofline (§2) is the unifying frame for all three.

---

## 11. The Throughput–Latency Pareto Frontier

The defining operational tension deserves its own treatment because every tuning decision (File 11) lives on this frontier.

Consider sweeping the batch size on a fixed deployment:

- **At small batch** (few concurrent requests): each request enjoys low **TPOT** (Time Per Output Token) because the decode step is short and the memory bandwidth is shared among few sequences. But **throughput** (tokens/sec/GPU) is poor — you are deep in the memory-bound regime, weights are re-read for almost no work, and most of the GPU's FLOP capacity is wasted.
- **As batch grows:** throughput rises nearly linearly (the fixed weight-read cost is amortized across more sequences) while TPOT rises slowly at first — adding sequences is "free" in compute until you approach the ridge point. This is the sweet spot the scheduler chases.
- **At large batch** (approaching KV-memory or ridge-point limits): throughput saturates (you become compute-bound, or you run out of KV cache and start preempting), and TPOT degrades sharply as each sequence now competes for bandwidth and compute. **TTFT** (Time To First Token) also suffers because new requests queue behind a saturated batch.

The operator does not actually want to maximize throughput *or* minimize latency in isolation — they want to maximize **goodput**: the request rate served *while meeting latency SLOs* (e.g. P95 TTFT < 2 s and P99 TPOT < 100 ms). Goodput is the business-relevant metric because latency violations mean failed requests regardless of raw token throughput. Much of the sophistication in vLLM's and SGLang's schedulers (Files 04, 09) — chunked prefill, priority policies, preemption choices, disaggregation — exists to push the goodput frontier outward: to serve more requests per GPU without breaching the latency budget. File 11 makes this quantitative with latency-under-load curves and the tuning knobs that shift them.

A useful intuition: training optimization is a *scalar* problem (maximize tokens/sec); inference optimization is a *constrained, multi-objective* problem (maximize goodput subject to latency SLOs under a stochastic, variable-length request stream). That difference is why inference serving is a genuine systems discipline and not merely "training, but forward-only."

---

## 12. The Anatomy of a Serving Engine

To orient the reader before the deep dives, here is the component decomposition shared (with naming differences) by vLLM and SGLang. Every serious inference engine has these layers, and the rest of this database is organized around them.

1. **API / frontend layer.** Accepts requests (OpenAI-compatible HTTP, or SGLang's program DSL), applies chat templates, tokenizes input, and streams output. Covered in File 07 (vLLM) and Files 08–09 (SGLang frontend and TokenizerManager).
2. **Scheduler.** The brain: decides each iteration which requests run, admits new requests from a waiting queue, preempts or swaps under memory pressure, and balances prefill against decode (chunked prefill). This is where continuous batching lives. Files 04 (vLLM) and 09 (SGLang).
3. **KV cache / memory manager.** Allocates and frees KV blocks, maintains block tables, implements copy-on-write and prefix caching/RadixAttention, and handles CPU swap. Files 03 (PagedAttention) and 08 (RadixAttention).
4. **Model executor / runner.** Builds the packed batch (token IDs, positions, slot mappings, block tables), runs the forward pass, applies sampling, and returns tokens. Files 06 (vLLM) and 09 (SGLang).
5. **Attention & compute kernels.** FlashAttention/FlashInfer/Triton attention, fused norm/activation/RoPE kernels, quantized matmuls, sampling kernels, CUDA graphs. File 10.
6. **Distributed runtime.** Tensor/pipeline/expert/sequence parallelism, collective communication (NCCL/RCCL), multi-node coordination. File 05.
7. **Observability & operations.** Metrics, health checks, autoscaling, graceful shutdown, multi-model and LoRA serving. Files 07, 18, 19.

The request's journey — HTTP in, tokenize, schedule, allocate KV, forward pass through kernels (possibly across many GPUs), sample, detokenize, stream out, free KV — touches every layer. Holding this map in mind makes each subsequent file legible as a zoom-in on one box. The two protagonists differ most in layers 2–3 (SGLang's RadixAttention and program-aware scheduling vs vLLM's PagedAttention and broad scheduler policies) and in layer 1 (SGLang's frontend DSL has no vLLM equivalent), and they increasingly converge in layers 4–6 as both adopt each other's best ideas. Where they diverge and why is the recurring theme of Files 03–11 and the explicit subject of File 17's comparison.

---

## 13. How to Read This Database

This is a reference, not a tutorial, but it rewards a deliberate path. A reader new to serving should start with this file (especially §§2 and 4), then File 02 for the architecture-as-cost-model lens, then File 03 (PagedAttention) and File 04 (the scheduler) to see how the two central resources — KV memory and GPU time — are managed. From there, the SGLang pair (Files 08–09), the kernels-and-hardware file (10), and the performance file (11) form the systems core. The remaining files are deep dives that can be read in any order as needs arise: distributed inference (05), quantization and model support (06), serving APIs (07), speculative decoding (12), long context (13), multimodal (14), research frontiers (15), hardware (16), framework comparison (17), LoRA (18), operations (19), and business/ecosystem (20). The README provides curated reading paths for specific goals.

Throughout, the same small set of cost models recurs — the roofline, the KV-cache formula, the prefill/decode duality, the throughput–latency frontier. They are introduced here precisely because they are the load-bearing intuitions. An engineer who can apply them fluently will find that the engines' design choices, far from arbitrary, are nearly forced by the physics of memory bandwidth and the economics of GPU time.
