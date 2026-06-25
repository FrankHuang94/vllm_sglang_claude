# vLLM & SGLang Technical Knowledge Database

> A structured, deeply technical knowledge base on **vLLM** and **SGLang** — the two dominant open-source LLM inference engines — for **AI inference engineers** who need to understand these systems at the kernel, scheduler, memory-management, and distributed-systems level.

## What This Is

This database covers LLM inference serving from first principles (the roofline physics that bounds it) through the foundational systems (PagedAttention, RadixAttention, continuous batching, distributed inference, attention kernels), the performance methodology (benchmarking and tuning), the specialized techniques (speculative decoding, long context, multimodal, the research frontier), and the practical context (hardware, framework comparison, fine-tuned serving, production operations, and business/ecosystem).

It is written for engineers who need to *reason about and optimize* these systems — not a user manual, but a mechanism-level reference grounded in the cost models (the roofline, the KV cache, the prefill/decode duality) that make the engines' design choices legible. Total length is **~152,000 words** across 20 files plus this index.

**How deep:** kernel-level (FlashAttention tiling, CUDA graphs, custom all-reduce, quantized matmul), algorithmic (PagedAttention block management, RadixAttention's radix tree, speculative sampling), systems-level (the scheduler, tensor/pipeline/expert parallelism, the multi-process runtime), and quantitative (worked memory budgets, latency-throughput curves, roofline analyses, cost calculations) throughout.

---

## Reading Paths

Curated paths for specific goals:

1. **New to vLLM:** `01` (overview) → `02` (transformer internals) → `03` (PagedAttention) → `04` (scheduler) → `07` (serving/APIs). The foundation for understanding vLLM end to end.
2. **Deep systems:** `03` (PagedAttention) → `04` (scheduler) → `05` (distributed) → `08` (RadixAttention) → `09` (SGLang architecture) → `10` (kernels). The systems core of both engines.
3. **SGLang focus:** `08` (RadixAttention) → `09` (runtime architecture) → `10` (kernels) → `11` (performance). SGLang's distinctive contributions and how to run it fast.
4. **Performance tuning:** `10` (kernels) → `11` (benchmarking/tuning) → `12` (speculative decoding) → `13` (long context). Making a deployment fast and cheap.
5. **Production deployment:** `07` (serving) → `11` (tuning) → `16` (hardware) → `17` (frameworks) → `19` (operations). Deploying and operating reliably and economically.

If reading cover to cover, the files build on each other: `01` establishes the cost models; `02` the architecture; `03–07` vLLM; `08–09` SGLang; `10` kernels; `11` tuning; `12–15` specialized/advanced; `16–20` hardware, frameworks, LoRA, operations, business.

---

## Table of Contents

| # | File | Description |
|---|------|-------------|
| 01 | [Overview & Landscape](01_overview_and_landscape.md) | Why LLM inference is a distinct systems problem; the roofline model; the prefill/decode duality; the KV cache as the central resource; vLLM and SGLang origins; the ecosystem and market context; the anatomy of a serving engine. |
| 02 | [Transformer Inference Fundamentals](02_transformer_inference_fundamentals.md) ⭐ | Attention math and variants (MHA/MQA/GQA/MLA/SWA); FlashAttention; FFN/MoE; positional encoding (RoPE/YaRN/ALiBi); KV cache layout and quantization; sampling; speculative/constrained decoding foundations; quantization fundamentals — the architecture as a cost model. |
| 03 | [PagedAttention & vLLM Memory](03_paged_attention_and_vllm_memory.md) ⭐ | The memory-fragmentation problem; the paging algorithm; block tables and the CUDA kernel; copy-on-write; prefix caching; the block allocator; CPU swap; preemption (swap vs recompute); the OS virtual-memory analogy. |
| 04 | [vLLM Scheduler](04_vllm_scheduler.md) ⭐ | Continuous batching; the scheduler design and `schedule()`; token budgets; chunked prefill (Sarathi); priority/fairness; preemption; prefill-decode disaggregation; multi-step/async scheduling; the V1 engine; TTFT/TPOT decomposition. |
| 05 | [vLLM Distributed Inference](05_vllm_distributed_inference.md) ⭐ | Tensor/pipeline/expert/sequence parallelism; collective communication (AllReduce, all-to-all); the Megatron partitioning; interconnect tiers; multi-GPU KV management; communication-computation overlap; topology selection. |
| 06 | [Model Execution & Quantization](06_vllm_model_support_and_quantization.md) | The execution pipeline; attention metadata; quantization (GPTQ/AWQ/FP8/INT8) and kernels; weight loading; the model registry; MoE and MLA execution; speculative decoding and VLM support; fused kernels. |
| 07 | [Serving & APIs](07_vllm_serving_and_apis.md) | AsyncLLMEngine; the OpenAI-compatible API; streaming (SSE); the request lifecycle; sampling parameters; chat templates and tool calling; metrics; multi-LoRA; deployment; the V1 architecture; warmup/readiness. |
| 08 | [SGLang RadixAttention](08_sglang_radix_attention.md) ⭐ | The multi-call problem; the radix tree; insert/match/evict; the RadixAttention algorithm; tree-structured sharing; fork/join; vs hash-based caching; KV block management; XGrammar (structured output); the frontend DSL. |
| 09 | [SGLang Architecture & Internals](09_sglang_architecture_and_internals.md) ⭐ | The multi-process architecture; ZeroMQ and shared-memory communication; the scheduler and executor; attention backends (FlashInfer/Triton); CUDA graphs; the overlap scheduler; DP attention; the GIL rationale; the zero-overhead scheduler. |
| 10 | [Attention Kernels & Hardware](10_attention_kernels_and_hardware.md) ⭐ | The attention IO problem; FlashAttention (1/2/3); paged-attention kernels; FlashInfer; H100/Hopper and MI300X; Triton; fused kernels; multi-GPU attention; CUDA profiling; quantized matmul (Marlin/FP8); roofline analysis. |
| 11 | [Benchmarking & Tuning](11_performance_benchmarking_and_tuning.md) ⭐ | The metrics taxonomy; goodput; roofline analysis; vLLM and SGLang tuning parameters; benchmarking methodology; the latency-throughput curve; framework comparison; multi-GPU scaling; cost optimization; the tuning playbook. |
| 12 | [Speculative Decoding](12_speculative_decoding.md) | The theory and losslessness; the speedup model; tree speculation; Medusa/EAGLE/MTP; vLLM and SGLang implementations; draft-model management; MoE considerations; the batch-regime dependence. |
| 13 | [Long Context & Memory Optimization](13_long_context_and_memory_optimization.md) | The long-context challenge; StreamingLLM/attention sinks; KV offloading; sparse attention and eviction (H2O/SnapKV); GQA/MLA reduction; KV quantization; context extension (YaRN/LongRoPE); the decode-latency cost. |
| 14 | [Multimodal & Embedding Inference](14_multimodal_and_embedding_inference.md) | VLM architecture patterns; image caching; vLLM and SGLang multimodal; embedding and reranking serving; audio/video; the compute-bound vs memory-bound distinction; multimodal prefix caching. |
| 15 | [Advanced Features & Research](15_advanced_features_and_research.md) ⭐ | Prefill-decode disaggregation (Mooncake/DistServe); MoE optimization and DeepSeek-V3; continuous-batching enhancements; KV reuse/compression; MLA deep dive; reasoning-model inference; the research frontier; agent-oriented inference. |
| 16 | [Hardware Backends & Deployment](16_hardware_backends_and_deployment.md) | NVIDIA generations (A100→Blackwell); AMD MI300X; Intel Gaudi; Apple Silicon; Groq; AWS Inferentia; edge inference; the bandwidth-decode relationship; interconnect; TCO; hardware selection. |
| 17 | [Framework Comparison](17_model_serving_frameworks_comparison.md) | vLLM vs SGLang vs TensorRT-LLM vs TGI vs DeepSpeed-MII vs MLC-LLM vs Ollama vs LMDeploy; the feature table; architectural philosophies; convergence/divergence; orchestration layers; open vs closed dynamics; framework selection. |
| 18 | [LoRA & Fine-Tuned Serving](18_lora_and_finetuned_model_serving.md) | LoRA fundamentals; multi-LoRA serving (Punica/SGMV kernels); adapter caching; multi-LoRA workload patterns; QLoRA; the SaaS economics; LoRA scheduling; PEFT beyond LoRA. |
| 19 | [Production Operations & Reliability](19_production_operations_and_reliability.md) | Graceful shutdown; health checks; autoscaling (and the cold-start challenge); cost optimization; versioning/rollout; multi-model serving; chaos engineering; observability; circuit breaking; SLO management; security; capacity planning; an on-call runbook. |
| 20 | [Business, Market & Ecosystem](20_business_market_and_ecosystem.md) | Inference market economics; the vLLM and SGLang ecosystems/governance; hosted providers; competitive dynamics; open-source inference as a moat; model-serving co-design; the future trajectory; build-vs-buy; the inference engineer's role. |

⭐ = PRIMARY file (≥12,000 words, full kernel-level/algorithmic/systems depth).

---

## Key Concepts Index

Where the central concepts are developed:

- **PagedAttention:** File 03 (algorithm, block tables, CoW, prefix caching, kernel — §§2–6, 16); File 01 §4 (motivation); File 10 §5 (kernel).
- **RadixAttention:** File 08 (radix tree, algorithm, XGrammar, frontend — §§3–12); File 03 §14 (vs APC).
- **Continuous batching:** File 04 §3 (mechanics); File 09 §9 (SGLang); File 01 §11 (the throughput-latency frontier).
- **Chunked prefill:** File 04 §7, §17 (Sarathi, stall-free); File 03 §20 (memory mechanics).
- **Speculative decoding:** File 12 (full); File 02 §9 (foundations); File 06 §8 (vLLM impl); File 09 §25 (SGLang/EAGLE).
- **Tensor parallelism:** File 05 §2 (Megatron); §14 (worked AllReduce); File 11 §9 (scaling efficiency).
- **FlashAttention:** File 10 §§2–4, 13 (the family, derivation); File 02 §3 (foundations); FlashInfer File 10 §6.
- **KV cache:** File 01 §4 (the central resource); File 02 §6 (layout); File 03 (paging); File 13 (long context); File 15 §29 (the unifying theme — every technique manages it).
- **MoE inference:** File 02 §4, §21 (routing); File 05 §5 (expert parallelism); File 15 §§5–6 (optimization, DeepSeek-V3).
- **Quantization:** File 02 §11 (fundamentals); File 06 §3 (vLLM); File 10 §16 (kernels — Marlin/FP8).
- **MLA:** File 02 §2.4 (architecture); File 06 §7 (execution); File 10 §17 (kernel); File 15 §10 (deep dive).
- **Disaggregation:** File 04 §10, §30 (scheduling); File 15 §§1–4 (production systems, Mooncake/DistServe).
- **CUDA graphs:** File 09 §8 (capture/replay); File 10 §15.2 (launch overhead); File 05 §30 (with TP collectives).
- **Goodput & tuning:** File 11 (the methodology, the latency-throughput curve, the playbook); File 01 §11 (the frontier).

---

## Glossary

- **APC** — Automatic Prefix Caching (vLLM's hash-based prefix cache).
- **BGMV / SGMV** — Batched/Segmented Gather Matrix-Vector (Punica LoRA kernels).
- **CoW** — Copy-on-Write (shared KV blocks copied on divergence).
- **CUDA graph** — captured GPU operation sequence, replayed to eliminate launch overhead.
- **DP** — Data Parallelism (independent model replicas).
- **EP** — Expert Parallelism (MoE experts distributed across GPUs).
- **FA / FA-2 / FA-3** — FlashAttention versions.
- **FP8 / FP4** — 8-bit / 4-bit floating-point (E4M3/E5M2 / E2M1).
- **GQA / MQA / MHA / MLA** — Grouped-Query / Multi-Query / Multi-Head / Multi-head Latent Attention.
- **HBM** — High-Bandwidth Memory (GPU memory).
- **KV cache** — the cached key/value projections of prior tokens.
- **NCCL / RCCL** — NVIDIA / AMD collective-communication libraries.
- **PD disaggregation** — Prefill-Decode disaggregation.
- **PP** — Pipeline Parallelism (model layers across GPUs).
- **RoPE / YaRN** — Rotary Position Embedding / a RoPE context-extension method.
- **SLO** — Service Level Objective (latency/availability target).
- **SP** — Sequence Parallelism (the sequence/activations split across GPUs).
- **TP** — Tensor Parallelism (weight matrices split across GPUs).
- **TTFT / TPOT** — Time To First Token / Time Per Output Token.
- **W4A16 / W8A8** — 4-bit-weight/16-bit-activation / 8-bit-weight/8-bit-activation quantization.
- **XGrammar** — SGLang's grammar-constrained decoding engine.

---

## Recurring Themes

Five throughlines unify the database:

1. **The roofline** (File 01 §2): decode is memory-bandwidth-bound, prefill compute-bound; every optimization is a move on this chessboard.
2. **The KV cache as the central resource** (File 01 §4, File 15 §29): nearly every technique reduces, reuses, places, evicts, or spreads it.
3. **The prefill/decode duality** (File 01 §3): the two phases' opposite characteristics drive scheduling (chunked prefill) and architecture (disaggregation).
4. **Communication-computation overlap** (File 05 §34): hide communication (TP AllReduce, EP all-to-all, KV transfer) and CPU work (scheduling) behind GPU compute.
5. **Composition** (File 15 §21): the techniques stack — frontier deployments combine paging, prefix caching, quantization, speculation, parallelism, and disaggregation, each addressing a bottleneck.

The durable lesson (File 15 §39): the specific techniques and tools churn, but these principles endure — and the engineer who holds them can master each new technique, tune any deployment, and follow the field's relentless advance.

---

*This database covers vLLM and SGLang as of the 2024–2026 state of the art. The field moves fast; the principles endure. Built for AI inference engineers — from the roofline physics to the cost structure, the full discipline of LLM inference engineering.*
