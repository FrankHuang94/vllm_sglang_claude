# Hardware Backends — NVIDIA, AMD, Intel, and Edge Deployment

> **Standard reference file.** The hardware determines the rooflines (File 01 §2) that bound inference. This file surveys NVIDIA GPU generations (A100 → Blackwell), AMD ROCm/MI300X, Intel Gaudi, Apple Silicon, Groq, AWS Inferentia/Trainium, and edge deployment — their specs, software stacks, and inference suitability. Prerequisites: File 01 §2 (roofline), File 05 (distributed), File 10 §§7–8 (H100/MI300X kernels).

---

## 1. NVIDIA GPU Generations for Inference

NVIDIA dominates inference; knowing the generations and their trade-offs is essential for hardware selection.

### 1.1 A100 SXM (Ampere, 2020)

- 80 GB HBM2e at **2 TB/s**; **312 TFLOP/s BF16** (624 with sparsity); 6912 CUDA cores, 3rd-gen tensor cores.
- No FP8 (INT8 for W8A8 quantization). NVLink 3.0 (~600 GB/s).
- The prior-generation workhorse — solid inference, still widely deployed. PCIe and SXM variants (SXM has 2× the bandwidth). Cost: ~$10K new, less refurbished. Roofline ridge: `312/2 = 156` FLOP/byte (File 01 §2).

### 1.2 H100 SXM5 (Hopper, 2022)

- 80 GB HBM3 at **3.35 TB/s**; **989 TFLOP/s BF16** (1979 FP8 via Transformer Engine); 132 SMs, 4th-gen tensor cores.
- FP8 (E4M3/E5M2), TMA, WGMMA, warp specialization (File 10 §§4, 21). NVLink 4.0 (900 GB/s).
- The current inference standard — FP8 (2× compute, half weight bytes), high bandwidth (1.67× A100 decode), and the Hopper features FA-3/FlashInfer exploit (File 10 §7). Cost ~$30K. Ridge: `989/3.35 ≈ 295` BF16, ~590 FP8.

### 1.3 H200 SXM (2024)

- **141 GB HBM3e at 4.8 TB/s** (1.43× H100 bandwidth); same compute as H100.
- The bandwidth/capacity upgrade: 1.43× H100 decode throughput (decode is bandwidth-bound, File 01 §2), more KV/larger models from 141 GB. Same kernels, faster memory (File 10 §28). Ideal for decode-heavy and long-context (more KV) workloads.

### 1.4 GH200 (Grace Hopper Superchip)

- 96 GB HBM3e (GPU) + 480 GB LPDDR5X (Grace CPU), NVLink-C2C at 900 GB/s CPU↔GPU → **624 GB unified memory**.
- The large unified memory suits long-context inference (huge KV capacity) and CPU-offload scenarios (fast CPU↔GPU). Limited availability. The CPU-GPU coherent memory eases KV offloading (File 13 §3).

### 1.5 B100/B200 (Blackwell, 2024/2025)

- B200: **192 GB HBM3e at ~8 TB/s**; **~4500 TFLOP/s FP4**; NVLink 5.0 (1.8 TB/s); 2nd-gen Transformer Engine.
- Adds **FP4** (E2M1) — 4-bit float for inference, 2× FP8 compute, with minor quality loss (File 10 §28). ~8 TB/s nearly doubles H100 decode bandwidth; NVLink 5.0 scales TP further. New FP4 kernels required. The next inference tier — 2× decode throughput vs H100 per GPU, FP4 for compute-bound prefill.

### 1.6 L40S (Ada Lovelace, 2023)

- 48 GB GDDR6; **864 TFLOP/s FP8**; PCIe (no SXM/NVLink).
- Lower cost per GPU, suitable for medium models (13B–34B) or as cheaper decode hardware (disaggregation, File 15 §3). 8× L40S can compete with 2× H100 at lower cost for some inference. No NVLink limits TP efficiency (File 05 §2.6) — better for DP/smaller models.

### 1.7 Selecting an NVIDIA GPU

- **Decode-heavy / long context:** prioritize bandwidth/capacity → H200, GH200 (decode is bandwidth-bound).
- **Prefill-heavy / compute:** prioritize FLOP/s and low precision → H100/B200 FP8/FP4.
- **Cost-sensitive medium models:** L40S (cheaper, no NVLink — use DP).
- **Frontier large models:** H100/B200 with NVLink for TP, lots of HBM.

The roofline (File 01 §2) drives the choice: match the GPU's bandwidth (decode) and compute (prefill) to the workload's profile (File 11 §22).

---

## 2. AMD ROCm and MI300X

AMD's MI300X is the leading NVIDIA alternative for inference (File 10 §8).

- **Specs:** **192 GB HBM3 at 5.3 TB/s** (more capacity and bandwidth than H100); ~1307 TFLOP/s FP16, ~2614 FP8; CDNA3, 8 chiplets.
- **Bandwidth advantage:** 5.3 TB/s vs H100's 3.35 (1.58×) → faster decode in principle (~1.2–1.4× realized, kernel maturity, File 10 §8.2). **192 GB capacity** (2.4× H100) fits 70B in FP16 on one GPU, or larger models on fewer GPUs (less TP overhead).
- **Software (ROCm):** HIP (compile CUDA-style to AMD), RCCL (collectives), Triton on ROCm (portable kernels, File 10 §9), FlashAttention ROCm port. vLLM and SGLang have ROCm builds and AMD CI (File 10 §8.3).
- **Performance:** ~85–90% of CUDA in LLM workloads (kernel maturity gap), but the capacity/bandwidth advantage offsets this for memory-bound large-model serving.
- **The case for MI300X:** memory-bound, capacity-hungry inference (large models, long context, high concurrency) where the 192 GB and 5.3 TB/s shine, and where the Triton-based kernels (portable) are mature enough. The most credible NVIDIA alternative, supported in both engines.

---

## 3. Intel Gaudi 2/3

Intel's Habana Gaudi accelerators target AI inference (and training):

- **Software:** SynapseAI SDK; vLLM Gaudi backend (Intel's `vllm-fork`); Intel's optimized Gaudi attention kernel; TGI-Gaudi. Gaudi-specific quantization (Intel Neural Compressor).
- **Gaudi 3:** 128 GB HBM2e; performance competitive with A100-class at lower cost.
- **The case:** cost-competitive with NVIDIA at the A100 tier, for organizations seeking NVIDIA alternatives. Limited community support vs NVIDIA/AMD — the software ecosystem is smaller, and the engines' Gaudi support is via Intel forks rather than mainline. A viable option for cost-sensitive deployments willing to use Intel's stack, but the ecosystem maturity lags NVIDIA/AMD.

---

## 4. Apple Silicon (M-series)

Apple Silicon enables capable *local* inference on Macs:

- **Stack:** MLX (Apple's ML framework, File 17), llama.cpp Metal backend. **vLLM/SGLang do not support Apple Silicon** (Linux + CUDA/ROCm only) — Apple is a local-inference, not server-class, platform.
- **Unified memory:** CPU and GPU share the same physical memory → simple memory management, no CPU↔GPU transfer. M2 Ultra has up to 192 GB unified memory — enough to hold large models (a 70B in 4-bit).
- **Limits:** memory bandwidth (~800 GB/s, vs HBM3's 3+ TB/s) and compute (~27 TFLOP/s FP16 M2 Ultra vs 989 H100) are far below datacenter GPUs → suitable for local single-user inference of models up to ~70B (quantized), not high-throughput serving.
- **The case:** local, private, single-user inference on a Mac — privacy (data stays local), no cloud cost, offline. The large unified memory is the distinguishing advantage (hold big models); the bandwidth/compute limits cap throughput. Part of the edge/local-inference regime (File 17), distinct from server-class serving.

---

## 5. Groq LPU

Groq's Language Processing Unit is a specialized inference accelerator (File 15 §22):

- **Architecture:** deterministic execution, **SRAM-only (no DRAM)**, ~750 GB/s on-chip SRAM bandwidth. Latency-optimized.
- **Performance:** very high single-stream token rates (~200K tokens/sec for LLaMA-2 7B, vs ~5K for H100 single-stream) — by keeping everything in fast SRAM, eliminating the DRAM bottleneck.
- **Limits:** SRAM capacity is small, so models must fit in SRAM (small models, or many chips for large models — a 70B needs many LPUs). Proprietary compiler; closed API only (you use Groq's API, not self-host).
- **The case:** latency-optimized inference (fastest tokens/sec for a single stream) for small-to-medium models — Groq's API offers very fast responses. The bet: SRAM-centric, deterministic hardware beats GPUs for low-latency inference (File 15 §22). Latency-optimized, not throughput-optimized (a GPU serves more concurrent users; Groq serves one user faster).

---

## 6. AWS Inferentia and Trainium

AWS's custom inference (Inferentia) and training (Trainium) chips:

- **Stack:** Neuron SDK; `neuronx-cc` compiler (ahead-of-time compilation required — unlike dynamic PyTorch, the model is compiled to the chip, with caching of compiled models). vLLM Inferentia support (AWS-maintained fork).
- **Inf2:** 384 GB HBM across 12 Inferentia2 chips.
- **The case:** **2–3× cheaper per token than H100** for supported models on AWS — a cost advantage for AWS deployments willing to use the Neuron stack and accept the AOT-compilation constraint (compile each model, cache it). The catch: AOT compilation (less flexible than dynamic execution), AWS-only, and the Neuron ecosystem (smaller than CUDA). For cost-sensitive AWS inference of supported models, Inferentia offers real savings; the trade-off is the compilation workflow and ecosystem lock-in.

---

## 7. The Edge/Local Inference Stack

Distinct from server-class serving (vLLM/SGLang), edge/local inference (File 15 §23, File 17) runs on consumer devices:

- **llama.cpp:** the reference CPU/edge engine — GGUF quantization (Q4_K_M etc., File 02 §11.5), Metal/CUDA/Vulkan backends, runs on laptops/phones/Raspberry Pi.
- **Ollama:** llama.cpp-based, dead-simple local UX (`ollama run llama3`).
- **MLC-LLM:** TVM-compiled to any device (iOS, Android, browser via WebGPU, File 17).
- **Apple MLX:** Apple Silicon (§4).
- **Constraints:** single-user (no continuous batching), memory-frugal (consumer RAM), battery/thermal (mobile), extreme quantization (4-bit and below). Prioritizes single-user latency and footprint, not multi-tenant throughput.
- **The case:** privacy (local data), offline, no cloud cost, low latency (no network) — for on-device LLM features (autocomplete, local assistants). As small models improve (File 15 §23), edge inference grows, complementing cloud serving (large models for hard tasks, edge for private/offline/simple).

---

## 8. Hardware Selection Framework

Synthesizing into a selection framework:

| Workload / Need | Hardware |
|---|---|
| Frontier large models, high throughput | H100/B200 (NVLink TP, FP8/FP4) |
| Decode-heavy / long context | H200, GH200, MI300X (bandwidth/capacity) |
| Large models on fewer GPUs (capacity) | MI300X (192 GB), B200 (192 GB) |
| Cost-sensitive medium models | L40S, Gaudi 3 (cheaper) |
| AWS cost optimization | Inferentia (2–3× cheaper, supported models) |
| Lowest single-stream latency | Groq LPU (small-medium models) |
| Local / private / offline | Apple Silicon, llama.cpp/Ollama (edge) |

The decision drivers: the workload's roofline profile (decode-bound → bandwidth/capacity; prefill-bound → compute/FP8), the model size (capacity needs), the cost target, the ecosystem (NVIDIA most mature, others trade ecosystem for cost/specialization), and the deployment context (datacenter server vs edge device). NVIDIA's dominance comes from the combination (good hardware + the mature CUDA/engine ecosystem); the alternatives (AMD, Intel, AWS, Groq) compete on specific advantages (MI300X capacity, Inferentia cost, Groq latency) but trade ecosystem maturity. The choice is rarely just hardware specs — it's specs *plus* ecosystem (do the engines/kernels support it well?) *plus* cost *plus* deployment context. For most server-class GPU inference today, NVIDIA H100/H200 (with the mature vLLM/SGLang + FlashInfer + CUDA stack) is the default; MI300X is the credible alternative (capacity, if Triton kernels suffice); the specialized options (Inferentia, Groq) suit specific needs (AWS cost, latency).

---

## 9. Memory Bandwidth and the Decode Roofline, Across GPUs

The single most important hardware spec for inference is **memory bandwidth**, because decode is bandwidth-bound (File 01 §2). The decode step time is bounded by `weight_bytes / bandwidth` (File 02 §15.2), so decode throughput scales directly with bandwidth. Compare the generations for LLaMA-3 70B decode (FP8 weights, 70 GB, read once per step):

- **A100 (2 TB/s):** min decode step ≈ `70 GB / 2 TB/s = 35 ms` (FP8) — wait, weights read shared across batch; the point is the per-step weight read is `35 ms` floor.
- **H100 (3.35 TB/s):** ≈ `70/3.35 ≈ 21 ms` — 1.67× faster decode than A100.
- **H200 (4.8 TB/s):** ≈ `70/4.8 ≈ 14.6 ms` — 1.43× faster than H100.
- **MI300X (5.3 TB/s):** ≈ `70/5.3 ≈ 13.2 ms` — 1.58× faster than H100 (kernel maturity caps realized gains, §2).
- **B200 (~8 TB/s):** ≈ `70/8 ≈ 8.75 ms` — ~2.4× faster than H100.

This is why bandwidth is the headline decode spec: decode throughput tracks it almost linearly (the step time is the weight read). For decode-heavy workloads (chat, reasoning, File 11 §§13, 15), the bandwidth ranking (B200 > MI300X > H200 > H100 > A100) is roughly the decode-throughput ranking. The compute (FLOP/s) matters for prefill (compute-bound) but not decode — so a decode-heavy deployment should prioritize bandwidth (H200/MI300X/B200), while a prefill-heavy one (summarization, File 11 §14) values compute/FP8 (H100/B200). The roofline (File 01 §2) makes this precise: decode below the ridge point is bandwidth-bound (the GPU's bandwidth is the limit), prefill above is compute-bound (the GPU's FLOP/s is the limit) — and the hardware choice follows the workload's regime.

---

## 10. Interconnect: NVLink, NVSwitch, and InfiniBand

Beyond per-GPU specs, the **interconnect** determines multi-GPU efficiency (File 05):

- **NVLink** (intra-node GPU↔GPU): 600 GB/s (3.0, A100), 900 GB/s (4.0, H100), 1.8 TB/s (5.0, Blackwell). Low latency — essential for tensor parallelism's frequent small AllReduces (File 05 §14). NVSwitch connects all GPUs in a node at full NVLink bandwidth (any-to-any).
- **PCIe** (GPU↔GPU without NVLink): ~64 GB/s (5.0), higher latency — TP scales poorly over PCIe (File 05 §2.6, ~60–70% at TP=8 vs ~90% NVLink). GPUs without NVLink (L40S) are better for DP than TP.
- **InfiniBand / RoCE** (inter-node): ~400 Gbps (50 GB/s), higher latency — for PP/EP across nodes (File 05 §8, §11), not TP (which needs NVLink's low latency).

The interconnect tier (File 05 §8.1) dictates the parallelism placement: TP within NVLink nodes, PP/EP across InfiniBand. A GPU's NVLink presence and bandwidth are as important as its compute/bandwidth for multi-GPU serving — an H100 with NVSwitch scales TP=8 at ~90% efficiency; the same H100s on PCIe scale poorly. So hardware selection for multi-GPU includes the interconnect: NVLink/NVSwitch nodes for TP-heavy large-model serving, and the InfiniBand fabric for multi-node. The fabric must be configured correctly (NCCL over IB, GPUDirect RDMA, File 05 §15) — a misconfiguration silently degrades all collectives. The interconnect is the often-overlooked half of multi-GPU hardware: the GPUs do the compute, but the interconnect determines how efficiently they cooperate, and a fast GPU on a slow interconnect (PCIe TP) underperforms a balanced configuration.

---

## 11. Total Cost of Ownership (TCO)

Hardware selection ultimately comes down to **$/token** (File 11 §16, File 20 §economics), which depends on more than the GPU price:

- **GPU cost:** purchase (A100 ~$10K, H100 ~$30K, B200 higher) or cloud hourly (~$2–4/GPU-hr for H100). 
- **Throughput per GPU:** the goodput (File 11 §2) the GPU delivers for the workload — drives the $/token (cost / throughput).
- **Power and cooling:** datacenter GPUs draw 400–700W; power is a real operating cost.
- **Utilization:** the GPU must be well-utilized (70–85%, File 11 §29) — an expensive GPU at 30% utilization has poor $/token.
- **Ecosystem/engineering cost:** a cheaper GPU (Gaudi, Inferentia) with a less-mature stack may need more engineering effort (forks, fewer optimizations) — a hidden cost.

The TCO calculus: a more expensive GPU (H100) with higher throughput and a mature ecosystem may have *lower* $/token than a cheaper GPU (A100, or an alternative) with lower throughput or more engineering overhead. For example, H100's FP8 (2× throughput, File 10 §16.2) and mature kernels can make it cheaper per token than A100 despite the higher purchase price. Conversely, for memory-bound large-model serving, MI300X's 192 GB (fitting a model on one GPU, avoiding TP overhead) or Inferentia's cost advantage can win on $/token. The right hardware minimizes $/token *for the specific workload* (decode-bound → bandwidth; prefill-bound → compute; large model → capacity; cost-sensitive on AWS → Inferentia), accounting for throughput, price, power, utilization, and ecosystem. This is the hardware analog of the tuning methodology (File 11): measure the goodput per GPU for the workload, compute $/token, and choose the hardware (and configuration) that minimizes it. Hardware is not chosen on specs alone but on delivered $/token for the workload — which requires benchmarking the actual workload on the candidate hardware (File 11 §6), not just comparing datasheets.

---

## 12. Precision Formats Across Hardware

The supported precision formats (File 02 §11) vary by hardware and determine the quantization options:

- **A100:** BF16/FP16, TF32, **INT8** (no FP8). W8A8 quantization uses INT8 (SmoothQuant, File 02 §11.3); W4A16 uses Marlin INT4 (File 10 §16.1). No FP8 path.
- **H100/Ada (L40S):** adds **FP8** (E4M3/E5M2) via the Transformer Engine — 2× BF16 compute, the preferred low-precision path (File 02 §11.4). FP8 weights, activations, and KV.
- **Blackwell (B200):** adds **FP4** (E2M1) — 2× FP8 compute, 4-bit float for inference (File 10 §28). The next precision tier.
- **MI300X:** BF16/FP16, **FP8** (~2614 TFLOP/s), INT8. FP8 support comparable to H100.

The precision support gates the quantization strategy: on A100, use INT8 W8A8 or W4A16 (no FP8); on H100/MI300X, prefer FP8 (cleaner, 2×, File 02 §11.4); on Blackwell, FP4 for further compute. So the hardware generation determines the best quantization (File 06 §12) — an A100 deployment uses INT8/W4A16, an H100 deployment uses FP8, a Blackwell deployment can use FP4. This couples hardware and quantization choices: the FP8 advantage of H100/MI300X over A100 (2× compute, half bytes, cleaner than INT8) is a reason to prefer them for quantization-heavy serving, and Blackwell's FP4 extends this. The trend (BF16 → FP8 → FP4, each generation adding a lower-precision format that doubles compute and halves bytes, File 10 §28) means each hardware generation enables more aggressive quantization — and the kernels (FP8 GEMM, FP4 GEMM) must follow to realize it (File 10 §16.3).

---

## 13. H100 vs MI300X: A Detailed Comparison

The two leading inference GPUs, compared in depth (File 10 §8):

| Spec | H100 SXM5 | MI300X |
|---|---|---|
| HBM capacity | 80 GB | 192 GB (2.4×) |
| HBM bandwidth | 3.35 TB/s | 5.3 TB/s (1.58×) |
| BF16/FP16 | 989 TFLOP/s | 1307 TFLOP/s |
| FP8 | 1979 TFLOP/s | 2614 TFLOP/s |
| Ecosystem | CUDA (most mature) | ROCm/HIP (~85–90%) |
| Kernels | FlashInfer (native) | Triton (portable) |

**MI300X advantages:** 2.4× capacity (fit 70B FP16 on one GPU, larger models on fewer, less TP overhead), 1.58× bandwidth (faster decode in principle). For memory-bound, capacity-hungry serving (large models, long context, high concurrency), the memory wins.

**H100 advantages:** more mature ecosystem (CUDA, FlashInfer native, more optimizations, larger community), so realized performance is often closer to peak; broader feature support (some features land on NVIDIA first). The ecosystem maturity means H100 often delivers more of its theoretical performance.

**The verdict:** MI300X is the credible alternative — its memory/bandwidth advantages are real and suit memory-bound large-model serving, but the ~85–90% kernel maturity (File 10 §8) and smaller ecosystem mean H100 is often the safer choice for getting peak performance. The gap narrows as ROCm/Triton kernels mature. For a deployment, the choice depends on: is the workload memory-bound (MI300X's capacity/bandwidth wins) and are the AMD kernels mature for your model (check)? If yes, MI300X can be cost-effective; if the workload needs the latest features or peak-of-peak performance, H100's ecosystem is the safer bet. Both are supported in vLLM/SGLang (File 10 §8.3), so the choice is workload-and-ecosystem-driven, and benchmarking the actual workload on both (File 11 §6) is the way to decide.

---

## 14. Hardware Trends and the Trajectory

The hardware trajectory (File 10 §28, File 15 §22) shapes the future of inference:

- **Bandwidth growth:** A100 (2 TB/s) → H100 (3.35) → H200 (4.8) → B200 (~8) → future. Each generation's bandwidth increase directly speeds decode (bandwidth-bound, §9). Bandwidth is the inference-critical spec, and its growth makes decode progressively faster.
- **Precision reduction:** BF16 → FP8 (Hopper) → FP4 (Blackwell) → possibly lower. Each doubles compute and halves bytes (with quality validation, File 06 §12). The precision trend makes both prefill (compute) and decode (bytes) faster, requiring matching kernels (File 10 §16.3, §28).
- **Capacity growth:** 80 GB (H100) → 141 (H200) → 192 (B200, MI300X) → future. More capacity holds more KV (long context, concurrency) and larger models on fewer GPUs.
- **Interconnect growth:** NVLink 600 → 900 → 1800 GB/s; larger NVLink domains. Scales TP further (File 05).
- **Specialization:** Groq, Cerebras, Inferentia, and others target inference specifically (§§5–6, File 15 §22) — the bet that inference deserves purpose-built silicon, challenging GPU generality.

The trends compound to drive inference cost down (File 01 §8): faster bandwidth (decode), lower precision (compute and bytes), more capacity (concurrency, larger models), faster interconnect (TP scaling). Each hardware generation, combined with the software advances (the engines, the kernels, the techniques of Files 03–15), lowers $/token. For the inference engineer, the hardware trajectory means: track the new generations (each shifts the optimal configuration — more bandwidth favors decode-heavy, more capacity enables less TP, new precision needs new kernels), and re-derive the optimal hardware/config for the workload as the hardware evolves (the roofline reasoning, File 01 §2, stays constant; the numbers shift). The hardware is the foundation the rooflines rest on, and its relentless improvement — alongside the software — is half the story of inference's order-of-magnitude cost reduction (File 01 §8, File 20 §economics). Understanding the hardware (its bandwidth, compute, capacity, precision, interconnect) and how it maps to the workload (via the roofline) is essential to hardware selection and to predicting how a new generation will affect a deployment — the hardware half of the inference-engineering discipline that the software half (Files 03–15) complements.

---

## 15. Synthesis

Hardware sets the rooflines (File 01 §2) that bound inference, and its selection is a workload-driven, TCO-minimizing decision. NVIDIA dominates via the combination of strong hardware and the mature CUDA/engine/kernel ecosystem (H100 the current standard, H200/B200 the upgrades). AMD's MI300X is the credible alternative (capacity/bandwidth advantages, ~85–90% kernel maturity). Intel Gaudi, AWS Inferentia, and Groq offer specific advantages (cost, AWS savings, latency) trading ecosystem maturity. Apple Silicon and the edge stack (llama.cpp/Ollama/MLX) serve local/private inference, a distinct regime. The selection drivers are the workload's roofline profile (decode-bound → bandwidth/capacity; prefill-bound → compute/precision), the model size (capacity), the cost target (TCO/$/token), the ecosystem maturity, and the deployment context (server vs edge). The recurring lesson: hardware is chosen for delivered $/token on the workload, not specs alone — match the GPU's bandwidth/compute/capacity/precision/interconnect to the workload via the roofline, account for the ecosystem, and benchmark the actual workload (File 11 §6). The hardware trajectory (more bandwidth, lower precision, more capacity, faster interconnect) continuously improves inference economics, and tracking it — re-deriving the optimal hardware as generations evolve — is part of the inference engineer's ongoing work, grounded in the stable roofline reasoning that makes each new generation's impact predictable. Hardware and software together (this file and Files 03–15) determine inference cost and capability; mastery of both is the full discipline.

---

## 16. A Worked Hardware TCO Comparison

Quantify the TCO reasoning (§11) for serving a 70B model, decode-heavy chat workload, comparing A100 and H100.

- **A100 (BF16, ~$2/GPU-hr cloud):** 70B BF16 needs 140 GB → TP=2 (2× A100-80GB). Decode bound by 2 TB/s. Suppose the 2-GPU replica delivers ~4,000 output tok/s goodput. Cost: `2 × $2 / 4000 tok/s` → `$4/hr / 14.4M tok/hr ≈ $0.28/1M tokens`.
- **H100 (FP8, ~$3.50/GPU-hr cloud):** 70B FP8 needs 70 GB → fits on 1 H100, or TP=2 for latency. Decode bound by 3.35 TB/s, FP8 (less bytes). Suppose 1 H100 delivers ~5,000 tok/s (FP8 + faster bandwidth). Cost: `$3.50 / 5000 tok/s` → `$3.50/hr / 18M tok/hr ≈ $0.19/1M tokens`.

Despite H100's higher hourly cost (~$3.50 vs $2), its **lower $/token** ($0.19 vs $0.28) wins — because FP8 (2× compute, half bytes) and higher bandwidth deliver more throughput per GPU, and FP8 fits the model on *one* GPU (no TP overhead) where A100 needs two. This is the TCO lesson (§11): the more expensive GPU can be cheaper per token via higher throughput and better precision support. The exact numbers depend on the workload and the actual benchmarked goodput (File 11 §6), but the principle holds — H100's FP8 and bandwidth often make it cheaper per token than A100 for 70B serving, despite the higher sticker price. The same analysis applied to MI300X (capacity fits 70B FP16 on one, high bandwidth) or Blackwell (FP4, ~8 TB/s) would show their $/token, and the cheapest-per-token option (for the workload) is the right choice — found by benchmarking goodput on each candidate and computing $/token, not by comparing sticker prices or datasheets. This worked comparison is the hardware instance of the cost-optimization discipline (File 11 §16): measure delivered throughput, compute $/token, choose the minimum — applied to hardware selection. The broader point: hardware decisions, like tuning decisions (File 11) and architecture decisions (File 02), are made by measurement against the goodput-per-dollar objective, grounded in the roofline — the same disciplined, quantitative approach applied at every layer of the inference stack, from kernel to cluster to hardware procurement. Whether choosing a kernel, a parallelism degree, a quantization scheme, or a GPU, the method is constant: reason from the roofline, measure the delivered goodput, and minimize $/token subject to the SLOs and quality bar — a unifying discipline that makes the many decisions of inference engineering coherent and tractable across the full stack from silicon to service. That coherence — one disciplined method spanning every decision — is what this database has aimed to build, and the hardware layer is where it grounds out in the physical machines that ultimately run every inference workload, generation after generation, each faster and more capable than the last but governed by the same enduring physics of bandwidth, compute, and memory capacity that the roofline model captures and that the inference engineer reasons from when selecting and deploying any hardware platform, today's or tomorrow's, server-class or edge, mainstream GPU or specialized accelerator, on-premises or in the cloud — the method transfers across them all, which is exactly what makes it worth mastering.


