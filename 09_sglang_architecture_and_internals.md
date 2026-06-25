# SGLang Runtime Architecture — Components, Communication, and Execution

> **PRIMARY reference file.** This file dissects SGLang's runtime: the multi-process architecture (TokenizerManager, Scheduler, model executors), ZeroMQ and shared-memory communication, the attention backend abstraction (FlashInfer, Triton, CUDA-graph), CUDA-graph capture and replay, the continuous-batching implementation, data-parallel attention for long context, and CUDA stream management. Prerequisites: File 08 (RadixAttention and the frontend the runtime executes), File 04 (scheduling concepts), File 05 (distributed), File 10 (kernels, referenced throughout).

---

## Table of Contents

1. Design Philosophy: Off the Critical Path
2. Multi-Process Architecture
3. ZeroMQ Communication Patterns
4. Shared-Memory Batch Transfer
5. The Scheduler Process
6. The Model Executor
7. Attention Backend Architecture
8. CUDA Graph Capture and Replay
9. Continuous Batching Implementation
10. The Overlap Scheduler
11. DP Attention for Long Context
12. CUDA Stream Management
13. Custom Kernels

---

## 1. Design Philosophy: Off the Critical Path

SGLang's runtime architecture is organized around one principle: **keep everything that isn't GPU matrix math off the GPU's critical path.** Tokenization, detokenization, HTTP handling, scheduling, and the radix-tree management (File 08) are all CPU work; if they run synchronously between GPU steps, they idle the GPU. SGLang's answer is a **multi-process architecture** with efficient inter-process communication, where the CPU work happens in separate processes overlapped with GPU execution. This is the same insight vLLM's V1 engine arrived at (File 04 §14, File 07 §9) — convergent evolution toward process separation — but SGLang adopted it earlier as a core design choice. The result is that the GPU-bound model executor processes spend their time on the forward pass, while tokenization/scheduling/detokenization proceed concurrently in other processes, communicating through low-overhead channels (ZeroMQ for control, shared memory for bulk data).

---

## 2. Multi-Process Architecture

SGLang's server is a set of cooperating processes:

- **`TokenizerManager`** (main process): handles incoming HTTP requests, tokenizes prompts, and detokenizes outputs. Runs an asyncio event loop to handle many concurrent connections. It's the front door — requests enter here, get tokenized, and are dispatched to the scheduler; generated tokens come back here to be detokenized and streamed to clients.
- **`Scheduler`** (one process): receives tokenized requests, runs RadixAttention prefix matching (File 08), builds batches via continuous batching, and dispatches work to the model executors. It owns the radix tree and the KV memory pools. It's CPU-bound and not on the GPU's critical path.
- **Model executor processes** (`TpModelWorker` / `ModelRunner`, one per TP rank/GPU): run the actual model forward pass on the GPU. They receive batch metadata from the scheduler, execute the forward pass (attention, FFN, etc.), and return logits/tokens.

This separation means: HTTP + tokenization (TokenizerManager) || scheduling + RadixAttention (Scheduler) || GPU forward pass (executors) can all proceed concurrently, each in its own process, rather than serialized in one process between GPU calls. The processes communicate via ZeroMQ (control/metadata) and shared memory (bulk batch data) — §§3–4.

### 2.1 Why processes, not threads

Python's Global Interpreter Lock (GIL) prevents true CPU parallelism among threads in one process — two threads can't run Python bytecode simultaneously. So to actually overlap tokenization, scheduling, and the (GPU-launching) executor, SGLang uses separate *processes*, each with its own GIL. This sidesteps the GIL and lets the CPU-bound stages (tokenization, scheduling, detokenization) genuinely run in parallel with each other and with the GPU work. The cost is inter-process communication overhead, which SGLang minimizes with ZeroMQ and shared memory (§§3–4). The process model is also more robust (a crash in one stage is more contained) and language-agnostic (a future C++ scheduler could slot in via the same ZMQ interface).

---

## 3. ZeroMQ Communication Patterns

SGLang uses **ZeroMQ (ZMQ)** for inter-process messaging — a high-performance, brokerless messaging library with several socket patterns suited to different flows:

- **PUSH/PULL** for request dispatch: the TokenizerManager PUSHes tokenized requests; the Scheduler PULLs them. This is a load-balanced pipeline pattern — work flows one direction, distributed to the puller.
- **ROUTER/DEALER** for result return: the Scheduler routes results back to the TokenizerManager (and thus to the right client connection), using the identity-aware ROUTER/DEALER pattern for request/response correlation.
- **PUB/SUB** for broadcast: control messages that all relevant processes need (e.g. configuration changes, LoRA adapter updates) are published and subscribed.

ZMQ's advantages here: **zero-copy** message passing where possible, **non-blocking** asynchronous I/O (fits the asyncio TokenizerManager), and **language-agnostic** wire protocol (enabling future native-code components). The messages over ZMQ are control/metadata (request descriptors, output tokens, status) — small messages where ZMQ's low overhead shines. The *bulk* data (the batch tensors the executor needs) goes via shared memory (§4), not ZMQ, because copying megabytes of batch tensors through a socket each step would be too slow. This split — ZMQ for control, shared memory for bulk — is the key to low-overhead inter-process communication.

---

## 4. Shared-Memory Batch Transfer

The performance-critical communication is the Scheduler → ModelExecutor batch transfer, which happens every step and carries the batch metadata (token IDs, positions, KV block tables/token indices, attention metadata, sampling params).

### 4.1 The problem

Naively passing this batch via pickle + socket each step would incur a CPU memcpy of potentially millions of integers (e.g. 512 requests × thousands of tokens). At hundreds of steps per second, this serialization/copy overhead would dominate and idle the GPU between steps. For a batch of 512 requests × 4096 tokens, that's ~2M integers per step to serialize — far too slow via pickle (which would cost ~500 µs per transfer).

### 4.2 The solution

SGLang uses **shared memory**: the Scheduler writes the batch tensors directly into a shared-memory buffer; the ModelExecutor reads from the *same physical memory* via mmap — **zero copies**. The transfer is just a notification (over ZMQ) that the batch is ready, plus the executor reading the shared buffer. This drops the batch-transfer latency from ~500 µs (pickle+socket) to ~10 µs (shared memory + notification) — a ~50× reduction that keeps the per-step inter-process overhead negligible. Implementations use `multiprocessing.shared_memory` or torch's shared-storage mechanisms (`UntypedStorage.from_file` / shared tensors).

### 4.3 The batch tensor format

The batch is packed into tensors with the ragged-batch metadata (File 02 §20, File 06 §2): sequence lengths, packed token IDs, position IDs, the token→KV mappings (SGLang's `ReqToTokenPool`/`TokenToKVPool` indices, File 08 §9), and the attention metadata (`indptr`-style cumulative lengths for FlashInfer). Variable-length sequences are packed unpadded (the FlashAttention/FlashInfer varlen API consumes this directly), avoiding padding waste. The shared-memory layout is designed so the executor can construct the GPU tensors from it with minimal CPU work — the batch construction (File 04 §38) is itself optimized to be fast and zero-copy where possible.

---

## 5. The Scheduler Process

The Scheduler is the heart of the runtime (RadixAttention integration in File 08 §10).

- **Single-threaded event loop:** receives requests (ZMQ PULL from TokenizerManager), runs the scheduling loop, dispatches batches (shared memory + ZMQ notify to executors), receives outputs, and forwards them back (ZMQ to TokenizerManager). Being single-threaded simplifies the radix-tree and memory-pool management — all mutations happen in one loop, avoiding concurrent-modification races (File 08 §22).
- **Owns RadixAttention and the KV pools:** the radix tree, `ReqToTokenPool`, and `TokenToKVPool` (File 08 §9) live here. Each request triggers a `match_prefix`, allocation for the suffix, and tree updates.
- **Continuous batching:** builds each step's batch (§9), balancing prefill and decode under the token budget, with cache-aware scheduling (File 08 §10.1).
- **CPU-bound, off the GPU critical path:** the scheduler's work (matching, allocation, batch construction) runs in its own process, overlapped with the executors' GPU work (especially with the overlap scheduler, §10).

The scheduler is where policy (continuous batching, cache-awareness, chunked prefill, token budget) is implemented — the SGLang analog of vLLM's `Scheduler` (File 04), with RadixAttention woven in.

---

## 6. The Model Executor

The model executor processes (one per GPU/TP rank) run the forward pass.

- **`ModelRunner`:** receives the batch metadata (shared memory), constructs the GPU input tensors, runs `model.forward()` (the transformer, using the attention backend §7 and the model's parallel/fused layers, File 06), and returns logits → sampled tokens.
- **Per-TP-rank:** under tensor parallelism (File 05 §2), one executor process per rank, each holding its weight shard and KV head-shard, communicating via NCCL for the TP collectives. The scheduler dispatches the same batch plan to all ranks (broadcast), which execute in lockstep (File 05 §9.3).
- **CUDA streams:** the executor manages CUDA streams for overlapping compute with memory copies and (in some configurations) with the next step's preparation (§12).
- **CUDA graphs:** for decode batches of captured sizes, the executor replays a pre-captured CUDA graph instead of eager execution (§8), eliminating per-kernel launch overhead.

The executor is purely a GPU-work component — it receives a plan and executes it, returning results. This clean role (the scheduler decides, the executor executes) mirrors vLLM's worker design (File 05 §25) and is what lets the executor be CUDA-graph-captured and tightly optimized while the scheduler's CPU logic runs elsewhere.

---

## 7. Attention Backend Architecture

SGLang abstracts attention behind an `AttentionBackend` interface with multiple implementations, selected by hardware and configuration (`--attention-backend`).

### 7.1 FlashInfer backend (default on NVIDIA)

The default high-performance backend on H100/A100 is **FlashInfer** (Ye et al., arXiv 2501.01005, File 10) — a library of optimized attention kernels purpose-built for LLM serving:

- **Paged KV:** attention over the token-granular paged KV cache (File 08 §9), gathering K/V via the page/token indices.
- **Variable-length batches:** the `indptr` (cumulative-length) API handles ragged batches mixing prefill (multi-token queries) and decode (single-token queries) without padding (File 02 §20).
- **Specialized wrappers:** `BatchPrefillWithPagedKVCacheWrapper` for prefill and `BatchDecodeWithPagedKVCacheWrapper` for decode — each optimized for its shape regime (prefill: many queries, compute-bound; decode: one query, memory-bound). `begin_forward()`/`end_forward()` manage the wrapper's workspace state.
- **GQA-optimized:** reads each shared KV head once per group (File 02 §28), preserving GQA's bandwidth saving.

FlashInfer is significantly faster than naive paged attention on H100 (File 03 §16.4), which is why it's the default for both SGLang and vLLM on NVIDIA. Its workspace-buffer model requires allocating scratch space, managed by the backend.

### 7.2 Triton backend

A pure-**Triton** (File 10 §Triton) implementation of the attention kernels (`triton_attn_fwd` for prefill, etc.), JIT-compiled. It's the fallback for hardware where FlashInfer isn't available — notably **AMD ROCm** (HIP Triton, File 16) — and reaches ~80% of FlashInfer's CUDA performance. The Triton backend also serves as a portable reference and a base for implementing novel attention variants not yet in FlashInfer.

### 7.3 The torch (reference) backend

A pure-PyTorch reference implementation — correct but slow (no kernel fusion, materializes intermediate tensors). Used for debugging and correctness validation, not production.

### 7.4 Backend selection

SGLang auto-selects based on hardware (FlashInfer on supported NVIDIA, Triton on ROCm or as fallback) and the model (MLA, certain variants may need specific handling). `--attention-backend flashinfer|triton|torch` overrides. For production NVIDIA deployments, FlashInfer is the choice; the backend determines the exact metadata layout the scheduler/executor must produce (File 06 §13). A mismatch between the backend's page-size expectations and the configured KV block granularity surfaces as errors or slowdowns (File 03 §13).

---

## 8. CUDA Graph Capture and Replay

CUDA graphs are one of SGLang's most impactful performance features, especially for decode.

### 8.1 What CUDA graphs do

A CUDA graph captures a sequence of GPU operations (kernel launches, memory ops, collectives) as a single replayable unit. Replaying the graph re-issues all those operations with **one launch** instead of one launch per kernel. This eliminates the per-kernel **launch overhead** (~5 µs/kernel), which for a decode step with ~100+ kernels (matmuls, attention, norms, AllReduces) is `100 × 5 µs = ~500 µs` of pure overhead — a large fraction of a fast decode step. CUDA graphs cut this to a single replay launch, giving **2–3× speedup for small-batch decode** where launch overhead dominates relative to the small compute (File 02 §15, File 04 §31).

### 8.2 The static-shape requirement

CUDA graphs require **fixed tensor shapes** at capture time — a graph captured for batch size 16 only replays for batch size 16. For decode, `seq_len = 1` per sequence (one new token), so the only varying dimension is the batch size (number of running sequences). SGLang captures graphs for a **ladder of batch sizes** (1, 2, 4, 8, 16, 32, …, up to a max), and at runtime **pads** the actual batch up to the nearest captured size and replays that graph (trimming the padding's output). Prefill, with its variable sequence lengths, is harder to graph and typically runs eager; decode is the graph-captured path (File 04 §31). This is why chunked prefill's predictable step structure (File 04 §7.5) helps — it keeps steps decode-dominated and graph-friendly.

### 8.3 TP and CUDA graphs

Under tensor parallelism, the NCCL AllReduces (File 05 §2.4) must be *inside* the captured graph, or the graph would replay the matmuls but not the collectives. CUDA-graph-compatible NCCL (CUDA 11.3+) allows capturing collectives, but requires all ranks to capture **simultaneously** with communicators initialized beforehand (File 05 §30). SGLang captures the decode graph including the TP AllReduces, so a replay performs the full distributed decode step with no per-kernel launch overhead — a big win for TP decode, which is full of small kernels and AllReduces.

### 8.4 The CudaGraphRunner

SGLang's `CudaGraphRunner` (conceptually): at startup, `capture()` profiles the batch sizes and captures a graph for each; at runtime, `replay(batch)` pads the input to the nearest captured size, replays, and trims the output. If a batch size exceeds the max captured (or graphs are disabled), it falls back to eager mode. Memory cost: each captured graph holds a snapshot of GPU memory state (~tens to ~100 MB per graph), so capturing many batch sizes costs memory — a trade-off between graph coverage (more sizes captured → fewer eager fallbacks) and memory. The captured set is tuned to cover the common decode batch sizes. CUDA graphs are most beneficial for small models and small batches (where launch overhead is relatively large) and for TP decode (many AllReduces); their benefit shrinks for large models with long compute-heavy steps (where launch overhead is a small fraction).

---

## 9. Continuous Batching Implementation

SGLang's `Scheduler.schedule()` loop (RadixAttention integration in File 08 §10):

1. **Check the waiting queue** for new requests.
2. **For each new request:** run RadixAttention `match_prefix` to compute the prefix-hit length (File 08 §5).
3. **Optionally sort** by prefix-hit length (cache-aware scheduling, favoring cache-warm requests, File 08 §24).
4. **Allocate KV** (token slots in `TokenToKVPool`) for the non-cached suffix only.
5. **Merge** prefill and decode requests into one batch (varlen, File 02 §20).
6. **Enforce the token budget:** `max_total_num_tokens` per step bounds compute/memory.
7. **Build the `ModelWorkerBatch`** (the batch metadata) and dispatch via shared memory to the executors.

### 9.1 Running-batch management

A `running_batch` holds the active `Req` objects. After each step: update each `Req` (append the new token, check stop conditions, extend its radix-tree path, update its token→KV mapping), remove finished `Req`s (decrement ref_counts, File 08 §4.3), and promote from the waiting queue if budget allows. The state machine (`Waiting`, `Running`-prefill, `Running`-decode, `Finished`) mirrors vLLM's (File 03 §12, File 04 §12.4), with abort handling on client disconnect (File 04 §28).

### 9.2 Token budget enforcement

`max_total_num_tokens` (and `--max-running-requests`) bound the per-step work, preventing OOM and latency spikes. Decode tokens = number of running sequences (1 each); prefill tokens = the remaining budget, allocated to admitting/chunking new requests. As more requests enter decode, less budget remains for prefill — the same prefill/decode budget tension as vLLM (File 04 §6.3), managed with chunked prefill (`--chunked-prefill-size`).

---

## 10. The Overlap Scheduler

SGLang's **overlap scheduler** is a key performance feature that hides scheduling/CPU overhead behind GPU execution — the same idea as vLLM's async scheduling (File 04 §11.3) but a defining part of SGLang's design.

### 10.1 The problem

Even with the scheduler in its own process (§5), there's a dependency: planning step `t+1` requires knowing step `t`'s output (which token each sequence generated, whether any stopped, how the radix tree changed). Naively, the scheduler waits for the GPU to finish step `t`, processes the results, then plans `t+1` — serializing CPU scheduling and GPU compute, idling the GPU during the CPU work.

### 10.2 The overlap

The overlap scheduler **plans step `t+1` while the GPU executes step `t`**. It optimistically prepares the next batch (most of the batch composition is known — the running sequences continue, new requests can be admitted) before step `t`'s results arrive, then reconciles with the actual results (which sequences finished, what tokens were produced) when they do. By the time the GPU finishes step `t`, step `t+1` is largely prepared and can launch with minimal delay. This hides the scheduler's CPU latency (matching, allocation, batch construction) behind the GPU's compute, so the GPU stays busy step after step rather than stalling for CPU scheduling. The effect is largest for fast steps (small models, small batches, CUDA-graph decode) where the scheduler overhead would otherwise be a large fraction of step time (File 04 §35's analysis).

### 10.3 The reconciliation challenge

The complication is that some of step `t+1`'s plan depends on step `t`'s results — specifically, which sequences stopped (freeing their slots) and the new tokens (extending the radix tree, affecting future matches). The overlap scheduler handles this by preparing the parts that don't depend on `t`'s output and reconciling the dependent parts when results arrive — accepting a small amount of speculative preparation that's occasionally adjusted. This is more complex than synchronous scheduling but recovers the GPU idle time. Combined with the process separation (§2) and shared-memory transfer (§4), the overlap scheduler is why SGLang achieves low per-step overhead — the GPU executors stay saturated while tokenization, scheduling, and detokenization proceed concurrently in other processes.

---

## 11. DP Attention for Long Context and MoE

SGLang's `--enable-dp-attention` implements **data-parallel attention**, useful in two scenarios.

### 11.1 For MoE models

For MoE models (File 02 §4.2, File 05 §5), the experts dominate compute and are expert-parallel (EP), while attention is comparatively cheap. Tensor-parallelizing the attention (splitting its heads across ranks) incurs AllReduce overhead for little benefit. **DP attention** instead *replicates* the attention computation across ranks (each rank computes attention for its share of the *requests*, not a share of the heads), while the experts remain expert-parallel. This avoids the attention AllReduces and can be more efficient for MoE serving where attention isn't the bottleneck. It's distinct from request-level data parallelism (File 05 §7) — here it's attention-specific within a single model deployment. SGLang surfaced this optimization prominently for DeepSeek-style MoE serving.

### 11.2 For very long single sequences

DP attention also addresses very long context (100K+ tokens) where a single sequence's KV doesn't fit comfortably or the attention compute is large (File 05 §6, File 13). The single sequence is partitioned across GPUs (each holds a fraction of K/V), and attention is computed with cross-GPU exchange of K/V (ring-style) and online-softmax combination (File 02 §3.2) — conceptually like ring attention (File 05 §6.2). Each GPU processes its share with ~2× memory headroom for 2 GPUs (proportional for more), enabling contexts that wouldn't fit on one GPU. This makes SGLang capable of extreme long-context serving by spreading the per-sequence attention memory and compute across GPUs.

### 11.3 Implementation

Each GPU processes local Q (full heads, a subset of sequence positions for the long-sequence case, or a subset of requests for the MoE case), fetches K/V from neighbors as needed, and accumulates attention output using the numerically-stable log-sum-exp combination across chunks. The mode is selected by configuration and the workload (MoE serving, or long-context). DP attention is one of the distributed features SGLang shipped aggressively, reflecting its focus on MoE and long-context frontiers.

---

## 12. CUDA Stream Management

The model executor uses multiple **CUDA streams** to overlap independent GPU operations, a core technique for keeping the GPU busy (File 05 §34's overlap principle).

- **Main compute stream:** runs the matmuls, attention, and other forward-pass kernels.
- **Copy stream(s):** handle host↔device transfers — uploading the new token embeddings / batch data (H2D) and downloading output logits (D2H) — concurrently with compute, so transfers don't stall the compute stream.
- **Event synchronization:** CUDA events (`stream.wait_event`) coordinate dependencies between streams (e.g. compute must wait until the input copy completes). Proper event synchronization ensures correctness while maximizing overlap.

### 12.1 Async tokenization and detokenization

Beyond CUDA streams, the *process-level* overlap (§2) means tokenization (in TokenizerManager) runs concurrently with GPU execution of other requests, and **streaming detokenization** (File 02 §18.2) turns each new token into client-visible text as it's generated — also concurrent with the next step's compute. The detokenizer's incremental state (handling multi-token characters correctly) runs in the TokenizerManager process, off the executor's critical path. This layered overlap — process-level (tokenize/schedule/detokenize concurrent with GPU) and stream-level (copies concurrent with compute) — is how SGLang keeps the GPU as the only thing on the critical path.

### 12.2 Speculative decoding stream overlap

For speculative decoding (File 12), the draft and target model executions can be overlapped on CUDA streams — the draft proposing tokens on one stream while the target verifies on another (where the dependency structure allows), hiding some of the draft's cost behind the target's compute. EAGLE-style speculation (File 12) in SGLang uses such overlap to maximize the speedup.

---

## 13. Custom Kernels

SGLang ships custom CUDA/Triton kernels (`sgl-kernel` / `sglang._kernels`, File 10) for operations where fusion or specialization beats generic PyTorch:

- **RoPE fusion:** applying rotary embeddings inside/after the QKV projection (File 02 §5.1), avoiding a separate pass.
- **RMSNorm:** fused normalize + scale (+ residual), accumulating the sum-of-squares in FP32 (File 02 §13).
- **SiLU + multiply** for SwiGLU FFN (`act(gate) ⊙ up`, File 02 §24).
- **Quantized matmuls:** GPTQ Marlin, AWQ, FP8 GEMMs (File 06 §3, File 10).
- **Sampling kernels:** fused penalty/temperature/top-k/top-p sampling (File 02 §8.5).
- **Grammar-mask application** for XGrammar (File 08 §11.4) — applying per-sequence allowed-token masks across a heterogeneous batch.

These mirror vLLM's fused kernels (File 06 §17, File 10) — the same set of fusions (RoPE, RMSNorm, SwiGLU, quantized matmul, sampling) because the same bandwidth bottlenecks drive both engines. SGLang and vLLM both increasingly rely on FlashInfer for attention and share kernel ideas; some kernels are developed collaboratively or adopted across engines. The custom kernels are where the per-GPU compute efficiency (File 10) lives, beneath the architecture this file describes.

---

## 14. A Full Request-Flow Trace

To see the architecture in motion, trace a request end to end through the processes.

1. **HTTP arrival (TokenizerManager process):** a client POSTs to the OpenAI-compatible endpoint. The TokenizerManager's asyncio loop accepts the connection, applies the chat template (File 07 §11), and **tokenizes** the prompt — CPU work, running concurrently with other requests' processing and with the GPU's current step.
2. **Dispatch (ZMQ PUSH → Scheduler PULL):** the tokenized request (a small message: token IDs + sampling params + request ID) is PUSHed over ZMQ to the Scheduler. Zero bulk data here — just the request descriptor.
3. **Scheduling (Scheduler process):** the Scheduler receives the request, runs RadixAttention `match_prefix` (File 08 §5) to find its cached prefix, allocates KV slots for the suffix, and (when budget allows) includes it in the next batch. With the overlap scheduler (§10), this planning happens while the GPU is busy with the prior step.
4. **Batch dispatch (shared memory + ZMQ notify → Executors):** the Scheduler writes the batch metadata to the shared-memory buffer (§4) and notifies the executor processes over ZMQ. Zero-copy bulk transfer.
5. **Forward pass (Executor processes, one per GPU):** each executor reads the batch from shared memory, builds GPU tensors, and runs `model.forward()` — using the FlashInfer attention backend (§7), the model's parallel/fused layers (File 06), and (for decode) replaying a CUDA graph (§8). Under TP, the executors run in lockstep with NCCL AllReduces (File 05 §2). The sampler produces the next token.
6. **Results return (ZMQ → TokenizerManager):** the generated token(s) are sent back (via ZMQ) to the TokenizerManager.
7. **Detokenization and streaming (TokenizerManager):** the new token is **incrementally detokenized** (File 02 §18.2) — handling multi-token characters correctly — and streamed to the client via SSE (File 07 §13). This runs concurrently with the next step's GPU work.
8. **Loop:** steps 3–7 repeat each decode step until a stop condition (EOS, max_tokens, stop string), at which point the request's KV is freed (ref-counts decremented, radix-tree nodes become evictable, File 08 §4.3) and the stream is closed.

The key observation: at any instant, the TokenizerManager is tokenizing/detokenizing some requests, the Scheduler is planning the next batch, and the Executors are running the current forward pass — **three stages pipelined across processes**, so the GPU (the expensive resource) stays busy while the CPU work overlaps. This pipelining, enabled by the process separation and the overlap scheduler, is the architectural payoff.

---

## 15. The Overlap Scheduler, Worked

Quantify the overlap scheduler's benefit (§10) with a timing example on a fast model where a decode step is 3 ms of GPU time and the scheduler's per-step CPU work (match, allocate, build batch) is 1 ms.

**Without overlap (synchronous):**
```
per step = 3 ms (GPU) + 1 ms (CPU scheduling, GPU idle) = 4 ms
GPU utilization = 3/4 = 75%
```

**With overlap:**
```
the 1 ms CPU scheduling for step t+1 happens during step t's 3 ms GPU work
per step ≈ max(3 ms GPU, 1 ms CPU) = 3 ms
GPU utilization ≈ 100%   (CPU work fully hidden)
throughput gain ≈ 4/3 ≈ 1.33×
```

The overlap recovers the 25% the GPU would otherwise idle waiting for CPU scheduling. The gain is larger the bigger the CPU/GPU ratio — i.e. for fast steps (small models, small batches, CUDA-graph decode where GPU time is short) the scheduler's fixed CPU cost is a larger fraction, so hiding it matters more. For large models with long compute-heavy steps (tens of ms), the 1 ms scheduling is already a small fraction and the overlap matters less — but it never hurts. This is the same analysis as vLLM's multi-step vs async (File 04 §35); SGLang's overlap scheduler is the "async" approach (hide CPU behind GPU) rather than the "multi-step" approach (amortize CPU over many GPU steps), and it preserves responsiveness (no blindness window, unlike multi-step). Combined with shared-memory transfer (§4, ~10 µs vs ~500 µs) and process separation (tokenize/detokenize concurrent), SGLang minimizes every non-GPU overhead, approaching the ideal where the GPU forward pass is the only thing on the critical path.

---

## 16. CUDA-Graph Coverage and Memory Analysis

The CUDA-graph batch-size ladder (§8.2) involves a coverage/memory trade-off worth quantifying.

Suppose decode batch sizes in production range from 1 to 256. Capturing a graph for every size (256 graphs) would cover all cases (no eager fallback) but cost `256 × ~50 MB ≈ 12.8 GB` of memory for graph snapshots — prohibitive. Capturing a geometric ladder (1, 2, 4, 8, 16, 32, 64, 128, 256 — 9 graphs) costs `9 × ~50 MB ≈ 450 MB` and covers all sizes with padding (a batch of 40 pads up to 64's graph). The padding wastes some compute (running 64's graph for 40 real sequences — 24 wasted "slots"), but the launch-overhead saving dominates for small batches. The ladder spacing trades padding waste (finer ladder → less padding) against memory (finer ladder → more graphs). A geometric ladder is a good default: padding waste is at most ~2× the batch (worst case just above a captured size), bounded and acceptable for the launch-overhead win.

For batches exceeding the max captured size (256 here), the executor falls back to **eager mode** (no graph) — correct but slower (full launch overhead). So the max captured size should cover the typical operating range; rare larger batches pay the eager cost. The memory for graphs comes out of the budget that would otherwise be KV cache, so there's a second trade-off: more graph coverage vs more KV cache (more concurrency). SGLang's `--cuda-graph-max-bs` and related flags tune this (File 11). The net guidance: capture a geometric ladder up to the expected max decode batch, accepting ~hundreds of MB of graph memory for the 2–3× small-batch decode speedup — a strong trade for decode-heavy or small-model workloads, less impactful for large models where steps are compute-dominated.

---

## 17. Process Startup and Initialization

When an SGLang server starts, the processes initialize in a coordinated sequence (the analog of vLLM's warmup, File 07 §19):

1. **Launch processes:** the TokenizerManager, Scheduler, and per-GPU executor processes are spawned, and their ZMQ sockets + shared-memory buffers are wired up.
2. **Model loading:** each executor loads its shard of the weights (File 05 §18, File 06 §4) — minutes for large models.
3. **KV pool allocation:** the Scheduler/executors profile memory and allocate the KV pools (`ReqToTokenPool`/`TokenToKVPool`, File 08 §9), sized by `--mem-fraction-static` (File 11). The radix tree is initialized empty.
4. **CUDA graph capture:** the executors capture decode graphs for the batch-size ladder (§8, §16) — adds startup time and reserves graph memory.
5. **Warmup:** a few forward passes trigger lazy kernel compilation/autotuning (FlashInfer workspace allocation, Triton JIT) so the first real request doesn't pay that cost.
6. **Ready:** the server reports healthy and accepts requests.

This startup sequence (load + profile + capture + warmup) takes from tens of seconds (small models) to several minutes (large models with many graphs), the same cold-start consideration as vLLM (File 07 §19) — autoscaling must be proactive because a new replica isn't ready for minutes (File 11 §autoscaling, File 19). The multi-process startup also means the processes must rendezvous (establish ZMQ connections, shared memory) before serving, and a failure in any (e.g. an executor failing to load weights) must be detected and surfaced rather than leaving the server half-initialized.

---

## 18. FlashInfer Wrapper Internals

The FlashInfer backend (§7.1) deserves deeper treatment because its design shapes how SGLang executes attention. FlashInfer separates attention into a **plan** phase and a **run** phase:

- **Plan (`begin_forward` / `plan`):** given the batch's structure (the `indptr` arrays encoding each sequence's KV length, the page table, the number of heads), FlashInfer computes a *schedule* — how to partition the work across thread blocks for good GPU occupancy (the split-K-style partitioning for long contexts, File 03 §16.3). This planning is CPU/lightweight-GPU work that can be done once per batch shape and reused.
- **Run (`forward`):** execute the attention using the plan, gathering paged K/V via the page indices and computing the (online-softmax) attention.

This plan/run split lets SGLang amortize the planning: for CUDA-graph-captured decode (§8), the plan is baked into the graph for each captured batch size, so replay skips re-planning. The **workspace buffer** FlashInfer requires (scratch space for the partitioned reductions) is pre-allocated. The wrappers — `BatchPrefillWithPagedKVCacheWrapper` and `BatchDecodeWithPagedKVCacheWrapper` — encapsulate this state. The decode wrapper is GQA-aware (reads each KV head once per query-head group, File 02 §28), and the prefill wrapper handles the causal-with-cached-prefix attention (prefill tokens attending over cached prefix KV plus their own, File 02 §30). FlashInfer's efficiency on these paged, varlen, GQA workloads — better than the original vLLM paged kernel (File 03 §16.4) — is why it became the default for both engines. SGLang's tight integration with FlashInfer (using its plan/run model, CUDA-graph compatibility, and page layout) is a significant part of its per-GPU performance.

---

## 19. The Detokenizer

SGLang separates detokenization into its own concern (a `DetokenizerManager` or detokenization within the TokenizerManager process), reflecting that turning token IDs into streamed text is nontrivial CPU work that must stay off the critical path.

The detokenizer maintains **incremental state** per request (File 02 §18.2): a buffer of recent tokens, a read offset, and a prefix offset. As each new token arrives, it decodes the buffer and emits only the *stable* prefix of the decoded text — the part that cannot change when more tokens arrive. This correctly handles multi-token characters: a single emoji or CJK character may span several tokens, and emitting per-token decodes would produce replacement characters (`�`) or split graphemes. The incremental detokenizer waits until bytes are unambiguously complete before emitting them. It also handles stop-string matching (File 02 §18.3) — checking the decoded suffix against stop strings and truncating precisely at the match (not at a token boundary).

Running detokenization in a separate process (or at least off the executor's path) means it proceeds concurrently with the next GPU step — the executor doesn't wait for text to be produced and streamed. For high-throughput serving with many concurrent streams, this concurrency matters: detokenizing hundreds of streams' tokens each step is real CPU work that would otherwise compete with scheduling or stall the GPU. SGLang's (and vLLM V1's, File 07 §9) separation of the detokenizer is a recognition that even this "trivial" step deserves to be off the critical path.

---

## 20. Fault Tolerance Across Processes

The multi-process architecture has implications for reliability (File 05 §22, File 19):

- **Executor failure:** if a model executor process dies (GPU error, OOM), the model replica is down (under TP, all executors are needed). The server must detect this (the Scheduler notices the executor stopped responding) and fail/restart the replica. In-flight requests are lost (no checkpointing for inference). At the fleet level, multiple replicas behind a load balancer provide resilience (File 05 §22, §35).
- **Scheduler failure:** the Scheduler holds the radix tree and request state; its failure loses that state and requires a restart. It's a single point within a replica.
- **TokenizerManager failure:** loses in-flight HTTP connections; restart re-establishes the front door.
- **Detection:** the processes monitor each other (ZMQ heartbeats / liveness); a dead process is detected and the replica is marked unhealthy (failing the `/health` check, File 19) so the load balancer routes around it.

The process model is more robust than a monolith in that a crash is somewhat contained and detectable, but a replica is still all-or-nothing (any process failing takes the replica down). As with vLLM (File 05 §22), the reliability strategy is multiple replicas (DP) for fault tolerance, graceful health-check-driven routing, and load shedding under degradation (File 19). The process boundaries also ease *disaggregation* (File 04 §10, File 15) — separating prefill and decode is more natural when the architecture is already multi-process with defined communication.

---

## 21. Multi-Node SGLang

For models too large for one node or for scale-out, SGLang runs multi-node, with the same interconnect-tier discipline as vLLM (File 05 §8, §11): TP within NVLink nodes, larger parallelism (PP/EP) across nodes over InfiniBand. The executor processes span GPUs across nodes, with NCCL/RCCL handling the TP/EP collectives over the fabric. The Scheduler coordinates centrally and dispatches to all executors. For MoE at scale (DeepSeek-V3-class), SGLang combines DP attention (§11), EP across nodes, and the communication-overlap techniques (File 05 §21), tracking the reference EP128 deployment. The ZMQ + shared-memory communication is intra-node (between the local processes); cross-node is the NCCL collectives among executors and any cross-node coordination. Multi-node SGLang requires the same fabric configuration care as vLLM (NCCL over IB, GPUDirect RDMA, File 05 §15, §26) — misconfiguration silently degrades the collectives.

---

## 22. SGLang vs vLLM V1 Architecture

SGLang's architecture and vLLM's V1 engine (File 04 §14, File 07 §9) converged on strikingly similar designs — a case of independent teams reaching the same conclusions from the same constraints:

| Aspect | SGLang | vLLM V1 |
|---|---|---|
| Process model | TokenizerManager / Scheduler / Executors (separate processes) | API server / EngineCore / Detokenizer (separate processes) |
| Inter-process comm | ZeroMQ (control) + shared memory (batch) | efficient IPC + shared memory |
| Scheduling overlap | overlap scheduler (plan t+1 during t) | async scheduling (plan t+1 during t) |
| Tokenization | separate process (TokenizerManager) | separate from engine core |
| Detokenization | separate (incremental) | separate Detokenizer process |
| Attention | FlashInfer default (Triton fallback) | FlashInfer/FlashAttention default |
| CUDA graphs | decode-path ladder + TP collectives | decode-path ladder + TP collectives |
| Prefix cache | RadixAttention (token-granular tree) | APC (block-granular hash) |
| Frontend | program-aware DSL | request-level (OpenAI API) |

The convergence (separate processes, shared-memory batch transfer, scheduling/execution overlap, FlashInfer attention, CUDA-graph decode) reflects that both faced the same problem — keep CPU work off the GPU critical path, minimize per-step overhead — and found the same answers. The remaining differences are SGLang's RadixAttention (token-granular tree vs vLLM's block-hash, File 08 §8) and SGLang's program-aware frontend (no vLLM equivalent). Architecturally, the two engines are now close cousins; their distinctive value lies in the prefix-caching policy and the frontend, not in the process/execution architecture, which has converged. This convergence is healthy — it means the systems-architecture lessons (process separation, overlap, shared memory, FlashInfer, CUDA graphs) are settled best practices, and the engines compete on the higher-level policy and programming-model choices (File 17).

---

## 23. Memory Pool Internals

SGLang's two-level memory indirection (File 08 §9) is worth examining in execution detail, because it's how token-granularity sharing is realized efficiently.

- **`ReqToTokenPool`:** a 2D structure mapping `(request_index, token_position) → token_kv_index`. Each active request occupies a row; the row records, for each of its token positions, the index into the KV pool where that token's K/V lives. This is the request's token-granular "block table" (File 03 §3). For a shared prefix (RadixAttention), multiple requests' rows point to the *same* KV indices for the shared positions — that's how the physical KV is shared.
- **`TokenToKVPool`:** maps `token_kv_index → physical KV memory` (the actual K/V tensors in HBM). This is the physical pool, allocated once at startup (sized by `--mem-fraction-static`). A free list manages available KV indices (O(1) allocate/free).

The indirection (`request → token indices → physical KV`) decouples the logical token sequence from physical layout, enabling: (a) token-granular sharing (shared prefix tokens map to shared KV indices), (b) flexible reuse (freed KV indices return to the pool for any request), and (c) the radix tree's references (tree nodes hold KV index ranges). When RadixAttention matches a prefix, it sets the new request's `ReqToTokenPool` row to point at the cached KV indices for the matched positions (incrementing ref counts) and allocates fresh indices only for the suffix. This is the concrete mechanism behind §8's automatic reuse. The pool sizes and the indirection overhead (a token-index per token) are the cost of token granularity, kept manageable by the radix tree's structural compression (File 08 §9.4).

### 23.1 The Req object

Each request is a `Req` object tracking: the request ID, the prompt and generated token IDs, the `ReqToTokenPool` row (its token→KV mapping), the current position in the radix tree, the sampling parameters, the grammar matcher (if constrained, File 08 §11.4), the speculative-decoding state (if applicable), and the request state (`Waiting`/`Running`/`Finished`). The scheduler's `running_batch` is a collection of `Req`s; each step updates them (append token, check stop, extend tree path, update KV mapping). The `Req` is the unit of request state, analogous to vLLM's `Sequence`/`SequenceGroup` (File 04 §4.2). Its lifecycle — admitted from waiting, run through prefill then decode, finished and freed — drives the memory pool and radix tree mutations.

---

## 24. torch.compile Integration

SGLang supports `torch.compile` (`--enable-torch-compile`) as an additional optimization, particularly with the `reduce-overhead` mode (which itself uses CUDA graphs). torch.compile traces the model's forward pass and applies graph-level optimizations — operator fusion, layout optimization, and dead-code elimination — generating optimized kernels. It's most beneficial for **small models** (<7B) where the Python/framework overhead is a larger fraction of step time and where the fusion of many small operations yields a relatively bigger win. For large models, the hand-tuned kernels (FlashInfer attention, the custom fused kernels §13) already capture most of the benefit, and torch.compile's additional gain is smaller. torch.compile requires a warmup compilation step (adding startup time, File 07 §19) and can interact with CUDA graphs (both want static shapes). It's an opt-in optimization, complementary to the always-on CUDA-graph decode (§8); SGLang's default path (FlashInfer + custom kernels + CUDA graphs) is already well-optimized, with torch.compile an extra lever mainly for small-model deployments.

---

## 25. Speculative Decoding Architecture (EAGLE)

SGLang supports speculative decoding (File 12), notably **EAGLE** (feature-level drafting, File 02 §9.3, File 12). The architecture:

- **Draft and target in the executor:** the draft (EAGLE head/model) and target model both run in the executor process. The draft proposes tokens (EAGLE uses the target's hidden states as input, achieving high acceptance), and the target verifies them in one forward pass.
- **`SpecInfo` per request:** the speculative state (proposed tokens, the draft tree for tree-speculation) is tracked per `Req` (§23.1).
- **Tree attention:** for tree-structured speculation (proposing a tree of candidates verified together, File 02 §9.3), SGLang uses a tree-attention kernel (a FlashInfer extension) that verifies the whole candidate tree in one pass with the appropriate attention mask.
- **Stream overlap:** the draft and target executions overlap on CUDA streams (§12.2) where the dependency structure allows, hiding some draft cost behind target compute.
- **RadixAttention compatibility:** speculative requests reuse cached prefixes like any other (File 08 §27); only accepted tokens' KV becomes part of the permanent cached path.

EAGLE-2's dynamic draft-tree construction (File 12) adapts the speculation tree based on per-node acceptance probability, raising the effective speedup. SGLang's early and aggressive EAGLE support reflects its tendency to ship cutting-edge algorithms quickly (File 08 §39). The architecture integration — speculative state in the `Req`, tree attention in the backend, stream overlap in the executor — shows how a major feature threads through the runtime's components.

---

## 26. Quantization in SGLang

SGLang supports the same quantization landscape as vLLM (File 06 §3, File 02 §11): FP8 (the preferred low-precision path on H100, with FP8 KV cache), GPTQ/AWQ W4A16, INT8 W8A8, and others, with matching kernels (Marlin, FP8 GEMMs). The quantized linear layers and matmul kernels (§13) handle the dequantization, and the quantization composes with TP (consistent scale sharding, File 05 §19) and with the attention backend (FP8 KV in FlashInfer). For MoE models, quantized expert weights (FP8) are supported in the fused MoE kernels. DeepSeek-V3-class serving uses FP8 throughout (File 05 §29.2), which SGLang supports as part of its MoE/MLA focus. The quantization story is largely shared between the engines — the same methods, the same kernels (often the same FlashInfer/Marlin implementations), driven by the same memory and compute bottlenecks — so the choice of quantization (File 06 §12) is similar regardless of engine; the engine difference is in prefix caching and the frontend, not quantization.

---

## 27. What Makes SGLang Fast: A Synthesis

SGLang's performance comes from stacking several overhead-elimination techniques, each removing a different non-GPU cost from the critical path:

1. **RadixAttention** (File 08): eliminates redundant prefill compute and KV memory for shared prefixes — the biggest win for prefix-heavy workloads (order-of-magnitude prefill reduction).
2. **Process separation** (§2): tokenization, scheduling, and detokenization run concurrently with GPU execution, sidestepping the GIL (§28).
3. **Shared-memory batch transfer** (§4): ~10 µs instead of ~500 µs per step for the scheduler→executor handoff.
4. **The overlap scheduler** (§10): hides the scheduler's CPU latency behind GPU execution (~25% utilization recovery for fast steps).
5. **CUDA graphs** (§8): eliminates per-kernel launch overhead for decode (~2–3× for small batches).
6. **FlashInfer attention** (§7): the fastest paged-attention kernels on NVIDIA.
7. **Custom fused kernels** (§13): RoPE/RMSNorm/SwiGLU/quantized-matmul fusions cut HBM round-trips.
8. **XGrammar** (File 08 §11): near-free structured output.

Each addresses a specific overhead: redundant compute (RadixAttention), GIL serialization (processes), transfer cost (shared memory), CPU-on-critical-path (overlap scheduler), launch overhead (CUDA graphs), attention efficiency (FlashInfer), HBM traffic (fusion), constraint cost (XGrammar). The compounding of these — none individually dominant, but together substantial — is why SGLang achieves strong performance, particularly on the multi-call, structured, MoE, and long-context workloads it targets. The architecture (this file) provides the substrate (processes, communication, backends, graphs) on which these optimizations run; RadixAttention and the frontend (File 08) provide the algorithmic differentiators.

---

## 28. The GIL and Why Processes Matter (Deeper)

Section 2.1 noted the GIL motivation; here is why it's decisive. Python's Global Interpreter Lock serializes Python bytecode execution within a process — only one thread runs Python at a time. For an LLM server, the CPU-bound stages (tokenization, scheduling with radix-tree management, sampling-parameter processing, incremental detokenization, stop-string checking) are substantial Python work. In a single-process, multi-threaded design, these would serialize against each other *and* against the thread that launches GPU kernels — so while one thread tokenizes, no other thread can schedule, and the GPU-launching thread can't run, idling the GPU.

By splitting into separate *processes* (each with its own GIL), SGLang lets these stages run truly in parallel: the TokenizerManager tokenizes/detokenizes, the Scheduler schedules, and the executors launch GPU work — all simultaneously on different CPU cores, each holding its own GIL. This is the fundamental reason the architecture is multi-process rather than multi-threaded. The cost is inter-process communication (solved with ZMQ + shared memory, §§3–4) and the complexity of coordinating processes. But the payoff — genuine CPU parallelism that keeps the GPU fed — is large, especially as models get faster (smaller, CUDA-graphed) and the CPU work becomes a larger relative fraction. This is also why both SGLang and vLLM V1 (§22) chose multi-process: the GIL makes it the only way to truly overlap the CPU stages with each other and with the GPU. (A future where Python's GIL is removed, or where these stages are reimplemented in native code, could change this calculus — and indeed both engines are moving hot paths toward native code — but for now, processes are the answer.)

---

## 29. A Worked Latency Breakdown

Decompose where time goes in an SGLang decode step to see the architecture's effect. Consider a 7B model, decode batch 32, on an H100, with the optimizations active:

- **GPU forward pass:** ~2 ms (memory-bound decode reading 7B weights + KV, File 02 §15.2), executed as a replayed CUDA graph (no launch overhead, §8).
- **Scheduler CPU work:** ~0.8 ms (match, allocate, build batch) — but **hidden** by the overlap scheduler (§10, §15) behind the GPU's 2 ms, so it adds ~0 to the critical path.
- **Shared-memory transfer:** ~10 µs (§4) — negligible.
- **Tokenization/detokenization:** runs in the TokenizerManager process concurrently (§2, §19) — off the critical path.
- **Net per-step latency:** ~2 ms (the GPU forward pass), with everything else overlapped.

Without the architecture's overlap, the same step would be ~2 ms (GPU) + ~0.8 ms (scheduling) + ~0.5 ms (pickle transfer) + detokenization = ~3.3+ ms, with the GPU idle for ~40% of it. The architecture recovers that ~40%, approaching the ideal where the step latency equals the GPU forward pass alone. This is the concrete payoff of process separation + shared memory + overlap scheduler + CUDA graphs — and it's why these architectural choices, not just the attention kernel, determine real serving performance. For larger models (longer GPU steps), the overlapped overheads are a smaller fraction, but the architecture never hurts and always keeps the GPU as the bottleneck (the desirable state).

---

## 30. Observability and Common Pitfalls

### 30.1 Metrics

SGLang exposes metrics (File 11 §SGLang debug) covering: throughput (tokens/sec, requests/sec), latency (TTFT, TPOT, E2E), the **prefix-cache hit rate** (RadixAttention's key metric, File 08 §31), KV pool utilization, the running/waiting queue sizes, and CUDA-graph usage. The `--log-level debug` surfaces per-request RadixAttention hit info and scheduling decisions; `--enable-profiling` integrates torch.profiler.

### 30.2 Common pitfalls

- **Low prefix-cache hit rate on a sharing workload:** KV pool too small (raise the KV share via `--mem-fraction-static`), prefixes scattered across replicas (fix routing, File 08 §35), or token mismatches breaking the match (chat template/formatting differences, File 08 §37).
- **Eager-mode fallback (slow decode):** batch sizes exceeding `--cuda-graph-max-bs` fall back to eager (§16) — raise the max captured size or check why batches are large.
- **High per-step overhead on a small model:** ensure the overlap scheduler and CUDA graphs are active; consider `--enable-torch-compile` (§24).
- **OOM on static allocation:** `--mem-fraction-static` too high for the weights + graphs — lower it (but that shrinks the KV pool, File 11).
- **Attention-backend issues on AMD:** use the Triton backend (FlashInfer may be NVIDIA-only for some features, §7.2, File 16).
- **Multi-node slowness:** NCCL not using IB / TP spanning nodes (File 05 §27) — same fabric-config pitfalls as vLLM.

The diagnostic discipline parallels vLLM's (File 04 §32): distinguish latency vs throughput problems, check the relevant metric (hit rate, KV utilization, queue depth, graph usage), and map to the responsible component (RadixAttention sizing, scheduler, CUDA graphs, backend, fabric).

---

## 31. The Server and Engine APIs

SGLang exposes its runtime through two surfaces (the analog of vLLM's server + offline `LLM`, File 07):

- **HTTP server (`python -m sglang.launch_server`):** an OpenAI-compatible API server (the TokenizerManager front door, §2) with `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, streaming via SSE, and the guided-decoding parameters (`json_schema`, `regex`, etc.) backed by XGrammar (File 08 §11). This is the production-serving path, drop-in compatible with OpenAI clients.
- **Engine API (programmatic):** an in-process `Engine` (or `sgl.Runtime`) for offline/embedded use — generate over a list of prompts with the same continuous-batching engine, without the HTTP layer. Used for batch processing, evaluation, and as the backend for the frontend DSL (File 08 §12). The frontend programs execute against this runtime.
- **The frontend DSL** (File 08 §12) layers on top: `@sgl.function` programs run on the runtime via the interpreter, getting structural batching and RadixAttention reuse. This is SGLang's distinctive surface, atop the same engine the HTTP server uses.

The architecture this file describes (processes, communication, backends) underlies all three surfaces — the HTTP server, the programmatic Engine, and the frontend runtime all drive the same Scheduler + Executor core. This is why SGLang is both a strong general-purpose OpenAI-API server (the HTTP path, benefiting from RadixAttention/XGrammar automatically) *and* a structured-program runtime (the frontend path) — the underlying engine serves both.

---

## 32. Configuration and Tuning Reference

Key SGLang runtime flags (full methodology File 11):

- **`--mem-fraction-static`** (~0.88): fraction of GPU memory for static allocation (weights + CUDA graphs); the rest is the KV pool / radix cache. Lower → more KV cache (higher prefix-cache hit rate, more concurrency); raise if OOMing on static allocation.
- **`--max-running-requests`:** concurrency cap (like vLLM's `max_num_seqs`); set below the memory limit to avoid thrash (File 04 §27).
- **`--chunked-prefill-size`:** chunked-prefill token budget (File 04 §7); smaller → better decode-latency interleaving, lower prefill throughput.
- **`--attention-backend`** (`flashinfer` | `triton` | `torch`): the attention backend (§7); flashinfer for production NVIDIA, triton for AMD/fallback.
- **`--cuda-graph-max-bs`:** max decode batch size captured as a CUDA graph (§8, §16); higher → more graph coverage (fewer eager fallbacks) but more graph memory.
- **`--enable-torch-compile`:** extra fusion via torch.compile (§24); mainly helps small models.
- **`--enable-dp-attention`:** data-parallel attention for MoE / long context (§11).
- **`--tp-size`, `--dp-size`:** tensor and data parallelism (File 05).
- **`--enable-p2p-check`, NCCL env:** multi-node fabric configuration (File 05 §15).

The tuning principle mirrors vLLM (File 11): size the KV pool (`--mem-fraction-static`) to hold the hot-prefix working set plus active KV, cap concurrency below the memory limit, enable chunked prefill for mixed-traffic latency, ensure FlashInfer + CUDA graphs are active, and configure the parallelism per the topology rules (File 05). Monitor the prefix-cache hit rate and KV utilization as the primary signals.

---

## 33. The sgl-kernel Library

SGLang's custom kernels are increasingly organized in a dedicated **`sgl-kernel`** package — compiled CUDA/Triton extensions for the operations where fusion or specialization matters (§13): fused RoPE, RMSNorm, SwiGLU activation, quantized GEMMs (Marlin/AWQ/FP8), sampling, grammar-mask application, and MoE grouped GEMM. Packaging these separately (rather than inline) eases maintenance, allows sharing across SGLang components, and reflects the maturation of the kernel layer into a reusable library — paralleling vLLM's `csrc`/kernel organization (File 06). Some kernels are shared or co-developed with the broader ecosystem (FlashInfer, the vLLM kernels), since the same fusions address the same bandwidth bottlenecks in both engines (File 10). The kernel layer is the foundation of per-GPU efficiency beneath the runtime architecture; File 10 covers these kernels (FlashAttention/FlashInfer, the fused norm/activation/RoPE kernels, quantized matmuls, sampling) in depth, including their CUDA-level implementation and the hardware features they exploit.

---

## 34. Scheduler Policies in SGLang

SGLang's scheduler supports policies analogous to vLLM's (File 04 §8), with RadixAttention adding cache-aware options:

- **FCFS** (default): arrival order, fair, starvation-free.
- **Cache-aware (longest-prefix)** ordering: prioritize requests with large prefix-cache hits (cheap to run, File 08 §24), raising throughput on sharing workloads — tempered to avoid starving cache-cold requests (File 08 §24's risk).
- **Chunked prefill** for stall-free mixed-traffic latency (File 04 §7).
- **Token budget + concurrency caps** for admission control (§9.2).

The cache-aware policy is the SGLang-specific addition, enabled by RadixAttention's explicit prefix-hit-length computation — the scheduler knows each request's compute cost (proportional to its non-cached suffix) and can order accordingly. As with vLLM, policy shapes latency within capacity but cannot create capacity (File 04 §18); admission control and horizontal scaling (with prefix-cache-aware routing, File 08 §35) handle overload. The scheduler is where SGLang's continuous batching, RadixAttention, chunked prefill, and cache-aware ordering come together — the policy brain (File 08 §10) atop the runtime substrate this file describes.

---

## 35. How the Architecture Enables Disaggregation

The multi-process design (§2) naturally extends to **prefill–decode disaggregation** (File 04 §10, File 15). Because the runtime is already decomposed into communicating processes with defined interfaces (ZMQ control, shared-memory/transferable batch data), separating prefill and decode onto different process groups (or different machines) is an extension of the existing communication model rather than a redesign. A prefill instance computes the prompt KV (maintaining its radix tree for prefix caching, File 08 §14.3); the KV is transferred to a decode instance (File 03 §21, over NVLink/RDMA); the decode instance continues generation. The Scheduler's role splits between a prefill scheduler and a decode scheduler, coordinated by routing. SGLang has invested in disaggregation support, and the process-oriented architecture makes it tractable — the boundaries the architecture already draws (tokenize/schedule/execute) generalize to the prefill/decode boundary. This is a concrete benefit of the clean process separation: it's not just for overlapping CPU and GPU within one instance, but a structure that scales to disaggregated, heterogeneous deployments (File 05 §36, File 15). The same is true of vLLM V1's process architecture (§22) — both engines' move to processes eased their paths to disaggregation.

---

## 36. The Broader Convergence and What Remains Distinctive

Stepping back across Files 03–09, the two engines have converged on a shared systems architecture: paged/token-granular KV cache, continuous batching with chunked prefill, multi-process designs that overlap CPU work with GPU execution, shared-memory batch transfer, FlashInfer attention, CUDA-graph decode, the same fused kernels, and the same quantization methods. These are now settled best practices — the systems-level "how to build an LLM serving engine" is largely agreed.

What remains distinctive is *higher up*: SGLang's **RadixAttention** (token-granular tree prefix reuse, File 08) vs vLLM's APC (block-hash); SGLang's **program-aware frontend** (the DSL, structural batching) with no vLLM equivalent; and SGLang's emphasis on **structured output** (XGrammar) and **MoE/long-context** frontiers (DP attention, EP). vLLM's distinctive strengths are its **breadth** (the widest model and quantization support), its **ecosystem** (LF AI governance, the largest contributor base, the production stack/router), and its broad production hardening. So the engines compete not on systems architecture (converged) but on prefix-caching policy, programming model, and ecosystem — a healthy state where each pushes the other (vLLM adopting XGrammar; both adopting process separation and FlashInfer) and the field advances (File 17, File 20 §competitive dynamics).

---

## 37. Key Takeaways

1. **The organizing principle is keeping non-GPU work off the GPU's critical path** — achieved via multi-process architecture (sidestepping the GIL, §28), efficient communication (ZMQ + shared memory, §§3–4), and the overlap scheduler (§10).
2. **Three pipelined stages** — TokenizerManager (tokenize/detokenize), Scheduler (RadixAttention + continuous batching), Executors (GPU forward pass) — run concurrently in separate processes, keeping the GPU the bottleneck (the desirable state, §29).
3. **Shared-memory batch transfer** (§4) drops the per-step scheduler→executor handoff from ~500 µs to ~10 µs — essential at hundreds of steps/second.
4. **The attention backend** (FlashInfer default, Triton fallback, §7) provides the paged, varlen, GQA-optimized attention kernels, with a plan/run split (§18) that integrates with CUDA graphs.
5. **CUDA graphs** (§8) eliminate decode launch overhead (2–3× small-batch) by capturing a batch-size ladder (including TP collectives), padding to captured sizes at runtime.
6. **The overlap scheduler** (§10) hides scheduler CPU latency behind GPU execution, recovering ~25%+ utilization for fast steps.
7. **DP attention** (§11), **EAGLE speculative decoding** (§25), and **torch.compile** (§24) are specialized optimizations for MoE/long-context, decode-amortization, and small models respectively.
8. **The architecture converged with vLLM V1** (§22) — both reached multi-process, shared-memory, overlap-scheduled, FlashInfer, CUDA-graph designs — leaving RadixAttention and the frontend (File 08) as SGLang's true differentiators.

The runtime architecture is the substrate that makes RadixAttention and the frontend (File 08) run fast: without process separation, shared memory, the overlap scheduler, FlashInfer, and CUDA graphs, the algorithmic advantages would be bottlenecked by CPU overhead and kernel-launch costs. Together, File 08 (the algorithms) and File 09 (the architecture) constitute SGLang, the counterpart to vLLM (Files 03–07) in the open-source serving landscape.

---

## 38. Appendix: Source Pointers

- **Paper:** Zheng et al., arXiv 2312.07104 (SGLang, RadixAttention, frontend). XGrammar: arXiv 2411.15100. FlashInfer: arXiv 2501.01005.
- **Codebase:** `sglang/srt/managers/` (TokenizerManager, Scheduler, the process orchestration); `sglang/srt/mem_cache/` (radix tree, ReqToTokenPool, TokenToKVPool); `sglang/srt/model_executor/` (ModelRunner, CUDA graph runner); `sglang/srt/layers/attention/` (FlashInfer/Triton backends); `sgl-kernel/` (custom kernels); `sglang/lang/` (the frontend DSL and interpreter).
- **To trace execution:** HTTP → TokenizerManager (tokenize) → ZMQ → Scheduler (RadixAttention match, continuous batching, overlap-scheduled) → shared memory → Executors (FlashInfer attention, CUDA-graph decode, TP collectives) → sample → ZMQ → TokenizerManager (incremental detokenize, SSE stream). Each arrow is a section above; the pipeline keeps the GPU saturated while CPU work overlaps.

*Next: File 10 — the attention and compute kernels (FlashAttention, FlashInfer, Triton, fused kernels, quantized matmuls) and hardware backends that make each GPU fast, beneath both engines' architectures.*

---

## 39. Multimodal Handling in the Architecture

Vision-language models (File 06 §9, File 14) thread through SGLang's architecture with image processing in the TokenizerManager and feature caching via RadixAttention.

- **Image preprocessing in TokenizerManager:** when a request includes an image (via `sgl.image()` in the DSL, File 08 §12, or the multimodal API), the TokenizerManager process handles image loading, resizing, and normalization — CPU work overlapped with GPU execution like text tokenization (§2). The image-feature extraction (the ViT/CLIP/SigLIP encoder) runs on the GPU (in the executor) or is batched with text processing.
- **Image prefix caching via RadixAttention:** the image's KV (the LLM's KV for the image-feature tokens) is fixed for a given image, so RadixAttention caches it — keyed including the image (a hash, so different images don't cross-contaminate, File 08 §22). Multi-turn visual QA on the same image reuses the image KV across questions (File 08 §15) — a large saving for image-grounded multi-turn workloads. This is a natural fit: the same prefix-caching mechanism that helps text multi-call workloads helps image-grounded ones identically.
- **CUDA stream overlap:** the image encoding can overlap with the previous step's decode on a separate CUDA stream (§12), hiding the encoder cost. The architecture's stream and process overlap apply to multimodal exactly as to text.

So multimodal serving reuses the architecture's machinery — TokenizerManager for preprocessing, RadixAttention for feature caching, streams for overlap — with the visual encoder as an added GPU stage. SGLang's strength here (File 08 §15) is that the image prefix caching comes "for free" from RadixAttention, making repeated-image workloads (visual search, document QA, iterative editing) efficient.

---

## 40. Process Overhead vs Monolith, Quantified

To justify the multi-process complexity (§§2, 28), compare against a hypothetical single-process design on a fast workload (small model, decode batch 32, step ~2 ms GPU).

**Monolithic single-process (GIL-serialized):** within one process, the GIL serializes tokenization, scheduling, sampling-param processing, and GPU-launch. Suppose these CPU stages total ~1.5 ms per step (scheduling + detok + misc), serialized with the 2 ms GPU. Even with threads, the GIL prevents overlap → ~3.5 ms/step, GPU ~57% utilized.

**Multi-process (SGLang):** the ~1.5 ms of CPU work splits across the TokenizerManager and Scheduler processes (each its own GIL, true parallelism) and overlaps with the 2 ms GPU step (overlap scheduler) → ~2 ms/step, GPU ~100% utilized. The IPC overhead (shared memory ~10 µs, ZMQ control messages ~tens of µs) is negligible against the 2 ms step.

Result: the multi-process design delivers ~`3.5/2 = 1.75×` the throughput of the GIL-bound monolith on this fast workload. The gain shrinks for large models (where the 2 ms becomes 20+ ms and the 1.5 ms CPU is a small fraction), but never reverses — the IPC overhead is tiny. This quantifies why the complexity is worth it: for the fast-step regime (small models, CUDA-graphed decode, MoE with cheap attention) that's increasingly common, the GIL would otherwise cap utilization well below 100%, and only true CPU parallelism (processes) plus overlap recovers it. It's the architectural foundation of SGLang's (and vLLM V1's) efficiency on these workloads.

---

## 41. Inference-Engineer FAQ

**Q: Why is SGLang multi-process when that adds complexity?** The GIL (§28): Python serializes threads, so the only way to run tokenization, scheduling, and GPU-launch truly in parallel (and overlap them with the GPU) is separate processes. The ~1.75× utilization win on fast workloads (§40) justifies the IPC complexity.

**Q: My decode is slow and `nvidia-smi` shows the GPU not fully busy.** Check whether CUDA graphs are active (eager fallback for too-large batches, §16) and whether the overlap scheduler is engaged (§10). On a small model, also try `--enable-torch-compile` (§24). If utilization gaps remain, the CPU stages may not be overlapping — verify the multi-process setup is healthy.

**Q: How do I maximize prefix-cache hit rate?** Give RadixAttention enough KV memory (lower `--mem-fraction-static` to enlarge the KV pool, §32), route shared-prefix requests to the same replica (File 08 §35), and ensure consistent prompt formatting so tokens match (File 08 §37). Monitor the hit-rate metric (§30).

**Q: FlashInfer vs Triton backend?** FlashInfer for production NVIDIA (fastest, §7.1); Triton for AMD ROCm or as a fallback (§7.2). The backend must match the KV page size (File 03 §13).

**Q: Does the architecture differ much from vLLM?** Not anymore (§22, §36) — both converged on multi-process, shared-memory, overlap-scheduled, FlashInfer, CUDA-graph designs. The real differences are RadixAttention (vs APC) and the frontend DSL (File 08), not the runtime architecture.

**Q: Can I use SGLang as a plain OpenAI server without the DSL?** Yes (§31, File 08 §37) — the HTTP server is OpenAI-compatible, and RadixAttention + XGrammar benefit ordinary API traffic automatically. The DSL adds structural batching and fork/join but isn't required.

These reduce to the file's thesis: the architecture exists to keep the GPU the only thing on the critical path, and the operational questions are mostly about ensuring that overlap (processes, shared memory, overlap scheduler, CUDA graphs) is actually working and that RadixAttention has the memory and routing to be effective.

---

## 42. Closing

SGLang's runtime architecture is a careful exercise in overhead elimination: a multi-process design that sidesteps the GIL to run tokenization, scheduling, and GPU execution in genuine parallel; ZeroMQ for low-overhead control and shared memory for zero-copy bulk transfer; the overlap scheduler to hide CPU latency behind GPU compute; FlashInfer for fast paged attention; CUDA graphs to eliminate decode launch overhead; and a suite of fused custom kernels. The compounding of these keeps the GPU — the expensive resource — as the sole bottleneck, the ideal state for an inference engine. The architecture is the substrate on which SGLang's algorithmic differentiators (RadixAttention, the program-aware frontend, XGrammar — File 08) run efficiently, and it has converged with vLLM V1 to a shared set of systems best practices. Together, Files 08 and 09 complete the SGLang half of the database, mirroring the vLLM half (Files 03–07). The remaining files cover the cross-cutting layers both engines share or specialize: the kernels and hardware (File 10) beneath both architectures, performance tuning (File 11), and the specialized techniques and operational concerns (Files 12–20) that complete the picture of LLM inference engineering. The lesson carried forward from this file: in a memory-bandwidth-bound, fast-stepping workload, the *architecture* around the GPU — how CPU work is parallelized and overlapped — is as decisive for real-world performance as the kernels on the GPU, and getting it right means relentlessly removing everything but matrix math from the critical path.

---

## 43. The Zero-Overhead Batch Scheduler in Detail

SGLang's architecture is sometimes described as pursuing a "zero-overhead batch scheduler" — the goal that scheduling adds essentially nothing to the per-step latency. Achieving this required combining several of the techniques above into a coherent whole, and the design iterations are instructive.

Early serving engines (including early SGLang and vLLM V0) ran scheduling synchronously in the main process: finish GPU step, read results, schedule next step, launch — with the GPU idle during the CPU scheduling. As GPU steps got faster (CUDA graphs, FlashInfer, smaller models), this synchronous scheduling overhead became a dominant fraction (the §40 analysis). The zero-overhead scheduler attacks this on multiple fronts simultaneously:

1. **Move scheduling to its own process** (§2) so it doesn't contend with tokenization/detokenization for the GIL.
2. **Overlap the scheduling of step t+1 with the GPU execution of step t** (§10), so by the time the GPU finishes, the next batch is ready to launch immediately.
3. **Transfer the batch via shared memory** (§4), so the handoff to the executor is ~10 µs, not ~500 µs.
4. **Bake the per-batch-shape planning into CUDA graphs** (§§8, 18), so the executor's per-step preparation is minimal for the common decode shapes.

The combined effect approaches the ideal: the per-step latency converges to the GPU forward-pass time, with scheduling, transfer, and launch overheads all hidden or eliminated. This is why "zero-overhead scheduler" is an apt description of the goal — not that scheduling takes zero CPU time (it doesn't), but that it adds zero to the *critical-path* latency because it's fully overlapped and its handoff is near-instant. Reaching this required the architecture's full stack; no single technique suffices (process separation alone still has synchronous handoff; overlap alone still has the GIL if single-process; shared memory alone still idles the GPU during scheduling). It's the integration that delivers the result — a recurring theme in systems engineering, where the win comes from composing several techniques that each remove a different bottleneck, none sufficient alone.

### 43.1 Why this matters more over time

This architecture investment matters increasingly because the trend is toward *faster steps*: smaller efficient models, FP8/quantization shrinking compute time, CUDA graphs and FlashInfer cutting per-step GPU cost, and MoE models with cheap attention. As the GPU step shrinks, any fixed CPU overhead becomes a larger fraction — so the value of hiding it grows. An engine that was "fast enough" with synchronous scheduling on a slow 70B BF16 step becomes badly bottlenecked on a fast 7B FP8 CUDA-graphed step unless the scheduling overhead is eliminated. This is why both SGLang and vLLM invested heavily in the zero-overhead/async scheduler architecture (§22) — it's not premature optimization but a response to the steady acceleration of the GPU step, which keeps raising the bar for how little overhead the surrounding software can afford. The architecture future-proofs the engine against ever-faster steps, ensuring the GPU stays the bottleneck rather than the Python scheduler becoming it.

---

## 44. Reflection: Architecture as the Hidden Determinant

A theme worth making explicit, closing the SGLang half of the database: much attention (in blogs, papers, and intuition) goes to the *algorithms* — PagedAttention, RadixAttention, FlashAttention, speculative decoding. These are real and important. But the *architecture* — the unglamorous plumbing of processes, communication, overlap, and graph capture — is an equally decisive determinant of real-world performance, and it's often the difference between a benchmark number and a production reality. A serving engine with the best attention kernel but a synchronous, GIL-bound, pickle-transferring scheduler will be bottlenecked on CPU overhead and underperform a engine with a merely-good kernel but a zero-overhead architecture. SGLang's (and vLLM V1's) performance rests as much on the architecture this file describes as on the RadixAttention/PagedAttention algorithms — the two are complementary, and neither suffices alone. For an inference engineer, this means that diagnosing and optimizing a deployment requires understanding *both* layers: the algorithmic (is prefix caching working? is the attention kernel efficient?) and the architectural (is the GPU idling between steps? is scheduling overlapped? are CUDA graphs active?). The architecture is the hidden half — less discussed, equally important — and this file's purpose has been to make it visible, so that "the GPU is the bottleneck" can be verified and ensured rather than assumed. With both engines' internals now fully mapped (Files 03–09), the database turns to the kernels (File 10) that run beneath both, and then to the performance methodology (File 11) that ties the algorithmic and architectural layers into a tuning discipline.

---

## 45. Appendix: Component Responsibility Summary

A consolidated map of which component owns what, for quick reference:

| Component | Process | Owns / Responsible for |
|---|---|---|
| TokenizerManager | main (asyncio) | HTTP, chat templates, tokenization, incremental detokenization, image preprocessing, SSE streaming |
| Scheduler | dedicated | RadixAttention (radix tree, match/insert/evict), KV pools (ReqToToken/TokenToKV), continuous batching, chunked prefill, cache-aware scheduling, token budget, Req lifecycle |
| Model Executor(s) | one per GPU/TP rank | model forward pass, attention backend (FlashInfer/Triton), CUDA graphs, TP/EP collectives (NCCL), sampling, CUDA streams |
| sgl-kernel | (library) | fused RoPE/RMSNorm/SwiGLU, quantized GEMMs, sampling, grammar masks, MoE grouped GEMM |
| Frontend interpreter | (in runtime) | `@sgl.function` execution, structural batching, fork/join, gen/select |

Communication: TokenizerManager ↔ Scheduler via ZMQ (control); Scheduler ↔ Executors via shared memory (batch) + ZMQ (notify); Executors ↔ Executors via NCCL/RCCL (collectives). This table, with §38's source pointers, is the navigational map for the SGLang codebase.

The clean ownership boundaries are not incidental — they are what make the architecture both performant (each component optimized for its role, overlapped with the others) and maintainable (a change to detokenization doesn't touch the scheduler; a new attention backend slots behind the backend interface; a C++ rewrite of any one component is possible because the boundaries are explicit ZMQ/shared-memory interfaces). This modularity is itself an architectural asset: it allowed SGLang to evolve rapidly (shipping RadixAttention, XGrammar, DP attention, EAGLE in quick succession, File 08 §39) because each could be developed within a well-bounded component. Good architecture is not only fast but *evolvable*, and SGLang's component decomposition has served both ends.

## 46. Final Word

SGLang demonstrates that a serving engine is two things at once: a set of *algorithms* (RadixAttention, XGrammar, the frontend — File 08) and an *architecture* that runs them efficiently (this file). Its design — multi-process to beat the GIL, shared memory and ZMQ to communicate cheaply, the overlap scheduler to hide CPU latency, FlashInfer and CUDA graphs to make the GPU step fast, fused kernels to cut HBM traffic — is a textbook example of composing overhead-elimination techniques toward the goal of keeping the GPU the sole bottleneck. That it converged with vLLM V1 on this architecture (§22, §36) confirms these are the right answers to the shared problem. What distinguishes SGLang remains its algorithmic layer (token-granular tree prefix reuse, the program-aware frontend, near-free structured output) and its focus on the multi-call, structured, MoE, and long-context frontiers. For the inference engineer, the durable lessons are: parallelize and overlap the CPU work (or the GPU starves), make the per-step handoff near-instant (or it dominates fast steps), capture decode as CUDA graphs (or launch overhead dominates), and use the best paged-attention kernel available (FlashInfer) — and then layer the algorithmic optimizations (prefix reuse, constrained decoding) on top. With SGLang fully mapped, File 10 descends to the kernels — the matrix math and memory movement on the GPU itself — that both engines' architectures exist to keep fed and fast.

A final framing to carry forward: an LLM serving engine can be understood as three nested layers — the *algorithms* (what work to do and what to reuse: PagedAttention/RadixAttention, scheduling, speculative decoding), the *architecture* (how to orchestrate the work across processes and the GPU without overhead: this file), and the *kernels* (how to execute each operation at the hardware roofline: File 10). A deployment is only as fast as its weakest layer: the best kernels are wasted if the architecture idles the GPU between steps; the best architecture is wasted if the kernels are inefficient; and both are wasted if the algorithms recompute what they could reuse. SGLang and vLLM each optimize all three, and an inference engineer must reason across all three to diagnose and tune real systems. Files 03–09 have covered the algorithms and architecture of both engines; File 10 completes the stack with the kernels, after which File 11 unifies all three layers into a performance-tuning methodology. Holding this three-layer model — algorithms, architecture, kernels — is the organizing framework for everything that follows, and the lens through which any inference deployment, on any engine, can be understood and optimized. When a system underperforms, the question is always: which layer is the bottleneck — are we recomputing what we could reuse (algorithms), idling the GPU between steps (architecture), or running kernels below the roofline (kernels)? The answer points to the fix, and the three-layer model ensures the question is asked completely rather than fixating on whichever layer is most familiar. That completeness — checking algorithms, architecture, and kernels in turn — is the methodical habit that distinguishes systematic performance work from guesswork.







