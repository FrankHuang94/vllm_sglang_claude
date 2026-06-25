# vLLM Scheduler — Continuous Batching, Preemption, and Scheduling Algorithms

> **PRIMARY reference file.** The scheduler is the policy brain of vLLM: every iteration it decides which sequences run, admits new requests, preempts under memory pressure, and balances compute-bound prefill against bandwidth-bound decode. This file covers continuous batching, the code-level scheduler design, token budgets, chunked prefill, priority/fairness policies, prefill–decode disaggregation, multi-step and async scheduling, and sequence-group/beam-search scheduling. Prerequisites: File 03 (the block pool the scheduler spends) and File 01 §§2–3, 11 (roofline, prefill/decode duality, the throughput–latency frontier).

---

## Table of Contents

1. Why Scheduling Is the Hard Part
2. Static Batching and Its Failure
3. Continuous Batching (Iteration-Level Scheduling)
4. The vLLM Scheduler: Code-Level Design
5. The `schedule()` Method, Step by Step
6. Token Budget and Batch Construction
7. Chunked Prefill
8. Priority and Fairness Policies
9. Preemption Policies
10. Prefill–Decode Disaggregation
11. Multi-Step Scheduling and Async Scheduling
12. Sequence Groups, Parallel Sampling, and Beam Search
13. Tuning the Scheduler
14. The V1 Engine Rearchitecture

---

## 1. Why Scheduling Is the Hard Part

PagedAttention (File 03) solved *where to put* the KV cache. The scheduler solves *what to compute next*, and it is arguably harder because it is an online, multi-objective optimization under uncertainty:

- **Online:** requests arrive at unpredictable times with unpredictable prompt lengths and unknown output lengths. The scheduler cannot see the future and cannot redo past decisions.
- **Multi-objective:** it must simultaneously maximize throughput (tokens/sec/GPU), minimize TTFT (time to first token), and minimize TPOT (time per output token) — objectives that conflict (File 01 §11).
- **Constrained:** every decision is bounded by the KV block pool (File 03) and by a per-step token budget that bounds compute and activation memory.
- **Under uncertainty:** output length is unknown until a sequence emits EOS, so the scheduler cannot know how long a request will occupy memory — the core difficulty that makes optimal scheduling (e.g. shortest-job-first) impossible to implement exactly.

The scheduler is to an LLM serving engine what the process scheduler is to an OS — and like an OS scheduler, its quality is invisible when load is light and decisive when load is heavy. Most of the perceived performance difference between serving engines, and most production incidents, trace to scheduling behavior under load.

---

## 2. Static Batching and Its Failure

The naive approach — the one every "serve a model" tutorial starts with — is **static batching**: collect `N` requests, run them together until all finish, then collect the next batch. It fails badly for LLM serving:

- **Tail-latency coupling:** a batch finishes only when its *longest* sequence finishes. A batch of nine 10-token responses and one 2,000-token response runs for 2,000 steps; the nine short requests are done at step 10 but cannot return (or free their slots) until step 2,000. The GPU spends 1,990 steps processing a batch of 1.
- **Idle GPU:** as sequences finish at different times, the effective batch shrinks, and the GPU runs progressively under-utilized batches — exactly the wrong direction, since decode throughput needs *large* batches (File 02 §15.2).
- **TTFT explosion:** new requests cannot start until the entire current batch finishes. A request arriving just after a batch starts waits for the whole batch's longest sequence before its prompt is even prefilled. TTFT scales with `batch_size × max_output_length`.

Static batching wastes precisely the resource — GPU time on large batches — that the memory-bound decode regime makes valuable. The fix is to make batching *dynamic at the granularity of a single decode step*.

---

## 3. Continuous Batching (Iteration-Level Scheduling)

### 3.1 The Orca insight

**Orca** (Yu et al., OSDI 2022) introduced **iteration-level scheduling**: instead of scheduling at request granularity (run a whole request to completion), schedule at *iteration* granularity (decide the batch fresh before *each* forward pass). After every step, finished sequences leave the batch and waiting sequences join it. The batch composition changes every iteration — hence "continuous" (or "in-flight," NVIDIA's term) batching.

### 3.2 How it works

Each scheduler step:

1. **Remove** sequences that finished last step (hit EOS, max_tokens, or a stop string) — free their KV blocks immediately (File 03 §12).
2. **Admit** waiting requests if KV memory and token budget allow — these run their *prefill* this step.
3. **Continue** all running sequences — these run one *decode* step.
4. **Run** the forward pass on this dynamically-assembled batch (a mix of prefill and decode work).
5. **Repeat.**

The batch size is never fixed; it floats with the workload. A short request joins, runs 10 decode steps, and leaves while long requests around it keep going — no request waits for another's completion. This single change lifts GPU utilization dramatically: the batch stays large because finished sequences are immediately replaced, and new requests start within one iteration rather than waiting for a batch boundary.

### 3.3 Selective batching and the mixed-phase batch

Orca also introduced **selective batching**: because prefill (many tokens per sequence) and decode (one token per sequence) have different tensor shapes, naively batching them together requires padding. Orca batched only same-phase sequences for the attention operation while batching everything for the shape-uniform operations (the linear layers). vLLM goes further and schedules **mixed prefill+decode batches** directly, using a varlen attention kernel (File 02 §20) that handles per-sequence query lengths (length `n` for a prefilling sequence, length 1 for a decoding one) via a cumulative-length index. This avoids padding and lets a single forward pass serve both phases — though it reintroduces the head-of-line blocking that chunked prefill (§7) then addresses.

### 3.4 The throughput win, quantified

Recall (File 02 §15.2) that a decode step reads all model weights once and shares that read across the batch. With static batching the effective batch collapses toward 1 as sequences finish; with continuous batching it stays near the memory-imposed maximum. If the memory limit allows batch 64 and static batching averages an effective batch of ~12 (because of length variance), continuous batching delivers roughly `64/12 ≈ 5×` the decode throughput on the same hardware — the headline result that made it universal. The exact gain depends on the output-length distribution: the more variance, the worse static batching does and the larger continuous batching's advantage.

---

## 4. The vLLM Scheduler: Code-Level Design

### 4.1 Location and structure

The scheduler lives in `vllm/core/scheduler.py` (the V0 engine; the V1 rearchitecture, §14, restructures this). Its central class, `Scheduler`, owns three queues of `SequenceGroup`s:

- **`waiting`**: newly arrived requests that have not started prefill. FIFO by default.
- **`running`**: sequences currently executing (prefill or decode). These are the active batch.
- **`swapped`**: sequences preempted via swap (File 03 §8), awaiting swap-in.

It also holds a reference to the `BlockSpaceManager` (File 03 §11) for memory decisions and a `SchedulerConfig` carrying the budget parameters.

### 4.2 SequenceGroup

A **`SequenceGroup`** bundles all sequences originating from one request that share sampling parameters and a prompt — for a simple request, one sequence; for parallel sampling (`n>1`), `n` sequences; for beam search, `beam_width` sequences (§12). The scheduler operates on groups so that the shared-prompt CoW semantics (File 03 §5) are respected and the `n` samples are scheduled together. Each sequence within a group has its own state (File 03 §12) and block table, but they share prompt blocks.

### 4.3 The scheduler is pure CPU policy

A key architectural property: the scheduler does **no GPU work**. It produces a `SchedulerOutputs` object describing *what* to run (which sequences, their token ranges, block-table mutations, swaps to perform, blocks to free), and the worker(s) (File 06) execute it. This mechanism/policy separation (mirroring File 03's block-manager design) means the scheduler is a fast CPU component — but it also means scheduler latency sits on the critical path between GPU steps unless overlapped (§11, §14). For small models where a decode step is short, the Python scheduler's per-step cost can become a meaningful fraction of step time, which motivated multi-step and async scheduling.

---

## 5. The `schedule()` Method, Step by Step

The heart of the scheduler is `schedule()`, called once per iteration. Its logic, in order:

### 5.1 Schedule running (decode) sequences first

vLLM prioritizes keeping already-running sequences making progress (this protects TPOT and avoids wasting the prefill already invested). For each running sequence it checks `can_append_slots` (File 03 §26): does the block pool have room for the block(s) this sequence will need this step?

- If **yes**, the sequence is scheduled to decode.
- If **no** (memory full), the scheduler must **preempt** a sequence to free blocks (§9). Under the default FCFS policy it preempts the *most recently added* running sequence (LIFO victim), recomputing or swapping it, and retries.

### 5.2 Swap in swapped sequences

If there are sequences in the `swapped` queue and memory has freed up, swap them back in (File 03 §8) and add them to running. This restores preempted work before admitting brand-new requests, so preempted requests aren't starved.

### 5.3 Admit waiting (prefill) sequences

If token budget and memory remain, admit waiting requests for prefill. For each, `can_allocate` (File 03 §26) checks whether the prompt (minus any prefix-cache hits) fits. Admission stops when the **token budget** is exhausted (§6) or memory is full. Newly admitted sequences run their prefill (or first prefill chunk, §7) this step.

### 5.4 Produce SchedulerOutputs

The method returns the assembled batch: the list of sequences to run with their token ranges, the block-table updates, the set of blocks to swap in/out, and the set to free. The worker turns this into a forward pass.

### 5.5 The ordering rationale

The order — running first, then swapped, then waiting — encodes vLLM's default priorities: **protect in-flight work** (don't let running sequences stall, don't strand swapped sequences), then **grow the batch** with new admissions. This is work-conserving (it uses any spare capacity) and biased toward completing started work over starting new work, which keeps TPOT stable at the cost of potentially higher TTFT under heavy load. Different policies (§8) reorder these priorities.

---

## 6. Token Budget and Batch Construction

### 6.1 `max_num_batched_tokens`

The scheduler enforces a **token budget** per step: the total number of tokens processed (summed across all sequences, prefill tokens + decode tokens) cannot exceed `max_num_batched_tokens`. This bounds the compute per step (and thus the activation/workspace memory, protecting against OOM) and is the primary lever on the prefill/decode balance.

- A **decode** token counts as 1 (each running sequence contributes 1 token/step).
- A **prefill** of an `n`-token prompt counts as `n` tokens (all processed in one step, unless chunked).

So if `max_num_batched_tokens = 4096` and 200 sequences are decoding (200 tokens), there is budget for ~3,896 prefill tokens this step — enough for one ~3,896-token prompt or several short ones. If a single arriving prompt is 8,000 tokens and chunked prefill is *off*, it cannot fit the budget at all in one step alongside decodes — and without chunked prefill vLLM would either run it alone (blocking all decodes that step) or require `max_num_batched_tokens ≥ max_model_len`.

### 6.2 `max_num_seqs`

A separate cap, `max_num_seqs`, limits the number of concurrent sequences regardless of token budget. It bounds per-sequence bookkeeping and the batch dimension. The effective batch is `min(max_num_seqs, what fits in KV memory, what fits in token budget)`. These three limits interact, and File 11 covers setting them coherently.

### 6.3 The prefill-vs-decode tension in one budget

Because prefill and decode share one token budget, a large prefill *crowds out* decode capacity for that step, and a batch full of decodes *starves* prefill (new requests wait). This shared-budget coupling is the mechanism behind head-of-line blocking: a giant prefill consumes the budget, decodes get no room, and TPOT spikes. The budget is the knob; chunked prefill is the technique that makes the knob usable.

---

## 7. Chunked Prefill

### 7.1 The problem it solves

Without chunking, a long prefill is an atomic, budget-dominating, latency-spiking event. A 16,000-token prompt either needs `max_num_batched_tokens ≥ 16000` (a huge per-step compute spike that stalls all decodes for that step, spiking their TPOT) or cannot be served. Either way, decode requests sharing the engine suffer a latency hiccup whenever a long prompt arrives — a P99 TPOT/TTFT disaster under mixed traffic.

### 7.2 The algorithm

**Chunked prefill** (Agrawal et al., *Sarathi* arXiv 2308.16369; *Sarathi-Serve* arXiv 2403.02310; vLLM `--enable-chunked-prefill`) splits a long prefill into fixed-size chunks of `chunk_size` tokens, processed across consecutive scheduler steps:

```
prompt of P tokens → ceil(P / chunk_size) prefill steps
each step: process one chunk, attending over all previously-cached chunks
after the last chunk: the sequence enters decode
```

Crucially, each step's chunk is *bounded* (`chunk_size` tokens), so it leaves room in the token budget for decode tokens from other sequences. Decodes interleave with prefill chunks: while a long prompt is being prefilled over 8 steps, the other sequences keep generating tokens every step. No decode waits for the whole prefill — it waits at most one chunk.

### 7.3 The KV and position mechanics

From the block manager's view (File 03 §20), each chunk is an incremental `append_slots`: chunk `i` allocates blocks and writes KV for its tokens, attending over chunks `0..i-1`'s cached KV. Positions continue across chunks (File 02 §19). Peak KV memory is unchanged (the whole prompt's KV is resident by the end); what changes is the *temporal distribution* of the compute, smoothing the per-step spike into bounded increments.

### 7.4 The trade-off

Chunked prefill is not free:

- **Prefill throughput drops slightly:** more, smaller kernel launches mean more overhead per token, and the attention for later chunks re-reads earlier chunks' KV (the prefill becomes a sequence of prefix-attention operations). For a prompt split into `k` chunks, the cumulative KV read grows.
- **Decode latency improves substantially:** P99 TTFT/TPOT for decode requests waiting behind long prompts improves 2–3× (Sarathi-Serve's headline result), because the long prefill no longer monopolizes a step.

The net is a strong win for **mixed, latency-sensitive workloads** (chat with occasional long documents) and a slight loss for **pure-throughput offline prefill-heavy batch jobs** (where you'd want big atomic prefills). `chunk_size` (often via `max_num_batched_tokens` with chunked prefill on, or a dedicated flag) tunes the balance: smaller chunks → better decode interleaving, lower prefill efficiency; larger chunks → the reverse. Typical good values are 512–2,048 tokens.

### 7.5 Why it's becoming the default

In recent vLLM, chunked prefill is increasingly enabled by default because most production traffic is mixed and latency-sensitive, and the prefill-throughput cost is modest with well-chosen chunk sizes. It also simplifies the scheduler's budget reasoning: with all prefills chunked to ≤ `chunk_size`, no single sequence can blow the per-step budget, making per-step compute (and latency) predictable. Predictable step time is itself valuable — it is what makes CUDA graphs (File 09) and SLO guarantees tractable.

---

## 8. Priority and Fairness Policies

### 8.1 FCFS (First-Come-First-Served)

vLLM's default. Requests are served in arrival order; the waiting queue is FIFO. Properties: simple, starvation-free (every request eventually runs), and fair in the queuing sense. Weakness: it ignores latency SLOs and job size. A short, latency-critical request arriving behind a long batch job waits its turn — bad for interactive workloads mixed with batch. Under preemption, FCFS protects earlier (longer-waiting, closer-to-done) requests by preempting the most recently admitted (§9.4).

### 8.2 Priority scheduling

vLLM supports request priorities (`--scheduling-policy priority`, with a per-request `priority` value). Higher-priority requests are admitted first and preempt lower-priority running sequences when memory is tight. This serves latency-tiered workloads (interactive > batch). The risk is **starvation**: a stream of high-priority requests can indefinitely block low-priority ones. Mitigations include aging (raise priority with wait time) or reserving capacity for low-priority work — application-level policies layered on the engine.

### 8.3 SJF / SRTF and the output-length problem

The classic latency-optimal policies are **Shortest-Job-First (SJF)** and its preemptive form **Shortest-Remaining-Time-First (SRTF)**: run the request that will finish soonest, minimizing average completion time. The fatal obstacle for LLMs is that **output length is unknown at arrival** — you don't know how many tokens a request will generate until it emits EOS. So SJF/SRTF cannot be implemented exactly. Approaches:

- **Length prediction:** a small model or heuristic predicts output length from the prompt (FastServe, §10; File 15 §response-length-prediction). Prediction error degrades the policy gracefully toward FCFS.
- **User hints:** clients supply `max_tokens` as an upper bound; some systems use it as a proxy for length.
- **Online learning:** estimate length distributions from request history (per endpoint, per user).

Even approximate length information enables meaningful SRTF-style improvements (FastServe reports ~1.4× lower average job completion time vs FCFS), which is why length prediction is an active research and engineering area.

### 8.4 Work-conserving vs non-work-conserving

vLLM's default is **work-conserving**: if any GPU capacity is free, it is used (admit more requests, never deliberately idle). A **non-work-conserving** scheduler may *delay* admitting requests to assemble a better (larger, more uniform) batch later — trading a little latency for batching efficiency. Work-conserving is the right default for interactive serving (don't make a request wait when you could serve it), but non-work-conserving ideas appear in throughput-optimized and disaggregated systems where deliberate batching pays off.

---

## 9. Preemption Policies

### 9.1 When preemption happens

Preemption occurs when a running sequence needs a new KV block (to append a decode token or a prefill chunk) but the block pool is exhausted (at the watermark, File 03 §30). Something must give up memory. The scheduler selects a victim and reclaims its blocks.

### 9.2 Swap vs recompute (recap + scheduling view)

As detailed in File 03 §9, the victim's KV can be **swapped** to CPU (preserved, PCIe cost) or **recomputed** (discarded, re-prefilled on resume). The scheduler chooses based on configuration and sequence length: recompute for short sequences (cheap re-prefill, no PCIe), swap for long ones (expensive re-prefill). The chosen victims become `SWAPPED` (swap) or return to `waiting` (recompute) and are rescheduled when memory frees.

### 9.3 The preemption cost on throughput

Preemption is wasted work: a swapped sequence pays PCIe round-trips; a recomputed one re-runs prefill it already did. Heavy preemption (thrashing) means `max_num_seqs` admitted more concurrency than KV memory can sustain (File 03 §22). The cure is admission control — don't admit what you can't sustain. The `gpu_cache_usage_perc` metric near 100% with frequent preemptions is the signature of over-admission.

### 9.4 Victim selection

Under FCFS, the victim is the most-recently-admitted running sequence (LIFO), protecting older requests. Under priority, the lowest-priority running sequence. The choice matters for fairness: LIFO victimization means a late-arriving long request may be repeatedly preempted, which is acceptable (it arrived last) but can extend its tail latency significantly under sustained pressure.

---

## 10. Prefill–Decode Disaggregation

### 10.1 The motivation, from the duality

Prefill is compute-bound; decode is memory-bandwidth-bound (File 01 §3). In a single engine they share the same GPUs and the same batch, which causes two problems: (1) head-of-line blocking (long prefill delays decode, partly fixed by chunked prefill); and (2) **hardware mismatch** — the ideal GPU for prefill (high FLOP/s, e.g. H100) differs from the ideal for decode (high HBM bandwidth/capacity). Disaggregation runs them on *separate instances*.

### 10.2 Architecture

```
request → load balancer → PREFILL instance (compute prompt KV, emit first token)
                              │  transfer KV cache
                              ▼
                          DECODE instance (autoregressive generation, stream tokens)
```

The prefill pool is optimized for large-batch compute throughput; the decode pool for low-latency, high-bandwidth generation. KV cache is transferred between them over NVLink (intra-node), PCIe P2P, or RDMA/InfiniBand (inter-node) — File 03 §21, File 15 §transfer.

### 10.3 When it pays off

Disaggregation is worthwhile when the KV transfer time is small relative to the prefill compute it offloads. For a 1,000-token LLaMA-3 70B prompt: KV ≈ 320 MB, ~6.4 ms over 50 GB/s RDMA, versus ~1 s of prefill compute — transfer is <1% of prefill, easily amortized. It also enables **heterogeneous hardware**: H100 for prefill, cheaper high-bandwidth GPUs (or more of them) for decode, cutting cost 2–3× at equal throughput (File 15 §disaggregation rationale). And it lets prefill and decode pools scale *independently* to match the workload's prefill:decode ratio.

### 10.4 Systems: Mooncake, DistServe

- **DistServe** (Zhong et al., arXiv 2401.09670) provides the analytical model: treat prefill and decode as separate queues, choose the prefill:decode worker ratio to maximize goodput. The optimal ratio depends on the workload — ~1:1 for long-prompt/short-output (summarization), heavily decode-weighted (e.g. 1:4) for short-prompt/long-output (chat).
- **Mooncake** (Kimi.ai) is a production disaggregation system: RDMA P2P KV transfer, a global KV cache manager tracking which worker holds which sequence's KV, prefix-cache-aware routing (send requests sharing a prefix to the same prefill worker), and KV migration between workers.

### 10.5 vLLM status

vLLM exposes disaggregation experimentally via `--kv-transfer-config` and a pluggable connector framework (with backends like LMCache). It composes with speculative decoding (draft on prefill, target on decode) and with prefix caching (per-prefill-worker caches). Production-grade disaggregation has been a major roadmap item; File 15 covers the frontier in depth.

---

## 11. Multi-Step Scheduling and Async Scheduling

### 11.1 The scheduler-overhead problem

The scheduler runs on the CPU in Python between GPU steps (§4.3). For a large model, a decode step takes tens of milliseconds and the few-hundred-microsecond scheduler cost is negligible. But for a **small model** (or a heavily optimized one with CUDA graphs), a decode step can be ~1–2 ms, and the Python `schedule()` overhead plus the CPU↔GPU synchronization can become 20–40% of step time — pure overhead that idles the GPU between steps.

### 11.2 Multi-step scheduling

`--num-scheduler-steps N` lets the scheduler plan once and the worker execute `N` decode steps before returning to the scheduler. The Python `schedule()` cost is amortized over `N` GPU steps, and CPU↔GPU synchronization happens `N×` less often. For decode-heavy workloads this can substantially raise throughput. The cost: the scheduler is "blind" for `N` steps — it cannot admit new requests or react to completions until the multi-step block finishes, slightly increasing TTFT and making the engine less responsive. It is also awkward with streaming (tokens for all `N` steps return together) and with stop-condition checks (a sequence that should stop at step `k<N` still runs to `N`, wasting a little compute). Multi-step is most useful for offline/batch throughput; for interactive serving, `N` is kept small or the async approach (below) is preferred.

### 11.3 Async scheduling (overlapping CPU and GPU)

The more elegant fix is to **overlap** the scheduler with model execution: while the GPU runs step `t`, the CPU scheduler plans step `t+1`. When step `t` finishes, step `t+1` is already prepared and launches immediately, hiding scheduler latency entirely behind GPU compute. This requires decoupling scheduling from execution — a background thread/process plans ahead while the worker executes. `AsyncLLMEngine` (File 07) provides the asyncio wrapper, and the V1 engine (§14) makes async scheduling a first-class architectural property. The challenge is that planning step `t+1` requires knowing step `t`'s output (which token each sequence generated, whether any stopped) — so the overlap is partial (the scheduler can prepare most of the batch but must reconcile completions). SGLang's "overlap scheduler" (File 09) implements the same idea and is one reason SGLang achieves low scheduler overhead.

---

## 12. Sequence Groups, Parallel Sampling, and Beam Search

### 12.1 SequenceGroup scheduling

As introduced in §4.2, a `SequenceGroup` holds all sequences from one request. The scheduler schedules the *group*: all its sequences run together each step, sharing prompt KV via CoW (File 03 §5). This keeps the `n` parallel samples (or `beam_width` beams) in lockstep and ensures their shared blocks are accounted once.

### 12.2 Parallel sampling (n > 1)

For `SamplingParams(n=4)`, the group has 4 sequences. They share the prompt's KV blocks (refcount 4) and diverge as they sample differently, triggering CoW at divergence (File 03 §24). The scheduler treats the 4 as a unit for admission (they share the prompt) but each consumes its own decode-token budget and its own suffix blocks. Memory cost after divergence is ~`4 ×` the suffix KV (the prompt stays shared), far less than 4 independent requests.

### 12.3 Beam search scheduling

Beam search (File 02 §8.4) is the most complex scheduling case. Each step, all `beam_width` beams run a decode step; the engine computes logits for each, scores all `beam_width × V` candidate extensions, keeps the top `beam_width`, and prunes the rest. The scheduler must:

- Run all live beams each step (they share the common prefix via CoW).
- After scoring, **fork** surviving beams that branched (increment refcounts, CoW on write) and **free** pruned beams (decrement refcounts, release blocks).
- Handle the dynamic beam set: beams appear (forks) and disappear (prunes) every step, so the group's sequence membership changes within the request's lifetime.

The memory footprint is roughly `beam_width ×` the per-beam KV after the beams diverge from the shared prefix. Because beam search is `beam_width ×` more expensive and rarely improves open-ended generation quality, it is uncommon in production serving — but the SequenceGroup + CoW machinery makes it correct and as memory-efficient as possible when needed.

### 12.4 Stopping conditions

After each decode step, the scheduler/engine checks each sequence's stop conditions: EOS token, `max_tokens` reached, or a `stop` string matched (requiring incremental detokenization, File 02 §18.3). A sequence hitting any condition transitions to `FINISHED_*` (File 03 §12) and is removed from running, its blocks freed immediately so the next step can reuse them. Prompt-level `ignore_eos`, `min_tokens`, and `stop_token_ids` refine this. Prompt-and-group interactions matter: in beam search, the group finishes when enough beams have completed; in parallel sampling, each sample finishes independently and the group completes when all `n` are done (or `best_of` selection applies).

---

## 13. Tuning the Scheduler

The scheduler's behavior is shaped by a handful of interacting flags (full methodology in File 11):

- **`--max-num-batched-tokens`**: per-step token budget (§6). Higher → larger batches, more throughput, but bigger per-step latency spikes (worse for chunked-prefill latency goals). With chunked prefill, this also bounds chunk size.
- **`--max-num-seqs`**: concurrency cap (§6.2). Set to match sustainable KV memory; too high → preemption thrash (§9.3).
- **`--enable-chunked-prefill`** (+ chunk size): smooths prefill latency spikes (§7). Default-on in recent versions for mixed traffic.
- **`--scheduling-policy`** (`fcfs` | `priority`): fairness model (§8).
- **`--preemption-mode`** (`recompute` | `swap`): reclaim strategy (§9.2).
- **`--num-scheduler-steps`**: multi-step amortization (§11.2); raise for offline throughput, keep low for interactive.
- **`--gpu-memory-utilization`**: indirectly sets the block pool the scheduler spends (File 03 §7.1).

The central tuning tension: **batch size vs latency**. Bigger budgets and higher concurrency raise throughput and degrade per-request latency; chunked prefill decouples prefill latency from decode latency; preemption mode and policy shape behavior under overload. There is no universal setting — the right configuration depends on the workload's prompt/output length distribution and SLO targets, which is why File 11 frames tuning as fitting these knobs to a measured latency–throughput curve.

---

## 14. The V1 Engine Rearchitecture

vLLM's V1 engine (introduced around v0.6+ and progressively made default) restructured the scheduler and execution path to cut the Python overhead that the V0 design incurred per step.

### 14.1 Motivations

In V0, the scheduler, block manager, detokenizer, and request handling all ran in the main Python process on the critical path between GPU steps. As GPU kernels got faster (CUDA graphs, FlashInfer) and models got smaller, this Python overhead became a larger fraction of step time — sometimes the bottleneck. V1 attacks this.

### 14.2 Key changes

- **`EngineCore`**: a tight, optimized core loop for the scheduling hot path, isolated from the API/request-handling layers and designed to run with minimal per-step overhead (with paths implemented in/toward native code).
- **Separate processes**: the API server / request handling, the detokenizer, and the engine core run in **separate processes** communicating via efficient IPC, so tokenization/detokenization and HTTP handling don't block the scheduler, and the scheduler doesn't block I/O.
- **Async-by-design scheduling**: the architecture overlaps scheduling with execution (§11.3) as a first-class property rather than a bolt-on, planning the next step while the GPU runs the current one.
- **Simplified, faster block manager and scheduler**: V1 streamlined the data structures (building on BlockManager v2, File 03 §32) and the `schedule()` logic to reduce per-step CPU cost, and made prefix caching and chunked prefill default, well-integrated behaviors rather than optional add-ons.

### 14.3 The result

V1 delivers materially higher throughput on decode-heavy and small-model workloads (where overhead dominated) and lower, more consistent latency, while preserving the V0 feature set. Architecturally it mirrors a direction SGLang took (separate tokenizer/scheduler/executor processes with efficient IPC, File 09) — convergent evolution toward moving everything off the GPU's critical path. For an inference engineer, the practical upshot is that recent vLLM has much lower scheduler overhead than early versions, and the old workarounds (large `--num-scheduler-steps`, disabling features to cut Python cost) are largely unnecessary on V1.

---

## 15. Synthesis: The Scheduler as Goodput Maximizer

Every mechanism in this file serves one end: maximizing **goodput** (requests served within SLO, File 01 §11) on a fixed, memory-constrained, dual-phase workload.

- **Continuous batching** keeps the batch large (throughput) without making requests wait for each other (latency) — the foundational move.
- **Chunked prefill** decouples prefill latency from decode latency, fixing head-of-line blocking so long prompts don't spike everyone's TPOT.
- **Token budget + concurrency caps** bound per-step compute and admission to what memory and SLOs allow.
- **Preemption (swap/recompute)** provides graceful behavior at the memory cliff instead of OOM.
- **Priority/SRTF policies** bias service toward latency-critical or short jobs when the workload is tiered.
- **Disaggregation** separates the two phases onto ideal hardware and independently scalable pools.
- **Multi-step / async / V1** remove the scheduler itself from the critical path.

The scheduler is where the abstract tensions of File 01 (roofline, prefill/decode duality, throughput–latency frontier) become concrete control decisions, executed hundreds of times per second, spending the block pool of File 03. It is the component most responsible for how a deployment behaves under real, bursty, heterogeneous load — and therefore the one an inference engineer tunes most. File 05 next extends the execution substrate across many GPUs (tensor, pipeline, expert, and sequence parallelism), and File 11 turns the knobs surveyed here into a quantitative tuning methodology.

---

## 16. A Worked Scheduling Trace

Abstract policy becomes concrete in a timeline. Consider a single-engine vLLM with `max_num_batched_tokens = 2048`, `max_num_seqs = 8`, chunked prefill on with `chunk_size = 512`, FCFS, on a model where each decode step is ~20 ms. Requests:

- **R1** arrives at t=0: 1,500-token prompt, will generate 200 tokens.
- **R2** arrives at t=0: 100-token prompt, will generate 30 tokens.
- **R3** arrives at t=25 ms: 6,000-token prompt (a long document), will generate 50 tokens.
- **R4** arrives at t=30 ms: 80-token prompt, will generate 500 tokens.

**Step 0 (t=0):** waiting = [R1, R2]. The scheduler admits both for prefill. R1 needs 1,500 tokens but chunk_size is 512, so R1 prefills its first 512-token chunk; R2 prefills all 100 tokens. Token budget used: 512 + 100 = 612 ≤ 2048. Both have blocks allocated. R1 is mid-prefill (chunk 1 of 3); R2 finished prefill and will decode next step.

**Step 1:** R1 prefills chunk 2 (tokens 512–1023, 512 tokens, attending over chunk 1's KV); R2 decodes its first output token (1 token). Budget: 512 + 1 = 513. Note how the decode for R2 *interleaves* with R1's ongoing prefill — chunked prefill in action. Without it, R1's 1,500-token prefill would have consumed a whole step and R2 would have waited.

**Step 2 (≈t=40ms, R3 has arrived):** R1 prefills chunk 3 (tokens 1024–1499, 476 tokens), then transitions to decode-ready; R2 decodes (token 2). R3 is waiting. Budget so far 476 + 1 = 477; there's room, so the scheduler admits R3 and prefills R3's first 512-token chunk (budget now 477 + 512 = 989). R3 begins its 12-chunk prefill journey.

**Steps 3–13:** R3 grinds through its remaining 11 prefill chunks (512 tokens each), one per step, while R1, R2 (until it finishes at ~step 30 from its perspective), and later R4 *decode every step*. R4, arriving at t=30 ms, is admitted at the next step with budget room (its 80-token prefill fits alongside R3's chunk and the decodes) and starts generating. Crucially, **no decoding request stalls for R3's giant prefill** — each step R3 contributes only 512 prefill tokens, leaving budget for all the decodes. This is the P99-TTFT/TPOT protection chunked prefill buys.

**Contrast without chunked prefill:** R3's 6,000-token prefill would exceed `max_num_batched_tokens=2048` and either be rejected or require running alone in a step (or several), during which R1, R2, R4 would generate *nothing* — a 6,000-token prefill at ~prefill speed could stall all decodes for tens of milliseconds, spiking their inter-token latency. The trace makes vivid why chunked prefill is the default for mixed traffic.

This trace also illustrates the FCFS admission order (R1, R2 before R3, R4), the token-budget accounting (prefill chunks + decode tokens summed against 2048), and the lockstep decode of all running sequences each step. Add memory pressure (say `max_num_seqs` reached or KV pool full) and the scheduler would begin preempting the most-recently-admitted running sequence (R4, then R3) to keep R1/R2 progressing — FCFS victim selection (§9.4).

---

## 17. The Sarathi Stall-Free Batching Analysis

Chunked prefill's theoretical foundation is worth making precise, because it explains *why* the chunk size matters and what "stall-free" means.

### 17.1 The decode stall

Define a **decode stall** as a scheduler step in which a decoding sequence makes no progress because the step is monopolized by prefill. In a mixed batch without chunking, whenever a long prefill arrives, it consumes the entire token budget (or runs alone), and all decodes stall for that step. Stalls directly inflate TPOT: a request expecting a token every 20 ms instead waits 20 ms + (prefill step time) whenever a long prompt sneaks in.

### 17.2 Sarathi's claim

Sarathi (Agrawal et al., arXiv 2308.16369) observes that if every prefill is chunked to a bounded size and *each step always includes the running decodes*, then **no decode ever stalls** — every step makes decode progress because the bounded prefill chunk leaves budget for the decode tokens. The batch is "stall-free": decodes proceed at a steady cadence regardless of prefill arrivals. Sarathi-Serve (arXiv 2403.02310) productionizes this with SLO awareness and reports **2.6× lower P99 TTFT** (and improved TPOT) versus an Orca-style baseline that runs prefills atomically.

### 17.3 Choosing the chunk size

The chunk size sets a trade-off captured by a simple model. Let prefill efficiency (tokens/sec) be `E(c)`, increasing with chunk size `c` (bigger chunks amortize kernel-launch and attention-setup overhead, approaching the efficiency of a full prefill). Let the decode-stall risk fall as `c` shrinks (smaller chunks leave more budget for decodes). Then:

- **Small `c` (e.g. 256):** decodes never starve (low TPOT variance), but prefill is inefficient (you pay overhead on many small chunks, and later chunks re-read more accumulated KV), so prefill throughput drops and TTFT for the prefilling request itself rises.
- **Large `c` (e.g. 4096):** prefill is efficient (high throughput, low TTFT for that request), but a chunk this large can crowd out decodes in a step, reintroducing stalls.

Sarathi-Serve's insight is to size chunks just small enough to guarantee the decode SLO while keeping prefill as efficient as possible — effectively, choose `c` so the per-step time stays under the TPOT budget. In vLLM, this is governed by `max_num_batched_tokens` (with chunked prefill on) and tuned empirically (File 11). Typical production values land in 512–2,048. The deeper point: chunked prefill converts an *uncontrolled* latency coupling (prefill arrivals randomly spiking decode latency) into a *tunable* one (the chunk size directly bounds the spike), which is what makes SLO guarantees possible.

---

## 18. Continuous Batching as a Queuing System

Viewing the scheduler through queuing theory clarifies admission control and capacity planning.

### 18.1 The model

Treat the engine as a server with a service rate that *depends on the batch size* (unusual — most queues have fixed service rate). Requests arrive at rate `λ` (often modeled as Poisson). Each request requires a prefill (compute proportional to prompt length) and a number of decode steps (proportional to output length). The "service" of a decode step is shared across all running sequences, so per-request service rate *falls* as the batch grows (more sequences share the fixed weight-read bandwidth), while aggregate throughput *rises* until the ridge point or memory limit.

### 18.2 The capacity cliff

There is a maximum sustainable arrival rate `λ_max` — the request rate at which the engine's aggregate token throughput equals the rate at which arriving requests demand tokens. Below `λ_max`, queues stay bounded and latency is stable. Above `λ_max`, the waiting queue grows without bound: requests arrive faster than they can be served, KV memory fills, preemption thrashing begins, and latency diverges. This is the **capacity cliff** — and it is sharp, because the memory-bound throughput is roughly flat up to the limit and then collapses (preemption overhead) past it. The operator's job (File 11, File 19) is to keep `λ < λ_max` with margin, via load balancing across replicas, admission control (rejecting/shedding when full, returning 503), and autoscaling.

### 18.3 Why admission control matters more than scheduling order

A subtle but important consequence: once `λ > λ_max`, *no scheduling policy can save you* — the work simply exceeds capacity, and any policy just chooses *which* requests suffer. The scheduler's clever policies (SRTF, priority, chunked prefill) shape latency *within* capacity; they cannot create capacity. This is why production systems pair the scheduler with **load shedding** (reject when `gpu_cache_usage_perc` is critical or the queue exceeds a threshold) and **horizontal scaling** (add replicas to raise aggregate `λ_max`). The scheduler is necessary but not sufficient; admission control and capacity are the other half (File 19 §circuit breaking).

---

## 19. Scheduler Interaction with Speculative Decoding

Speculative decoding (File 12) changes the scheduler's per-step accounting because each step now produces *multiple* tokens per sequence (the accepted draft tokens) instead of one.

- **Variable tokens per step:** a speculative step proposes `K` draft tokens and the target verifies them, accepting between 1 and `K+1` tokens. So a running sequence advances by a variable, data-dependent number of positions each step. The scheduler's block accounting (`append_slots`) must allocate for up to `K+1` new tokens per sequence per step, not 1 — raising the per-step KV growth and changing the `can_append_slots` check.
- **Budget accounting:** the target model's verification forward pass processes `K+1` positions per sequence (like a mini-prefill), so the token budget per step is consumed faster. With batch `B` and `K=4`, a verification step processes up to `B·5` tokens — the scheduler sizes the batch so this fits the budget.
- **Draft execution:** the draft model runs `K` cheap forward passes (or a single tree-draft) before the target verification. In vLLM's `SpecDecodeWorker` (File 06, File 12), this is orchestrated within the worker; the scheduler sees the spec step as a unit producing variable output. Multi-step scheduling (§11) and spec decoding interact carefully because both change the tokens-per-step relationship.
- **Acceptance-rate-dependent throughput:** because accepted tokens vary, the *effective* batch progress varies, making step time and memory growth less predictable than plain decode — a complication for CUDA graphs (which want fixed shapes, File 09) and for SLO estimation. The scheduler benefits from speculative decoding's higher tokens-per-step (better goodput) but must tolerate its variance.

---

## 20. Scheduler Interaction with Prefix Caching

Prefix caching (File 03 §6) directly changes admission cost. When the scheduler evaluates `can_allocate` for a waiting request, it accounts for prefix-cache hits: a request whose 2,000-token system prompt is already cached needs blocks only for its non-cached suffix, so it is far cheaper to admit (less memory, near-zero prefill compute). This has two scheduling consequences:

1. **Cheaper admission for cache-warm requests:** they consume little prefill budget (only the suffix is computed) and little new memory (the prefix blocks are shared), so the scheduler can admit many of them per step. A burst of requests sharing a hot system prompt is dramatically cheaper to onboard than the same burst with cold prefixes.
2. **Cache-aware scheduling opportunities:** SGLang's scheduler (File 08, File 09) explicitly computes each request's prefix-hit length and can *prioritize* cache-warm requests (they're cheap to run) — a policy vLLM's scheduler is also evolving toward. At the cluster level, **prefix-cache-aware routing** (File 05 §load balancing) sends requests with shared prefixes to the same replica to maximize hit rate, turning prefix caching into a routing concern as well as a scheduling one.

The interaction with chunked prefill: the first chunk of a cache-warm request may begin partway through the prompt (after the cached prefix), so the chunking starts from `num_computed_tokens` (File 03 §25.1), not from 0.

---

## 21. Scheduler Interaction with LoRA Serving

When serving multiple LoRA adapters on one base model (File 18), the scheduler gains an extra dimension: which adapters' requests are in the batch. A batch can contain requests using *different* adapters (Punica/SGMV kernels, File 18, apply the right adapter per request via batched grouped GEMV). The scheduler must respect `--max-loras` (the number of adapters that can be resident in GPU memory simultaneously): if admitting a request would require a not-yet-loaded adapter and the adapter cache is full, the engine must evict (LRU) and load an adapter — a cost the scheduler weighs. In practice the scheduler tries to batch requests for already-resident adapters and admits new-adapter requests when adapter-cache room exists, balancing adapter-load overhead against admission latency. This makes multi-LoRA scheduling a two-resource problem (KV blocks *and* adapter slots), a mild generalization of the single-resource admission control of §18.

---

## 22. The SchedulerOutputs Structure

The concrete product of `schedule()` is a `SchedulerOutputs` (V0) / scheduler-output object (V1) describing everything the worker must do this step. Its key contents:

- **Scheduled sequence groups** with, per sequence: the token IDs to process this step (the prompt chunk for a prefilling sequence, or the single last token for a decoding one), the sequence's block table, and its sampling parameters.
- **`num_prefill_groups` / prefill vs decode split:** so the model runner (File 06) can build the right attention metadata (`cu_seqlens`/`indptr`, per-sequence query lengths).
- **Blocks to swap in** and **blocks to swap out** (CPU↔GPU copy operations, File 03 §8).
- **Blocks to copy** (the CoW copies triggered by `append_slots`/`fork`, File 03 §5).
- **Blocks to free** (from finished or preempted sequences).
- **Token budget consumed** and bookkeeping for metrics.

The worker executes these in order: perform the swaps/copies/frees on the KV pool, build the batch tensors (`input_ids`, `positions`, `slot_mapping`, block tables, attention metadata — File 06), run the forward pass, sample, and return the generated tokens. The scheduler never touches the GPU; it only emits this declarative plan. This clean interface is what allows the executor to be a separate process (V1, §14) and the workers to be distributed across GPUs (File 05) — the plan is broadcast and each worker applies its shard.

---

## 23. Scheduler Observability

The scheduler exposes the metrics that reveal serving health (full list in File 07 §metrics, used in File 11/19):

- **Queue depth** (`num_waiting`, `num_swapped`): how many requests are queued or preempted. A growing waiting queue signals approaching/exceeding `λ_max` (§18.2).
- **Running batch size** (`num_running`): the live concurrency; compare to `max_num_seqs` to see if you're concurrency-bound or memory-bound.
- **KV cache utilization** (`gpu_cache_usage_perc`): the block-pool occupancy (File 03 §22). Sustained >90% with preemptions = over-admission/thrash.
- **Preemption count**: the rate of swap/recompute preemptions; nonzero-and-rising is a thrash warning.
- **Prefix-cache hit rate**: fraction of prompt tokens served from cache; central to prefix-heavy workload performance.
- **TTFT / TPOT / E2E latency histograms**: the SLO metrics the scheduler ultimately serves (File 11 §metrics taxonomy).

These let an operator diagnose whether a latency problem is queuing (raise capacity), memory (reduce `max_num_seqs` or quantize KV), prefill-interference (enable/tune chunked prefill), or genuine overload (shed load). The scheduler's internal state is, in effect, the dashboard of the whole engine.

---

## 24. Comparison with the SGLang Scheduler

vLLM and SGLang schedulers share the continuous-batching, chunked-prefill, token-budget core but differ in emphasis (full SGLang treatment in File 09):

| Aspect | vLLM scheduler | SGLang scheduler |
|---|---|---|
| Prefix reuse | hash-based APC, block-granular | RadixAttention, token-granular, tree-structured |
| Cache-aware scheduling | evolving | explicit prefix-hit-length computation, can prioritize cache-warm |
| Process model | V0 in-process; V1 separate EngineCore/detokenizer processes | TokenizerManager / Scheduler / ModelExecutor separate processes (ZMQ) from the start |
| Overlap | V1 async scheduling | "overlap scheduler" overlaps CPU scheduling with GPU execution |
| Program awareness | none (request-level) | frontend DSL enables structural batching across program calls |
| Chunked prefill | `--enable-chunked-prefill` | `--chunked-prefill-size`, radix-tree-aligned |

The biggest conceptual difference is that SGLang's scheduler is **program-aware** (File 08 §frontend): because SGLang sees the structure of multi-call LLM programs, it can batch calls across users with the same program structure and exploit the radix tree for fine-grained reuse — capabilities a request-level scheduler like vLLM's cannot match without the frontend. Conversely, vLLM's request-level model is simpler and its broad policy support (priority, configurable preemption) and V1 rearchitecture make it highly competitive for general OpenAI-API serving. The two have converged substantially (both do chunked prefill, prefix caching, async/overlap scheduling) — File 17 compares them end to end.

---

## 25. Research Lineage in Detail

### 25.1 Orca (OSDI 2022)

Orca's two contributions deserve elaboration because vLLM inherited and extended both.

**Iteration-level scheduling** broke the assumption that a request is the unit of scheduling. By rescheduling before every forward pass, Orca let requests join and leave the batch mid-flight, eliminating the convoy effect of static batching (§2). vLLM's continuous batching *is* iteration-level scheduling with PagedAttention memory underneath.

**Selective batching** addressed a shape problem: in a batch mixing requests at different sequence positions, the attention operation has per-request shapes (different KV lengths) that don't batch as a single dense tensor, while the linear layers (QKV, FFN, output projections) *do* batch uniformly (they're per-token). Orca therefore batched the linear ops across all requests but ran attention per-request (or per-phase). vLLM generalizes this with varlen attention kernels (File 02 §20) that handle the ragged attention shapes directly via `cu_seqlens`, so even attention is "batched" through a single kernel call over the packed layout — a refinement of Orca's idea enabled by FlashAttention/FlashInfer's varlen APIs.

### 25.2 FastServe (arXiv 2305.05920)

FastServe attacked **head-of-line blocking at the request level** with preemptive, size-aware scheduling. Its core is a multi-level feedback queue approximating **Shortest-Remaining-Processing-Time (SRPT)**: requests are demoted to lower-priority queues as they consume more service, so short requests (which finish before demotion) are effectively prioritized over long ones — without needing to know output length in advance (the demotion *discovers* length online). This requires **iteration-level preemption** (pause a long request to let a short one through), which PagedAttention makes cheap (preempt via recompute/swap, File 03 §9). FastServe reported ~1.4× lower average job completion time vs FCFS. vLLM's priority scheduling and preemption machinery provide the building blocks to implement such policies; the MLFQ/SRPT approximation is the kind of policy layered on top.

### 25.3 The synthesis vLLM represents

vLLM = Orca's iteration-level scheduling + PagedAttention's memory management + (increasingly) Sarathi's chunked prefill + optional FastServe-style priorities. Each prior system contributed a piece; vLLM integrated them into a production engine and added the broad model/quantization/distributed support that made it the default. Understanding the lineage clarifies that the scheduler is not one idea but a *stack* of scheduling ideas, each addressing a specific failure of the previous (static → iteration-level → memory-managed → stall-free → size-aware).

---

## 26. Decode-Prioritized vs Prefill-Prioritized Scheduling

A scheduler choice with large latency consequences is whether, within a step, to favor decode or prefill when budget is contended.

- **Decode-prioritized** (vLLM's default leaning, §5.1): schedule running decodes first, then fill remaining budget with prefill. Protects TPOT (in-flight requests keep generating) at the cost of TTFT (new requests' prefill waits for budget). Good for interactive workloads where smooth token streaming matters.
- **Prefill-prioritized:** admit and run prefills aggressively, even if it means fewer decodes per step. Lowers TTFT (new requests start fast) at the cost of TPOT (existing requests stutter). Good when TTFT is the dominant SLO (e.g. short-output, high-arrival-rate workloads).

Chunked prefill (§7) softens this dichotomy by making prefill *bounded*, so you can serve both prefill and decode every step without either monopolizing. The residual choice — how much of each step's budget to allocate to prefill vs decode — is a tuning decision (`max_num_batched_tokens` vs the running decode count) that File 11 fits to the workload's TTFT/TPOT priorities. Some systems expose explicit knobs (a target prefill-token fraction per step); vLLM controls it implicitly through the budget and the decode-first ordering.

---

## 27. A Worked Memory-Aware Admission Example

Make admission control concrete. Suppose (per File 03 §17's budget) the engine has **40,000 KV blocks** of 16 tokens (640,000 token-slots), `max_num_seqs = 256`, and a workload averaging 1,000-token contexts (prompt + generation).

- Average KV per sequence ≈ `1000 / 16 = 62.5 blocks`.
- Memory-bound concurrency ≈ `40000 / 62.5 = 640` sequences.
- But `max_num_seqs = 256` caps concurrency below the memory limit → the engine is **concurrency-bound**, not memory-bound, here. The operator could safely raise `max_num_seqs` toward ~600 to use the spare KV memory (more batch → more throughput), leaving a watermark margin.

Now suppose contexts average 8,000 tokens instead:

- KV per sequence ≈ `8000 / 16 = 500 blocks`.
- Memory-bound concurrency ≈ `40000 / 500 = 80` sequences.
- With `max_num_seqs = 256`, the engine is now **memory-bound**: it can only run ~80 sequences before the pool fills, and admitting more triggers preemption thrash. The operator should *lower* `max_num_seqs` toward ~75 (below the memory limit, with margin) to prevent thrashing, or reduce per-sequence KV (FP8 KV cache, GQA model, shorter `max_model_len`).

This is the essential admission-control reasoning: the binding constraint flips between concurrency and memory depending on context length, and the right `max_num_seqs` is `min(concurrency_cap, memory_limit_with_margin)` computed from the *actual* context distribution. Setting it blindly (the default 256) is fine for short-context chat and disastrous for long-context workloads (thrash) or wasteful for very short ones (under-batching). File 11 turns this into a measurement-driven procedure.

---

## 28. Aborts, Cancellation, and Timeouts

Real serving must handle requests that should stop before natural completion:

- **Client disconnect:** the HTTP client closes the connection (user navigated away, gateway timeout). The API layer detects this and signals the engine to **abort** the request: the scheduler transitions it to `FINISHED_ABORTED` and frees its KV blocks immediately (File 03 §12), reclaiming memory for other requests. Failing to detect disconnects leaks KV (a "zombie" request generating tokens nobody reads, holding blocks) — a real production bug class.
- **Cancellation API:** some deployments expose explicit cancellation (cancel request ID); same abort path.
- **Timeouts:** a max generation time or `max_tokens` cap bounds how long a request can occupy resources. `max_tokens` is enforced as a stop condition (§12.4); wall-clock timeouts are typically enforced at the API/gateway layer with a corresponding engine abort.

Abort handling interacts with everything: an aborted sequence mid-chunked-prefill must free its partial blocks and decrement refcounts on shared prefix blocks; an aborted beam-search group must prune all beams; an aborted speculative step must discard draft state. The scheduler's `free` path (File 03 §11.3) is the common cleanup, and its correctness under abort is essential to avoiding slow memory leaks under churny traffic (many short-lived, frequently-cancelled requests — common with interactive UIs that cancel on each keystroke-triggered regeneration).

---

## 29. Guided/Structured Decoding and the Scheduler

Structured decoding (File 02 §10, File 08 XGrammar) adds per-sequence grammar state that the scheduler and executor must thread through each step. Each constrained sequence carries a **grammar matcher** (FSM/PDA state) that, after each generated token, advances and produces the allowed-token mask for the next step. Scheduling implications:

- **Per-sequence state:** the matcher state is part of the sequence's metadata, carried across steps and through preemption (a swapped/recomputed sequence must restore its grammar state — for recompute, the state is re-derived by replaying the generated tokens through the grammar).
- **Batched masking:** a batch may contain sequences with different grammars (one doing JSON, another a regex, others unconstrained). The executor applies each sequence's mask to its logits before sampling, via a batched logit-processing step (XGrammar's CUDA kernel handles this efficiently across heterogeneous grammars).
- **Mask computation overlap:** advancing the grammar and computing the next mask is CPU work that can overlap with the GPU forward pass (compute the current step's mask while the previous step's matmuls run), keeping structured decoding's overhead off the critical path — the same overlap principle as async scheduling (§11.3). XGrammar's precomputed per-state masks (File 08) make this near-free.

The scheduler treats constrained sequences like any other for admission/batching; the constraint machinery rides along in the per-sequence metadata and the logit-processing stage. This composability — structured decoding "just works" within continuous batching — is itself a design achievement, since a naive per-step FSM evaluation over a 128K vocabulary would have made constrained decoding a scheduling bottleneck.

---

## 30. Disaggregation: The Prefill:Decode Ratio, Worked

DistServe's central planning decision (§10.4) is how many prefill workers vs decode workers to provision. Work it numerically.

Consider a chat workload: average prompt 500 tokens, average output 500 tokens, on LLaMA-3 70B at TP=4 per instance.

- **Prefill cost per request:** ~`2 · 70e9 · 500 ≈ 70 TFLOP`. On a 4×H100 prefill instance (~4×989 ≈ 3,956 TFLOP/s, say 80% util → ~3,165 TFLOP/s), ~22 ms of prefill compute per request.
- **Decode cost per request:** 500 decode steps. Each step (batched) is memory-bound; per-request decode time depends on the decode batch size. Suppose the decode instance sustains a batch of 64 with ~40 ms/step → each request gets a token every 40 ms → 500 tokens × 40 ms = 20 s of wall-clock decode, but the *instance* serves 64 requests concurrently, so per-request decode *throughput cost* is `20 s / 64 ≈ 0.31 s` of instance time per request.

So one request needs ~22 ms of prefill-instance time and ~310 ms of decode-instance time → the workload is **~14× more decode-heavy than prefill-heavy** in instance-time terms. To balance, you'd provision roughly **1 prefill instance per ~14 decode instances** (rounded by integer instances and burst headroom, often expressed as ~1:4 to 1:8 after accounting for prefill batching efficiency and decode-instance cheaper hardware). For a summarization workload (8,000-token prompt, 100-token output) the balance flips toward prefill-heavy (closer to 1:1 or prefill-dominant). The lesson: the optimal P:D ratio is *workload-specific* and derived from the relative prefill and decode instance-time per request — exactly the queuing analysis (§18) applied to two separate pools. Mis-provisioning (e.g. equal P and D for a decode-heavy chat workload) leaves prefill instances idle while decode instances saturate, wasting the disaggregation benefit. This is why DistServe frames it as a goodput-maximizing resource-allocation problem and why production disaggregation (Mooncake) includes dynamic rebalancing.

---

## 31. Step-Time Predictability and CUDA Graphs

The scheduler's batch-composition decisions interact with CUDA graphs (File 09), which capture a fixed-shape forward pass to eliminate per-kernel launch overhead. CUDA graphs require **static shapes** — a graph captured for batch size 16 can only replay for batch size 16. The scheduler's dynamically-varying batch size therefore forces a choice:

- **Pad to captured sizes:** capture graphs for a set of batch sizes (1, 2, 4, 8, 16, 32, …); at runtime, pad the actual batch up to the nearest captured size and replay that graph. The padding wastes a little compute but gains the launch-overhead elimination. This is the standard approach (vLLM and SGLang both pre-capture decode graphs for a ladder of batch sizes).
- **Decode-only graphs:** prefill has variable sequence lengths (hard to graph), so graphs are captured for the *decode* path (query length 1, batch = number of running sequences) where shapes are regular. Prefill runs in eager mode. This is why chunked prefill's predictable per-step structure (§7.5) helps: a step that's "all decode plus a bounded prefill chunk" is more graph-friendly than wildly varying atomic prefills.

The scheduler doesn't manage graphs directly, but its tendency to produce regular, bounded-size decode batches (helped by chunked prefill and concurrency caps) is what makes the graph approach effective. An erratic scheduler producing wildly varying batch sizes would force constant graph-size switching or eager fallback, losing the benefit. So scheduler design and kernel-execution strategy are coupled: predictable scheduling enables CUDA-graph speedups, which especially help small models and decode-heavy steps where launch overhead dominates (File 02 §15, File 09).

---

## 32. Common Scheduler Pitfalls and Their Signatures

A field guide tying symptoms to scheduler causes (operational depth in File 11, File 19):

- **High P99 TPOT under mixed traffic, fine P50:** long prefills are stalling decodes. *Fix:* enable/tune chunked prefill (§7); reduce chunk size.
- **High TTFT under load, fine TPOT:** decode-prioritized scheduling is starving prefill admission, or you're near `λ_max`. *Fix:* allow more prefill budget, add capacity, or shed load.
- **Throughput collapses past a load threshold (capacity cliff):** `λ > λ_max` → preemption thrash. *Fix:* admission control / load shedding, add replicas (§18).
- **Frequent preemptions, `gpu_cache_usage_perc` pinned ~100%:** `max_num_seqs` over-admits for the context length. *Fix:* lower `max_num_seqs` to the memory limit with margin (§27), or reduce per-seq KV.
- **Slow memory leak under churny traffic:** aborted/disconnected requests not freeing KV. *Fix:* ensure disconnect detection and the abort→free path (§28).
- **Low throughput on a small model despite low latency:** Python scheduler overhead dominates. *Fix:* upgrade to V1 engine (§14), enable CUDA graphs, or use multi-step scheduling for offline jobs.
- **One request's tail latency is terrible under sustained load:** FCFS LIFO victim selection repeatedly preempts a late-arriving long request. *Fix:* priority scheduling, or accept it as fair (it arrived last).

The diagnostic discipline: distinguish *latency* problems (TTFT vs TPOT — which one? at which percentile?) from *throughput/capacity* problems (is the waiting queue growing?), then map to the responsible mechanism. The scheduler metrics (§23) provide the evidence.

---

## 33. Appendix: Scheduler Configuration Reference

- **`--max-num-batched-tokens`**: per-step token budget (prefill + decode). Primary throughput/latency lever and chunk-size bound with chunked prefill.
- **`--max-num-seqs`**: max concurrent sequences. Set to `min(concurrency_need, memory_limit_with_margin)` for the actual context distribution (§27).
- **`--enable-chunked-prefill`**: bounded prefill chunks; default-on in recent versions; the key to stall-free mixed-traffic latency (§7, §17).
- **`--scheduling-policy`** (`fcfs` | `priority`): fairness model (§8).
- **`--preemption-mode`** (`recompute` | `swap`): reclaim strategy under memory pressure (§9).
- **`--num-scheduler-steps`**: multi-step amortization of scheduler overhead (§11.2); raise for offline throughput, keep at 1 for interactive streaming.
- **`--enable-prefix-caching`**: cheaper admission for cache-warm requests (§20).
- **`--enable-lora` / `--max-loras`**: multi-adapter serving and the adapter-slot resource (§21).
- **`--gpu-memory-utilization`**: sizes the block pool the scheduler spends (File 03 §7).
- **`--max-model-len`**: per-sequence length cap; bounds block-table size and (with the memory budget) the context distribution the scheduler must support.

These are not independent. A coherent configuration starts from the workload (arrival rate, prompt/output length distributions, SLO targets), computes the memory-bound concurrency (§27, File 03 §17), sets `max_num_seqs` below it, sizes `max_num_batched_tokens`/chunk to meet the TPOT SLO under expected prefill arrivals (§17.3), and chooses a policy and preemption mode for the fairness/overload behavior desired. File 11 walks this end to end.

---

## 34. Key Takeaways

1. **Scheduling is online, multi-objective optimization under output-length uncertainty** — harder than memory management, and the main determinant of behavior under load.
2. **Continuous (iteration-level) batching** keeps the batch large without coupling request latencies — the foundational departure from static batching, inherited from Orca.
3. **Chunked prefill** makes prefill bounded, decoupling prefill latency from decode latency and enabling "stall-free" batching (Sarathi) and SLO guarantees.
4. **Token budget + concurrency caps** are the admission-control knobs; the binding constraint flips between concurrency and memory with context length (§27).
5. **Preemption (swap/recompute)** gives graceful memory-pressure behavior; thrashing signals over-admission, not a scheduling-policy problem.
6. **Policies (FCFS, priority, SRTF-approximations)** shape latency *within* capacity; they cannot create capacity — admission control and horizontal scaling do (§18).
7. **Disaggregation** separates the compute-bound and bandwidth-bound phases onto ideal, independently-scaled hardware, with the P:D ratio set by workload (§30).
8. **Multi-step / async / V1** remove the scheduler from the GPU's critical path, mattering most for fast steps (small models, CUDA graphs).

The scheduler spends the block pool of File 03 to maximize the goodput defined in File 01, mediating the prefill/decode duality through every decision. Master it and you can predict — and tune — how a vLLM deployment will behave when the traffic gets real. Next, File 05 extends the execution substrate across many GPUs, where the scheduler's plan must be coordinated across tensor-, pipeline-, expert-, and sequence-parallel workers.

---

## 35. Multi-Step Scheduling, Worked

To make §11.2 concrete, compare per-step accounting with and without multi-step on a fast model where a decode step takes 2 ms of GPU time and `schedule()` + CPU↔GPU sync costs 0.8 ms.

**Single-step (`--num-scheduler-steps 1`):**
```
per iteration = 0.8 ms (schedule) + 2 ms (GPU) = 2.8 ms
GPU utilization = 2 / 2.8 ≈ 71%   (29% lost to scheduler/sync)
```

**Multi-step (`--num-scheduler-steps 8`):**
```
per 8 GPU steps = 0.8 ms (schedule once) + 8 × 2 ms = 16.8 ms
GPU utilization = 16 / 16.8 ≈ 95%   (scheduler amortized over 8 steps)
throughput gain ≈ 95/71 ≈ 1.34×
```

The cost: between the 8 GPU steps the scheduler is blind — it cannot admit the request that arrived during them until the block completes (adding up to ~16 ms to that request's TTFT), and a sequence that hits EOS at sub-step 3 still runs sub-steps 4–8 (wasting 5 steps of compute on it, though its tokens are simply discarded). For **offline batch throughput** (no streaming, latency-insensitive) multi-step is a clear win; for **interactive streaming** it harms responsiveness and conflicts with per-token streaming (tokens for all 8 sub-steps arrive together). This is why the better long-term answer is **async scheduling** (§11.3) / the V1 engine (§14), which overlaps the 0.8 ms schedule with the 2 ms GPU step (planning step `t+1` during step `t`'s execution), recovering the 29% *without* the blindness — utilization approaches 100% while staying responsive. Multi-step is the simpler, coarser tool; async/overlap is the refined one.

---

## 36. Scheduling Under Pipeline Parallelism

Pipeline parallelism (File 05 §PP) partitions the model's layers across GPUs, so a forward pass flows through stages. This complicates scheduling because a single step is no longer one synchronous GPU call but a pipeline with multiple micro-batches in flight.

- **Scheduler runs on the driver (rank 0):** the scheduler makes batch decisions centrally; the assembled batch is split into **micro-batches** that flow through the pipeline stages to keep all stages busy (hiding the pipeline bubble, File 05 §PP). 
- **Micro-batch scheduling:** with PP degree `P`, the scheduler must supply enough concurrent micro-batches (≥ `P`) to fill the pipeline, or stages idle in the bubble. This couples the scheduler's batch-size choice to the pipeline depth — too small a batch can't fill the pipeline.
- **In-flight steps across stages:** while stage 0 processes micro-batch `i`, stage 1 processes micro-batch `i−1`, etc. The scheduler reasons about multiple micro-batches in flight, and completion/admission decisions must account for the pipeline latency (a token isn't "done" until it exits the last stage). vLLM's PP support integrates this with continuous batching by having the scheduler emit micro-batched work and the runtime manage the inter-stage send/recv (File 05).

The upshot: pipeline parallelism adds a scheduling dimension (micro-batch count to fill the pipe) on top of the continuous-batching logic, and the scheduler must produce batches large enough to amortize the pipeline bubble. This is one reason PP is less common than TP for latency-sensitive inference — it needs sufficient concurrency to be efficient, whereas TP works even at batch 1 (File 05 §TP vs PP trade-offs).

---

## 37. Multi-Tenancy and Fairness Across Users

Production deployments often serve many tenants (customers, API keys, internal teams) on shared capacity, raising fairness questions the base FCFS/priority policies only partly address:

- **Per-tenant fairness:** FCFS is fair *per request* but not *per tenant* — a tenant flooding the queue with requests gets proportionally more service than a low-volume tenant. True per-tenant fairness needs weighted fair queuing (round-robin across tenants' queues, or deficit-weighted scheduling), typically implemented at a **router/gateway layer above the engine** rather than inside vLLM's scheduler. The router admits requests to engines in a tenant-fair order; the engine schedules whatever it's given.
- **Rate limiting and quotas:** per-tenant rate limits (requests/sec, tokens/sec) are enforced upstream, shaping the arrival stream the scheduler sees so no tenant can monopolize.
- **Priority tiers:** mapping tenants to priority classes (paid > free, interactive > batch) uses the scheduler's priority policy (§8.2) plus upstream admission. The engine's priority scheduling preempts lower-tier running requests for higher-tier arrivals.
- **Isolation vs sharing:** strict isolation (dedicated replicas per tenant) wastes capacity but guarantees no cross-tenant interference; sharing (one pool, fair scheduling) is efficient but requires careful fairness/QoS to prevent noisy-neighbor effects. Most deployments share with priority tiers and per-tenant rate limits, reserving dedicated capacity only for the largest or most latency-sensitive tenants.

The division of labor — engine scheduler handles *mechanism and per-request/priority policy*, the router/gateway handles *per-tenant fairness, quotas, and load balancing* — keeps the engine simple and puts multi-tenancy policy where it belongs (File 05 §load balancing, File 19 §multi-model, File 20 §SaaS patterns).

---

## 38. Building the Batch Tensor: Where Scheduling Meets Execution

A final concrete detail: how the scheduler's plan becomes the tensors the model consumes (the handoff to File 06). Given the `SchedulerOutputs` (§22), the model runner constructs, for the mixed prefill+decode batch:

- **`input_ids`**: a flat (packed) array of all tokens to process this step — for each prefilling sequence, its chunk's tokens; for each decoding sequence, its single new token. No padding (packed/varlen layout, File 02 §20).
- **`positions`**: the position ID for each token, continuing from each sequence's `num_computed_tokens` (File 02 §19) — essential for correct RoPE.
- **`slot_mapping`**: for each token, the physical KV slot to write its K/V into (File 03 §3.2), derived from the sequence's block table.
- **`block_tables`**: per-sequence logical→physical maps, for the attention kernel to read cached KV.
- **`cu_seqlens` / `seq_lens` / query-start indices**: the ragged-batch metadata telling the varlen attention kernel each sequence's query length (n for prefill chunks, 1 for decode) and KV length (context length).
- **Sampling metadata**: per-sequence sampling params, grammar masks (§29), LoRA adapter IDs (§21).

This construction — gathering scattered per-sequence state into packed tensors with the ragged indices — is pure CPU work done each step, and doing it with minimal overhead (and overlapping it with the GPU, §11.3) is a real performance concern (SGLang uses shared memory for zero-copy transfer of this batch to the executor, File 09 §shared memory). The scheduler decides *what* runs; this construction encodes *how* the GPU sees it. Together they close the loop from "a queue of requests" to "a forward pass," every 2–40 ms, for the life of the server — the relentless heartbeat that the rest of the stack (kernels, distributed runtime, memory) exists to make fast.

---

## 39. Decomposing TTFT and TPOT into Scheduler Terms

To close the loop between scheduling decisions and the SLO metrics (File 11), decompose the two latencies into their scheduler-controlled components.

**TTFT (Time To First Token)** for a request decomposes as:
```
TTFT = queue_wait + prefill_compute + scheduling_overhead
```
- `queue_wait`: time in the waiting queue before admission — set by load (`λ` vs `λ_max`, §18), admission policy (§8), and whether decode-prioritized scheduling delays prefill (§26). This is the term that explodes near the capacity cliff.
- `prefill_compute`: time to process the prompt — set by prompt length, prefix-cache hits (§20, which can slash it), and whether chunked prefill spreads it across steps (which can *increase* TTFT slightly for the prefilling request itself, the cost of protecting others' TPOT).
- `scheduling_overhead`: the per-step Python/sync cost (§11), reduced by V1/async/multi-step.

**TPOT (Time Per Output Token)** decomposes as:
```
TPOT = decode_step_time = f(batch_size, context_length, model, hardware) + stall_time
```
- The base decode step time grows with batch size (more sequences sharing bandwidth) and context length (more KV to read) — the memory-bound physics of File 02 §15.2.
- `stall_time`: extra latency when a step is monopolized by prefill (§17.1), driven to ~0 by chunked prefill. This is the term chunked prefill exists to eliminate.

This decomposition is the scheduler's report card. A TTFT problem points to queue_wait (capacity/admission) or prefill_compute (prompt length/caching). A TPOT problem points to batch size (throughput–latency trade-off, dial back concurrency), context length (long-context cost, File 13), or stalls (enable chunked prefill). Each term maps to a specific scheduler knob, which is why diagnosing *which percentile of which metric* is degraded (§32) immediately narrows the fix. The scheduler cannot change the memory-bound physics of `decode_step_time`, but it controls every other term — queue wait, prefill spreading, stall elimination, and the batch-size point on the throughput–latency curve — which is the full extent of what software can do to shape latency on fixed hardware.

### 39.1 A note on percentiles

Means hide the scheduler's most important behavior. A scheduler can deliver excellent P50 latency while P99 is catastrophic — because the tail is exactly where preemption, stalls, and queue buildup bite (the unlucky request that got preempted twice, or arrived during a long-prefill burst). Production SLOs are almost always stated at P95/P99 precisely because that is where scheduling quality shows. Chunked prefill's headline result (2.6× P99 TTFT, §17.2) is a *tail* improvement that barely moves the mean. When evaluating a scheduler change, always look at the tail percentiles under load, not the average — the average will mislead you into thinking a head-of-line-blocking problem is solved when only the lucky-majority case improved.

---

## 40. Closing

The vLLM scheduler is the policy engine that turns the memory mechanism of File 03 and the kernels of Files 05/10 into a system that behaves well under real traffic. Its evolution — from Orca's iteration-level scheduling, through PagedAttention-enabled cheap preemption, to Sarathi's stall-free chunked prefill and the V1 async rearchitecture — is a layered response to the specific ways naive serving fails: convoy effects, head-of-line blocking, memory thrash, and scheduler overhead. Every knob it exposes corresponds to a term in the TTFT/TPOT decomposition above, and every policy it implements is a strategy for spending finite KV memory and finite GPU time to maximize goodput under latency constraints. An inference engineer who internalizes this file can read a latency dashboard and name the responsible scheduler mechanism, and can configure an engine to fit a workload's SLOs rather than accepting defaults. That diagnostic-and-tuning fluency, grounded in the queuing and roofline models, is the practical payoff — and it carries directly into the distributed setting of File 05, where the same scheduling logic must now coordinate a plan across many GPUs without letting communication become the new bottleneck.

---

## 41. Appendix: Key Papers and Source Pointers

- **Orca: A Distributed Serving System for Transformer-Based Generative Models** — Yu et al., OSDI 2022. Iteration-level scheduling and selective batching; the conceptual origin of continuous batching.
- **Efficient Memory Management for LLM Serving with PagedAttention** — Kwon et al., SOSP 2023. The memory layer the scheduler spends; introduces continuous batching atop paged KV.
- **Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills** — Agrawal et al., arXiv 2308.16369; **Sarathi-Serve** — arXiv 2403.02310. Chunked prefill and stall-free batching with SLO guarantees.
- **FastServe: Fast Distributed Inference Serving for LLMs** — Wu et al., arXiv 2305.05920. Preemptive, SRPT-approximating, iteration-level-preemptive scheduling.
- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving** — Zhong et al., arXiv 2401.09670. The analytical model for prefill:decode disaggregation.
- **Mooncake** — Kimi.ai, 2024. Production disaggregation with RDMA KV transfer and a global KV cache manager.

Read these in order — Orca for the batching insight, PagedAttention for the memory substrate, Sarathi for chunked prefill, FastServe for size-aware preemption, DistServe/Mooncake for disaggregation — and the scheduler's design appears not as a single invention but as the accreted response of a research community to each successive bottleneck exposed by the previous generation of systems.

In the vLLM codebase, the scheduler is `vllm/core/scheduler.py` (V0) and the V1 engine core under `vllm/v1/` (`EngineCore`, the async scheduler); `SchedulerConfig` carries the budget/concurrency/policy parameters; `SequenceGroup` and `Sequence` (in `vllm/sequence.py`) model the request/sample/beam structure; and the model runner that turns `SchedulerOutputs` into batch tensors lives in `vllm/worker/` and `vllm/model_executor/` (File 06). To trace a request end to end: `AsyncLLMEngine.generate` (File 07) → enqueue → `Scheduler.schedule` (this file) → block-manager mutations (File 03) → worker `execute_model` → forward pass (Files 05, 06, 10) → sample → detokenize/stream (File 07). Each arrow is a section in this database; the scheduler is the hub they all pass through, which is why understanding it is prerequisite to understanding the system as a whole. The scheduler is the one component that touches every other: it reads the block pool's free state, drives the model runner's batch, gates admission against capacity, and shapes every latency percentile the user experiences — the integration point where memory, compute, communication, and policy meet.

*Next: File 05 — distributed inference, where tensor, pipeline, expert, and sequence parallelism extend this execution model across many GPUs, and where collective communication becomes a first-class scheduling and performance concern.*





