# Benchmarking, Performance Analysis, and Production Tuning

> **PRIMARY reference file.** This file is the practical synthesis: how to measure LLM serving performance, analyze it against the roofline, and tune vLLM and SGLang for a target workload and SLO. It covers the metrics taxonomy (TTFT/TPOT/throughput/goodput), roofline analysis, the vLLM and SGLang tuning parameters in depth, benchmarking methodology, production deployment patterns, and profiling/debugging. It unifies the three layers — algorithms (Files 03–09), architecture (Files 04, 09), and kernels (File 10) — into a measurement-driven tuning discipline. Prerequisites: File 01 (roofline, goodput), and Files 03–10 for the mechanisms being tuned.

---

## Table of Contents

1. The Metrics Taxonomy
2. Goodput: The Real Objective
3. Roofline Analysis for LLM Serving
4. vLLM Tuning Parameters
5. SGLang Tuning Parameters
6. Benchmarking Methodology
7. Latency Under Load
8. Comparing vLLM and SGLang
9. Multi-GPU Scaling Efficiency
10. Production Deployment Patterns
11. Profiling and Debugging
12. A Tuning Playbook

---

## 1. The Metrics Taxonomy

You cannot tune what you don't measure correctly. LLM serving has a specific set of metrics, and conflating them causes bad decisions.

### 1.1 TTFT (Time To First Token)

The time from request arrival to the first output token. Decomposes (File 04 §39) as `queue_wait + prefill_compute + scheduling_overhead`. Dominated by prefill latency for long prompts. For **streaming** applications (chat), TTFT is the dominant perceived-latency metric — the user waits for the response to *start*. Typical SLO: 200 ms–2 s depending on the application. Affected by: queue depth (load), prompt length, prefix-cache hits (which slash prefill), and whether decode-prioritized scheduling delays prefill admission (File 04 §26).

### 1.2 TPOT (Time Per Output Token)

The average time between consecutive output tokens after the first: `TPOT = (E2E_latency − TTFT) / (num_output_tokens − 1)`. Dominated by decode-step latency. For streaming, TPOT determines the *streaming speed* (tokens/sec the user sees) — it should exceed reading speed (~5–10 tokens/sec for humans, but applications often want much faster). Typical SLO: <50–100 ms (i.e. >10–20 tokens/sec). Affected by: batch size (more sequences share decode bandwidth → higher TPOT), context length (more KV to read), model size, and stalls from prefill interference (eliminated by chunked prefill, File 04 §17). Also called ITL (Inter-Token Latency).

### 1.3 E2E Latency

End-to-end: `TTFT + TPOT × (num_output_tokens − 1)`. The total time the user waits for the complete response. For non-streaming requests, this is what matters. Dominated by TPOT × output length for long generations (reasoning models, File 15 §long CoT).

### 1.4 Throughput

- **Output token throughput:** generated tokens/sec across all requests — the headline capacity metric.
- **Total token throughput:** including prompt (prefill) tokens.
- **Request throughput (RPS):** requests/sec.
- **Per-GPU normalization:** tokens/sec/GPU — the cost-relevant metric (more GPUs trivially raise absolute throughput; per-GPU is what determines $/token).

Throughput trades off against latency: larger batches raise throughput but raise TPOT (File 01 §11). The two cannot be maximized independently.

### 1.5 Why means mislead

Always look at **percentiles** (P50, P95, P99), not just means (File 04 §39.1). A deployment can have great P50 TTFT but terrible P99 — the tail is where queuing, preemption, and prefill-stalls bite, and where SLOs are usually defined. Reporting only averages hides the head-of-line-blocking and capacity-cliff behaviors that determine real user experience. A latency histogram (the engines' Prometheus metrics, File 07 §6) is the right tool.

---

## 2. Goodput: The Real Objective

Raw throughput is the wrong optimization target because it ignores latency constraints. The right target is **goodput**: the request rate served *while meeting the latency SLOs* (File 01 §11). For example: "requests/sec served with P95 TTFT < 2 s AND P99 TPOT < 100 ms." A configuration that achieves higher raw throughput by using huge batches (raising TPOT past the SLO) has *lower* goodput than one with smaller batches that meets the SLO — because the SLO-violating requests are effectively failures (a chat response streaming too slowly is a bad experience regardless of aggregate token throughput).

Goodput maximization is the actual objective of all the scheduling and tuning machinery (Files 04, 09): chunked prefill, priority policies, the batch-size point on the throughput-latency curve, disaggregation — all exist to push the goodput frontier outward (serve more requests per GPU *within* the latency budget). This reframing is important: when tuning, you're not maximizing throughput or minimizing latency in isolation, but maximizing the rate of SLO-meeting requests. The benchmarking methodology (§6–7) measures goodput by sweeping load and finding the maximum rate that keeps the latency percentiles under their SLOs.

---

## 3. Roofline Analysis for LLM Serving

The roofline (File 01 §2) is the analytical foundation for understanding where performance comes from and where it's lost.

### 3.1 Arithmetic intensity of the operations

- **Single decode step (batch 1):** `2·num_params FLOP / (2·num_params bytes + KV bytes) ≈ 1 FLOP/byte` (BF16) — deeply memory-bound.
- **Large-batch prefill (batch B, seqlen S):** `≈ B·S FLOP/byte` — compute-bound for large `B·S`.
- **Break-even (ridge point):** compute-bound when `B·S ≥ peak_FLOPS / peak_bandwidth`. A100: `312e12 / 2e12 = 156` tokens in flight; H100 BF16: ~295; H100 FP8: ~590 (File 01 §2.2).

### 3.2 Memory-bandwidth utilization (decode)

For decode, the achievable performance is `bandwidth × arithmetic_intensity`, and the step time is bounded below by the bytes read divided by bandwidth. Worked (File 01 §15.2, File 02 §15.2): LLaMA-3 70B, batch 32, A100 (2 TB/s): bytes read ≈ `2 · 70e9 · 2 (BF16 weights) = 280 GB`; min decode step time `≥ 280 GB / 2 TB/s = 140 ms`. If the measured step time is 150 ms → `140/150 ≈ 93%` bandwidth utilization (well-optimized — the kernel is near the bandwidth roofline). If it's 280 ms → 50% utilization, indicating a kernel inefficiency (poor coalescing, File 10 §30) or overhead (launch overhead without CUDA graphs, File 10 §15.2). This is how you check whether decode is hitting its roofline: compute the theoretical min step time from bytes/bandwidth, compare to measured.

### 3.3 FLOP utilization (prefill)

For prefill, compute the theoretical min time from FLOPs/peak-FLOP-rate. Worked: LLaMA-3 70B, prefill batch 512 × seqlen 512, A100: FLOPs ≈ `2 · 512 · 512 · 70e9 ≈ 3.7e16`... (using `2·P·tokens`, tokens = 262144) ≈ `2 · 70e9 · 262144 ≈ 3.7e16 = 37 PFLOP`; min time `37e15 / 312e12 ≈ 118 ms`. If measured is 145 ms → `118/145 ≈ 81%` FLOP utilization (good). Below ~70% suggests inefficient GEMM kernels or small-batch underutilization. This checks whether prefill hits its compute roofline.

### 3.4 Using the roofline to direct tuning

The roofline tells you the *regime* (compute- or memory-bound) and the *gap to optimal*, which directs the fix:
- Decode far below the bandwidth roofline → kernel inefficiency (FlashInfer? CUDA graphs? coalescing?) or overhead.
- Prefill far below the compute roofline → GEMM efficiency, or batch too small (raise batch to fill the tensor cores).
- Decode *at* the bandwidth roofline → to go faster, reduce bytes (quantization, GQA, FP8 KV) — you can't beat the bandwidth, only move less data.
- Prefill *at* the compute roofline → to go faster, raise the roofline (FP8) — you can't beat the FLOP rate, only use faster tensor cores.

This roofline-directed reasoning is the analytical core of performance tuning: measure where you are relative to the roofline, identify the regime and gap, and apply the matching lever.

---

## 4. vLLM Tuning Parameters

The key vLLM flags and how to set them (mechanisms in Files 03–07):

### 4.1 `--max-num-batched-tokens`

The per-step token budget (prefill + decode, File 04 §6). Higher → larger effective batches → higher throughput, but bigger per-step latency spikes (worse for chunked-prefill latency goals). With chunked prefill on, this also bounds the chunk size. Tune: raise for throughput-oriented offline jobs; for latency-sensitive serving, set so the per-step time meets the TPOT SLO (a step processing this many tokens shouldn't exceed the inter-token budget). Typical: 2048–8192, or higher for prefill-heavy throughput.

### 4.2 `--max-num-seqs`

The concurrency cap (File 04 §6.2, §27). The effective batch is `min(max_num_seqs, KV-memory limit, token budget)`. Set to `min(concurrency_need, memory-bound concurrency with margin)` for the *actual* context distribution (File 04 §27): for short context, raise it (memory allows more); for long context, lower it (memory binds, over-admission thrashes). The default 256 is fine for short-context chat, wrong for long-context (thrash) or wasteful for very short (under-batch).

### 4.3 `--gpu-memory-utilization`

The fraction of GPU memory for the KV pool after weights (File 03 §7). Default 0.9. Raise to 0.95 for max KV/throughput (risk: OOM on unexpected long sequences or profiling underestimate, File 03 §17.1); lower to 0.85 for safety. The single most impactful memory knob — it directly sets `num_gpu_blocks` and thus concurrency. Profile to find the safe maximum.

### 4.4 `--enable-chunked-prefill` (+ chunk size)

Enables chunked prefill (File 04 §7), smoothing prefill latency spikes for mixed traffic — default-on in recent versions. Smaller chunks → better decode-latency interleaving (lower P99 TPOT) but lower prefill throughput; larger → the reverse. Tune the chunk (via `max_num_batched_tokens` with chunked prefill on) so the per-step time meets the TPOT SLO (File 04 §17.3). Typical good values: 512–2048.

### 4.5 `--tensor-parallel-size` and `--data-parallel-size`

TP and DP degrees (File 05 §12). For a model that fits on few GPUs: TP=lowest-that-meets-latency × DP=rest (File 05 §16). For latency-critical: higher TP (more parallel compute). For throughput: lower TP × higher DP (less AllReduce overhead, linear scaling). Check head-divisibility (File 05 §2.3) and benchmark scaling efficiency (§9). For a 70B on 8×H100: TP=8 single replica (latency) or TP=4×DP=2 (throughput) — choose by whether TP=4 meets the TPOT SLO.

### 4.6 `--num-scheduler-steps`

Multi-step scheduling (File 04 §11.2). Default 1. Raise (4–8) for decode-heavy *offline* throughput (amortizes scheduler overhead) — but it harms interactive streaming (blindness window, File 04 §35). On the V1 engine (async scheduling, File 04 §14), this is largely unnecessary. Use for offline batch jobs, not interactive serving.

### 4.7 Speculative decoding

`--speculative-model`, `--num-speculative-tokens` (File 12). The draft size vs acceptance-rate trade-off: a 7B draft for a 70B target gets ~60–70% acceptance on chat, ~80% on code. Memory overhead: the draft weights + speculation KV. Tune `num_speculative_tokens` (K) — higher K proposes more but with diminishing acceptance; typical 3–5. Most beneficial when decode is the bottleneck (long outputs, low-to-moderate batch where the target has spare compute to verify).

### 4.8 CUDA graphs and KV dtype

`--enforce-eager` disables CUDA graphs (for debugging — leave graphs on for production, File 10 §15.2). `--kv-cache-dtype fp8_e5m2` halves KV memory, ~doubling KV-bound concurrency on H100+ (File 13). `--quantization fp8|awq|gptq` for weight quantization (File 06 §3, §12) — choose by the binding constraint (memory → W4A16; prefill compute → FP8).

---

## 5. SGLang Tuning Parameters

The SGLang equivalents (mechanisms in Files 08–09):

### 5.1 `--mem-fraction-static`

The fraction of GPU memory for static allocation (weights + CUDA graph buffers); the rest is the KV pool / radix cache (File 09 §32). Default ~0.88. Lower it to give RadixAttention more KV (higher prefix-cache hit rate, more concurrency); raise if OOMing on static allocation. The analog of vLLM's `--gpu-memory-utilization` (inverted sense — this is the *static* fraction). The key knob for prefix-cache effectiveness (File 08 §18).

### 5.2 `--max-running-requests`

Concurrency cap (File 09 §9.2), like vLLM's `--max-num-seqs`. Set below the memory-bound concurrency to avoid thrash. Tune to `num_kv_slots / avg_kv_per_request` for the actual context distribution.

### 5.3 `--chunked-prefill-size`

The chunked-prefill token budget (File 04 §7, File 09 §9). Smaller → better decode-latency interleaving; larger → higher prefill throughput. Radix-tree-aligned where possible. Tune for the TPOT SLO under prefill arrivals.

### 5.4 `--attention-backend`

`flashinfer` (default, fastest on NVIDIA), `triton` (AMD ROCm or fallback), `torch` (reference, debugging) — File 09 §7, File 10 §27. Always use FlashInfer for production NVIDIA. The backend must match the KV page size (File 03 §13).

### 5.5 `--cuda-graph-max-bs`

The max decode batch size captured as a CUDA graph (File 09 §16). Higher → more graph coverage (fewer eager fallbacks) but more graph memory (competing with KV). Set to cover the expected decode batch range. Geometric ladder up to this max (File 09 §16).

### 5.6 `--enable-torch-compile` and `--enable-dp-attention`

`--enable-torch-compile` (File 09 §24): extra fusion, mainly helps small models (<7B). `--enable-dp-attention` (File 09 §11): data-parallel attention for MoE serving (cheap attention, expert-parallel FFN) or long context. Enable for DeepSeek-style MoE.

### 5.7 The shared tuning principle

For both engines: **size the KV pool to hold the hot-prefix working set plus active KV**, **cap concurrency below the memory limit** (to avoid thrash), **enable chunked prefill** for mixed-traffic latency, **ensure FlashInfer + CUDA graphs are active**, and **configure parallelism** per the topology rules (File 05). Monitor the prefix-cache hit rate, KV utilization, and queue depth. The parameters differ in name but the underlying levers (memory split, concurrency, chunk size, parallelism, precision) are the same because the underlying mechanisms (paged/radix KV, continuous batching, chunked prefill, TP, quantization) are the same.

---

## 6. Benchmarking Methodology

Measuring serving performance correctly is harder than it looks; naive benchmarks mislead.

### 6.1 Realistic traffic patterns

The request distribution profoundly affects results:
- **ShareGPT:** real chat traffic — mean input ~200 tokens, mean output ~200 tokens, but a *heavy tail* (some very long). The standard realistic chat benchmark.
- **Alpaca:** shorter (input ~50, output ~100) — instruction-following.
- **Code generation:** longer inputs (context + code), medium outputs.
- **Document summarization:** very long inputs (8K+), short outputs — prefill-heavy.
- **Reasoning (o1/R1-style):** short inputs, very long outputs (1K–10K) — decode-heavy (File 15 §long CoT).

Benchmark on a distribution matching your *actual* workload — results from ShareGPT don't transfer to a summarization workload (opposite prefill/decode balance). The engines provide benchmark suites: vLLM's `benchmarks/benchmark_serving.py` and `benchmark_throughput.py`; SGLang's `python -m sglang.bench_serving`. Both can replay ShareGPT and synthetic distributions.

### 6.2 Throughput vs latency benchmarks

- **Throughput benchmark:** send all requests at once (or as fast as possible), measure total tokens/sec. Measures peak capacity but ignores latency. Use for capacity comparison, not SLO validation.
- **Latency-under-load benchmark:** send requests at a controlled arrival rate (Poisson), measure the latency percentiles. This is the SLO-relevant benchmark (§7).

### 6.3 Cold vs warm cache

For prefix-sharing workloads (File 08), the prefix-cache hit rate depends on warmup:
- **Cold benchmark** (empty cache): *underestimates* production throughput (no cache hits yet).
- **Warm benchmark** (pre-populated cache): *overestimates* (all hits).
- **Realistic:** replay actual traffic so the cache warms naturally, reaching the steady-state hit rate. Reporting throughput without specifying cache state is meaningless for prefix-heavy workloads (File 03 §31, File 08 §23). Always state the cache condition.

### 6.4 Common benchmarking mistakes

- Reporting throughput without latency (or vice versa) — they trade off; report the latency-throughput *curve* (§7).
- Means instead of percentiles (§1.5) — hides the tail.
- Unrealistic distributions (fixed-length prompts) — real traffic has variance that exercises scheduling.
- Cold cache for prefix-heavy workloads — underestimates.
- Too-short runs — don't reach steady state (queues, cache).
- Ignoring warmup — the first requests pay kernel-compilation/autotuning cost (File 10 §23).

A rigorous benchmark: realistic distribution, controlled arrival rate, warmed cache (or replayed traffic), steady-state measurement, latency percentiles reported across a load sweep, on the target hardware and parallelism config.

---

## 7. Latency Under Load

The most informative benchmark is the **latency-throughput curve**: sweep the arrival rate λ from low to saturation, measuring the latency percentiles at each.

### 7.1 The curve

At low λ, latency is flat and low (no queuing — requests served immediately). As λ rises toward `λ_max` (the capacity, File 04 §18.2), latency stays roughly flat (the engine absorbs load by batching) until near saturation, where it rises sharply (the **capacity cliff**) — queues grow, KV pressure causes preemption thrash, and latency diverges. The curve is roughly flat-then-cliff, not gradual.

### 7.2 Reading goodput off the curve

**Goodput** (§2) is the maximum λ at which the latency percentiles still meet the SLOs. On the curve, draw horizontal lines at the SLO thresholds (P95 TTFT < 2 s, P99 TPOT < 100 ms); the goodput is the λ where the curve crosses the first (binding) SLO line. This is the capacity-planning number: how many requests/sec one replica can serve within SLO, which (with the workload's λ) determines how many replicas you need (§10). A configuration with a higher goodput (the curve crosses the SLO at higher λ) is better, even if its peak throughput is lower.

### 7.3 What shifts the curve

- **More batch capacity** (higher `max_num_seqs`, more KV via quantization) → higher `λ_max`, but possibly higher TPOT (the throughput-latency trade) — net effect on goodput depends on which SLO binds.
- **Chunked prefill** → lower P99 TTFT/TPOT (no prefill stalls, File 04 §17) → the curve meets the latency SLO at higher λ → higher goodput.
- **Prefix caching** (warm) → cheaper requests → higher `λ_max` and lower latency → higher goodput for sharing workloads.
- **Speculative decoding** → more tokens/step → higher decode throughput → higher goodput when decode-bound.
- **More GPUs (TP/DP)** → higher `λ_max` (capacity) and/or lower latency (TP) → higher goodput.

The tuning goal is to shift the curve so it meets the SLOs at the highest possible λ — maximizing goodput. The latency-throughput curve is the single most useful artifact for capacity planning and tuning validation: it shows the goodput, the cliff, and the effect of each change.

---

## 8. Comparing vLLM and SGLang

General observations from published benchmarks (both improve rapidly — always benchmark your workload, File 08 §29, §36):

- **SGLang typically leads 10–30% on prefix-sharing workloads** (multi-turn chat, few-shot, RAG, agents) — RadixAttention's token-granular tree reuse (File 08) captures more sharing than vLLM's block-hash APC.
- **vLLM is competitive or ahead on workloads without prefix sharing** (independent unique prompts) — RadixAttention's tree overhead is pure cost when there's no reuse (File 08 §19).
- **SGLang often has lower TTFT variability** (aggressive chunked prefill, the overlap scheduler).
- **vLLM has broader model support and the larger ecosystem** (File 20) — more models, more quantization methods, more community.
- **Structured output:** SGLang's XGrammar (File 08 §11) is excellent; vLLM adopted XGrammar as a backend, so both are strong.
- **MoE/long-context:** SGLang's DP attention and aggressive feature adoption (File 09 §11) suit DeepSeek-style MoE; both support it.

The decisive factor is the **workload's prefix-sharing fraction** (File 08 §36): high sharing → SGLang's edge; independent requests → roughly even, vLLM's ecosystem an advantage. Both have continuous batching, chunked prefill, prefix caching, CUDA graphs, FlashInfer, and the converged architecture (File 09 §22), so they're close on fundamentals. The honest guidance: benchmark both on your actual traffic with realistic cache warming (§6.3); the prefix-sharing fraction will usually decide.

### 8.1 Why "it depends" is the right answer

Benchmark numbers comparing engines are workload-, version-, and config-dependent, and both engines improve monthly. A benchmark showing engine A 20% faster can reverse next release or on a different workload. So rather than memorizing a winner, understand the *mechanisms* (RadixAttention vs APC, the converged architecture) and *measure on your workload*. This is why this database emphasizes mechanisms over benchmark numbers — the numbers churn, the mechanisms (and which workload each suits) endure (File 08 §36, File 17).

---

## 9. Multi-GPU Scaling Efficiency

When scaling across GPUs (File 05), measure the **scaling efficiency** to detect communication bottlenecks.

### 9.1 Measuring it

Benchmark throughput at TP=1, 2, 4, 8 and compute `efficiency = (throughput_TP=N / N) / throughput_TP=1`. Ideal is 1.0 (linear); reality is sublinear due to AllReduce overhead (File 05 §2.6).

- **NVLink (NVSwitch node):** ~90%+ efficiency at TP=8 — the small decode AllReduces stay cheap (File 05 §14).
- **PCIe (no NVLink):** ~60–70% at TP=8 — the AllReduce latency dominates (File 05 §14's worked numbers).

### 9.2 Diagnosing poor scaling

If efficiency is below ~80% at TP=8 with NVLink expected:
- **Check NVLink is detected/used:** `nvidia-smi topo -m` should show `NV#` links (not `PHB`/`SYS`); NCCL should use NVLink (`NCCL_DEBUG=INFO`).
- **Check custom all-reduce is on** (File 05 §3.4) — `--disable-custom-all-reduce` off.
- **Check NCCL isn't falling back** to PCIe/sockets (a common silent degradation, File 05 §27).
- **Consider lower TP × higher DP** (File 05 §16): if a model fits on 4 GPUs, TP=4×DP=2 often beats TP=8 for throughput (less AllReduce per replica).

### 9.3 The TP vs DP decision, validated by measurement

The latency-headroom-vs-throughput trade (File 05 §16) is settled empirically: benchmark TP=8 vs TP=4×DP=2 on your workload. If TP=4 meets the TPOT SLO, TP=4×DP=2 usually delivers more goodput (it meets latency with margin and scales throughput linearly). If TP=4 violates the SLO, you need TP=8 for latency. The measurement — does the lower TP meet the latency SLO? — decides. This is the multi-GPU instance of the general principle: use the minimum model parallelism that meets latency, scale throughput with DP (File 05 §28).

---

## 10. Production Deployment Patterns

Translating tuning into a deployed service (operations in File 19):

### 10.1 High availability

Run ≥2 replicas per model behind a load balancer (nginx/HAProxy/Envoy or the vLLM router). Health checks on `/health` (File 07 §6); the LB routes around unhealthy replicas. Graceful shutdown drains in-flight requests before termination (File 19). Rolling updates: update one replica while others serve (zero downtime). DP replicas provide both throughput and fault tolerance (File 05 §22).

### 10.2 Autoscaling

Kubernetes HPA on GPU utilization, or KEDA on queue depth (`num_requests_waiting` via Prometheus). **Critical caveat:** cold start is slow (model load + graph capture + warmup = minutes for large models, File 07 §19). So autoscaling must be *proactive* — scale on leading indicators (rising queue depth before saturation), pre-warm replicas, or maintain headroom — not reactive (a replica added at saturation isn't ready for minutes, by which time the spike may have caused SLO violations). This multi-minute cold start fundamentally shapes autoscaling policy (File 07 §19, File 19).

### 10.3 Capacity planning

From the goodput (§7.2) and the expected peak λ: `replicas = ceil(peak_λ / goodput_per_replica × safety_factor)`. The safety factor (e.g. 1.3–1.5) accounts for traffic burstiness and the cold-start lag (you need headroom because you can't scale instantly). Monitor `gpu_cache_usage_perc` (File 03 §22) — sustained >90% with preemptions means under-provisioned (scale up). This is the capacity arithmetic: measure goodput per replica, divide peak demand by it, add headroom for bursts and cold-start lag.

### 10.4 Memory and OOM safety

Monitor KV cache utilization; alert at >90% to trigger scaling before OOM (which is catastrophic — kills all in-flight requests, File 03 §17.1). Leave a `--gpu-memory-utilization` margin (0.9, not 0.97) for unexpected long sequences (§4.3). Set `--max-model-len` to bound per-request KV. The goal: never OOM in production; always have headroom and scale before the cliff.

---

## 11. Profiling and Debugging

When performance is below expectation, a systematic diagnostic workflow (kernel profiling in File 10 §12):

### 11.1 The top-down workflow

1. **Check the metrics** (File 07 §6): is the problem latency (which percentile? TTFT or TPOT?) or throughput/capacity (growing queue)?
2. **Map to the layer** (File 09 §44): is the GPU idling between steps (architecture — scheduling overhead, check with nsys, File 10 §12.1)? Is a kernel slow (kernel layer — check with ncu's roofline, File 10 §12.2)? Is work being recomputed (algorithm — low prefix-cache hit rate)?
3. **Apply the matching fix and re-measure.**

### 11.2 Specific diagnostics

- **High P99 TPOT, fine P50:** prefill stalls (File 04 §32) — enable/tune chunked prefill.
- **Throughput collapse past a load threshold:** capacity cliff (File 04 §18) — preemption thrash; reduce `max_num_seqs` or add capacity. Check `gpu_cache_usage_perc` (pinned ~100% + preemptions = over-admission).
- **Low decode throughput, GPU "busy":** memory-bound decode (File 02 §FAQ) — `nvidia-smi` GPU-util shows kernels running but tensor cores idle; raise batch or reduce bytes (quantization).
- **Low decode throughput, GPU idling between steps:** scheduler/launch overhead — ensure V1/overlap scheduler and CUDA graphs are active (File 04 §35, File 10 §15.2).
- **Poor multi-GPU scaling:** §9.2 (NVLink, NCCL, custom all-reduce).
- **Low prefix-cache hit rate on a sharing workload:** KV pool too small (§5.1) or prefixes scattered across replicas (prefix-aware routing, File 08 §35).
- **Slow memory leak:** aborted requests not freeing KV (File 04 §28) — check disconnect handling.

### 11.3 Tools

- **Engine metrics** (`/metrics`, Prometheus + Grafana): the first place to look — queue depth, KV utilization, latency histograms, preemptions, hit rate.
- **nsys:** system timeline — GPU idle gaps, communication, CPU bottlenecks (File 10 §12.1).
- **ncu:** kernel roofline — which kernel is slow and why (File 10 §12.2).
- **`nvidia-smi -l 1` / `nvitop`:** live GPU memory/util — spot leaks (stable after warmup) and utilization.
- **Engine debug flags:** vLLM `VLLM_LOG_LEVEL=DEBUG`, SGLang `--log-level debug` — scheduler decisions, per-request stats, prefix-cache hits.
- **`torch.cuda.memory_summary()`:** detailed memory breakdown for OOM debugging.

### 11.4 Deadlock and thrash

A scheduler "deadlock" (latency spikes + GPU util dropping) usually means all KV is held by running requests that can't progress — over-admission (File 04 §32). Resolution: reduce `max_num_seqs`, add GPUs, or use priority scheduling with preemption. The watermark (File 03 §30) prevents true allocation deadlock, but over-admission causes thrash that looks like a stall. The signature is `gpu_cache_usage_perc` near 100% with high preemption rate — the cure is admission control (cap concurrency below the memory limit, File 04 §27).

---

## 12. A Tuning Playbook

A step-by-step procedure to tune a deployment for a workload and SLO, synthesizing the file:

1. **Characterize the workload:** measure the prompt/output length distributions, the arrival rate (peak λ), the prefix-sharing fraction, and the SLOs (P95 TTFT, P99 TPOT). This determines everything downstream.
2. **Choose the model precision:** by the binding constraint — memory/fit → W4A16 (AWQ/GPTQ) or FP8 weights; prefill compute on H100 → FP8 W8A8; long-context KV → add FP8 KV (File 06 §12).
3. **Choose the parallelism:** minimum TP to fit + meet latency, then DP for throughput/HA (File 05 §16, §9). Validate scaling efficiency (§9.1).
4. **Compute the memory-bound concurrency** from the context distribution and KV-per-token (File 03 §17), and set `--max-num-seqs` below it with margin (File 04 §27).
5. **Set `--gpu-memory-utilization`** to the safe maximum (profile; 0.9 default, margin for long sequences).
6. **Enable chunked prefill** (and tune the chunk via `max_num_batched_tokens`) so the per-step time meets the TPOT SLO (File 04 §17.3).
7. **Ensure the fast path:** FlashInfer backend, CUDA graphs on, prefix caching on (for sharing workloads), V1/overlap scheduler.
8. **Benchmark the latency-throughput curve** (§7) with realistic traffic and warmed cache; read off the goodput (the λ meeting the SLOs).
9. **Iterate:** if goodput is below target, diagnose (§11) — is it latency (which term? File 04 §39) or capacity? — and apply the matching lever (chunked prefill for stalls, more batch/KV for capacity, speculative decoding for decode-bound, more GPUs for both).
10. **Size the fleet:** `replicas = peak_λ / goodput_per_replica × safety_factor` (§10.3), with proactive autoscaling for the cold-start lag.
11. **Set up monitoring:** alert on `gpu_cache_usage_perc`, queue depth, latency percentiles, preemptions — to catch the capacity cliff before it causes SLO violations.

This playbook turns the mechanisms of Files 03–10 into a repeatable tuning process: characterize → choose precision/parallelism → size memory/concurrency → enable fast-path features → benchmark goodput → iterate → size the fleet → monitor. Each step rests on a mechanism covered earlier, and the whole is driven by measurement (the latency-throughput curve and the metrics) rather than guesswork.

---

## 13. Worked Tuning Example 1: High-QPS Chat

**Workload:** chatbot, mean prompt 300 tokens (with a shared 200-token system prompt), mean output 250 tokens, peak 50 req/s, SLO P95 TTFT < 1 s, P99 TPOT < 50 ms. Model: LLaMA-3 70B. Hardware: 8×H100.

**Tuning:**
1. **Precision:** FP8 weights (70 GB → fits with room for KV on fewer GPUs; ~1.8× BF16 throughput, File 06 §12).
2. **Parallelism:** FP8 70B is 70 GB; at TP=2 that's 35 GB/GPU, leaving ~37 GB/GPU for KV. Benchmark TP=2 vs TP=4: if TP=2 meets P99 TPOT < 50 ms, use TP=2 × DP=4 (8 GPUs, 4 replicas) for max throughput/HA; if not, TP=4 × DP=2. Suppose TP=4 is needed for the TPOT SLO → TP=4 × DP=2.
3. **Concurrency:** at ~550-token average context, KV/token (FP8 KV, GQA-8, TP=4) ≈ 40 KB; with ~40 GB KV/GPU → ~1M tokens → ~1800 sequences' worth, but cap `--max-num-seqs` ~256 (above which scheduling/batch overhead grows) — concurrency-bound, fine.
4. **Prefix caching ON:** the 200-token system prompt is shared by all → cached once. With prefix-aware routing (File 08 §35) directing to the same replica, hit rate is high → prefill drops from 300 to ~100 tokens/request (File 03 §31).
5. **Chunked prefill ON:** prompts are short (300 tokens), so prefill stalls are minor, but enable it for the occasional long prompt; chunk ~512.
6. **Fast path:** FlashInfer, CUDA graphs, V1/overlap scheduler.

**Result:** benchmark the latency-throughput curve; suppose goodput is ~15 req/s/replica within SLO → 2 replicas serve 30 req/s, short of 50. Add replicas: `ceil(50 / 15 × 1.3) = 5` replicas → TP=4 × DP=5 (20 GPUs). Or, if TP=2 had met the SLO, more replicas per GPU-count would be cheaper. The chat workload benefits most from **prefix caching** (shared system prompt) and **moderate TP** (latency) — the levers that matter here.

---

## 14. Worked Tuning Example 2: Document Summarization

**Workload:** summarize long documents — mean prompt 8,000 tokens, mean output 200 tokens, peak 10 req/s, SLO P95 TTFT < 5 s (users tolerate longer for long docs), P99 TPOT < 100 ms. Model: LLaMA-3 70B. Hardware: H100s.

**Tuning:**
1. **This is prefill-heavy** (8000-token prompts, short outputs) — the opposite balance from chat. Prefill dominates compute (File 01 §3).
2. **Precision:** FP8 W8A8 — benefits *prefill* compute (2× tensor-core throughput, File 06 §12), which is the bottleneck here.
3. **Parallelism:** prefill is compute-bound, so TP helps (parallel prefill compute); the long prompts also need KV memory. TP=4 or 8. Disaggregation could help (prefill-heavy → more prefill capacity, File 05 §30) but adds complexity.
4. **Chunked prefill ESSENTIAL:** the 8,000-token prefills would each monopolize a step and spike TPOT for any concurrent decode (File 04 §7). Chunk to ~2048 (larger chunks OK here since prefill throughput matters and the TTFT SLO is loose). This keeps decode interleaving while processing the long prompts.
5. **`--max-num-batched-tokens`:** larger (e.g. 8192) since prefill throughput matters and the TTFT SLO is loose — bigger chunks process the long prompts faster.
6. **Concurrency:** at 8,000-token context, KV/token (FP8 KV) ≈ 40 KB → 8000 × 40 KB = 320 MB/request; KV-bound concurrency is low (~125 requests at 40 GB) — set `--max-num-seqs` accordingly (memory-bound, File 04 §27), lower than chat.
7. **Prefix caching:** documents are usually unique (low sharing) unless the same document is summarized repeatedly — modest benefit; the system prompt (instructions) is shared.

**Result:** the summarization workload is **prefill-bound and KV-memory-bound** (long contexts) — the levers are FP8 W8A8 (prefill compute), chunked prefill (latency), larger chunks (prefill throughput), and lower concurrency (long-context memory). Goodput is limited by prefill compute and KV memory, not decode — a completely different tuning profile from chat, illustrating why the workload characterization (§12 step 1) drives everything.

---

## 15. Worked Tuning Example 3: Reasoning Model (Long CoT)

**Workload:** a reasoning model (o1/R1-style, File 15 §long CoT) — short prompts (~100 tokens), very long outputs (mean 4,000 tokens, some 10,000+), peak 20 req/s, SLO P99 TPOT < 80 ms (the long outputs make TPOT critical — at 4000 tokens, 80 ms/token = 320 s E2E, so TPOT directly sets the user wait). Model: 70B-class. Hardware: H100s.

**Tuning:**
1. **This is decode-heavy** (long outputs) — TPOT and decode throughput dominate (File 15 §long CoT).
2. **Speculative decoding HIGH-VALUE:** decode is the bottleneck, and speculative decoding (File 12) amortizes weight reads across multiple tokens — directly improving TPOT and decode throughput. Use a 7B draft (or EAGLE/MTP heads) → ~2× decode speedup if acceptance is good. This is the single most impactful lever for reasoning workloads.
3. **Precision:** FP8 weights (faster decode — less weight to read, File 06 §12). FP8 KV too — the long outputs grow KV (4000 tokens × KV/token), so FP8 KV doubles concurrency (File 13).
4. **KV memory grows during the request:** a 10,000-token reasoning chain is ~400 MB KV (FP8). Concurrency is KV-bound and *grows* per request as it generates — set `--max-num-seqs` for the *peak* per-request KV (the long tail), or risk OOM/preemption mid-generation. This is the reasoning-specific challenge: KV grows large within a single request (File 15 §long CoT).
5. **Parallelism:** TP for the model + decode latency; DP for throughput. The long outputs mean each request occupies a decode slot for a long time, so concurrency turnover is slow — provision for the sustained concurrent count.
6. **Chunked prefill:** prompts are short, so minor; but the long *decodes* are the load.

**Result:** the reasoning workload is **decode-bound with growing KV** — the levers are speculative decoding (the big win for decode), FP8 (faster decode, more KV), and KV-aware concurrency sizing (long outputs grow KV, risking OOM). Goodput is set by decode throughput (helped most by speculation) and KV capacity (helped by FP8 KV). Again, a distinct profile — reasoning, summarization, and chat each tune differently because their prefill/decode/KV balances differ, which is the central lesson: **there is no universal configuration; tune to the workload's measured profile.**

---

## 16. Cost Optimization and $/Token

The ultimate metric is **$/token** (or $/1M tokens) — the cost-relevant figure for any production deployment (File 20 §economics).

### 16.1 Computing it

```
$/1M tokens = (GPU_hourly_cost × num_GPUs) / (output_tokens_per_hour) × 1e6
            = (GPU_hourly_cost × num_GPUs) / (throughput_tokens_per_sec × 3600) × 1e6
```

Worked: 8×H100 at ~$2/GPU-hr (cloud), serving 8,000 output tokens/sec → `(2 × 8) / (8000 × 3600) × 1e6 = $16 / 28.8M × 1e6 ≈ $0.56/1M tokens` of compute cost. Commercial APIs mark this up 5–15× (File 20 §economics). The goal of tuning is to *minimize* $/token while meeting SLOs — i.e. maximize SLO-meeting throughput per GPU-dollar (goodput per GPU, §2).

### 16.2 The cost levers

- **Quantization (W4A16/FP8):** 1.5–2× throughput → ~30–50% cost reduction (File 06 §12). The biggest single lever for memory-bound decode.
- **Speculative decoding:** ~2× decode throughput with no quality loss (lossless, File 12) → ~50% cost reduction for decode-bound workloads — "free" speedup.
- **Batching (continuous batching, larger batches):** maximizes GPU utilization → more tokens/GPU → lower $/token (the foundational lever, File 04 §3).
- **Prefix caching:** for sharing workloads, slashes redundant prefill → more effective capacity per GPU → lower cost (File 08 §23).
- **Right-sizing parallelism:** minimum TP (less AllReduce overhead, higher efficiency) × DP → more goodput per GPU (File 05 §16).
- **Disaggregation + heterogeneous hardware:** prefill on expensive compute GPUs, decode on cheaper high-bandwidth GPUs → 2–3× cost reduction at scale (File 05 §36, File 15).
- **Higher GPU utilization target:** 70–85% (higher → SLO violations; lower → wasted cost, File 19). The sweet spot maximizes goodput per GPU without breaching latency.

Each lever's value depends on the workload (speculation helps decode-bound; prefix caching helps sharing; quantization helps memory-bound) — the worked examples (§§13–15) show which dominate for each. Stacking the applicable levers (e.g. FP8 + speculation + prefix caching + right-sized parallelism for a reasoning chat workload) compounds the cost reduction. $/token is the scoreboard; the levers are how you move it.

---

## 17. Speculative Decoding Tuning

Speculative decoding (File 12) is a high-value but nuanced lever, worth detailed tuning treatment.

### 17.1 The speedup model (recap)

`speedup ≈ E[accepted tokens + 1] / (1 + K·draft_cost_ratio)` (File 02 §9.2, File 12). Higher acceptance rate `β` and more speculative tokens `K` raise the numerator; a cheaper draft (lower cost ratio) keeps the denominator low.

### 17.2 The knobs

- **Draft model choice:** must share the target's vocabulary; ~10× smaller (7B draft for 70B target). Same-family drafts (LLaMA draft for LLaMA target) have higher acceptance. EAGLE/MTP (feature/built-in drafts, File 12) often beat separate drafts on acceptance.
- **`K` (num_speculative_tokens):** more proposed tokens = more potential speedup, but acceptance decays with depth (later tokens less likely to match), and verification cost grows. Typical sweet spot K=3–5. Tune by measuring acceptance at each K.
- **Draft temperature:** draft greedy (T=0) maximizes acceptance (the draft commits to its best guess, most likely to match the target); the target uses the user's temperature for the final accepted-token distribution.

### 17.3 When it helps and when it doesn't

Speculative decoding helps when **decode is the bottleneck and the target has spare compute to verify** — i.e. low-to-moderate batch (the target's verification of K+1 tokens is cheap relative to its idle compute) and long outputs (decode-bound, §15). It helps *less* at high batch (the target is already compute-saturated by the batch, no spare for verification — File 12) and not at all for prefill-bound workloads (§14). So measure: at your operating batch size, does the target have spare compute? If yes (decode-bound, moderate batch), speculation is a big win; if no (compute-saturated high batch), it may not help. The acceptance rate (measured per workload — ~60–70% chat, ~80% code) and the batch regime determine the benefit. This is why speculation is the headline lever for reasoning/chat (decode-bound, §15, §13) and irrelevant for summarization (prefill-bound, §14).

---

## 18. KV Cache Offloading and Long-Context Tuning

For long-context workloads (File 13), KV memory is the binding constraint, with specific tuning levers:

- **FP8 KV cache** (`--kv-cache-dtype fp8_e5m2`): halves KV memory → doubles KV-bound concurrency, <0.5% quality impact (File 02 §7, File 13). The first lever for long context.
- **GQA/MLA models:** choosing a model with GQA (8× KV reduction) or MLA (DeepSeek, ~64×) over MHA is a model-selection lever that hugely affects long-context serving (File 02 §26).
- **KV offloading to CPU** (vLLM swap, File 03 §8): a relief valve for transient pressure, but PCIe-bandwidth-limited — fine occasionally, catastrophic if frequent (thrash). Tune `--swap-space`; prefer recompute preemption for short sequences (File 03 §9).
- **Sliding window / attention sinks** (File 02 §2.5, §29; File 13): for streaming, bounds KV to a window + sinks — enables unbounded streaming at fixed memory.
- **`--max-model-len`:** bound the context to what you'll actually serve — over-setting reserves block-table capacity and risks OOM on a max-length request (File 03 §22).

Long-context tuning is fundamentally about fitting the KV (reduce per-token bytes via FP8/GQA/MLA, bound the length, offload transiently) — the memory-bound regime where the levers all target KV size (File 13).

---

## 19. Regression Detection and Safe Rollout

Performance tuning is ongoing — models, engine versions, and traffic change — so detecting regressions and rolling out safely matters (File 19 §versioning):

- **Continuous benchmarking:** run the latency-throughput benchmark (§7) on each engine-version upgrade or config change, comparing goodput against the baseline. An engine upgrade can regress (or improve) performance; measure, don't assume.
- **Canary deployment:** route a small fraction (5%) of traffic to the new version/config, monitor latency and error rates, and roll forward only if metrics hold (File 19). Catches regressions on real traffic before full rollout.
- **Shadow mode:** send the same requests to old and new versions, compare outputs (for correctness, e.g. after a quantization change) and latency, without affecting users (File 19).
- **A/B testing:** for tuning changes (e.g. chunked-prefill chunk size), split traffic and compare the latency percentiles and goodput statistically.
- **Regression alerts:** track the latency percentiles and throughput over time; alert on degradation (a sudden P99 TPOT rise after a deploy signals a regression).

The discipline: every change (engine version, config, model, quantization) is validated by measurement against a baseline, rolled out via canary/shadow, and monitored for regression. This prevents the common failure of a "performance improvement" config change silently regressing the tail latency or a quantization change degrading quality (File 06 §12's "always validate"). Tuning isn't a one-time activity but a continuous measure-validate-rollout loop.

---

## 20. A Worked Latency-Throughput Curve

Make §7 concrete with numbers for a hypothetical LLaMA-3 70B (TP=4) chat deployment, SLOs P95 TTFT < 1 s, P99 TPOT < 50 ms. Sweep the Poisson arrival rate λ:

| λ (req/s) | running batch | P50 TPOT | P99 TPOT | P95 TTFT | KV util | preemptions |
|---|---|---|---|---|---|---|
| 2 | ~6 | 18 ms | 22 ms | 0.2 s | 15% | 0 |
| 5 | ~15 | 22 ms | 28 ms | 0.3 s | 35% | 0 |
| 8 | ~28 | 30 ms | 42 ms | 0.5 s | 60% | 0 |
| 10 | ~40 | 38 ms | 52 ms | 0.8 s | 80% | rare |
| 11 | ~50 | 45 ms | 70 ms | 1.4 s | 92% | frequent |
| 12 | thrash | 60 ms+ | 150 ms+ | 4 s+ | ~100% | heavy |

**Reading it:** latency is flat-and-low up to λ≈8, rises through λ≈10, and falls off the cliff at λ≈11–12 (KV saturates ~90%+, preemption thrash begins, latency diverges). The **goodput** is where the binding SLO is crossed: P99 TPOT < 50 ms is crossed between λ=10 (52 ms — just over) and λ=8 (42 ms — under), and P95 TTFT < 1 s is crossed near λ=10–11. So goodput ≈ **8–9 req/s** (the highest λ where *both* SLOs hold). Note that raw throughput keeps rising past goodput (more tokens/sec at λ=11 than λ=8) but those requests violate the SLO — goodput, not peak throughput, is the usable capacity. This single table is the deliverable of a latency-under-load benchmark: it gives the goodput (for fleet sizing, §10.3), shows the cliff (where to alert, §11.4), and would show the effect of any tuning change (re-run with the change, compare the goodput).

### 20.1 The effect of a tuning change on the curve

Suppose you enable speculative decoding (decode-bound chat, §17.3). The decode step effectively produces ~1.8× tokens, so at each λ the running batch clears faster → lower TPOT at a given λ → the P99 TPOT < 50 ms line is crossed at higher λ → goodput rises (say to ~12 req/s). Re-running the curve quantifies this: the table shifts right (each λ now has lower latency), and the goodput (the SLO crossing) moves to higher λ. This is how every tuning change is validated — re-measure the curve, compare the goodput. A change that raises goodput is good; one that raises peak throughput but not goodput (or worsens the tail) is not.

---

## 21. How Batch Size Shapes the Curve

The batch size (set by `max_num_seqs` and the natural load) is the master throughput-latency dial (File 01 §11), and understanding its effect explains the curve's shape.

At a given load, the running batch size emerges from the arrival rate and the service rate. As batch grows (higher load or higher `max_num_seqs`):
- **Decode throughput rises** nearly linearly (the weight read is amortized across more sequences, File 02 §15.2) — until the ridge point (compute-bound) or KV limit.
- **TPOT rises** slowly at first (adding sequences is "free" in the memory-bound regime, File 01 §2.2) then faster as the batch competes for bandwidth/compute.
- **TTFT rises** as new requests queue behind the larger running batch.

So the curve's flat region (low λ) is where the batch is small and adding load is nearly free; the rise (moderate λ) is where the batch grows and latency climbs; the cliff (high λ) is where KV saturates and preemption thrash sets in. Setting `max_num_seqs` *caps* the batch: too low caps throughput (under-batching, leaving goodput on the table); too high admits batches that violate TPOT or thrash KV (File 04 §27). The optimal `max_num_seqs` is the batch size where the marginal sequence still meets the TPOT SLO and KV has margin — found by the sweep (§20). This is why `max_num_seqs` and the batch it permits are the central tuning dial: they directly position you on the throughput-latency curve.

---

## 22. Workload Characterization Methodology

Everything starts with characterizing the workload (§12 step 1), which deserves a methodology:

1. **Length distributions:** collect (or estimate) the prompt-length and output-length distributions — not just means but the *tail* (P95, P99), since the tail drives KV peaks and scheduling stress. Histograms from production logs or representative traces.
2. **Arrival pattern:** the request rate over time — mean, peak, burstiness. Peak λ drives capacity; burstiness drives the safety margin (§10.3).
3. **Prefix-sharing fraction:** how much of the prompts is shared (system prompts, few-shot, common documents)? This determines prefix-caching value and the SGLang-vs-vLLM choice (§8). Measure by analyzing prompt overlap in traces.
4. **Prefill/decode balance:** mean output/prompt ratio — chat ~1:1, summarization prefill-heavy, reasoning decode-heavy (§§13–15). Determines which levers matter (speculation for decode-heavy, FP8 W8A8 for prefill-heavy).
5. **SLOs:** the latency targets (P95 TTFT, P99 TPOT) and any throughput/cost targets — from product requirements.
6. **Quality bar:** the acceptable quality (for quantization validation, File 06 §12) — what task accuracy must be preserved.

This characterization is the input to the tuning playbook (§12) — every downstream decision (precision, parallelism, concurrency, chunked prefill, speculation, prefix caching) follows from it. Skipping it ("just use the defaults") is why deployments are often badly mistuned: the defaults suit a generic short-context chat workload and are wrong for long-context, prefill-heavy, decode-heavy, or sharing-heavy workloads (§§13–15). The discipline: characterize first, then tune to the characterization, then validate by benchmark. Production traces (real prompt/output lengths, arrival patterns, prefixes) are the gold standard; representative public datasets (ShareGPT for chat, §6.1) are a fallback when traces aren't available.

---

## 23. The Capacity Cliff, Quantified

The capacity cliff (§7.1, File 04 §18) deserves quantification because it's the most dangerous operating region. Below `λ_max` (the sustainable rate), the system is stable — queues are bounded, latency flat. Above it, the waiting queue grows without bound (arrivals exceed service), and three things compound:

1. **Queue wait grows** — new requests wait longer (TTFT rises).
2. **KV saturates** — the running batch fills the pool; new admissions trigger preemption.
3. **Preemption thrash** — preempted requests swap/recompute (wasted work, File 04 §9.3), reducing effective service rate, which makes the queue grow faster — a vicious cycle.

The cliff is *sharp* because the throughput is roughly flat up to the limit (the memory-bound batch is efficient) and then collapses (thrash overhead). This is why the curve (§20) shows flat-then-cliff, not gradual degradation. Operationally: you must keep `λ < λ_max` with margin (the goodput is below `λ_max`, since the SLO is crossed before the cliff). The defenses (File 04 §18.3): **admission control / load shedding** (return 503 when at capacity, rather than queuing into the cliff), and **horizontal scaling** (add replicas to raise aggregate `λ_max`). Crucially, *no scheduling policy saves you past `λ_max`* — the work exceeds capacity, and the only fixes are shed load or add capacity (File 04 §18.3). Monitoring must alert *before* the cliff — `gpu_cache_usage_perc` > 90% and rising preemptions are the leading indicators (§11.2) — so you scale or shed before latency diverges. The cliff is why capacity planning (§10.3) sizes for peak with margin: operating near the cliff is operating one traffic spike away from a latency catastrophe.

---

## 24. A Profiling Worked Example

Walk through diagnosing a slow decode with the tools (File 10 §12). Symptom: decode throughput is ~60% of the roofline estimate (§3.2 says min step time 140 ms, but measured is ~230 ms).

1. **nsys timeline:** capture a few steps. Observe the GPU timeline — is it busy the whole step, or are there gaps? Suppose you see ~40 ms of gap between steps (GPU idle) plus the kernels taking ~190 ms. The 40 ms gap points to *architecture* overhead (scheduling not overlapped — File 09 §10) or launch overhead.
2. **Check CUDA graphs:** is the decode using a captured graph, or eager? If eager (e.g. batch exceeded `cuda-graph-max-bs`, File 09 §16), the ~190 ms includes launch overhead for ~900 kernels (~4.5 ms, File 10 §26) plus the gap. Enable/extend graph capture → the gap and launch overhead shrink.
3. **ncu on the attention kernel:** if the kernels themselves are slow (not just overhead), profile the attention kernel. Check its roofline position: is it near the bandwidth roofline (memory-bound, optimal) or below (inefficient)? Check global load efficiency (coalescing, File 10 §30) — if low, the paged access is scattered (File 03 §16.5); consider FlashInfer (better coalescing) if not already used.
4. **Check the backend:** is FlashInfer active, or the slower native paged kernel (File 03 §16.4)? Switching to FlashInfer can be the 1.5–2× difference.
5. **Re-measure:** after enabling CUDA graphs (gap → ~0) and FlashInfer (kernels faster), the step drops toward the 140 ms roofline. If it's now ~150 ms → 93% of roofline (§3.2), the decode is optimized.

This top-down workflow (nsys for gaps/overhead → check graphs/backend → ncu for kernel efficiency → re-measure) is how you close the gap to the roofline. The key is the roofline gives a *target* (140 ms), and the tools localize *where* the gap is (overhead vs kernel), directing the fix (graphs/backend vs kernel tuning). Without the roofline target, you wouldn't know 230 ms is slow; without the tools, you wouldn't know why.

---

## 25. Benchmark Tools in Practice

The engines provide benchmark suites for the methodology (§6):

- **vLLM:** `python benchmarks/benchmark_serving.py --model <model> --dataset-name sharegpt --request-rate <λ>` runs a latency-under-load benchmark at a controlled arrival rate, reporting TTFT/TPOT/E2E percentiles and throughput. `benchmark_throughput.py` measures peak throughput (all requests at once). `benchmark_latency.py` for single-request latency.
- **SGLang:** `python -m sglang.bench_serving --backend sglang --dataset-name sharegpt --request-rate <λ>` — the equivalent, reporting the same metrics. `--profile` enables torch.profiler during the benchmark.
- **Common options:** `--num-prompts` (how many), `--request-rate` (λ, or "inf" for peak throughput), dataset selection (ShareGPT, random, custom), input/output length controls.

The workflow: pick the dataset matching your workload (§6.1), sweep `--request-rate` from low to saturation, collect the latency percentiles at each, plot the latency-throughput curve (§20), read off the goodput. For cross-engine comparison (§8), run the *same* dataset and rates against both (with warmed cache, §6.3). For tuning validation, run before and after a config change and compare the curves/goodput. These tools, used with realistic datasets and a load sweep, produce the latency-throughput curve that drives capacity planning and tuning — they're the practical instrument behind the methodology.

---

## 26. Prefix Caching's Goodput Impact, Worked

Quantify prefix caching's (File 08, File 03 §6) effect on goodput for a sharing workload. Chat with a 2,000-token shared system prompt, 50-token user queries, 200-token outputs:

- **Without prefix caching:** each request prefills 2,050 tokens. Prefill is a large per-request cost; at the engine's prefill throughput, this limits how many requests/sec can be admitted (prefill competes with decode for the token budget, File 04 §6.3).
- **With prefix caching (warm, 95% hit):** each request prefills only ~150 tokens (50 query + 5% of the 2,000 prompt, File 03 §31) — a ~14× prefill reduction. This frees the token budget for more decode and more admissions → higher `λ_max` and lower TTFT (cache-warm requests start fast).

The goodput impact: the prefill reduction directly raises the sustainable request rate (less prefill work per request → more requests served) and lowers TTFT (cache-warm prefill is fast). On the latency-throughput curve (§20), prefix caching shifts the curve right (higher goodput) and down (lower TTFT) for sharing workloads. The magnitude depends on the sharing fraction and hit rate — for this heavily-shared workload, goodput could rise 2–3×. This is why prefix caching is the dominant lever for sharing workloads (§8, §13) and why the cache-warm benchmark (§6.3) is essential — a cold-cache benchmark would miss this entire effect, badly underestimating production goodput for sharing workloads. Conversely, for non-sharing workloads (unique prompts), prefix caching does nothing (no hits) — the workload characterization (§22) tells you whether to expect this lever to matter.

---

## 27. Common Tuning Anti-Patterns

Mistakes that recur, as a checklist of what *not* to do:

- **Using defaults blindly:** the defaults suit generic short-context chat; long-context, prefill-heavy, decode-heavy, and sharing workloads need different tuning (§§13–15, §22).
- **Maximizing throughput instead of goodput:** huge batches raise throughput but violate TPOT — those requests are failures (§2). Tune for goodput.
- **Reporting/optimizing means, not percentiles:** the tail (P99) is where SLOs are defined and where scheduling problems show (§1.5, File 04 §39.1).
- **Cold-cache benchmarks for sharing workloads:** underestimates goodput by missing prefix caching (§6.3, §26).
- **Over-setting `--gpu-memory-utilization`:** risks OOM (catastrophic) for marginal capacity gain (§4.3, File 03 §17.1).
- **Over-setting `--max-num-seqs` for long context:** admits more than KV supports → preemption thrash → throughput collapse (§21, File 04 §27).
- **TP across nodes:** the frequent AllReduces over slow inter-node links tank efficiency (File 05 §27).
- **Quantizing without quality validation:** quantization error is data-dependent; a throughput win can hide a tail-quality regression (File 06 §12).
- **Reactive autoscaling:** scaling at saturation, when cold start is minutes — too late (§10.2, File 07 §19).
- **Micro-optimizing a non-bottleneck kernel:** profile top-down first (§24, File 10 §12) — optimize the actual bottleneck, not a familiar-but-irrelevant kernel.
- **Ignoring the workload's prefill/decode balance:** applying decode levers (speculation) to a prefill-bound workload (or vice versa) wastes effort (§17.3, §§13–15).

Each anti-pattern is the inverse of a principle in this file; avoiding them is most of good tuning. The meta-anti-pattern is *tuning by folklore* (copying someone else's flags) rather than *tuning by measurement* (characterize, benchmark the curve, diagnose the bottleneck, apply the matching lever, re-measure) — the latter is the discipline this file teaches.

---

## 28. The Metrics to Monitor (Reference)

The Prometheus metrics (File 07 §6) that drive production monitoring and tuning, with what each tells you:

- **`vllm:time_to_first_token_seconds`** (histogram): TTFT percentiles — the streaming-start SLO. Rising P95 → queue wait (capacity) or prefill cost.
- **`vllm:time_per_output_token_seconds`** (histogram): TPOT percentiles — the streaming-speed SLO. Rising P99 → batch too large, prefill stalls, or long context.
- **`vllm:e2e_request_latency_seconds`** (histogram): end-to-end — the total wait.
- **`vllm:num_requests_running` / `_waiting` / `_swapped`:** the scheduler state (File 04 §23). Growing `waiting` → approaching/exceeding `λ_max` (the cliff, §23).
- **`vllm:gpu_cache_usage_perc`:** KV pool occupancy (File 03 §22) — the capacity gauge. Sustained >90% + preemptions = over-admission/under-provisioning. The single most important alert.
- **`vllm:num_preemptions`:** preemption rate (File 04 §9.3) — the thrash indicator.
- **`vllm:prefix_cache_hit_rate`** (or equivalent): prefix-cache effectiveness (File 03 §31, File 08 §31) — for sharing workloads, the key efficiency metric.
- **Throughput counters:** prompt and generation tokens/sec, requests/sec.

The alerting strategy: alert on `gpu_cache_usage_perc > 90%` (approaching the cliff) and rising preemptions (thrash) — these are *leading* indicators that let you scale or shed *before* latency diverges (§23). Alert on the latency percentiles breaching the SLOs (the symptom). Dashboard the queue depth and KV utilization for capacity visibility. SGLang exposes analogous metrics. These metrics are the production instrument — the latency-throughput curve (§20) is the benchmark-time artifact, and these metrics are its runtime counterpart, showing where on the curve you're currently operating and whether you're approaching the cliff.

---

## 29. Batch Size, Utilization, and Cost: The Core Trade-off Restated

The central tuning trade-off, restated with the cost lens (§16): GPU utilization target should be **70–85%**. Below 70%, you're paying for idle GPU (high $/token — under-batched, leaving goodput unused). Above 85%, you're operating near the cliff (§23) — one spike from SLO violations, and the marginal requests likely violate latency anyway (low *good*put despite high utilization). The 70–85% band is where goodput per GPU-dollar is maximized: the GPU is busy enough to be cost-efficient but with enough headroom to absorb bursts and meet latency. This is why "maximize GPU utilization" is wrong as a literal goal (it pushes past the cliff); the goal is the *utilization that maximizes goodput per dollar*, which is the 70–85% band, found by the latency-throughput sweep (the utilization at the goodput point). Capacity planning (§10.3) provisions replicas so that *peak* load lands in this band — not at 100% (cliff) nor at 30% (waste). The interplay of batch size (the dial, §21), utilization (the operating point), goodput (the SLO-meeting rate, §2), and $/token (the scoreboard, §16) is the heart of production tuning: dial the batch (via `max_num_seqs` and load) to the utilization that maximizes goodput per dollar, size the fleet so peak load lands there, and monitor to stay off the cliff.

---

## 30. Key Takeaways

1. **Measure the right metrics:** TTFT (streaming start), TPOT (streaming speed), E2E, throughput, and especially **goodput** (the SLO-meeting rate, §2) — at percentiles, not means (§1.5).
2. **The roofline** (§3) tells you the regime (compute/memory-bound) and the gap to optimal, directing the fix — compute the theoretical min step time and compare to measured.
3. **The latency-throughput curve** (§7, §20) is the central artifact: sweep load, read off goodput, see the cliff, validate every tuning change against it.
4. **Tune to the workload** (§22, §§13–15): chat (prefix caching, moderate TP), summarization (FP8 W8A8, chunked prefill, prefill-bound), reasoning (speculation, FP8 KV, decode-bound) — no universal config.
5. **The levers** (§4–5, §16): quantization (memory-bound), speculation (decode-bound), chunked prefill (latency/stalls), prefix caching (sharing), parallelism (fit/latency/throughput), batch size (the master dial). Apply the ones the workload's profile makes relevant.
6. **Goodput per GPU-dollar** is the objective (§16, §29) — maximize SLO-meeting throughput per GPU, operating at 70–85% utilization (off the cliff).
7. **Capacity planning** (§10.3): size replicas for peak goodput with margin for bursts and cold-start lag; autoscale proactively; alert on leading indicators (`gpu_cache_usage_perc`, preemptions) before the cliff (§23, §28).
8. **Validate every change** by re-benchmarking the curve, roll out via canary/shadow, monitor for regression (§19) — tuning is a continuous measure-validate loop, not a one-time setting.

This file is the synthesis of the database: it takes the mechanisms of Files 03–10 (paged/radix KV, scheduling, chunked prefill, distributed execution, quantization, speculation, kernels) and turns them into a measurement-driven discipline for making a real deployment hit its SLOs at minimum cost. The three layers — algorithms (what to reuse/compute), architecture (keep the GPU fed), kernels (hit the roofline) — all surface here as tuning levers, diagnosed through the metrics and the roofline, validated through the latency-throughput curve. The durable skill is not a set of magic flags but the *methodology*: characterize the workload, reason from the roofline, benchmark the curve, diagnose the bottleneck layer, apply the matching lever, re-measure. That methodology, grounded in the cost models of File 01, is what makes an inference engineer able to tune any model on any engine on any hardware — the practical payoff of the entire database.

---

## 31. Appendix: Quick-Reference Tuning Table

| Symptom | Likely cause | Lever |
|---|---|---|
| High P99 TPOT, fine P50 | prefill stalls | enable/tune chunked prefill |
| High TTFT, fine TPOT | queue wait (capacity) or prefill-starved | add capacity / allow more prefill budget |
| Throughput collapse past λ | capacity cliff (thrash) | admission control, more replicas, lower `max_num_seqs` |
| Low decode throughput, GPU "busy" | memory-bound decode | raise batch, quantize (less bytes) |
| Low decode throughput, GPU idling | scheduler/launch overhead | V1/overlap scheduler, CUDA graphs |
| Poor multi-GPU scaling | NVLink/NCCL/all-reduce | check topology, custom all-reduce, lower TP×higher DP |
| Low prefix-cache hit (sharing workload) | KV pool too small / scattered routing | raise KV share, prefix-aware routing |
| OOM in production | over-set memory util / unbounded length | lower `gpu_memory_utilization`, set `max_model_len` |
| High $/token | under-batched / wrong precision | raise utilization to 70–85%, quantize, speculate |
| Slow cold start affecting autoscaling | model load + warmup | proactive scaling, pre-warm, headroom |

This table maps symptoms (from the metrics, §28) to causes (from the mechanisms, Files 03–10) to levers (from §§4–5, §16) — the diagnostic shortcut for the common cases, backed by the deeper reasoning in the body. It's the on-call reference; the methodology (§30) is how you handle the cases not in the table.

---

## 32. Cold vs Warm Cache: Worked Impact

To make §6.3 concrete, contrast a cold-cache and warm-cache benchmark for a sharing workload (2,000-token shared system prompt, 50-token queries, 200-token outputs, §26).

- **Cold-cache run** (empty cache at start): the first request prefills the full 2,050 tokens and caches the system prompt; subsequent requests hit it. But over a short benchmark, the *measured* throughput is dragged down by the cold start (the early requests paying full prefill) — and if the benchmark is too short, it never reaches the warm steady state. A cold-cache benchmark might report, say, 8 req/s goodput.
- **Warm-cache run** (system prompt pre-populated): every request hits the cached prefix from the start, prefilling only ~150 tokens. The measured goodput reflects the steady state — say 20 req/s.

The 2.5× difference is entirely an artifact of cache state, not engine capability. Production runs warm (the system prompt is hit constantly), so the warm number (20 req/s) is the realistic capacity, and the cold number badly underestimates it. The correct methodology (§6.3): replay realistic traffic long enough to reach the warm steady state, or pre-warm the cache, and report the steady-state goodput. Reporting the cold number would lead to over-provisioning (sizing the fleet for 8 req/s when 20 is achievable — 2.5× too many GPUs, 2.5× the cost). This is why specifying and controlling the cache state is non-negotiable for benchmarking sharing workloads, and why the prefix-cache hit rate (§28) must be reported alongside throughput — a throughput number without the hit rate is uninterpretable for sharing workloads.

---

## 33. TTFT and TPOT Decomposition, Worked

Apply the latency decomposition (File 04 §39) with numbers to localize a latency problem. Suppose P95 TTFT is 1.8 s (SLO 1 s — violated) for a chat workload (300-token prompts).

**TTFT = queue_wait + prefill_compute + scheduling_overhead.**
- Measure `prefill_compute`: a 300-token prefill on the model/hardware ≈ 0.1 s (compute-bound, fast for a short prompt). With prefix caching (system prompt cached), the effective prefill is ~100 tokens ≈ 0.03 s.
- `scheduling_overhead`: ~0.01 s (negligible with V1/overlap).
- So `queue_wait ≈ 1.8 − 0.03 − 0.01 ≈ 1.76 s` — **the queue wait dominates.**

The diagnosis: the TTFT problem is *queuing*, not prefill cost — the system is near `λ_max` (requests wait ~1.76 s in the queue before admission). The fix is *capacity* (more replicas to raise `λ_max`) or *admission/scheduling* (prioritize, or shed load), **not** prefill optimization (the prefill is already fast). Had the decomposition shown `prefill_compute` dominating (e.g. a 5,000-token prompt taking 1.5 s to prefill), the fix would be different (chunked prefill won't help TTFT of the prefilling request itself; prefix caching would if the prompt is shared; faster prefill via FP8). The decomposition tells you *which term* is the problem, directing the fix — without it, you might wrongly "optimize prefill" when the issue is queuing (capacity). Similarly for TPOT: decompose into the base decode-step time (batch/context/model — the memory-bound physics, §3.2) plus stall time (prefill interference, §17.1); if stalls dominate, chunked prefill fixes it; if the base step time is the issue, reduce batch (lower TPOT, lower throughput — the trade) or reduce bytes (quantization). The decomposition is the diagnostic that turns "latency is high" into "this specific term is high, so this specific lever applies" — the precision that separates effective tuning from flailing.

---

## 34. Benchmark Reproducibility

A practical concern: benchmarks must be *reproducible* to be useful for comparison and regression detection (§19). Sources of variance to control:

- **Warmup:** discard the first requests (kernel compilation/autotuning, File 10 §23; cache warming, §32) — measure steady state only.
- **Cache state:** specify and control (cold/warm/replayed, §6.3, §32).
- **Arrival pattern:** use a fixed seed for the Poisson arrival generator so runs are comparable.
- **Hardware/software versions:** pin the GPU type, driver, CUDA, engine version, model — all affect results.
- **Run length:** long enough to reach steady state (queues, cache) and to get stable percentiles (P99 needs many samples).
- **Isolation:** no other load on the GPUs; consistent power/thermal state.

Reproducibility lets you compare engines (§8), validate tuning changes (§19), and detect regressions (a re-run that differs signals a change). Without it, benchmark numbers are noise — a 10% difference could be variance, not a real effect. The discipline: fix all the controllable variables, warm up, run long enough, report percentiles with the configuration (hardware, versions, cache state, dataset, arrival rate) so the result is interpretable and reproducible. This rigor is what makes a benchmark a *measurement* rather than an anecdote, and it's prerequisite to the measurement-driven tuning the file advocates — you can only tune by measurement if the measurements are trustworthy.

---

## 35. Worked Multi-GPU Scaling and Cost Comparison

Tie scaling efficiency (§9) and cost (§16) together with a worked comparison for LLaMA-3 70B (FP8), choosing between TP=8 (one replica) and TP=4×DP=2 (two replicas) on 8×H100.

Suppose benchmarks show:
- **TP=8:** TPOT 25 ms (well under a 50 ms SLO), throughput 6,000 tok/s, scaling efficiency ~88% (8 GPUs → 7× a single GPU). Goodput (within SLO) ~5,500 tok/s.
- **TP=4×DP=2:** TPOT 42 ms (under 50 ms SLO), throughput per replica ~3,800 tok/s × 2 = 7,600 tok/s, scaling efficiency ~94% per replica (less AllReduce). Goodput ~7,000 tok/s.

**TP=4×DP=2 wins on goodput** (7,000 vs 5,500 tok/s) because it meets the SLO (42 ms < 50 ms) with less AllReduce overhead per replica and scales throughput linearly across two replicas. The $/token (§16) follows: same 8 GPUs, so `$/token` is inversely proportional to goodput → TP=4×DP=2 is `7000/5500 ≈ 1.27×` cheaper per token. *Unless* the SLO were tighter (e.g. 35 ms TPOT), in which case TP=4's 42 ms would violate it and TP=8 (25 ms) would be required, accepting lower throughput for the latency. This is the latency-headroom-vs-throughput trade (File 05 §16) quantified: **use the lowest TP that meets the latency SLO, scale throughput with DP** — TP=4×DP=2 here, because TP=4 has latency headroom under the 50 ms SLO. The decision is made by measurement (the TPOT at each TP vs the SLO), not by rule of thumb, and it directly determines $/token. This worked comparison is the multi-GPU instance of the file's thesis: measure (the scaling efficiency and TPOT), reason from the goal (goodput per dollar), and choose the configuration the measurements support.

---

## 36. The Tuning Mindset

Closing the file, the durable mindset for performance work:

1. **Start from the workload, not the knobs.** Characterize (§22) before tuning. The workload's prefill/decode balance, sharing fraction, length distributions, and SLOs determine which levers matter.
2. **Reason from the roofline.** Compute the theoretical limits (§3); know whether you're compute- or memory-bound; the gap to the roofline directs the fix.
3. **Optimize goodput, not throughput.** The SLO-meeting rate per GPU-dollar is the objective (§2, §16), not raw tokens/sec.
4. **Measure the curve.** The latency-throughput curve (§20) is the central artifact — it gives goodput, shows the cliff, validates changes.
5. **Diagnose top-down.** Metrics → which layer (algorithm/architecture/kernel, File 09 §44) → which kernel/parameter → fix → re-measure (§11, §24).
6. **Apply the matching lever.** Each symptom maps to a cause to a lever (§31); don't apply decode levers to prefill problems or micro-optimize non-bottlenecks.
7. **Validate and monitor.** Benchmark every change, roll out via canary, monitor leading indicators, stay off the cliff (§19, §23, §28).

This mindset — workload-first, roofline-grounded, goodput-targeted, measurement-driven, top-down-diagnostic — is what this file teaches, and it's the practical capstone of the database. Files 01–10 built the understanding of *how* the systems work (the cost models, the mechanisms, the kernels); this file turns that understanding into the *discipline* of making a real deployment fast and cheap while meeting its SLOs. An inference engineer who internalizes both — the mechanisms and the tuning methodology — can take any model, any engine, any hardware, any workload, and systematically drive it to its goodput-per-dollar optimum. That capability, not any specific benchmark number or flag value, is the goal. The remaining files (12–20) cover specialized techniques (speculative decoding, long context, multimodal, advanced research, hardware, framework comparison, LoRA, operations, business) that deepen specific areas, but the tuning methodology here is the lens through which all of them connect to the practical question every deployment must answer: *how do I serve this workload within its SLOs at the lowest cost?*

---

## 37. Appendix: Source Pointers

- **Benchmark tools:** vLLM `benchmarks/` (`benchmark_serving.py`, `benchmark_throughput.py`, `benchmark_latency.py`); SGLang `python -m sglang.bench_serving`.
- **Metrics:** vLLM `/metrics` (Prometheus); the `vllm:*` metric families (§28). SGLang analogous metrics + `--enable-metrics`.
- **Profiling:** Nsight Systems (`nsys`), Nsight Compute (`ncu`) — File 10 §12; engine `--profile`/`--enable-profiling`.
- **Datasets:** ShareGPT (realistic chat), Alpaca, custom traces — §6.1.
- **Roofline:** File 01 §2 (the analytical foundation); per-kernel roofline via ncu (File 10 §12.2, §20).
- **The mechanisms tuned:** Files 03 (KV memory), 04 (scheduler), 05 (distributed), 06 (quantization), 08–09 (SGLang), 10 (kernels), 12 (speculation), 13 (long context).

The tuning workflow ties these together: characterize the workload → consult the mechanism files for the relevant levers → set the configuration → benchmark the curve with the tools → read the metrics → diagnose → iterate. This file is the hub; the mechanism files are the spokes; the measurement tools are the instrument; goodput-per-dollar is the destination.

---

## 38. Worked Example: Cost Reduction Campaign

To synthesize the cost levers (§16) into a realistic narrative, trace a cost-reduction campaign on a chat deployment initially running LLaMA-3 70B in BF16 at TP=8, serving a workload at $X/1M tokens.

1. **Baseline:** BF16, TP=8, no prefix caching, no speculation. Measure goodput and $/token. Suppose $1.20/1M tokens.
2. **Enable prefix caching** (shared system prompt, §26): prefill drops ~14× for cache-warm requests → goodput rises ~1.8× → $/token falls to ~$0.67. *Biggest single win for this sharing workload.*
3. **Switch to FP8 weights** (§16.2): ~1.8× decode throughput, and fits on fewer GPUs — re-benchmark to TP=4×DP=2 (better scaling, §35) → goodput rises further, $/token to ~$0.45.
4. **Add speculative decoding** (decode-bound chat, §17): ~1.7× decode speedup at ~65% acceptance → $/token to ~$0.30.
5. **FP8 KV cache** (§18): doubles KV-bound concurrency → larger batches within memory → modest further gain → ~$0.27.
6. **Right-size utilization to 70–85%** (§29): ensure replicas operate in the cost-optimal band, not under-batched → final ~$0.25/1M tokens.

The campaign stacked the applicable levers (prefix caching for sharing, FP8 for memory/compute, speculation for decode, KV FP8 for concurrency, utilization for efficiency) to cut $/token ~4.8× ($1.20 → $0.25), each validated by re-benchmarking goodput (§19). The order matters — prefix caching first (biggest win for sharing), then the precision and speculation levers, each re-measured. This is cost optimization as a *campaign*: identify the applicable levers from the workload profile (§22), stack them, validate each by the goodput/$ it delivers, and arrive at the optimum. The 4.8× reduction is typical of what's achievable moving from a naive BF16 baseline to a fully-tuned deployment — and it's why mastery of these levers (the content of Files 03–10, applied through this file's methodology) directly determines whether an inference business has a cost advantage or a 5× disadvantage (File 20 §moat). The levers are the same ones the entire field uses to drive down inference cost (File 01 §8's price trajectory); applying them systematically to a specific workload is the inference engineer's core value.

### 38.1 Why the order and validation matter

A subtle point in the campaign: the levers interact, so the order and per-step validation matter. Switching to FP8 weights *before* re-evaluating the parallelism would miss the TP=8→TP=4×DP=2 opportunity that FP8's smaller footprint enables (§35) — the precision change unlocked a parallelism change. Enabling speculation *before* prefix caching would show a smaller relative gain (the prefill cost would still dominate). And applying speculation to a high-batch compute-saturated regime would show little benefit (§17.3) — its value depends on the batch regime the *other* levers produce. So the campaign isn't "apply all levers" but "apply the applicable levers in an order that accounts for their interactions, validating each by re-measured goodput/$." This is why cost optimization is a measurement-driven *campaign*, not a checklist — the levers compound and interact, and only re-measuring after each reveals the true effect and the next opportunity. The discipline (§30) — measure, apply the matching lever, re-measure — applied iteratively, is what converts the mechanisms into the 4.8× cost reduction, and it's the same discipline that, applied continuously as models and traffic evolve (§19), keeps a deployment at its cost-optimal point over time.

A final caution: each lever must also pass the *quality* gate (File 06 §12). FP8 weights, FP8 KV, and aggressive quantization can shift output quality, and speculative decoding (while lossless to the target distribution, File 12) depends on the draft's acceptance. So the campaign's per-step validation isn't only goodput/$ but also a quality check on the target task — a cost reduction that degrades quality below the acceptable bar is not a win. The full validation is therefore two-dimensional: does the lever improve goodput/$ *and* preserve quality? Only levers passing both are kept. This is why the workload characterization (§22) includes a quality bar, and why cost optimization and quality validation proceed together — the cheapest deployment that still meets the quality requirement is the actual optimum, not the cheapest deployment outright. This two-dimensional optimization — minimize $/token subject to both the latency SLOs (goodput) and the quality bar — is the complete statement of the production tuning problem, and it is the problem every serious inference deployment, and every hosted-inference business (File 20), is continuously solving. Holding that complete objective — goodput within SLO, at minimum cost, above the quality bar — in mind is what keeps tuning honest: it prevents chasing throughput that violates latency, cost reductions that degrade quality, or benchmark wins that don't transfer to production. The objective is the anchor; the methodology (§30) is the path; the mechanisms (Files 03–10) are the levers; and the measurement (the curve, the metrics) is the feedback that closes the loop. Master that loop and the daunting space of flags, engines, models, and hardware collapses into a tractable, repeatable engineering practice — which is the practical liberation this file aims to deliver: not a memorized recipe that ages with the next release, but a methodology that adapts to whatever models, engines, and accelerators the field produces next. That adaptability — the ability to re-derive the right configuration from first principles for each new situation — is the difference between an engineer who can only follow recipes and one who can author them — and authoring them, from the workload and the roofline, is the craft this database exists to teach.








