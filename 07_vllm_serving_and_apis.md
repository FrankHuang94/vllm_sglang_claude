# vLLM Serving Infrastructure — OpenAI API, Async Engine, and Production Deployment

> **Standard reference file.** This file covers vLLM's serving surface: the `AsyncLLMEngine`, the OpenAI-compatible API server, the request lifecycle, sampling parameters, offline batched inference, metrics and observability, multi-LoRA serving, deployment configurations, and the V2/V1 engine architecture. Prerequisites: Files 03–06 (the engine internals this layer exposes).

---

## 1. AsyncLLMEngine

The core `LLMEngine` (which runs the scheduler + workers, Files 04–06) is synchronous: call `step()`, get outputs. Production serving needs to handle many concurrent HTTP requests without blocking, so vLLM wraps it in **`AsyncLLMEngine`** (`vllm/engine/async_llm_engine.py`).

- **Background loop:** `AsyncLLMEngine` runs the engine's `step()` in a background asyncio task, continuously executing the continuous-batching loop (File 04 §3).
- **`generate()` as an async generator:** a caller `await`s `engine.generate(prompt, sampling_params, request_id)`, which returns an `AsyncGenerator` yielding `RequestOutput` objects as tokens are produced. Each yielded output carries the new token(s), enabling **streaming**: the API layer forwards each token to the client as it's generated, rather than waiting for the full completion.
- **Concurrency:** many `generate()` calls can be in flight simultaneously; each adds its request to the engine's waiting queue, and the background loop's continuous batching interleaves them. The asyncio model lets one process handle hundreds of concurrent streaming requests without threads-per-request.

This async wrapper is what turns the batch-processing engine into a responsive multi-client server. The V1 engine (File 04 §14, §11 below) makes this async/overlap structure first-class, moving the engine core and detokenizer to separate processes.

---

## 2. The OpenAI-Compatible API Server

vLLM ships a FastAPI-based server (`vllm serve <model>` or `python -m vllm.entrypoints.openai.api_server`) implementing the OpenAI API, so existing OpenAI client code works against a self-hosted vLLM with only a base-URL change. This API compatibility is a major adoption driver.

### 2.1 Endpoints

- **`/v1/completions`:** text completion (prompt → completion).
- **`/v1/chat/completions`:** chat completion (messages → assistant message), applying the model's **chat template** (a Jinja2 template that formats the system/user/assistant turns into the token sequence the model expects, with the right special tokens and role markers).
- **`/v1/embeddings`:** embedding generation (File 14).
- **`/v1/models`:** list served models.
- **`/health`, `/metrics`:** operational endpoints (§6, File 19).

### 2.2 Streaming via SSE

Streaming responses use **Server-Sent Events (SSE)**: the server sends `data:` chunks as tokens are generated, terminated by `data: [DONE]`. This is the OpenAI streaming protocol; clients receive tokens incrementally for low perceived latency (TTFT matters most for streaming UX, File 11 §metrics). Non-streaming requests block until the full completion and return one JSON response.

### 2.3 Authentication and configuration

- **`--api-key`:** a bearer token for authentication.
- **`--served-model-name`:** the name clients use (can differ from the checkpoint path).
- **Chat template:** auto-loaded from the model's tokenizer config, or overridable with `--chat-template`. A wrong chat template is a common bug — the model receives mis-formatted role markers and produces degraded or confused output.

---

## 3. The Request Lifecycle

End-to-end, a chat request flows:

```
HTTP POST /v1/chat/completions
  → FastAPI handler: validate, apply chat template, tokenize
  → AsyncLLMEngine.generate(prompt_token_ids, sampling_params, request_id)
  → request enters the engine's waiting queue
  → scheduler admits it (File 04 §5), prefill runs
  → continuous-batching loop generates tokens
  → each token: detokenized incrementally (File 02 §18.2), streamed via SSE
  → stop condition met → final RequestOutput → SSE [DONE]
  → KV blocks freed (File 03 §12)
```

Key points: tokenization happens at the API layer (CPU, ideally overlapped with GPU work — the V1 engine and SGLang both run it in a separate process, File 04 §14, File 09); the request gets a unique `request_id` for tracking, streaming, and cancellation; a client disconnect triggers an abort (File 04 §28) freeing the KV. The detokenization is incremental and stateful (File 02 §18.2) to handle multi-token characters correctly in the stream.

---

## 4. Sampling Parameters

The API exposes the full sampling-parameter set (File 02 §8, File 06 §10) via `SamplingParams`:

- **`temperature`, `top_p`, `top_k`:** the core sampling controls.
- **`max_tokens`:** output length cap (also a stop condition and a scheduling hint, File 04 §8.3).
- **`stop`, `stop_token_ids`:** stop strings/tokens (File 02 §18.3).
- **`presence_penalty`, `frequency_penalty`, `repetition_penalty`:** repetition controls.
- **`logprobs`, `prompt_logprobs`:** return token log-probabilities (more expensive — needs the full logit distribution, File 05 §33).
- **`n`:** number of parallel samples (shares prompt KV via CoW, File 03 §5).
- **`best_of`:** generate `best_of` candidates, return the top `n` by cumulative logprob.
- **`seed`:** per-request RNG seed for reproducibility.
- **`guided_json` / `guided_regex` / `guided_grammar` / `guided_choice`:** structured-output constraints (File 02 §10, File 08 XGrammar) — the API accepts a JSON schema, regex, grammar, or choice list and the engine constrains generation accordingly.

These map directly to the sampler's per-sequence parameters; the engine supports a heterogeneous batch where each request has different sampling settings (File 02 §8.3).

---

## 5. Offline Batched Inference

For non-serving use — dataset processing, evaluation, synthetic data generation — vLLM offers the `LLM` class for **offline batched inference**:

```python
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3-8B", tensor_parallel_size=2)
outputs = llm.generate(list_of_prompts, SamplingParams(temperature=0.8, max_tokens=256))
```

`LLM.generate()` accepts a list of prompts and runs them through the same continuous-batching engine internally — so you get the throughput benefits of batching without managing a server. This is simpler than the serving path (no HTTP, no async, no streaming) and ideal for throughput-oriented offline jobs where you can use aggressive settings (large batches, multi-step scheduling, File 04 §11.2). It's the recommended path for evaluation harnesses and bulk processing.

---

## 6. Metrics and Observability

vLLM exposes Prometheus metrics at `/metrics`, the backbone of production monitoring (File 11, File 19).

- **Latency histograms:** TTFT, TPOT, end-to-end latency — the SLO metrics (File 11 §metrics taxonomy). Histograms (not just averages) so you can track P50/P95/P99 (File 04 §39.1).
- **Throughput:** tokens/sec (prompt and generation separately), requests/sec.
- **Queue metrics:** `num_requests_waiting`, `num_requests_running`, `num_requests_swapped` — the scheduler state (File 04 §23), revealing queuing and capacity pressure.
- **KV cache:** `gpu_cache_usage_perc` (block-pool occupancy, File 03 §22), the single most important capacity gauge; `cpu_cache_usage_perc` for swap.
- **Prefix cache:** hit rate (File 03 §31), central to prefix-heavy workload performance.
- **Preemptions:** count/rate (File 04 §9.3), the thrash indicator.

These feed Grafana dashboards and alerting. The diagnostic discipline (File 04 §32, File 11): a latency problem → check whether it's queue wait (capacity), KV pressure (memory), or prefill interference (chunked prefill); the metrics distinguish these. Alerting on `gpu_cache_usage_perc > 90%` plus rising preemptions catches the capacity cliff (File 04 §18) before it causes OOM or latency divergence.

### 6.1 Tracing and debugging

- **`VLLM_LOG_LEVEL=DEBUG`:** verbose scheduler/engine logs.
- **`VLLM_TRACE_FUNCTION=1`:** function-level tracing for deep debugging.
- **`--disable-log-stats`:** quiet the periodic throughput logging.
- Per-request logs report prompt length, output length, TTFT, and throughput — useful for spotting pathological requests.
- **torch.profiler / NVTX:** vLLM has profiling hooks (`--profile`-style integration) and NVTX range annotations for nsys timelines (File 10 §profiling).

---

## 7. Multi-LoRA Serving

vLLM serves many LoRA adapters on one base model (full treatment File 18, model-layer mechanics File 06 §14):

- **`--enable-lora`** turns on LoRA support; **`--max-loras`** caps concurrently-resident adapters; **`--max-lora-rank`** sets the max adapter rank.
- A request specifies its adapter via a `LoRARequest` (adapter name + path); the engine routes the request through that adapter's low-rank update.
- **Punica/SGMV kernels** (File 06 §14, File 18) let a single batch mix requests using different adapters efficiently.
- Adapters are loaded on demand (from HuggingFace Hub or local path), cached in GPU memory with LRU eviction, and kept in CPU pinned memory for fast reload. Per-adapter memory is small (rank × d_model × layers × bytes, e.g. ~64 MB for rank 16 on a 7B-class model).

This enables the SaaS pattern: one base model, many customer-specific fine-tunes, routed per request by customer ID → adapter name (File 18 §multi-LoRA, File 20). The scheduler treats adapter slots as a second resource (File 04 §21).

---

## 8. Deployment Configurations

Production deployment combines the engine flags into a coherent configuration (full tuning methodology File 11, operations File 19):

- **`--max-model-len`:** maximum sequence length (context window).
- **`--max-num-seqs`, `--max-num-batched-tokens`:** concurrency and per-step token budget (File 04 §6, §13).
- **`--gpu-memory-utilization`:** KV pool sizing (File 03 §7).
- **`--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size`:** parallelism topology (File 05).
- **`--quantization`, `--kv-cache-dtype`:** precision (File 06).
- **`--enable-chunked-prefill`, `--enable-prefix-caching`:** scheduling/caching features (File 04 §7, File 03 §6).

### 8.1 Containerized and orchestrated deployment

- **Docker:** vLLM ships official images; mount the model, expose the port, pass flags as args.
- **Kubernetes:** deploy with the NVIDIA GPU Operator (for GPU scheduling), a Deployment + Service, readiness/liveness probes on `/health` (File 19 §health), and an HPA or KEDA for autoscaling on queue depth / GPU utilization (File 11 §autoscaling). A vLLM Helm chart simplifies this.
- **Ray Serve:** vLLM integrates with Ray Serve for multi-replica, multi-model serving with Ray-managed scaling and routing.
- **The vLLM Production Stack / router:** a prefix-cache-aware, KV-aware router for multi-replica deployments (File 05 §7.2) that improves prefix-cache hit rates by routing requests with shared prefixes to the same replica.

---

## 9. The V1 / EngineCore Architecture

vLLM's V1 engine (File 04 §14) restructured the serving stack to cut Python overhead and improve concurrency:

- **`EngineCore`:** a tight scheduling/execution core, isolated from request handling, designed for minimal per-step overhead (with hot paths optimized toward native code).
- **Separate processes:** the API server, the **detokenizer**, and the engine core run as separate processes communicating via efficient IPC. This keeps tokenization/detokenization and HTTP handling off the engine's critical path (the same separation SGLang uses, File 09) — the detokenizer process turns token IDs into text streams without blocking the scheduler.
- **Async-by-design scheduling:** the architecture overlaps scheduling with GPU execution (plan step `t+1` during step `t`, File 04 §11.3), removing the scheduler from the critical path.
- **Result:** materially higher throughput on decode-heavy and small-model workloads (where Python overhead dominated) and lower, more consistent latency, while preserving the API and feature set.

For the operator, V1 means recent vLLM has far lower per-step overhead than early versions — the old workarounds (large `--num-scheduler-steps`, disabling features for speed) are largely unnecessary. The architecture also makes the engine more amenable to disaggregation (separate prefill/decode processes, File 04 §10, File 15) since the process boundaries already exist.

---

## 10. Synthesis

The serving layer turns vLLM's batch-processing engine into a production, multi-client, OpenAI-compatible service. `AsyncLLMEngine` provides the async, streaming, concurrent interface; the FastAPI server exposes the OpenAI endpoints with chat templates and SSE streaming; the request lifecycle threads each request from HTTP through tokenization, scheduling, generation, incremental detokenization, and streaming back, freeing KV on completion or abort. Sampling parameters, structured-output constraints, and multi-LoRA give clients fine control. Metrics expose the scheduler and KV state for monitoring and capacity management. The V1 architecture moves tokenization, detokenization, and request handling into separate processes and overlaps scheduling with execution, minimizing the overhead between the client and the GPU. This layer, plus the deployment patterns (Docker/K8s/Ray, autoscaling, health checks), is what makes vLLM a *service* rather than a library — the surface that File 19 (operations) and File 20 (the hosted-inference market built on exactly this) build upon. Next, File 08 turns to SGLang's distinctive contributions — RadixAttention and XGrammar — and File 09 to its runtime architecture, where many of the same serving concerns are solved with a different process model and a program-aware frontend.

---

## 11. Chat Templates in Depth

The chat template is one of the most error-prone parts of serving, and worth understanding deeply because a wrong template silently degrades every chat response. A chat template is a Jinja2 template, stored in the model's tokenizer configuration, that converts a list of role-tagged messages into the exact token sequence the model was fine-tuned to expect. Different model families use different formats: LLaMA-3 uses `<|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|eot_id|>`-style markers; ChatML (used by many models) uses `<|im_start|>role ... <|im_end|>`; Mistral uses `[INST] ... [/INST]`. The template inserts the system prompt, wraps each user and assistant turn in the correct delimiters, and appends the generation prompt (the tokens that cue the model to start the assistant turn).

If the served template doesn't match what the model was trained with, the model receives unfamiliar formatting and produces worse, sometimes incoherent, output — and crucially this fails *silently* (no error, just degraded quality), making it a frequent and frustrating production bug. vLLM auto-loads the template from the tokenizer config when present; `--chat-template` overrides it for models that ship without one or with a wrong one. The template also governs **tool/function-calling** formatting (§12) and multimodal content insertion (image placeholders, File 14). Verifying the rendered prompt for a sample conversation against the model's documented format is a worthwhile pre-deployment check — it catches the silent-degradation class of bug that metrics won't surface.

---

## 12. Tool Calling and Structured Output

Modern LLM applications rely on **function/tool calling** — the model emits a structured call (function name + JSON arguments) that the application executes. vLLM supports the OpenAI tool-calling API: clients pass a `tools` list (function schemas); the model is prompted (via the chat template's tool formatting) to emit tool calls, which vLLM parses from the generated text and returns in the structured `tool_calls` field. Reliability of the emitted JSON is the challenge — a model can generate malformed JSON or hallucinate fields. This is where **structured/guided decoding** (File 02 §10, File 08 XGrammar) becomes essential: by constraining generation to a JSON schema (`guided_json` or the tool's schema), the engine *guarantees* syntactically valid, schema-conforming output, eliminating the parse-failure class of bug. The combination — tool schemas + grammar-constrained decoding — is what makes function calling production-reliable, and it is near-free in latency with XGrammar's precomputed masks (File 02 §10.2). vLLM exposes guided decoding through `guided_json`, `guided_regex`, `guided_grammar`, and `guided_choice`, backed by a grammar backend (Outlines, XGrammar, or lm-format-enforcer). This is one of the most operationally important features for agentic and API-integration workloads (File 20 §agent-oriented inference).

---

## 13. A Worked Streaming Interaction

To make the SSE streaming concrete, trace a streaming chat completion. The client sends `POST /v1/chat/completions` with `"stream": true` and a messages list. The server:

1. Applies the chat template, tokenizes the rendered prompt, and calls `AsyncLLMEngine.generate` with a fresh `request_id`.
2. The request enters the waiting queue; the scheduler admits it and runs prefill (File 04 §5). The first token is produced — this is the moment that determines **TTFT** (File 11), which for a streaming UX is the dominant perceived-latency metric.
3. As each token is generated, the engine yields a `RequestOutput`; the API layer incrementally detokenizes (File 02 §18.2) and emits an SSE chunk: `data: {"choices":[{"delta":{"content":"<new text>"}}]}`. The inter-token interval here is **TPOT**.
4. On a stop condition (EOS, `max_tokens`, stop string), the server emits a final chunk with the finish reason and then `data: [DONE]`, and the engine frees the request's KV blocks (File 03 §12).
5. If the client disconnects mid-stream, the server detects the closed connection and aborts the request (File 04 §28), freeing KV immediately — essential to avoid zombie generations holding memory.

The streaming path's correctness rests on the stateful incremental detokenizer (a token may complete a multi-byte character only when a later token arrives, File 02 §18.2) — emitting raw per-token decodes would produce flickering or corrupted output for non-Latin scripts. This is why the detokenizer is a dedicated, stateful component (its own process in V1, §9).

---

## 14. Embeddings and Reranking APIs

Beyond generation, vLLM serves **embedding** and **classification/reranking** models (full treatment File 14):

- **`/v1/embeddings`** (`--task embed`): runs the model as an encoder, pooling the last hidden states (mean pooling or `[CLS]`) into a fixed-size embedding vector. No autoregression, no KV cache growth — a single forward pass per input, so throughput is maximized by large batches that saturate the tensor cores (a compute-bound, not memory-bound, workload — File 14 §embedding serving).
- **Reranking / classification:** cross-encoder scoring of (query, document) pairs, returning relevance scores — used in RAG pipelines to rerank retrieved candidates. Each pair is a single forward pass; latency scales with the number of candidate documents per query.

These tasks reuse the same engine, model-loading, and batching infrastructure but bypass the generation loop (no sampler, no KV cache, no streaming). Serving embeddings and generation from the same stack lets a deployment consolidate its model-serving infrastructure.

---

## 15. Rate Limiting, Quotas, and the Gateway Layer

Production serving needs request governance that the engine itself doesn't provide — it is layered at a **gateway/router** above vLLM (File 04 §37, File 19):

- **Rate limiting:** per-client/API-key request and token rate caps, shaping the arrival stream so no client overwhelms the engine (keeping `λ < λ_max`, File 04 §18).
- **Quotas and billing:** token accounting per client for usage-based billing (the commercial API pattern, File 20).
- **Authentication and routing:** validating API keys, routing to the right model/replica (prefix-cache-aware, File 05 §7.2).
- **Load shedding / circuit breaking:** returning 503 with `Retry-After` when the engine is at capacity, rather than queuing unboundedly (File 19 §circuit breaking).

The division of labor: the engine maximizes goodput on the requests it's given; the gateway controls *which* requests it's given and enforces fairness/quotas/limits (File 04 §37). vLLM provides the `/metrics` and `/health` endpoints the gateway and orchestrator use to make routing and scaling decisions.

---

## 16. Deployment Patterns, Expanded

Tying the deployment options to operational goals (File 19):

- **Single replica, single node:** simplest; TP within the node (File 05 §8.2). Suitable for moderate load where one replica meets throughput.
- **Multiple replicas (DP), load-balanced:** scale throughput and gain fault tolerance (File 05 §22). A load balancer (nginx/HAProxy/Envoy, or the vLLM router) distributes requests; health checks on `/health` route around failures; prefix-cache-aware routing maximizes cache hits.
- **Autoscaling:** Kubernetes HPA on GPU utilization, or KEDA on queue depth (`num_requests_waiting` via Prometheus). Because cold start is slow (model load + warmup, minutes for large models, File 19 §readiness), pre-warm replicas or scale proactively rather than reactively.
- **Multi-model:** serve several models behind one endpoint, either time-sharing (load/unload) or spatial-sharing (multiple models resident), often via Ray Serve or Triton orchestration (File 19 §multi-model). Multi-LoRA (§7) is the special case of many fine-tunes sharing one base model — far more memory-efficient than separate full models.
- **Disaggregated:** separate prefill and decode deployments (File 04 §10, File 15) for cost efficiency at scale — the V1 process architecture (§9) eases this.

The configuration that's right depends on load, SLOs, and budget — File 11 derives it from a measured workload, and File 19 covers the reliability and cost-optimization practices around it.

---

## 17. Closing

vLLM's serving layer is the bridge from the high-performance engine to real clients: async, streaming, OpenAI-compatible, observable, and deployable on standard orchestration. The `AsyncLLMEngine` and FastAPI server handle concurrency and the API surface; chat templates and tool-calling-with-guided-decoding make chat and agentic use cases reliable; the request lifecycle threads each request through tokenization, scheduling, generation, and streaming with correct incremental detokenization and abort handling; metrics expose the internal state for monitoring and capacity management; multi-LoRA and embedding/reranking broaden the workloads served from one stack; and the V1 architecture minimizes the overhead between client and GPU. The gateway layer above adds rate limiting, quotas, and load shedding. This is the surface on which the entire hosted-inference market (File 20) is built — providers take this serving layer (often forked and customized) and wrap it in their own gateways, billing, and routing. With both engines' internals (vLLM, Files 03–07) now covered, Files 08–09 turn to SGLang, whose RadixAttention, XGrammar, and program-aware runtime architecture represent the other major design lineage in open-source LLM serving.

---

## 18. The Engine API Surface

Beneath the HTTP layer, the engine exposes a programmatic API that the server (and library users) call:

- **`add_request(request_id, prompt, sampling_params, ...)`:** enqueue a request. The prompt can be raw text (tokenized by the engine) or pre-tokenized token IDs (skipping tokenization — useful when the caller controls tokenization or for non-text inputs). Multimodal inputs (File 06 §9, File 14) attach pixel/feature data here.
- **`abort_request(request_id)`:** cancel a request, freeing its KV (File 04 §28). The API layer calls this on client disconnect.
- **`step()`** (sync) / the background loop (async): run one continuous-batching iteration, returning the `RequestOutput`s with newly generated tokens.
- **`RequestOutput`:** carries the request ID, the generated token IDs and their (incrementally detokenized) text, cumulative logprobs if requested, the finish reason (stop/length/abort), and per-sequence outputs for `n > 1` parallel samples.

The async server wraps these: `generate()` calls `add_request` and yields `RequestOutput`s from the background loop as they arrive for that request ID, translating engine outputs into SSE chunks. This clean engine API is also what makes vLLM usable as a *library* (the offline `LLM` class, §5) and embeddable in larger systems (Ray Serve deployments, custom servers) — the HTTP server is just one consumer of the engine API.

### 18.1 Prompt handling and `prompt_logprobs`

Inputs can be text or token IDs; the engine also supports returning **`prompt_logprobs`** — the log-probabilities the model assigns to the *prompt* tokens themselves (computed during prefill). This is used for: scoring/ranking candidate prompts, perplexity evaluation, classification-by-likelihood (compare the model's probability of different completions), and some RAG/retrieval-scoring schemes. It's more expensive (requires the full logit distribution over the prompt positions, File 05 §33) and is off by default. The combination of `prompt_logprobs` and pre-tokenized input makes vLLM usable for evaluation and scoring tasks, not just generation.

---

## 19. Warmup and Readiness

When a vLLM server starts, several time-consuming steps precede readiness (File 19 §readiness):

1. **Weight loading** (File 06 §4): minutes for large models, even with sharded parallel loading (a 70B BF16 is 140 GB).
2. **KV pool profiling and allocation** (File 03 §7): a profiling forward pass to size the block pool.
3. **CUDA graph capture** (File 09): capturing decode graphs for a ladder of batch sizes — adds startup time and memory but speeds steady-state decode.
4. **Warmup runs:** a few forward passes to trigger lazy CUDA kernel compilation/autotuning so the first real request doesn't pay that cost.

The `/health` endpoint returns ready only after these complete. For Kubernetes, this means a **startup probe** with a generous timeout (model load can take 2–5 minutes for 70B), distinct from the liveness probe. The slow cold start is why autoscaling must be proactive (pre-warm replicas, scale on leading indicators) rather than purely reactive — a replica added in response to a load spike won't be ready for minutes (File 11 §autoscaling, File 19). Caching loaded weights in CPU pinned memory can speed subsequent GPU reloads (File 19 §multi-model). The practical consequence is that capacity planning must treat a replica as unavailable for the first several minutes of its life, sizing the warm pool and scaling triggers so that demand spikes are met by already-warm capacity rather than by replicas still loading — a constraint that distinguishes LLM serving operations from stateless web services that start in milliseconds. This single fact — multi-minute cold starts driven by gigabyte-scale weight loads and graph capture — shapes the entire operational posture of an inference deployment, from autoscaling policy to deployment strategy to capacity headroom — a constraint that File 19's operations practices are largely organized around accommodating.

---

## 20. Final Synthesis

The serving and API layer is what users and operators actually touch. It exposes the OpenAI-compatible surface that makes vLLM a drop-in replacement for hosted APIs; it streams tokens with correct incremental detokenization for responsive UX; it threads each request through its full lifecycle with proper abort handling so misbehaving clients don't leak memory; it makes chat and tool-calling reliable through chat templates and guided decoding; it serves many fine-tunes via multi-LoRA and other tasks via embedding/reranking endpoints; and it exposes the metrics and health signals that orchestration and gateways depend on. The V1 architecture's process separation and async scheduling minimize the client-to-GPU overhead. Above it sits the gateway (rate limiting, quotas, routing); below it the engine internals of Files 03–06. This layer is, in the end, the product surface — the thing a developer calls and an SRE operates — and getting it right (correct templates, reliable structured output, proper streaming and abort, good observability) is as important to a successful deployment as the kernel-level optimizations that make it fast. With vLLM's full stack now covered from kernel to API, the database turns to SGLang (Files 08–09) and then to the cross-cutting concerns — kernels (10), performance (11), and the specialized and operational topics (12–20) — that complete the picture of modern LLM inference engineering.

