# Production Operations — Reliability, Observability, and Cost Optimization

> **Standard reference file.** Running LLM inference in production requires operational discipline beyond the engine: graceful shutdown, health checks, autoscaling, multi-model serving, chaos engineering, versioning, observability, and cost optimization. This file covers the SRE/ops side of LLM serving. Prerequisites: File 07 (serving/APIs), File 11 (tuning, capacity planning), File 05 §22 (fault tolerance).

---

## 1. Graceful Shutdown and Request Draining

LLM requests are long-lived (a generation can take seconds to minutes), so shutting down a replica must not kill in-flight requests:

- **SIGTERM handler:** on receiving a termination signal, the engine stops accepting new requests (returns 503) and **drains** in-flight requests — lets them finish generating before terminating. A `max_drain_time` bounds the wait (e.g. 30–120 s, matching the max output-length SLO).
- **Kubernetes `preStop` hook:** a hook (often a sleep) gives the load balancer time to stop routing new requests to the draining pod before the engine shuts down — avoiding routing requests to a terminating replica.
- **Why it matters:** killing a replica mid-generation loses all its in-flight requests (no checkpointing for inference, File 05 §22) — a bad user experience (dropped responses). Graceful draining ensures in-flight requests complete. The drain time trades shutdown speed for request completion — long enough to finish typical generations, bounded to avoid indefinite waits.

Graceful shutdown is essential for rolling updates (§5) and autoscaling down (§3) — both terminate replicas, and both must drain to avoid dropping requests. The combination (stop accepting + drain + bounded wait + LB de-registration) is the standard pattern.

---

## 2. Health Checks and Readiness

Kubernetes (and load balancers) use health probes to route traffic only to healthy replicas:

- **`/health` endpoint:** returns 200 when the engine is loaded and ready, non-200 otherwise. The load balancer routes only to healthy replicas, routing around unhealthy ones (File 05 §22).
- **Readiness probe** (`/health`): is the replica ready to serve? Returns ready only after model load + KV profiling + CUDA graph capture + warmup (File 07 §19) — minutes for large models.
- **Liveness probe** (`/health`, longer period): is the replica alive (not hung)? A failed liveness probe restarts the replica.
- **Startup probe:** for slow-loading models (70B takes 2–5 min, File 07 §19), a startup probe with a generous timeout prevents the liveness probe from killing a still-loading replica. The startup probe gates the others until startup completes.

The readiness delay (model_load + warmup, minutes) is the key operational fact (File 07 §19): a replica isn't ready for minutes after starting, which shapes autoscaling (§3 — must be proactive). The probes (startup → readiness → liveness) handle the slow-start, ready-detection, and hang-detection respectively, ensuring traffic goes only to ready, alive replicas.

---

## 3. Autoscaling

Scaling replicas to match load (File 11 §10):

- **Horizontal Pod Autoscaler (HPA):** scale on GPU utilization or a custom metric.
- **KEDA (Kubernetes Event-Driven Autoscaler):** scale on queue depth (`num_requests_waiting` via Prometheus) — a better signal for LLM serving (queue depth leads saturation, File 04 §18).
- **The cold-start challenge:** model load + warmup is minutes (§2, File 07 §19), so a replica added at saturation isn't ready for minutes — by which time the spike may have caused SLO violations. So autoscaling must be **proactive**: scale on *leading* indicators (rising queue depth before saturation), maintain warm headroom, or pre-warm replicas. Reactive scaling (scale at saturation) is too late for the multi-minute cold start.
- **Scale-down:** drain (§1) before terminating; scale down gradually to avoid thrashing (scaling up again immediately).

The cold-start lag (minutes) fundamentally shapes LLM autoscaling vs stateless web services (milliseconds start). The strategy: scale proactively (leading indicators, headroom, pre-warming) to have ready capacity before the spike, since you can't add capacity instantly. Capacity planning (File 11 §10.3) sizes for peak with margin precisely because of this — you operate with headroom because you can't scale on demand. KEDA on queue depth (a leading indicator) plus a warm pool is the common pattern.

---

## 4. Cost Optimization

The operational cost goal: minimize $/token (File 11 §16) while meeting SLOs:

- **GPU utilization target 70–85%** (File 11 §29): higher → SLO violations (near the cliff); lower → wasted cost (under-utilized). The band maximizing goodput per dollar.
- **The cost levers** (File 11 §16): quantization (1.5–2× throughput, ~30–50% cost), speculative decoding (~2× decode for decode-bound, File 12), batching (maximize utilization), prefix caching (sharing workloads), right-sized parallelism (File 05 §16), disaggregation + heterogeneous hardware (2–3× at scale, File 15 §3).
- **$/1M tokens** as the metric (File 11 §16.1): `(GPU_cost × num_GPUs) / (throughput × time)`. Commercial APIs mark up 5–15× over this compute cost (File 20 §economics).
- **Multi-model packing** (§6): serve multiple models on shared GPUs to raise utilization.
- **Spot/preemptible instances:** cheaper GPUs that can be reclaimed — usable for fault-tolerant, interruptible workloads (with checkpointing/draining), risky for latency-critical.

Cost optimization is the continuous campaign (File 11 §38): stack the applicable levers (quantization, speculation, prefix caching, right-sized parallelism, disaggregation), operate at 70–85% utilization, validate by $/token, and monitor. The operational side: monitor utilization and $/token, alert on inefficiency (low utilization = wasted cost; near-cliff = SLO risk), and continuously tune as models/traffic evolve. Cost is the bottom line (File 20 §economics), and the ops discipline keeps the deployment at its cost-optimal operating point.

---

## 5. Versioning and Rollout

Updating models or engine versions safely (File 11 §19):

- **Shadow mode:** send the same requests to the old and new versions, compare outputs (correctness, e.g. after a quantization or model change) and latency — without affecting users. Catches regressions before exposure.
- **Canary deployment:** route a small fraction (5%) of traffic to the new version, monitor latency/error rates, roll forward only if metrics hold. Catches regressions on real traffic at limited blast radius.
- **Blue-green deployment:** run the new version alongside the old, switch traffic atomically (zero-downtime model updates), keep the old for fast rollback.
- **Rolling updates:** update replicas one at a time (drain each, §1, then update), keeping the rest serving — zero downtime, gradual rollout.
- **Compatibility:** vLLM/SGLang APIs are backward-compatible within minor versions; breaking changes in major versions — pin versions and test upgrades (engine upgrades can change performance, File 11 §19).

The discipline: every change (model, engine version, config, quantization) is validated (shadow/canary), rolled out gradually (canary → full, or rolling), with fast rollback (blue-green) — minimizing the risk of a regression reaching all users. This is standard SRE practice applied to LLM serving, with the LLM-specific wrinkles (long requests need draining for rolling updates; quality validation for model/quantization changes, File 06 §12; performance validation for engine upgrades, File 11 §19).

---

## 6. Multi-Model Serving

Serving multiple models on shared infrastructure:

- **Time-sharing:** load one model at a time, swap models on demand (load/unload). The model-load time (minutes, File 07 §19) makes swapping expensive — viable for low-frequency model switching, not per-request. Caching loaded model weights in CPU pinned memory speeds reload.
- **Spatial-sharing:** multiple models resident in GPU memory simultaneously (if they fit) — serve all without swapping. Limited by GPU memory (each model's weights).
- **Multi-LoRA** (File 18): the special case of many fine-tunes sharing one base — far more memory-efficient than separate full models (one base + adapters vs N models). The preferred multi-model approach when the models share a base.
- **Orchestration:** Triton Inference Server, Ray Serve (File 17 §13) orchestrate multi-model routing — route each request to the right model/replica, manage loading, scale per-model.

The multi-model decision: if the models share a base (fine-tunes), use multi-LoRA (File 18, hugely more efficient); if they're distinct models, spatial-share (if they fit) or time-share (if not, accepting swap cost) or dedicate replicas per model (isolation). For a platform serving many distinct models, an orchestration layer (Triton/Ray Serve) routes and manages them; for many fine-tunes of one base, multi-LoRA is the efficient path. Multi-model serving consolidates infrastructure (one cluster, many models) but requires managing the loading, routing, and memory across models — the orchestration layer's job (File 17 §13).

---

## 7. Chaos Engineering and Failure Testing

Testing the deployment's resilience (File 05 §22):

- **OOM handling:** send an oversized request — does the engine reject it gracefully (not crash)? OOM is catastrophic (kills in-flight requests, File 03 §17.1) — test that the memory limits (`--max-model-len`, `--gpu-memory-utilization`) prevent it.
- **GPU failure:** disable a GPU mid-run — does the replica fail and the load balancer route around it (File 05 §22)? Test the fault-detection and rerouting.
- **KV cache exhaustion:** saturate with long requests — does the scheduler preempt gracefully (swap/recompute, File 04 §9) or thrash/crash? Test the capacity-cliff behavior (File 04 §18).
- **Slow clients:** a consumer not reading the output stream — does the engine handle backpressure (not block other requests)? Test the streaming/abort handling (File 04 §28).
- **Request storms:** a burst exceeding capacity — does the engine shed load (503) rather than queue infinitely (File 04 §18.3)? Test admission control / load shedding.

Verify: graceful degradation (shed load, don't crash), no stale requests (aborts free resources, File 04 §28), proper error responses (503 for overload, not 500 crashes). Chaos engineering validates that the deployment fails gracefully under the inevitable production stresses (OOM attempts, GPU failures, overload, slow clients) — turning "it works in the happy path" into "it degrades gracefully under stress." The LLM-specific failure modes (OOM from long sequences, KV exhaustion, slow streaming clients, request storms) are the ones to test, and graceful handling (reject/shed/drain, don't crash) is the goal.

---

## 8. Observability

Comprehensive monitoring (File 07 §6, File 11 §28):

- **Metrics (Prometheus):** latency histograms (TTFT/TPOT/E2E percentiles), throughput (tokens/sec, RPS), queue depth (waiting/running/swapped), KV utilization (`gpu_cache_usage_perc`), preemptions, prefix-cache hit rate, GPU utilization (`nvidia-smi`/DCGM). Dashboarded (Grafana).
- **Alerting:** on leading indicators — `gpu_cache_usage_perc > 90%` (approaching cliff), rising preemptions (thrash), rising queue depth (capacity), latency percentiles breaching SLOs (symptom). Alert *before* the cliff (leading indicators) to scale/shed in time (File 04 §18, File 11 §23).
- **Logging:** per-request (prompt/output length, TTFT, throughput) for diagnosing pathological requests; engine debug logs (File 11 §11) for deep issues.
- **Tracing:** distributed tracing (request through gateway → engine → response) for latency attribution; nsys/ncu for kernel-level (File 10 §12).
- **GPU health:** monitor GPU memory (leak detection — stable after warmup, File 11 §11), temperature, power, errors (DCGM).

Observability is the foundation of operations — you can't operate what you can't see. The key signals (KV utilization, queue depth, latency percentiles, preemptions) reveal the deployment's state (capacity, latency, health), and alerting on leading indicators enables proactive response (scale/shed before the cliff). The diagnostic discipline (File 11 §11): from the metrics, distinguish latency (which percentile/metric?) from capacity (queue growing?) problems, map to the cause (File 09 §44's three layers), and act. Observability + alerting + the diagnostic discipline is how the deployment stays healthy and how problems are caught and fixed before they impact users.

---

## 9. Request Timeout and Circuit Breaking

Protecting the system from overload and cascading failures (File 04 §18.3):

- **Client timeout:** set the client's timeout greater than the max expected E2E latency (so legitimate long generations aren't cut off) but bounded (so a hung request doesn't wait forever). For long-output workloads (reasoning, File 11 §15), the timeout must accommodate the long generation.
- **Circuit breaking (Hystrix pattern):** if latency spikes or errors rise (the system is overloaded or failing), the circuit breaker *opens* — rejecting new requests (fast-fail with 503) rather than queuing them into the cliff (File 04 §18) or cascading the failure. This prevents an overloaded system from getting worse (more queuing → more latency → more retries → more load).
- **Load shedding:** when at capacity (`gpu_cache_usage_perc` critical, queue depth high), return 503 with a `Retry-After` header rather than admitting requests that would violate SLOs (File 04 §18.3). Better to reject some requests cleanly than to degrade all.
- **Client-side retry with backoff:** clients retry 503s with exponential backoff — spreading the retry load rather than hammering. The `Retry-After` header guides the backoff.

Circuit breaking and load shedding are the defenses against the capacity cliff (File 04 §18, File 11 §23): past `λ_max`, no scheduling saves you, so the system must *shed load* (reject cleanly) rather than collapse (queue into divergent latency). The circuit breaker fast-fails when overloaded/failing, preventing cascade; load shedding returns 503 when at capacity, protecting the SLO-meeting requests; client backoff spreads the retry load. Together they make the system *degrade gracefully* under overload (reject some, serve the rest within SLO) rather than *collapse* (admit all, violate all SLOs, possibly OOM). This is essential operational protection — the engine's scheduler handles within-capacity load, but the gateway's circuit breaking and load shedding handle *over*-capacity, the regime where the engine alone can't help (File 04 §18.3).

---

## 10. SLO Management

Operating to meet Service Level Objectives:

- **Define the SLOs:** the latency targets (P95 TTFT, P99 TPOT, File 11 §1), availability (uptime), and any throughput targets — from product requirements. The SLOs define "good enough."
- **Measure against them:** the latency percentiles (§8) vs the SLO thresholds — is the deployment meeting them? Track the SLO compliance over time.
- **Goodput as the operating metric** (File 11 §2): the rate of SLO-meeting requests — the real capacity. Operate below the goodput limit (with margin) to meet the SLOs (File 11 §10.3).
- **Error budgets:** allow a small fraction of SLO violations (an error budget) — operate to stay within it. When the budget is exhausted (too many violations), prioritize reliability over features.
- **Capacity for the SLO:** size the fleet (File 11 §10.3) so peak load lands within the goodput (SLO-meeting) capacity, with margin for bursts and cold-start lag (§3).

SLO management ties the operations to the product requirements: the SLOs define the targets, the metrics measure against them, the capacity planning (File 11 §10.3) provisions to meet them, and the alerting/scaling/shedding (§§3, 8, 9) keeps the deployment within them. The goodput (File 11 §2) is the unifying metric — it's the SLO-meeting capacity, and operating below it (with margin) is meeting the SLOs. The ops discipline is to define the SLOs, measure compliance, provision for the goodput with margin, and respond (scale/shed) to stay compliant — turning the abstract "meet the SLOs" into the concrete operations (capacity, monitoring, scaling, shedding) that achieve it.

---

## 11. Security and Multi-Tenancy Operations

Operational security for shared inference (File 04 §37):

- **Authentication and authorization:** API keys (File 07 §2.3), per-tenant access control — at the gateway (File 07 §15).
- **Rate limiting and quotas:** per-tenant request/token limits (File 07 §15) — shape the arrival stream so no tenant overwhelms the engine (File 04 §18) or exceeds its quota.
- **Tenant isolation:** prevent cross-tenant interference (noisy neighbor) — via fair scheduling (File 04 §37), per-tenant rate limits, or dedicated capacity for sensitive tenants. Prevent data leakage (one tenant's prompts/outputs visible to another) — important for prefix caching (a tenant's cached prefix must not serve another's request — the cache key must include tenant/adapter identity, File 03 §25.2, File 08 §22).
- **Input validation:** validate request inputs (size limits, content) at the gateway — prevent oversized requests (OOM, §7) and malicious inputs.
- **PII and data handling:** prompts/outputs may contain sensitive data — handle per the data-governance requirements (logging, retention, the deployment's privacy policy).

Multi-tenant operations layer security and fairness above the engine (at the gateway, File 07 §15): auth, rate limits, quotas, isolation, validation. The engine serves; the gateway governs. The prefix-caching isolation is an LLM-specific concern (a cached prefix must not cross tenants — the cache key includes tenant/adapter, File 03 §25.2, File 08 §22) — a subtle but important security consideration for shared serving. Operating a multi-tenant inference service requires this governance layer (auth, limits, isolation) alongside the engine, ensuring tenants are authenticated, rate-limited, isolated, and their data handled appropriately.

---

## 12. A Worked Capacity-Planning Example

Quantify capacity planning (File 11 §10.3) for a chat service. Suppose: peak load 100 req/s, measured goodput 15 req/s per replica (within SLO, from the latency-throughput benchmark, File 11 §7), cold-start 4 minutes.

- **Base replicas:** `100 / 15 ≈ 6.7 → 7 replicas` to handle peak goodput.
- **Burst margin:** traffic bursts above the mean peak; add ~30% margin → `7 × 1.3 ≈ 9 replicas`. This headroom absorbs bursts without hitting the cliff (File 04 §18).
- **Cold-start headroom:** since scaling takes 4 minutes (§3), you can't add capacity instantly for a spike — so the headroom (the 9 vs 7) must cover the spike magnitude that can occur within the scaling lag. If spikes can be large/fast, more headroom or pre-warmed standby replicas.
- **Autoscaling:** KEDA on queue depth (§3), scaling proactively (before saturation) within the 9-replica provisioned range, with the headroom covering the cold-start lag.
- **Monitoring:** alert on `gpu_cache_usage_perc > 90%` and queue depth (§8) to catch approaching saturation and scale/shed in time.

So the deployment runs ~9 replicas (7 for peak goodput + ~30% burst/cold-start margin), autoscaling proactively within that range, monitoring the leading indicators. The capacity arithmetic (peak / goodput-per-replica × margin) plus the cold-start consideration (headroom for the scaling lag) gives the fleet size. This is the operational realization of File 11 §10.3 — turning the measured goodput into a provisioned fleet that meets peak load within SLO, with margin for the burstiness and the cold-start lag that LLM serving's slow start imposes. Get the goodput from the benchmark, divide peak by it, add margin, and account for the cold-start lag — the capacity-planning recipe.

---

## 13. An On-Call Runbook

A practical runbook for common LLM-serving incidents (synthesizing the diagnostics, File 11 §11):

- **Latency SLO breach (P99 TPOT high):** check `gpu_cache_usage_perc` (high + preemptions → over-admission/thrash → scale up or lower `max_num_seqs`); check for prefill stalls (P99 TPOT but fine P50 → chunked prefill, File 04 §32); check queue depth (high → capacity → scale).
- **Latency SLO breach (TTFT high):** check queue depth (high → near `λ_max` → scale/shed); check prefill cost (long prompts → prefix caching if shared).
- **Throughput collapse:** capacity cliff (File 04 §18) — `gpu_cache_usage_perc` pinned, preemptions high → shed load (circuit breaker, §9), scale up.
- **OOM / replica crash:** check `--gpu-memory-utilization` (too high?), `--max-model-len` (unbounded?), unexpected long sequences. Lower utilization, bound length (File 03 §17.1). The replica restarts (liveness probe, §2); the LB routes around it (§2).
- **Replica unhealthy / not ready:** check startup (model load failing? OOM at load? §2). Check the startup probe timeout (too short for the model's load time?).
- **GPU failure:** the LB routes around the failed replica (§2); investigate/replace the GPU; the fleet degrades (one fewer replica) — scale up if needed.
- **Memory leak (GPU memory growing):** aborted requests not freeing KV (File 04 §28) — check disconnect handling; restart the replica as mitigation.

The runbook maps incidents (latency breach, throughput collapse, OOM, crash, GPU failure, leak) to diagnostics (the metrics, §8) and actions (scale, shed, lower limits, restart, route around). The on-call discipline: from the alert/symptom, consult the metrics to diagnose (File 11 §11), apply the runbook action, and escalate if it's a novel issue. Most LLM-serving incidents reduce to capacity (scale/shed), memory (limits/OOM), or replica health (restart/route-around) — and the runbook handles these, with the deeper diagnostics (File 11 §11, File 10 §12) for novel performance issues. A good runbook, grounded in the metrics and the mechanisms, makes on-call response fast and reliable.

---

## 14. Synthesis

Production operations turn a working inference engine into a reliable, observable, cost-efficient service. The disciplines: **graceful shutdown/draining** (§1) so updates and scale-downs don't drop requests; **health checks** (§2) so traffic goes only to ready replicas (with the slow-start consideration); **proactive autoscaling** (§3) accounting for the multi-minute cold start; **cost optimization** (§4) operating at 70–85% utilization with the stacked levers; **safe rollout** (§5, shadow/canary/blue-green/rolling) validating every change; **multi-model serving** (§6, multi-LoRA for shared bases); **chaos engineering** (§7) verifying graceful degradation; **observability** (§8) with leading-indicator alerting; **circuit breaking/load shedding** (§9) defending the capacity cliff; **SLO management** (§10) operating to the goodput; **security/multi-tenancy** (§11) governing shared serving; **capacity planning** (§12) sizing for peak with margin; and an **on-call runbook** (§13) for incidents. The recurring LLM-specific themes: the **slow cold start** (minutes) shaping autoscaling and capacity (proactive, headroom); the **capacity cliff** (File 04 §18) requiring leading-indicator alerting and load shedding; the **long requests** requiring draining for updates; **OOM** being catastrophic (prevent with limits); and **prefix-caching isolation** for multi-tenancy. Operations is the layer that makes the engine *production-grade* — reliable (graceful degradation, fault tolerance), observable (metrics, alerting), cost-efficient (utilization, levers), and SLO-compliant (capacity, monitoring, shedding). It's the SRE discipline applied to LLM serving, with the LLM-specific wrinkles (cold start, capacity cliff, long requests, OOM, isolation) handled. For an inference engineer, operations is half the job (the other half being the engine/tuning, Files 03–18) — a fast engine that isn't operated well (no draining, reactive scaling, no shedding, poor observability) will have outages, SLO breaches, and wasted cost, while a well-operated deployment (graceful, proactive, observable, defended) delivers reliable, cost-efficient service. The mechanisms (Files 03–18) make it fast; the operations (this file) make it reliable and cost-efficient in production — both are essential, and together they constitute running LLM inference in the real world.

---

## 15. Closing

Running LLM inference in production is the SRE discipline applied to a workload with specific characteristics — slow cold starts, long requests, a sharp capacity cliff, catastrophic OOM, and multi-tenant isolation needs — atop the engine and tuning of Files 03–18. The operational practices (graceful shutdown, health checks, proactive autoscaling, cost optimization, safe rollout, multi-model serving, chaos engineering, observability, circuit breaking, SLO management, security, capacity planning, and on-call runbooks) make the deployment reliable, observable, cost-efficient, and SLO-compliant. The LLM-specific considerations (cold-start lag, capacity cliff, long-request draining, OOM prevention, prefix-cache isolation) distinguish LLM serving operations from stateless web services, and handling them well (proactive scaling, leading-indicator alerting, load shedding, draining, memory limits, tenant-aware caching) is what separates a fragile deployment from a robust one. Operations is the bridge from "the engine works" to "the service is reliable and economical at scale" — the final practical layer atop the mechanisms and tuning. With operations covered, File 20 turns to the business and ecosystem context — the market, the economics, the providers, and the competitive dynamics — in which all this engineering operates, completing the database's arc from the roofline physics (File 01) through the systems internals (Files 03–11), the advanced techniques (File 15), the hardware (File 16), the frameworks (File 17), fine-tuned serving (File 18), operations (this file), and finally the business context (File 20) that motivates and funds it all.

---

## 16. Deployment Infrastructure in Practice

The concrete infrastructure for deploying the engines (File 07 §8):

- **Containerization:** vLLM/SGLang ship Docker images; mount the model, expose the port, pass the engine flags as args. The container packages the engine, dependencies, and CUDA runtime.
- **Kubernetes:** deploy with the NVIDIA GPU Operator (GPU scheduling/drivers), a Deployment (the replicas) + Service (the load balancer), readiness/liveness/startup probes (§2), an HPA or KEDA (§3), resource requests (GPU count, memory), and the `preStop` hook for draining (§1). A Helm chart parameterizes this.
- **GPU scheduling:** request GPUs (`nvidia.com/gpu: N`) per replica matching the TP degree (File 05); node selectors/affinity to place replicas on GPU nodes with the right hardware (NVLink for TP, §File 05 §8).
- **Persistent model storage:** models are large (GBs–TBs); store on fast persistent storage (or a model cache) and mount/load at startup (the load time, §2). A model cache (CPU/local SSD) speeds reload across restarts.
- **Networking:** for multi-node (File 05 §11), configure the InfiniBand/RoCE fabric (NCCL over IB, GPUDirect RDMA) — critical for multi-node collective performance (File 05 §15).

The infrastructure realizes the operational practices: the Deployment + probes + HPA/KEDA implement the replicas, health, and autoscaling; the `preStop` hook implements draining; the GPU Operator and resource requests handle GPU scheduling; the fabric config enables multi-node. A production deployment is this Kubernetes (or equivalent) setup — containerized engines, probed and autoscaled, GPU-scheduled, on fast storage and (for multi-node) a configured fabric — operationalizing the engine and the practices (§§1–13) into a running, managed service. The Helm chart / IaC (infrastructure-as-code) makes it reproducible and version-controlled, completing the operational picture: the engine (Files 03–18), the practices (§§1–13), and the infrastructure (this section) together constitute a production LLM-inference deployment — the full stack from the kernel to the Kubernetes manifest, all in service of reliably and economically serving the model to users within their latency and quality expectations, at a cost the business can sustain and grow on.


