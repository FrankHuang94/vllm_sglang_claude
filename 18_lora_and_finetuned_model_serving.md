# LoRA, PEFT, and Serving Fine-Tuned Models

> **Standard reference file.** LoRA enables serving many fine-tuned model variants from one base model — the foundation of multi-tenant fine-tuned serving. This file covers LoRA fundamentals, the vLLM and SGLang multi-LoRA implementations (Punica/SGMV kernels), adapter caching, multi-LoRA workload patterns, and QLoRA. Prerequisites: File 06 §14 (vLLM LoRA layer), File 04 §21 (LoRA scheduling), File 07 §7 (LoRA serving API).

---

## 1. LoRA Fundamentals

**LoRA (Low-Rank Adaptation, Hu et al. 2022, arXiv 2106.09685)** fine-tunes a model by learning a low-rank *update* to its weights, rather than updating the full weights. For a weight matrix `W_0 ∈ ℝ^{d×k}`, LoRA learns a low-rank decomposition `ΔW = B·A` where `B ∈ ℝ^{d×r}`, `A ∈ ℝ^{r×k}`, and the rank `r ≪ min(d, k)`:

```
W = W_0 + ΔW = W_0 + B·A
output = x·W = x·W_0 + x·(B·A) = base_output + lora_output
```

The fine-tuned model's output is the base model's output plus a low-rank correction. During training, only `A` and `B` are learned (the base `W_0` is frozen); during inference, the adapter (`A`, `B`) modifies the base.

### 1.1 The parameter efficiency

The rank `r` is typically 8–64. The adapter parameters (`r·d + r·k`) are a tiny fraction of the full matrix (`d·k`):

```
parameter ratio = (r·d + r·k) / (d·k) = r·(d+k)/(d·k) = r·(1/k + 1/d)
For d = k = 4096, r = 16:  16 · (2/4096) = 0.0078 = 0.78%
```

So a LoRA adapter is <1% of the base model's parameters — a 70B model's adapter is a few hundred MB (vs 140 GB base). This efficiency is why LoRA dominates fine-tuning: cheap to train (few parameters), cheap to store (small adapter), and — crucially for serving — **many adapters share one base model**.

### 1.2 Why LoRA matters for serving

The serving payoff: instead of hosting N separate fine-tuned full models (N × base size), host *one* base model + N small adapters. For a SaaS with 1,000 customer-specific fine-tunes of a 7B model: 1,000 full models = 14 TB (impossible); 1 base + 1,000 adapters = 14 GB + 1,000 × ~64 MB = ~78 GB (feasible). LoRA turns "N fine-tuned models" into "1 base + N small adapters," enabling **multi-tenant fine-tuned serving** — the key use case (§5). The challenge: serving requests for *different* adapters efficiently in one engine (batching across adapters), which the Punica/SGMV kernels solve (§2).

---

## 2. vLLM Multi-LoRA Serving

vLLM serves multiple LoRA adapters on one base model (File 06 §14, File 07 §7).

### 2.1 The architecture

- **`LoRAManager`** manages the adapter registry — adapters loaded on demand (from HuggingFace Hub or local path), tracked by name.
- **`LRUCacheWorkerLoRAManager`** handles LRU eviction of adapters from GPU memory when the cache is full.
- **`LoRARequest`** in the API specifies `lora_name` and `lora_path` per request — routing the request through that adapter.
- Adapters stored separately from the base model; dynamic switching per request.

### 2.2 The Punica kernels

The core challenge: a single batch may contain requests using *different* adapters. Naively, you'd compute `x·(B·A)` separately per adapter (or materialize the full `ΔW` per adapter — huge). **Punica** (Chen et al. 2023, arXiv 2310.18547) provides CUDA kernels for **batched LoRA with mixed adapters in one batch**:

- **`bgmv` (batched grouped GEMV):** applies each request's adapter in one batched operation, handling a batch where different requests use different adapters.
- **`bgmv_shrink` (`A·x`, the shrink to rank r)** then **`bgmv_expand` (`B·(A·x)`, the expand back)**: the low-rank computation done in two batched grouped GEMVs, avoiding materializing the full `ΔW` per request.

So a batch with requests for adapters 1, 2, 3 runs the base matmul once (shared) plus the Punica kernels applying each request's adapter — efficiently, in one batched pass. This is what makes mixed-adapter batching practical (the alternative — separate passes per adapter — would destroy batching efficiency).

### 2.3 Memory and limits

- **Per-adapter memory:** `rank × d_model × num_layers × bytes` ≈ for r=16, d=4096, 32 layers, BF16: ~64 MB. Small relative to the base.
- **`--max-loras`** limits the number of adapters resident in GPU memory simultaneously. LRU eviction when the cache is full; adapters kept in CPU pinned memory for fast reload.
- **`--max-lora-rank`** sets the maximum supported adapter rank.
- The scheduler treats adapter slots as a second resource (File 04 §21) — admitting a request needing a non-resident adapter may require evicting/loading one.

---

## 3. SGLang LoRA Serving

SGLang's LoRA support mirrors vLLM's architecture:

- **`LoRAManager`** with an adapter registry, on-demand loading, LRU eviction.
- **SGMV kernels** (Segmented Gather Matrix-Vector, similar to Punica's BGMV) handle batched mixed-adapter LoRA — the same problem (apply different adapters in one batch) with an analogous kernel solution.
- **`--enable-lora`** flag, `lora_paths` configuration.
- Integration with RadixAttention: requests using different adapters must not share prefix-cache nodes (the adapter is part of the cache key, File 08 §22) — adapter-A's KV differs from adapter-B's even for the same tokens.

Both engines solve the same core problem (efficient mixed-adapter batching) with analogous kernels (Punica BGMV / SGMV) and management (registry + LRU cache). The mechanisms are nearly identical because the problem (serve many adapters on one base, batch across adapters) is the same.

---

## 4. Adapter Caching

Adapters are managed like a cache (the analog of KV caching, File 03 §6):

- **Active adapters in GPU HBM:** the adapters currently in use are resident in GPU memory (small, ~64 MB each).
- **LRU eviction:** when the adapter cache (`--max-loras` slots) is full and a new adapter is needed, evict the least-recently-used adapter.
- **CPU pinned memory backing:** evicted adapters are kept in CPU pinned memory for fast reload (faster than re-fetching from disk/Hub).
- **Pre-loading:** popular adapters can be pre-loaded at startup to avoid first-request latency.
- **Hit rate:** the adapter cache has a hit rate like any cache — popular adapters stay hot (always resident), the long tail is evicted and reloaded. For power-law adapter popularity (a few hot adapters, many cold), the hot adapters stay cached and the cold ones reload on demand.

The adapter cache is a second cache alongside the KV cache — both LRU-managed, both with GPU/CPU tiering. The adapter cache's working set (the hot adapters) should fit in `--max-loras` slots for a high hit rate (low reload overhead). For a workload with many distinct adapters exceeding `--max-loras`, the cache thrashes (frequent reloads) — the cure is more adapter slots (more GPU memory for adapters) or adapter-aware routing (route requests for the same adapter to the same replica, concentrating each adapter's usage, the LoRA analog of prefix-cache-aware routing, File 08 §35).

---

## 5. Multi-LoRA Workload Patterns

The dominant LoRA serving use case is **SaaS multi-tenant fine-tuned serving**:

- **One base model, N customer adapters:** a SaaS offers fine-tuned models to N customers, each with their own LoRA adapter on a shared base. A request is routed to the customer's adapter (customer ID → `lora_name`).
- **Power-law popularity:** adapter request frequency follows a power law — a few popular adapters (large customers) are always hot; the long tail (small customers) is infrequent. The hot adapters stay cached; the long tail is evicted/reloaded (§4).
- **Routing:** route requests to the correct adapter (customer ID → adapter); adapter-aware routing across replicas concentrates each adapter's usage (better cache hit rate, §4).
- **Economics:** hosting N full models is infeasible (N × base); hosting 1 base + N adapters is feasible and cheap (§1.2). LoRA is what makes per-customer fine-tuned serving economical — the core enabler of fine-tuned-model SaaS (File 20).

Other patterns: A/B testing (serve two adapter versions, compare), task-specific adapters (different adapters for different tasks on the same base), and personalization (per-user adapters). All exploit the same mechanism — many small adapters on one base, batched across adapters (Punica/SGMV). The multi-LoRA capability transforms fine-tuned serving from "host N models" (expensive) to "host 1 base + N adapters" (cheap), enabling business models (per-customer fine-tunes, File 20) that full-model serving couldn't support.

---

## 6. QLoRA and 4-bit Base + LoRA

**QLoRA** combines a quantized base model with LoRA adapters:

- **The setup:** the base model in 4-bit (NF4/GPTQ, File 06 §3), the LoRA adapters in FP16. This is memory-efficient — the base is 4× smaller (4-bit), and the adapters are small (FP16, but tiny).
- **The inference path:** load the base model quantized; compute the base output (dequant + matmul, or quantized matmul, File 10 §16.1); compute the LoRA update (`x·B·A` in FP16); add them. The LoRA must be computed in higher precision (FP16) and added to the quantized base's output — so it's not perfectly fused (the base is quantized, the LoRA is FP16, combined at the output).
- **The overhead:** ~5–10% vs a non-quantized base + LoRA, from the mixed-precision combination (the LoRA path in FP16 alongside the quantized base). But the memory saving (4-bit base) is large.
- **The use case:** serving fine-tuned models when memory is constrained — a 4-bit base + LoRA fits where an FP16 base + LoRA wouldn't. vLLM supports QLoRA-style serving (quantized base, FP16 LoRA).

QLoRA extends multi-LoRA to memory-constrained settings: the quantized base (small) + many adapters (tiny) fits in less memory, enabling multi-tenant fine-tuned serving on smaller GPUs. It combines two efficiency techniques (quantization for the base, LoRA for the adapters) — the quantized base shrinks the dominant memory (the base), and LoRA enables the many adapters, together making memory-efficient multi-tenant fine-tuned serving possible.

---

## 7. The Punica Kernel Mechanics, Deeper

The Punica BGMV kernel (§2.2) solves a specific batching problem worth detailing, because it's what makes mixed-adapter serving efficient. Consider a decode batch of 4 sequences using adapters A1, A1, A2, A3 respectively. The base matmul `x·W_0` is shared (all sequences use the same base) — computed once for the batch. The LoRA updates differ per sequence:

```
seq 0, 1: output += x·(B_A1 · A_A1)
seq 2:    output += x·(B_A2 · A_A2)
seq 3:    output += x·(B_A3 · A_A3)
```

The naive approach computes each adapter's update separately (3 separate matmuls for 3 adapters) — losing the batching across sequences. Punica's **BGMV** instead does this as *batched grouped* operations: `bgmv_shrink` computes `A_i · x_i` for all sequences `i` in one kernel (each sequence uses its own adapter's `A`, gathered by the adapter index), producing the rank-`r` intermediate per sequence; `bgmv_expand` computes `B_i · (intermediate_i)` for all sequences in one kernel, producing the LoRA output per sequence. The "grouped" aspect: the kernel handles a batch where different sequences index different adapters (a gather by adapter index), computing all in one pass. This is the same idea as MoE's grouped GEMM (File 02 §21.3, File 10 §25) — handle a batch where different elements use different weights, in one batched kernel — applied to LoRA adapters. The result: mixed-adapter batches run nearly as efficiently as single-adapter batches (the base matmul shared, the small LoRA updates batched via BGMV), which is what makes serving hundreds of adapters in one engine practical. Without Punica/BGMV, mixed-adapter batching would require per-adapter passes, destroying the batching efficiency that continuous batching (File 04) provides.

### 7.1 Why low-rank makes it cheap

The LoRA updates are cheap because of the low rank: `bgmv_shrink` is a `[r × d]·[d]` GEMV (small, r≈16), `bgmv_expand` is `[d × r]·[r]` (small). The intermediate is rank-`r` (tiny). So the per-adapter LoRA computation is small relative to the base matmul (`[d × k]·[k]`) — the LoRA adds only `~2·r/k` fraction to the matmul cost (for r=16, k=4096: ~0.78%). This is why multi-LoRA is nearly free in compute (the LoRA updates are tiny) — the cost is the memory (adapters resident) and the slight kernel overhead (BGMV), not the compute. The low-rank structure (LoRA's defining feature, §1) is what makes both the storage (small adapters) and the compute (small updates) cheap, enabling many adapters per engine at minimal cost beyond the base model's.

---

## 8. A Worked Multi-LoRA Memory Budget

Quantify multi-LoRA serving for a 7B base model with rank-16 adapters on an 80 GB GPU.

- **Base model (FP16):** 14 GB. (Or 3.5 GB in 4-bit QLoRA, §6.)
- **Per-adapter memory:** rank 16, d_model 4096, ~32 layers, applied to several matrices (q,k,v,o,gate,up,down ≈ 7 matrices) → roughly `16 · 4096 · 2 (B,A) · 32 · 7 · 2 bytes ≈ 117 MB` per adapter (varies by which matrices the adapter targets; often just attention → less).
- **Adapter cache:** with `--max-loras 32` (32 resident adapters): `32 · ~64–117 MB ≈ 2–4 GB`.
- **KV cache:** the remainder (80 − 14 − 4 ≈ 62 GB) for KV — plenty for the 7B base.

So a 7B base + 32 resident adapters uses ~18 GB, leaving ~62 GB for KV — easily fitting. The adapter cache (2–4 GB for 32 adapters) is small relative to the base and KV. For more adapters (say 1,000 total, power-law popularity, §5): 32 resident (hot) + 968 in CPU/disk (cold, reloaded on demand). The GPU holds the base + hot adapters + KV; the cold adapters tier to CPU (fast reload). This budget shows multi-LoRA is memory-cheap: hundreds-to-thousands of adapters served with only the hot ones (tens) resident, each tiny, the base shared. The economics (§1.2): this 7B + 1,000 adapters on one GPU replaces 1,000 separate 7B models (14 TB) — a ~700× memory reduction, the enabler of fine-tuned-model SaaS (§5, File 20).

---

## 9. LoRA and Scheduling

Multi-LoRA adds a scheduling dimension (File 04 §21): the scheduler must consider which adapters are resident when forming a batch.

- **Adapter slots as a resource:** beyond KV blocks, the scheduler tracks the `--max-loras` adapter slots. Admitting a request needing a non-resident adapter requires evicting an LRU adapter and loading the new one (a cost).
- **Batching for resident adapters:** the scheduler prefers to batch requests for already-resident adapters (no load cost), admitting new-adapter requests when adapter-cache room exists. This is a two-resource admission problem (KV blocks AND adapter slots).
- **Adapter-load overhead:** loading an adapter (from CPU/disk to GPU) adds latency; the scheduler weighs this against admission. Frequent adapter loads (thrashing the adapter cache) hurt — the cure is more adapter slots or adapter-aware routing (§4).
- **Mixed-adapter batches:** the scheduler forms batches with mixed adapters (Punica handles them, §7); there's no requirement that a batch use one adapter. So the scheduler batches freely across resident adapters.

The scheduling interaction is a mild generalization of the single-resource (KV) admission control (File 04 §27) to two resources (KV + adapter slots). For most multi-LoRA workloads, the adapter cache (with adequate `--max-loras` and adapter-aware routing) keeps adapters resident, so the adapter-slot resource isn't the binding constraint — KV is, as usual. But for workloads with many distinct adapters and tight `--max-loras`, the adapter slots can bind (frequent reloads), and the scheduler/routing must manage it. Adapter-aware routing (route each adapter's requests to the same replica, §4) is the cluster-level lever, concentrating each adapter's usage so each replica's adapter working set fits its slots — the LoRA analog of prefix-cache-aware routing (File 08 §35).

---

## 10. Performance and Trade-offs

Multi-LoRA's performance characteristics:

- **Compute overhead:** ~0.78% per the low-rank analysis (§7.1) — the LoRA updates are tiny relative to the base matmul. Multi-LoRA is nearly free in compute.
- **Memory overhead:** the resident adapters (tens of MB to a few GB for the cache) — small relative to the base + KV (§8).
- **Adapter-load latency:** loading a cold adapter (CPU→GPU) adds latency to the first request using it — mitigated by caching/pre-loading (§4).
- **Mixed-adapter batching:** Punica/SGMV (§7) keep mixed-adapter batches nearly as efficient as single-adapter — so serving many adapters doesn't sacrifice batching throughput.
- **Quality:** LoRA adapters are fine-tuned models — their quality is the fine-tuning's, served faithfully (the adapter is applied exactly). QLoRA (§6) adds the quantized-base quality consideration (the base is quantized, validate, File 06 §12).

The net: multi-LoRA serves many fine-tuned variants at near the cost of the base model alone — tiny compute overhead, small memory overhead (resident adapters), efficient mixed-adapter batching. This is why it's the dominant fine-tuned-serving approach: the alternative (separate full models) is vastly more expensive (memory, no sharing), while multi-LoRA shares the base and adds only tiny per-adapter cost. For any deployment serving multiple fine-tuned variants (per-customer, per-task, per-user), multi-LoRA is the economical choice, and the engines' support (Punica/SGMV kernels, adapter caching, the scheduling integration) makes it production-ready.

---

## 11. Synthesis

LoRA (low-rank weight updates) enables serving many fine-tuned model variants from one base model — the foundation of multi-tenant fine-tuned serving (§5). The key insight: a fine-tune is a tiny low-rank adapter (<1% of the base, §1), so N fine-tunes = 1 base + N small adapters (not N full models), a massive memory reduction (§8). The serving challenge — batching requests for *different* adapters in one engine — is solved by the Punica (vLLM) / SGMV (SGLang) kernels (§§2, 7), which apply each request's adapter in one batched grouped operation (like MoE's grouped GEMM, File 10 §25), keeping mixed-adapter batches nearly as efficient as single-adapter. Adapters are cached (LRU, GPU/CPU tiered, §4) like the KV cache, with adapter-aware routing (§9) concentrating usage. QLoRA (§6) extends this to memory-constrained settings (quantized base + FP16 adapters). The result: multi-LoRA serves hundreds-to-thousands of fine-tuned variants at near the base model's cost (tiny per-adapter compute and memory, §10) — the enabler of fine-tuned-model SaaS (per-customer fine-tunes, §5, File 20). LoRA exemplifies a recurring theme: sharing a common resource (the base model) across many variants (adapters), batching across them efficiently (Punica/SGMV) — the same sharing-and-batching pattern as prefix caching (share the prefix KV) and MoE (share the routing). The base model is shared like a prefix; the adapters are the per-request variation; the kernels batch across the variation. Multi-LoRA is the fine-tuned-serving instance of the database's broader theme — share what's common, batch across what varies, and serve the result efficiently — applied to fine-tuned model variants, turning the expensive "many models" into the cheap "one base, many adapters."

---

## 12. LoRA Serving vs Merged Models vs Full Fine-Tunes

There are three ways to serve a fine-tuned model, with different trade-offs:

- **LoRA serving (adapter applied at inference):** the base + adapter, with the adapter applied dynamically (§§1–2). Enables multi-tenant serving (many adapters, one base, §5) and dynamic adapter switching. Slight per-adapter overhead (§10). The choice for *multi-tenant* fine-tuned serving.
- **Merged model (adapter merged into the base):** `W = W_0 + B·A` computed once, producing a merged full model. No per-adapter overhead (it's just a model), but it's a *separate full model* (no base sharing) — so for *one* fine-tune served at scale, merging avoids the LoRA overhead, but for *many* fine-tunes, merging loses the base-sharing benefit (N merged models = N × base). Merge for a single high-volume fine-tune; keep adapters separate for many fine-tunes.
- **Full fine-tune:** the model fully fine-tuned (all weights updated) — a separate full model, no sharing, maximum fine-tuning capacity (not limited to low rank). For when LoRA's low-rank capacity is insufficient (rare) or a single model is served at scale. Most fine-tuning uses LoRA (sufficient capacity, far cheaper); full fine-tunes are for cases needing maximum adaptation.

The decision: **many fine-tunes, multi-tenant → LoRA serving** (base sharing, the §5 SaaS case); **one high-volume fine-tune → merge** (no per-adapter overhead, but a separate model); **maximum adaptation capacity → full fine-tune** (rare, separate model). For the dominant use case (multi-tenant fine-tuned serving, many adapters), LoRA serving is the clear choice — the base sharing is the whole point. Merging and full fine-tunes apply when there's a single fine-tune served at scale (where the multi-adapter benefit doesn't apply) or when low-rank capacity is insufficient.

---

## 13. A Worked SaaS Multi-Tenant Example

Quantify the SaaS pattern (§5) end to end. A SaaS offers fine-tuned 7B models to 500 customers, each with a rank-16 adapter, on H100 GPUs.

- **Without LoRA (separate models):** 500 × 14 GB (7B FP16) = 7 TB — needs ~88 H100s just to hold the models (impossible/absurd for the workload).
- **With multi-LoRA:** 1 base (14 GB) + 500 adapters (500 × ~100 MB = 50 GB total, but only the hot ones resident, §8). On one H100: base (14 GB) + 32 hot adapters (~3 GB) + KV (~60 GB). The 500 adapters tier (hot in GPU, cold in CPU/disk, §4). **One H100 serves all 500 customers' fine-tuned models** (with adapter caching for the long tail) — vs 88 H100s for separate models. An ~88× hardware reduction.
- **The economics:** the SaaS can offer per-customer fine-tuned models economically (one base serves all) — a business model (custom fine-tunes per customer) that separate-model serving couldn't support (88× the hardware). Multi-LoRA is what makes fine-tuned-model SaaS viable (File 20).
- **Operations:** route by customer ID → adapter (§5), adapter-aware routing across replicas (§9) to keep each replica's adapter working set in its slots, pre-load popular customers' adapters (§4). The hot customers (high volume) are always resident; the long tail reloads on demand (fast, from CPU).

This worked example shows the transformative economics: multi-LoRA turns "500 fine-tuned models" from an 88-GPU impossibility into a 1-GPU (+ caching) reality, enabling the per-customer-fine-tune business model. It's the concrete payoff of LoRA's base-sharing (§1.2) — the reason multi-LoRA serving is a key capability for any platform offering customized models (File 20 §SaaS). The engineering (Punica/SGMV kernels §7, adapter caching §4, adapter-aware routing §9) makes it production-ready, and the economics (88× hardware reduction) make it compelling.

---

## 14. Closing

LoRA and multi-LoRA serving turn fine-tuned-model serving from "host N models" (N × base, infeasible at scale) into "host 1 base + N small adapters" (feasible, cheap) — the foundation of multi-tenant fine-tuned serving (§5) and fine-tuned-model SaaS (§13, File 20). The mechanism: low-rank adapters (<1% of the base, §1) applied at inference, batched across adapters by the Punica/SGMV kernels (§§2, 7) so mixed-adapter batches run efficiently, with adapters cached (LRU, tiered, §4) and routed (adapter-aware, §9). The compute and memory overhead are tiny (§§7.1, 8, 10), so many adapters are served at near the base model's cost. QLoRA (§6) extends this to memory-constrained settings. The economics (§13) are transformative — an ~88× hardware reduction for a 500-customer SaaS — enabling business models that full-model serving couldn't support. Multi-LoRA exemplifies the database's sharing-and-batching theme (share the base, batch across adapters), the fine-tuned-serving counterpart to prefix caching (share the prefix) and MoE (share the routing). For any deployment serving multiple fine-tuned variants, multi-LoRA is the economical choice, production-ready in both engines. With LoRA serving covered, File 19 turns to production operations (reliability, observability, cost optimization) and File 20 to the business and ecosystem context — completing the practical and business picture of LLM inference engineering, of which efficient fine-tuned serving (this file) is an important capability.

---

## 15. PEFT Beyond LoRA

LoRA is the dominant Parameter-Efficient Fine-Tuning (PEFT) method, but the family is broader, and some variants affect serving:

- **LoRA variants:** **DoRA** (weight-decomposed LoRA), **rsLoRA** (rank-stabilized), **LoRA+** (different learning rates for A and B) — refinements that improve fine-tuning quality while keeping the same low-rank serving structure (the adapter is still `B·A`, served identically). Serving treats them like LoRA.
- **Prefix/prompt tuning:** learn soft prompt embeddings (virtual tokens) prepended to the input, rather than weight updates. Served by prepending the learned embeddings — a different mechanism (no weight adapter), with its own (small) per-variant cost.
- **(IA)³, adapters (Houlsby):** other PEFT methods inserting small trainable modules. Less common for serving than LoRA.

For serving, **LoRA dominates** because its low-rank weight-update structure (§1) enables the efficient multi-adapter serving (Punica/SGMV, §7) and base sharing (§1.2) that the SaaS use case needs. The LoRA variants (DoRA etc.) share this structure (served like LoRA). Other PEFT methods (prompt tuning, adapters) are less common in multi-tenant serving (LoRA's combination of quality, efficiency, and serving-friendliness won out). So for serving fine-tuned models, LoRA (and its variants) is the method to support, and the engines' LoRA infrastructure (this file) covers it. The PEFT landscape is broad, but LoRA's serving advantages made it the standard — which is why this file (and the engines) focus on it. An inference engineer serving fine-tuned models will overwhelmingly serve LoRA adapters, and the multi-LoRA capability (Punica/SGMV, adapter caching, adapter-aware routing) is the relevant infrastructure — the practical toolkit for the increasingly common requirement of serving many customized model variants from shared base models, economically and at scale, which the multi-tenant fine-tuned-serving market increasingly demands as per-customer and per-use-case customization becomes a standard product feature across the AI products industry, from enterprise assistants to consumer personalization to domain-specialized professional tools and well beyond into every customizable AI offering.

