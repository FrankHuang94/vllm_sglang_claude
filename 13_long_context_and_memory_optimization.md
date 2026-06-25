# Long Context Inference — Algorithms, Memory Optimization, and Efficient Attention

> **Standard reference file.** Long context (32K–1M+ tokens) is a memory problem first and a compute problem second. This file covers the challenges, StreamingLLM/attention sinks, KV offloading, sparse attention and KV eviction (H2O, SnapKV), GQA/MLA memory reduction, and context-window extension (YaRN, LongRoPE). Prerequisites: File 01 §4 (KV cache), File 02 §§2, 5, 7 (attention variants, RoPE, KV quantization), File 03 (paged KV).

---

## 1. The Long-Context Challenge

Long context strains inference along two axes (File 01 §4):

### 1.1 KV cache grows linearly

The KV cache is `2 · num_layers · num_kv_heads · head_dim · bytes` per token (File 01 §4). At long context, this dominates memory:
- LLaMA-3 70B (GQA-8): 320 KB/token → at **128K context: 40 GB** for a *single* request's KV — half an 80 GB GPU consumed by one long-context request.
- This caps concurrency: a GPU that serves 100+ short-context requests serves only a handful of 128K-context ones (File 02 §26.1).

So long-context serving is **KV-memory-bound** — the binding constraint is fitting the KV, and the levers (this file) all target KV size or placement.

### 1.2 Attention compute is O(n²) in prefill

Prefill attention is `O(n²·d)` (File 02 §1.1) — at 128K tokens, the attention compute is enormous (and the naive `n×n` score matrix would be ~32 GB — impossible to materialize, which is why FlashAttention's linear-memory tiling is essential, File 10 §13.1). Chunked prefill (File 04 §7) is also essential to avoid a single 128K prefill monopolizing the engine for seconds. Decode attention reads the full KV each step (File 02 §6.4) — at 128K context, ~40 GB of KV read *per decode step per request*, making long-context decode slow (bandwidth-bound on the huge KV read).

### 1.3 The control: `--max-model-len`

The engine bounds context with `--max-model-len`, which sets the maximum sequence length and thus the per-request KV reservation and block-table size (File 03 §22). Setting it correctly (to what you'll actually serve) is important — over-setting reserves capacity and risks OOM on a max-length request. Long context is enabled by setting `--max-model-len` high *and* applying the memory-reduction techniques below to make the resulting KV fit.

---

## 2. StreamingLLM and Attention Sinks

For *unbounded* streaming (chat that never ends, processing an endless document), the KV cache cannot grow forever. **StreamingLLM** (Xiao et al. 2023, arXiv 2309.17453) enables fixed-memory streaming.

### 2.1 The attention-sink observation

Trained transformers exhibit **attention sinks** (File 02 §29): the first few tokens receive disproportionate attention from all later tokens, regardless of content — because softmax must sum to 1 and the model learns to "dump" excess attention mass on the initial tokens. Naively keeping only a sliding window of recent tokens (dropping the early ones) *destroys quality* — the model's attention distribution collapses without its accustomed sink.

### 2.2 The fix

StreamingLLM retains the **attention-sink tokens (the first ~4) plus a sliding window of recent W tokens**: `KV cache = {sink tokens} ∪ {recent W tokens}`. This bounds the KV at `4 + W` tokens regardless of stream length, enabling streaming at *arbitrary* length with fixed memory, at slightly degraded quality vs full attention (the model loses access to the dropped middle, but retains the sinks it needs and the recent context). vLLM exposes this via sliding-window configuration with sink retention (File 02 §29, File 03 §18 — the block manager ages out blocks outside the window but pins the sink blocks). StreamingLLM is the technique for *unbounded* streams; for *bounded but long* context (a 128K document you want fully attended), you need the full KV (and the memory-reduction techniques §§5–7).

---

## 3. KV Cache Offloading

When the KV doesn't fit in GPU HBM, offload to slower tiers (File 03 §8, §28).

### 3.1 CPU swap (vLLM)

vLLM swaps LRU KV blocks to CPU RAM over PCIe when GPU memory is pressured (File 03 §8). Mechanics: evict LRU blocks to pinned CPU memory; swap back in when needed. Cost: PCIe-bandwidth-limited (~64 GB/s PCIe 5.0) — a 5 MB block (LLaMA-3 70B, 16 tokens) is ~78 µs (File 03 §8.2). Acceptable for *occasional* eviction (transient pressure), catastrophic if *frequent* (thrash). The swap-vs-recompute choice (File 03 §9) applies: for long sequences, the already-computed KV is expensive to recompute, so swapping can win; for short, recompute. `--kv-cache-dtype` affects swap size (FP8 KV halves it).

### 3.2 Tiered KV and external stores

Beyond CPU swap, KV can tier to SSD or a distributed store (File 03 §28, LMCache), and prefix-cached KV can survive across requests/machines. **FlexGen** (Sheng et al., arXiv 2303.06865) explored systematic GPU+CPU+SSD offloading for single-GPU large-model inference — very slow (SSD bandwidth), for research/evaluation, not production serving. The general principle: a memory hierarchy (HBM → CPU → SSD → network) with the hottest KV in HBM, tiering down the colder KV — trading access latency for capacity (File 03 §28). For long context, tiering can hold a context that exceeds HBM, at the cost of swap-in latency when the offloaded KV is needed.

---

## 4. Offloaded Model Weights (Beyond GPU Memory)

For models too large even for the GPU, weights can be offloaded to CPU (compute on CPU or stream weights to GPU) — but CPU FLOPS ≪ GPU, so this is very slow, for research/evaluation only (not production vLLM/SGLang serving). FlexGen (§3.2) systematized GPU+CPU+SSD offloading for single-GPU inference of huge models, accepting low throughput for the ability to run a model that doesn't fit. Production serving instead uses quantization (fit in less memory, File 06) and distribution (TP/PP across GPUs, File 05) rather than weight offloading. Weight offloading is the "run it at all, slowly" option for memory-constrained research, distinct from the "serve it fast" production path.

---

## 5. Sparse Attention and KV Eviction

Instead of storing the full KV, *prune* it — keep only the important KV entries, reducing both memory and the decode-time KV read.

### 5.1 Sparse attention patterns (historical)

Early efficient-attention work (BigBird, Zaheer et al. 2020; Longformer) combined *local window* + *global* + *random* attention to reduce the `O(n²)` cost to near-linear for encoder models. These are **not commonly used in modern decoder-only LLMs** — RoPE + full attention is standard, and FlashAttention made full attention efficient enough. They're relevant context but not the production path for vLLM/SGLang.

### 5.2 KV eviction: H2O and SnapKV

For decoder-only LLMs, the modern approach is **dynamic KV eviction** — drop KV entries with low attention scores during decoding:

- **H2O (Heavy-Hitter Oracle, Zhang et al. 2023, arXiv 2306.14048):** observes that a small set of "heavy hitter" tokens (high cumulative attention score) dominate attention. H2O evicts the non-heavy-hitter KV entries, keeping only the heavy hitters + recent tokens. Reports ~20× context extension at ~5% quality cost. The eviction is online (during decoding), based on accumulated attention scores.
- **SnapKV (arXiv 2404.14469):** cluster-based KV eviction — identifies and keeps the important KV positions (via attention patterns over a recent window), compressing the KV cache for long prompts.

These trade quality for memory: evicting KV loses information (the dropped tokens can't be attended to), but if the dropped tokens were low-attention, the quality cost is small. The eviction policy (cumulative attention score, recency, or combined) determines the quality/memory trade. KV eviction is a research-active area; production support varies. The principle: not all KV is equally important, so keeping only the important entries reduces memory (and decode KV-read bandwidth) at a quality cost proportional to what's dropped.

---

## 6. GQA and MLA: Architectural KV Reduction

The most effective KV reduction is *architectural* — choosing a model with a KV-efficient attention variant (File 02 §2, §26):

- **GQA (Grouped-Query Attention):** sharing K/V across query-head groups → `G/h` KV reduction (8× for LLaMA-3's 8 KV / 64 query heads). The standard for modern large models; turns a 70B's KV from MHA's huge cache to a servable 320 KB/token (File 02 §26). For long context, GQA is what makes it feasible at all.
- **MLA (Multi-head Latent Attention, DeepSeek):** low-rank latent KV → ~64× reduction vs MHA (File 02 §2.4, File 06 §7, File 15). MLA is the most aggressive, enabling DeepSeek's long context economically. vLLM/SGLang support MLA for DeepSeek models (caching the latent, File 06 §7).

For long-context serving, the model's attention variant is a first-order decision: an MHA model at 128K is nearly unservable (huge KV); a GQA model is feasible; an MLA model is comfortable. This is a model-*selection* lever (File 11 §18) — choosing a GQA/MLA model over MHA hugely affects long-context cost. Combined with FP8 KV (§7), GQA/MLA bring long-context KV into a manageable range.

---

## 7. KV Quantization for Long Context

KV quantization (File 02 §7) directly attacks the long-context KV-memory bottleneck:

- **FP8 KV cache** (`--kv-cache-dtype fp8_e5m2`): halves KV memory → doubles the context length (or concurrency) that fits, at <0.5% quality impact (File 02 §7). The first lever for long context — combine with GQA/MLA for compounding reduction.
- **INT8 KV:** per-token scaling for better quality than per-tensor; also halves KV.
- **INT4/2-bit KV (KIVI, File 02 §7):** more aggressive (4× KV reduction), research-stage — reconstruction error in K can shift attention, needs validation.

For a 128K-context LLaMA-3 70B request: BF16 KV is 40 GB; FP8 KV is 20 GB — the difference between fitting 2 requests vs 4 on an 80 GB GPU (after weights). Stacking FP8 KV with GQA (already in the model) and, for DeepSeek, MLA, brings the per-token KV down to where long context is practical. KV quantization is the highest-leverage *tuning* knob for long context (the architecture choice §6 is the model-selection lever; KV dtype is the serving-config lever, File 11 §18).

---

## 8. Context Window Extension (YaRN, LongRoPE)

A model trained at context `L_train` degrades beyond it, but RoPE extension techniques (File 02 §5.2) extend the usable context *without* full retraining:

- **Position Interpolation, NTK-aware scaling, YaRN** (Peng et al., arXiv 2309.00071), **LongRoPE, Dynamic NTK:** rescale the RoPE frequencies so high positions stay in the trained distribution, extending context 2–32×+ (File 02 §5.2).
- **Inference responsibility:** the engine must apply the exact `rope_scaling` the model was extended with — `rope_scaling={"type": "yarn", "factor": 4.0, ...}` in the config, consistent with `--max-model-len`. A mismatch silently degrades long-context quality (File 02 §5.2, §FAQ) — one of the most common long-context bugs.
- **Quality:** good for 2–4× extension, degrades beyond — the extended context isn't free quality-wise. Validate on long-context tasks (the "needle in a haystack" retrieval test is common).

Context extension is the technique that *enables* a model to use long context (the model must support the length); the memory techniques (§§3, 5–7) make the resulting long-context KV *fit*. Both are needed: extend the model's usable context (RoPE scaling) and fit the KV (quantization, GQA/MLA, offloading). The inference engineer sets `--max-model-len` and `rope_scaling` consistently to enable the length, then applies the memory levers to serve it economically.

---

## 9. Synthesis

Long context is fundamentally a **memory problem** (File 01 §4): the KV cache grows linearly with context, dominating memory and capping concurrency, while attention compute grows quadratically in prefill. The levers all target KV size or placement:

- **Architectural** (model selection): GQA (8×) or MLA (~64×) KV reduction — the first-order choice (§6).
- **Quantization** (serving config): FP8 KV (2×), the highest-leverage tuning knob (§7).
- **Eviction** (research): H2O/SnapKV drop low-attention KV (§5), trading quality for memory.
- **Offloading** (relief valve): CPU/SSD swap for transient pressure or tiering (§3), PCIe-limited.
- **Streaming** (unbounded): StreamingLLM's sinks + window for fixed-memory infinite streams (§2).
- **Compute** (prefill): FlashAttention's linear-memory tiling (essential at long context, File 10 §13) + chunked prefill (avoid monopolizing the engine, File 04 §7) + split-K decode (fill the GPU at low-batch-long-context, File 10 §24).
- **Extension** (enablement): YaRN/LongRoPE to make the model *use* the long context (§8).

The recurring theme is the database's central insight (File 01): decode is memory-bound, and long context makes it acutely so (the huge KV read per step). The techniques attack the KV from every angle — make each token's KV smaller (GQA/MLA, quantization), keep fewer tokens (eviction, streaming windows), place colder KV in slower tiers (offloading), and ensure the model can use the length (extension) and the kernels can handle it (FlashAttention, chunked prefill, split-K). As context windows grow toward 1M+ tokens as a commodity (File 20 §future), these techniques become more central — and the combination (a GQA/MLA model, FP8 KV, chunked prefill, split-K decode, possibly offloading and eviction) is how production systems serve long context economically. Long context is the regime where the memory-bandwidth bound of decode is most punishing, and where the KV-reduction toolkit matters most.

---

## 10. Worked Long-Context Memory Budget

Make the memory pressure concrete for LLaMA-3 70B (GQA-8) serving long context on 8×H100 (TP=8, FP8 weights).

- **Weights:** 70 GB FP8 / 8 = 8.75 GB/GPU. Leaves ~63 GB/GPU for KV (at 0.9 util on 80 GB).
- **KV/token per GPU** (TP=8 → 1 KV head/GPU, FP8 KV): `2 · 80 layers · 1 · 128 · 1 byte = 20,480 B ≈ 20 KB/token/GPU`. (Full-model KV/token is 8× this = 160 KB at FP8, vs 320 KB BF16.)
- **Per-GPU KV capacity:** `63 GB / 20 KB ≈ 3.15M tokens`.

So per GPU, ~3.15M tokens of KV. At **128K context**, that's `3.15M / 128K ≈ 24 concurrent requests`. At **256K context**, ~12. At **1M context**, ~3. Without FP8 KV (BF16 KV, 40 KB/token/GPU), halve these (12, 6, 1.5). And with MHA instead of GQA (8× more KV heads), divide by 8 (3, 1.5, 0.4 — barely servable). This budget shows the compounding: GQA (8×) + FP8 KV (2×) = 16× more long-context concurrency than MHA+BF16. The architecture (GQA/MLA) and the KV dtype (FP8) together determine whether long-context serving is economical (24 concurrent 128K requests) or impractical (a fraction of one). This is the long-context capacity arithmetic (File 03 §17, File 11 §22) — and it's why §§6–7 (GQA/MLA + FP8 KV) are the first-order long-context levers.

---

## 11. Sequence Parallelism for Extreme Context

At extreme context (1M+ tokens), even a single sequence's KV and attention may not fit one GPU, requiring **sequence parallelism** (File 05 §6):

- **Ring attention** (File 05 §6.2): partition the single sequence's Q/K/V across GPUs by sequence chunks; each GPU computes attention for its chunk while passing K/V around a ring, combining via online softmax. Memory per GPU is `O(seq_len / SP_degree)` — so SP_degree GPUs can hold an SP_degree× longer context.
- **Ulysses / DP attention** (File 05 §6.1, File 09 §11): all-to-all to redistribute heads vs sequence, computing attention for all tokens but a subset of heads per GPU, reducing per-GPU activation memory.

These let a context exceeding one GPU's memory be served across multiple GPUs — the long-context analog of tensor parallelism (which splits the model; SP splits the sequence/activations). For the common case (≤128K, fitting with GQA+FP8 KV, §10), SP isn't needed; for 1M+ tokens or when activations dominate, SP becomes necessary (File 05 §6.3). SGLang's `--enable-dp-attention` and vLLM's experimental SP support provide this. The combination — TP for the model, SP for the long sequence, FP8/GQA for the KV — is how the most extreme long-context (1M+) is served, spreading both the model and the sequence across GPUs.

---

## 12. Prefix Caching for Long Shared Documents

A specific long-context optimization: when many requests share a long document (RAG, document QA — File 08 §34), prefix caching (File 03 §6, File 08) caches the document's KV once and reuses it. For a 50K-token document queried by 100 questions: without caching, 100 × 50K = 5M token-prefills; with caching, 50K once + 100 × (question only) — a ~100× prefill reduction for the document. The long document is exactly the kind of large shared prefix that prefix caching (especially RadixAttention's token-granular tree, File 08 §8) excels at. The catch (File 08 §34): document *order* matters (causal attention → different order = different KV), so canonicalize the document order to maximize cache hits. For long-context RAG, prefix caching the documents is a major lever — turning the expensive long-document prefill into a one-time cost amortized across all queries on that document. This connects long context (the documents are long) with prefix caching (they're shared) — the two combine for RAG-style long-context workloads.

---

## 13. Evaluating Long-Context Quality

Long-context techniques (extension §8, eviction §5, quantization §7) can degrade quality, so evaluation matters:

- **Needle-in-a-haystack:** insert a specific fact ("the needle") at a random position in a long context ("the haystack") and test whether the model retrieves it. Sweeping the needle position and context length produces a heatmap of retrieval accuracy — the standard long-context test. Degradation at certain positions/lengths reveals where the extension or eviction breaks down.
- **Long-context benchmarks:** RULER, LongBench, and similar test long-context reasoning beyond simple retrieval.
- **Perplexity at length:** measure perplexity as context grows — a rise indicates the model (or the extension/eviction) failing at length.

The inference engineer's responsibility: validate that the long-context configuration (RoPE scaling, KV quantization, any eviction) preserves quality on these tests *at the served length* (File 06 §12's "always validate"). A common failure is a RoPE-scaling mismatch (File 02 §5.2) that passes short-context tests but fails needle-in-a-haystack at length — caught only by long-context evaluation. The two-dimensional optimization (File 11 §38.1) applies: minimize long-context serving cost (FP8 KV, eviction) subject to passing the long-context quality bar. Aggressive KV eviction (H2O, §5) or low-bit KV (INT4, §7) can save memory but must be validated to not drop the needle.

---

## 14. The Long-Context Tuning Profile

Synthesizing into a tuning profile (File 11 §22, §14 was prefill-heavy; long context overlaps but is KV-bound):

1. **Choose a GQA or MLA model** (§6) — the first-order KV-reduction decision.
2. **Enable FP8 KV cache** (§7) — halve the KV, double the capacity.
3. **Set `--max-model-len` and `rope_scaling`** consistently for the target length (§8, §1.3).
4. **Enable chunked prefill** (File 04 §7) — essential so a long prefill doesn't monopolize the engine.
5. **Size `--max-num-seqs` low** — long context is KV-memory-bound (§10, File 04 §27); over-admitting thrashes.
6. **Ensure split-K decode** (File 10 §24) — fills the GPU at low-batch-long-context (common for long context: few concurrent very-long requests).
7. **For shared long documents (RAG):** enable prefix caching (§12).
8. **For 1M+ context:** sequence parallelism (§11).
9. **For unbounded streams:** StreamingLLM sinks + window (§2).
10. **Validate quality** at the served length (§13).

Long context inverts the usual short-context tuning: concurrency is low (KV-bound), the KV read dominates decode (huge per-step KV bandwidth), prefill is heavy (long prompts), and memory is the constant constraint. The levers all target the KV (smaller via GQA/MLA/FP8, fewer via eviction/windows, spread via SP/offloading) — because in long context, the KV cache *is* the system's defining resource, even more than in short-context serving where it's already central (File 01 §4). Mastering long context is mastering KV-cache management at its most extreme — which is why every KV technique in the database (paging, GQA/MLA, quantization, eviction, offloading, prefix caching, sequence parallelism) converges here. Long context is the stress test that exercises the entire KV-management toolkit at once.

---

## 15. The Decode-Latency Cost of Long Context, Worked

Beyond memory, long context slows *decode* because each step reads the entire KV (File 02 §6.4). Quantify for LLaMA-3 70B (GQA-8) at various context lengths, single request, H100 (3.35 TB/s):

- **KV bytes read per decode step** = `2 · num_kv_heads · head_dim · bytes · context_len · num_layers` = `2 · 8 · 128 · 2 (BF16) · context · 80`.
- At **4K context:** `2·8·128·2·4096·80 ≈ 1.3 GB` → ~0.4 ms KV read (on top of the ~21 ms weight read for the 70B BF16, File 10 §26). KV is a small fraction.
- At **128K context:** `2·8·128·2·131072·80 ≈ 42 GB` → ~12.5 ms KV read — now *comparable to* the weight read. The KV read has become a major part of decode latency.
- At **1M context:** ~328 GB KV read → ~98 ms per step *just for KV* — dwarfing the weight read. Decode is now KV-bandwidth-bound.

So at long context, the per-step KV read grows until it dominates decode latency — TPOT rises with context length. This is why long-context decode is slow (high TPOT) independent of the memory-capacity issue: even if the KV fits, reading 42 GB (128K) or 328 GB (1M) every step is slow. FP8 KV (§7) halves this read (and the memory), directly improving long-context TPOT. GQA/MLA reduce it further (fewer KV heads / latent). This is a second reason (beyond memory capacity) that KV reduction matters for long context: it cuts both the memory footprint *and* the per-step decode-read latency. The worked numbers show the transition: short context (4K) is weight-read-bound (KV negligible); long context (128K+) becomes KV-read-bound (KV dominates) — a regime change that makes long-context decode fundamentally slower and KV reduction doubly valuable.

---

## 16. Why Long Context Is Getting Cheaper

Despite the challenges, long context is becoming a commodity (File 20 §future) — 128K is increasingly standard. The drivers:

- **GQA/MLA adoption** (§6): modern models ship with KV-efficient attention, cutting the per-token KV 8–64× vs MHA — the single biggest enabler.
- **FP8 KV** (§7): production-standard on H100+, halving KV.
- **Bigger/faster HBM** (File 16): H200 (141 GB), MI300X (192 GB), Blackwell (192 GB) hold more KV; faster bandwidth (4.8–8 TB/s) speeds the KV read.
- **Better kernels** (File 10): FlashAttention's linear memory, split-K decode, FlashInfer's paged long-context attention.
- **Context extension** (§8): YaRN/LongRoPE extend models to long context without full retraining.

The combination has brought 128K context from a research feat to a routine deployment, and the trajectory (bigger HBM, more KV reduction, better kernels) points toward 1M+ as commodity. For the inference engineer, this means long-context serving, once exotic, is increasingly a standard requirement — and the techniques in this file (GQA/MLA model choice, FP8 KV, chunked prefill, split-K, prefix caching for shared documents, SP for extreme lengths) are the standard toolkit for meeting it economically. Long context exemplifies how the field's combined progress — architecture (GQA/MLA), quantization (FP8 KV), hardware (bigger/faster HBM), and kernels (FlashAttention/FlashInfer) — compounds to make once-impractical capabilities routine, the same compounding that drove the order-of-magnitude inference cost reduction overall (File 01 §8). Each layer's contribution (smaller KV, more memory, faster reads, better kernels) multiplies, turning the daunting `O(n)` KV growth and `O(n²)` prefill compute into a manageable, economical long-context service.

---

## 17. Long-Context Pitfalls and FAQ

**Q: My model claims 128K context but quality is poor at long lengths.** Likely a RoPE-scaling mismatch (§8, File 02 §5.2) — the `rope_scaling` config must match how the model was extended, and `--max-model-len` must be consistent. Verify with needle-in-a-haystack (§13). Also check the model genuinely supports the length (some "128K" models degrade well before 128K regardless of config).

**Q: Long-context requests OOM the server.** Long context is KV-memory-bound (§10) — a single 128K request is ~40 GB KV (BF16) / ~20 GB (FP8). Enable FP8 KV (§7), lower `--max-num-seqs` (long context needs low concurrency, §14), and ensure `--max-model-len` matches reality (over-setting reserves capacity). Compute the per-request KV (§10) and size accordingly.

**Q: Long-context decode is slow (high TPOT) even at low load.** Expected — the per-step KV read grows with context (§15); at 128K it's ~12.5 ms/step just for KV. FP8 KV and GQA/MLA reduce it, but long-context decode is fundamentally slower (more KV to read each step). This is the KV-read-bound regime (§15).

**Q: Many queries share a long document — am I recomputing it?** Without prefix caching, yes. Enable it (§12) and canonicalize document order (File 08 §34) so the shared document's KV is cached once and reused — a ~100× prefill saving for document-QA workloads.

**Q: When do I need sequence parallelism?** Only at extreme context (1M+) where a single sequence's KV/activations exceed one GPU (§11). For ≤128K with GQA+FP8 KV, standard TP suffices. SP adds all-to-all overhead, so use it only when the sequence genuinely doesn't fit.

These pitfalls reduce to the file's theme: long context is a KV-memory-and-bandwidth problem, and the fixes are the KV-reduction levers (GQA/MLA model, FP8 KV, eviction), correct configuration (`max-model-len` + `rope_scaling`), low concurrency (KV-bound), prefix caching (shared documents), and SP (extreme lengths). Diagnosing a long-context problem means asking: is it memory (capacity — reduce/spread KV), decode latency (KV read — reduce KV), quality (RoPE scaling/eviction — validate), or prefill cost (chunked prefill + prefix caching)? Each maps to a lever, and the KV cache is at the center of all of them — as it is at the center of LLM inference generally (File 01 §4), just most acutely in the long-context regime.

