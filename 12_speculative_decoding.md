# Speculative Decoding — Algorithms, Implementations, and Integration

> **Standard reference file.** Speculative decoding amortizes the memory-bound cost of decode (read all weights to produce one token) by verifying *multiple* candidate tokens per target forward pass. This file covers the theory, the tree/feature variants (Medusa, EAGLE), the vLLM and SGLang implementations, draft-model management, and MoE considerations. Foundations: File 02 §9. Prerequisites: File 01 §2 (memory-bound decode), File 04 §19 (scheduler interaction), File 10 §18 (tree attention).

---

## 1. The Core Idea and Why It Works

Decode is memory-bandwidth-bound (File 01 §2): each step reads the *entire* model weights from HBM to produce *one* token, so the tensor cores idle while memory streams. The key insight of speculative decoding: a single target forward pass can verify *several* token positions in parallel (like a mini-prefill) for nearly the same cost as producing one token — because the cost is dominated by the weight read, which is shared across the verified positions. So if a cheap **draft** can propose several likely-correct tokens, the expensive **target** can verify them all in one weight-read, producing multiple accepted tokens per target pass — amortizing the memory-bound cost across multiple tokens.

This only helps because decode is memory-bound: the target has *spare compute* (the tensor cores were idle) to verify the extra positions almost for free. In a compute-bound regime (large batch, prefill) there's no spare compute and speculation doesn't help (File 02 §9, File 11 §17.3). Speculative decoding is, fundamentally, a way to convert the target's idle compute (during memory-bound decode) into useful work (verifying speculated tokens), trading the otherwise-wasted FLOPs for fewer weight reads per token.

---

## 2. The Algorithm (Leviathan et al. 2023)

Speculative decoding (Leviathan et al., arXiv 2211.17192; Chen et al. similar) works as follows:

1. **Draft:** a small draft model `q` autoregressively proposes `K` tokens `x_1, ..., x_K` (cheap — the draft is ~10× smaller).
2. **Verify:** the target model `p` processes all `K` proposed positions in *one* forward pass, producing the target's probability distribution `p(x)` at each position (the positions are processed in parallel, like prefill — this is the key efficiency).
3. **Accept/reject (speculative sampling):** for each position `i` in order, accept the draft token `x_i` with probability `min(1, p(x_i)/q(x_i))`. On the first rejection, sample a corrected token from the adjusted distribution `(p − q)_+` (normalized) and stop accepting further tokens.
4. **Output:** the accepted tokens (plus the one corrected/bonus token) are emitted; generation continues from there.

### 2.1 Why it's lossless

The crucial property: the accepted sequence is distributed **exactly as if sampled from the target model `p`**. The accept/reject rule (speculative sampling) is constructed so that the combination of "accept with `min(1, p/q)`" and "resample from `(p−q)_+` on rejection" yields the exact target distribution. So speculative decoding is **lossless** — it produces the same output distribution as standard target decoding, just faster. This is not an approximation; the outputs are statistically identical to the target's. This losslessness is why speculative decoding is "free" speedup — no quality trade-off (unlike quantization, which can degrade quality).

### 2.2 The speedup model

Let `β` = expected per-token acceptance rate (`β = Σ_x min(p(x), q(x))`, the overlap of the draft and target distributions) and `c` = draft-to-target cost ratio. Expected accepted tokens per target pass `≈ (1 − β^{K+1})/(1 − β)`. Speedup over standard decode:

```
speedup ≈ E[accepted + 1] / (1 + K·c)
```

Worked (File 02 §9.2): `β = 0.7, K = 4, c = 0.1` → numerator ≈ 3.8, denominator ≈ 1.4 → ~2.7×. The speedup rises with acceptance `β` (draft and target agree often) and falls if the draft is too expensive. So the design goals: high acceptance (use a same-family, same-tokenizer draft, run it greedy) and cheap draft (~10× smaller).

---

## 3. Tree Speculation

Linear speculation proposes a single chain of `K` tokens. **Tree speculation** proposes a *tree* of candidate continuations, verified together in one target pass, achieving higher expected acceptance for the same target compute.

The idea: instead of betting on one continuation, the draft proposes several alternative tokens at each position, forming a tree of candidate paths. The target verifies the entire tree in one forward pass using a **tree attention mask** (File 10 §18) — each candidate token attends only to its ancestors in the tree. The accept logic then finds the longest accepted path through the tree. Because the tree covers multiple alternatives, the chance that *some* path matches the target is higher than betting on a single chain → higher acceptance, more tokens per target pass. The cost is verifying more positions (the whole tree) per pass, but since decode is memory-bound (spare compute), verifying a tree is nearly as cheap as a chain. Tree speculation is used by Medusa and EAGLE (§4) and requires the tree-attention kernel (File 10 §18).

---

## 4. Variants: Medusa and EAGLE

### 4.1 Medusa (Cai et al. 2024, arXiv 2401.10774)

Medusa adds multiple lightweight **prediction heads** to the target model itself — each head predicts a token at a different future offset (head 1 predicts the next token, head 2 the token after, etc.). These heads, trained on top of the frozen target, generate a tree of candidates directly from the target's hidden state, verified with tree attention. The advantage: **no separate draft model** — no second set of weights to host (saving the draft's memory). The disadvantage: the heads' predictions are less accurate than a real draft model (they predict further-out tokens from one hidden state), so acceptance is lower than a good separate draft. Medusa trades draft quality for zero draft-model memory overhead.

### 4.2 EAGLE (arXiv 2401.15077)

EAGLE is a **feature-level** draft: instead of predicting tokens from scratch, the EAGLE draft takes the target model's *hidden states* (features) as input and predicts the next features (and thus tokens). Using the target's rich features gives higher acceptance than a token-level draft. EAGLE achieves strong speedups (2–3×+) with a small draft head. **EAGLE-2** improves further with *dynamic* draft-tree construction: it builds the speculation tree adaptively based on each node's acceptance probability, expanding promising branches and pruning unlikely ones — getting more accepted tokens per target pass than a static tree. EAGLE/EAGLE-2 are among the most effective speculative methods and are supported in both engines (SGLang notably, File 09 §25). DeepSeek-V3's MTP heads (File 02 §22) are a related built-in-draft approach.

### 4.3 The spectrum

The variants span a spectrum: separate draft model (highest quality draft, extra memory) → EAGLE (feature-level, high acceptance, small head) → Medusa (token heads, no separate model, lower acceptance) → MTP (built into the model at training). The choice trades draft quality (acceptance rate) against draft cost/memory. EAGLE is often the sweet spot (high acceptance, small overhead); Medusa/MTP avoid a separate model; a separate draft is simplest conceptually. All use tree attention for verification.

---

## 5. vLLM Implementation

vLLM's speculative decoding lives in `vllm/spec_decode/` (File 06 §8):

- **`SpecDecodeWorker`** wraps the draft and target models. Per step: (1) run the draft `K` times to propose `K` tokens (plus the logits/probabilities needed for the accept rule); (2) build a "proposal" covering all `K+1` positions; (3) run the target model over those positions in one forward pass (`run_target_model`), producing the target distributions; (4) apply the **`RejectionSampler`** (speculative sampling, §2) to accept/reject; (5) return the accepted tokens.
- **`BatchExpansionTop1Scorer`** and related machinery handle scoring the proposals against the target at batch scale.
- **Flags:** `--speculative-model` (the draft checkpoint), `--num-speculative-tokens` (K), `--speculative-draft-tensor-parallel-size` (the draft can use a smaller TP degree than the target, since it's small).
- **Scheduler integration** (File 04 §19): the scheduler accounts for the variable accepted-tokens-per-step (a sequence advances by 1 to K+1 positions per step) and the larger per-step KV growth (allocate for up to K+1 new tokens).

vLLM supports separate-draft, EAGLE, and Medusa-style speculation, with the draft and target potentially on different CUDA streams to overlap (File 09 §12.2). The rejection sampling at batch scale (many sequences each with their own draft/target distributions) is the implementation's core, and it must be exact (preserving the lossless property, §2.1).

---

## 6. SGLang Implementation

SGLang supports speculative decoding, notably EAGLE, from v0.3+ (File 09 §25):

- **`SpecInfo`** tracks the draft tokens / draft tree per sequence (the per-`Req` speculative state).
- **Tree attention** via a custom FlashInfer extension verifies the candidate tree in one pass (File 10 §18).
- **`draft_worker` and `target_worker`** alternate, with CUDA-stream overlap (the draft proposing while the target verifies, where the dependency structure allows, File 09 §12.2).
- **RadixAttention compatibility** (File 08 §27): speculative requests reuse cached prefixes like any other; only accepted tokens' KV becomes part of the permanent cached path, while rejected speculative KV is discarded.
- **EAGLE-2's dynamic tree** is supported, adapting the speculation tree to acceptance probabilities (§4.2).

SGLang's early and aggressive EAGLE support reflects its tendency to ship cutting-edge algorithms (File 08 §39). The integration threads speculation through the runtime (state in the `Req`, tree attention in the backend, stream overlap in the executor — File 09 §25).

---

## 7. Draft-Model Management

For separate-draft speculation, managing the draft model is a practical concern:

- **Vocabulary compatibility:** the draft *must* share the target's tokenizer/vocabulary — the accept rule compares `p(x)` and `q(x)` over the same token space. A mismatched vocabulary makes speculation impossible.
- **Size:** the optimal draft is ~10× smaller than the target (e.g. a 7B draft for a 70B target, or a 1B draft for an 8B target) — small enough that `K` draft steps are cheap relative to one target pass (keeping the speedup denominator low, §2.2), large enough for good acceptance.
- **Same family:** a draft from the same model family/training (LLaMA draft for LLaMA target) has higher acceptance (the distributions overlap more, higher `β`). EAGLE/MTP achieve this by being trained *on* the target.
- **Memory overhead:** the draft weights (~1/10 of the target) plus the draft's speculation KV. For a 70B target, a 7B draft adds ~14 GB (BF16) or less quantized — modest but non-zero.
- **Draft temperature:** run the draft greedy (T=0) to maximize acceptance — the draft commits to its single best guess, most likely to match the target; the target uses the user's temperature for the final accepted-token distribution (the losslessness, §2.1, holds regardless).

---

## 8. MoE and Speculative Decoding

Speculative decoding for MoE models (File 02 §4.2, File 05 §5) has specific considerations:

- **Draft as a dense model approximating sparse MoE:** a natural draft for a MoE target is a small *dense* model approximating it. But MoE models have irregular per-token expert activation, so the acceptance rate can vary by which experts a token would activate.
- **MTP for MoE:** DeepSeek-V3 uses built-in MTP heads (File 02 §22) as the draft, avoiding a separate model and integrating speculation into the MoE architecture — the draft "knows" the MoE structure.
- **Verification cost:** verifying `K+1` positions through a MoE target involves the expert routing and (under EP) the all-to-all for each position — speculation interacts with the MoE communication (File 05 §5). The amortization still helps (one weight-read region per pass for multiple tokens) but the MoE communication per verified position adds cost.
- **Research area:** speculative decoding for MoE is an active area (per-expert drafts, MoE-aware speculation); the general principle (amortize the memory-bound target cost across verified tokens) holds, but the MoE structure (routing, EP communication) complicates the cost model.

---

## 9. When Speculative Decoding Helps

Synthesizing the conditions (File 11 §17.3):

- **Decode-bound workloads** (long outputs — reasoning, chat): speculation directly speeds the bottleneck (File 11 §15). High value.
- **Low-to-moderate batch:** the target has spare compute to verify the extra positions (§1). At high batch (compute-saturated), there's no spare compute and speculation doesn't help (File 02 §9).
- **High draft-target agreement** (good acceptance `β`): same-family draft, greedy draft, EAGLE/MTP. Low acceptance → little speedup (and wasted draft compute).
- **Not for prefill-bound workloads** (File 11 §14): prefill is already compute-bound and parallel; speculation is a decode optimization.

Measure the acceptance rate on your workload (~60–70% chat, ~80% code typically) and the batch regime to decide. When the conditions hold (decode-bound, moderate batch, good acceptance), speculation is a ~2× lossless speedup — one of the highest-value, lowest-risk optimizations (no quality cost, §2.1). When they don't (prefill-bound or compute-saturated), it adds draft overhead for little gain. This conditionality is why speculation is the headline lever for reasoning/chat tuning (File 11 §15, §13) and absent from prefill-heavy tuning (File 11 §14).

---

## 10. Synthesis

Speculative decoding amortizes the memory-bound cost of decode by verifying multiple draft-proposed tokens per target forward pass, exploiting the target's idle compute during memory-bound decode. It is **lossless** (the speculative-sampling accept rule preserves the target's exact output distribution, §2.1) — a rare "free" speedup with no quality trade-off. The variants — separate draft, tree speculation, Medusa (token heads), EAGLE (feature-level, the sweet spot), MTP (built-in) — trade draft quality (acceptance) against draft cost/memory, all using tree attention for verification. Both engines implement it (vLLM's `SpecDecodeWorker`, SGLang's EAGLE), with scheduler integration for the variable accepted-tokens-per-step (File 04 §19) and stream overlap for the draft/target (File 09 §12.2). It helps when decode is the bottleneck and the target has spare compute (moderate batch, long outputs) with good draft acceptance — the conditions of reasoning and chat workloads (File 11 §15). For those workloads it's a ~2× lossless speedup and a major cost lever (File 11 §16); for prefill-bound or compute-saturated workloads it doesn't apply. Speculative decoding exemplifies the database's theme of exploiting the memory-bound nature of decode: just as quantization and GQA *reduce* the bytes read per token, speculation *amortizes* the byte-read across multiple tokens — different mechanisms, same goal of beating the memory-bandwidth bound that dominates decode cost.

---

## 11. The Rejection Sampling Rule, Derived

The speculative-sampling accept/reject rule (§2.1) is worth deriving because its correctness is what makes speculation lossless. Given the draft proposes token `x` with probability `q(x)` and the target's true probability is `p(x)`:

- **Accept** `x` with probability `min(1, p(x)/q(x))`.
- **If rejected**, sample a replacement from the residual distribution `p'(x) = norm((p(x) − q(x))_+)` where `(·)_+ = max(·, 0)` and `norm` renormalizes to sum to 1.

**Why this yields exactly `p`:** the probability that token `x` is the final output is `P(propose x) · P(accept x) + P(reject the proposal) · P(resample x)`. The first term is `q(x) · min(1, p(x)/q(x)) = min(q(x), p(x))`. The second term works out (summing over the rejection cases) to exactly fill the gap so that the total equals `p(x)`. Concretely, when `q(x) ≤ p(x)` (draft underestimates), the token is always accepted (`min(1, p/q) = 1` since `p/q ≥ 1`)... wait, `min(1, p/q)` with `p ≥ q` gives `min(1, ≥1) = 1` — accept always, contributing `q(x)`; the remaining `p(x) − q(x)` comes from the resample path. When `q(x) > p(x)` (draft overestimates), accept with probability `p/q`, contributing `q · p/q = p(x)` directly. Either way the marginal is `p(x)` — the target distribution exactly. This is the mathematical guarantee of losslessness: the rule is designed so the output marginal equals the target's regardless of the draft `q`. A worse draft (lower `β`) just means more rejections (less speedup), never wrong outputs.

### 11.1 Acceptance rate intuition

`β = Σ_x min(p(x), q(x))` is the *overlap* (total variation complement) of the draft and target distributions — how much probability mass they agree on. A perfect draft (`q = p`) gives `β = 1` (every token accepted, maximal speedup); an uncorrelated draft gives low `β` (frequent rejections, little speedup). This is why draft quality (a same-family, greedy draft, or EAGLE's feature-level draft) matters: it raises the distribution overlap `β`, raising the accepted-tokens-per-pass and thus the speedup. The acceptance rate is measurable (count accepted/proposed during serving) and is the key tuning signal for speculation (File 11 §17.2).

---

## 12. Worked Speedup Examples

Concrete numbers for different scenarios (the model §2.2):

**Chat, separate 7B draft for 70B target, K=4, β=0.65, c=0.1:**
- Expected accepted ≈ `(1 − 0.65^5)/(1 − 0.65) ≈ (1 − 0.116)/0.35 ≈ 2.53` tokens, +1 bonus ≈ 3.5.
- Speedup ≈ `3.5 / (1 + 4·0.1) = 3.5/1.4 ≈ 2.5×`.

**Code, same setup, higher β=0.80** (code is more predictable):
- Expected accepted ≈ `(1 − 0.8^5)/0.2 ≈ (1 − 0.328)/0.2 ≈ 3.36`, +1 ≈ 4.36.
- Speedup ≈ `4.36/1.4 ≈ 3.1×`.

**EAGLE, β≈0.8, very cheap draft head c≈0.05, K=5:**
- Expected accepted ≈ `(1 − 0.8^6)/0.2 ≈ (1 − 0.262)/0.2 ≈ 3.69`, +1 ≈ 4.69.
- Speedup ≈ `4.69 / (1 + 5·0.05) = 4.69/1.25 ≈ 3.75×`.

The examples show: higher acceptance (code > chat; EAGLE > separate draft) and cheaper draft (EAGLE's small head) both raise the speedup. EAGLE's combination of high acceptance and cheap draft is why it often reaches 3×+ where a separate draft reaches ~2.5×. The numbers also show diminishing returns in `K` (acceptance decays with depth, `β^K` shrinks) — beyond K≈4–6 the marginal accepted token is unlikely, so more `K` adds draft/verification cost for little gain (File 11 §17.2). Tuning `K` to the acceptance-decay curve is part of optimizing speculation.

---

## 13. Speculative Decoding with Quantization and Disaggregation

Speculation composes with other optimizations:

- **+ Quantization:** the draft and/or target can be quantized (File 06 §3). A quantized target's distribution differs slightly from BF16, but speculation is lossless *with respect to whatever target it verifies against* (§2.1) — so speculating against a quantized target yields the quantized target's distribution exactly. The draft can be aggressively quantized (it's just a proposer; the target's accept rule corrects errors). Combining FP8 target + speculation stacks the cost levers (File 11 §16, §38).
- **+ Disaggregation** (File 04 §10, File 15): the draft can run on the prefill instance and the target on the decode instance, or both on decode — the placement is a design choice. Speculation overlaps with the disaggregated pipeline (File 02 §9.3). DeepSeek's MTP (built-in draft) integrates naturally with its disaggregated MoE serving.
- **+ Prefix caching:** orthogonal — speculative requests reuse cached prefixes (File 08 §27); only accepted tokens extend the cached path.

The composability is why speculation is a clean addition to a tuned stack: it's lossless (no quality interaction), it targets a different bottleneck (decode amortization) than quantization (bytes) or prefix caching (redundant prefill), and it stacks with all of them (File 11 §38). The main interaction to watch is the batch regime (it needs spare target compute, §9) and the acceptance rate (workload-dependent).

---

## 14. History and the Broader Landscape

Speculative decoding emerged in 2022–2023 (Leviathan et al., Chen et al.) and rapidly became standard:
- **Blockwise parallel decoding** (earlier work) predicted multiple tokens with auxiliary heads — a precursor to Medusa.
- **Speculative decoding** (Leviathan/Chen 2023) formalized the lossless draft-verify-accept scheme with a separate draft.
- **Medusa** (2024) added the multi-head no-separate-draft approach.
- **EAGLE/EAGLE-2** (2024) introduced feature-level drafting and dynamic trees — the current high-performers.
- **Lookahead decoding**, **SpecInfer**, **REST** (retrieval-based drafting), and others explored alternative drafting strategies (n-gram, retrieval, Jacobi iteration).
- **MTP** (DeepSeek-V3, 2024) built multi-token prediction into the model as a native draft.

The trajectory: from separate drafts → built-in heads/features → architecturally-integrated (MTP), each reducing the draft's overhead while maintaining or raising acceptance. The common thread is the lossless verify-accept core (§2.1, §11) — the drafting strategy varies, but the target's accept rule always guarantees the exact output distribution. Speculative decoding is now a standard feature in production engines (vLLM, SGLang, TensorRT-LLM), and its ideas (multi-token verification, tree attention) have become part of the inference toolkit, especially valuable as reasoning models (long decode, File 15) make decode amortization more important than ever.

---

## 15. Key Takeaways

1. **Speculation amortizes the memory-bound decode cost** by verifying multiple draft-proposed tokens per target weight-read, using the target's idle compute (§1).
2. **It is lossless** — the speculative-sampling accept rule (§2.1, §11) preserves the target's exact output distribution regardless of draft quality. Free speedup, no quality cost.
3. **Speedup ≈ accepted-tokens / draft-cost** (§2.2), maximized by high acceptance `β` (same-family/greedy/EAGLE draft) and cheap draft (~10× smaller, or built-in heads).
4. **Variants** trade draft quality vs cost: separate draft, Medusa (heads, no draft), EAGLE (feature-level, sweet spot), MTP (built-in). Tree speculation + tree attention raise acceptance (§§3–4).
5. **It helps when decode is the bottleneck with spare target compute** (long outputs, moderate batch, good acceptance) — reasoning and chat (File 11 §15) — and not for prefill-bound or compute-saturated workloads (§9).
6. **It composes** with quantization, disaggregation, and prefix caching (§13), stacking as a clean cost lever (File 11 §16).

Speculative decoding is the third major attack on the memory-bandwidth bound (alongside reducing bytes via quantization/GQA/MLA, and reusing KV via paging/radix caching): it amortizes the unavoidable weight-read across multiple tokens. As decode-heavy reasoning workloads grow, it's an increasingly central optimization — a ~2× lossless speedup that turns the target's wasted idle compute into useful verification, beating the memory bound that would otherwise cap decode at one-token-per-weight-read.

---

## 16. The Batch-Regime Dependence, Worked

The most important practical subtlety is that speculation's benefit *depends on the batch size* (§1, §9). Work through why with the roofline (File 01 §2).

At **batch 1** (single user), decode is deeply memory-bound — the target reads all weights to produce one token, tensor cores ~idle. Verifying K+1 positions costs almost the same as one (the weight read dominates), so speculation gives near-full speedup (~`accepted_tokens×`). Speculation is most valuable here.

At **moderate batch** (e.g. 16–32), still memory-bound but the batch already amortizes the weight read across 16–32 sequences. Verifying K+1 positions per sequence multiplies the batch's effective token count by K+1, which uses more of the (still spare) compute — speculation still helps, but the relative gain is smaller (the batch was already getting some amortization).

At **high batch** (e.g. 128+, approaching the ridge point, File 01 §2.2), the target is becoming compute-bound — the tensor cores are busy with the large batch, *no spare compute* to verify speculative positions. Speculation now *competes* for compute with the batch's own work, and the K+1× verification positions can actually slow things down. Speculation stops helping and can hurt.

So speculation's value is **inversely related to batch size**: maximal at batch 1, diminishing as batch grows, gone (or negative) at compute-saturated high batch. This is why it's the headline lever for *low-concurrency, decode-bound* workloads (a single user with a long reasoning chain, File 11 §15) and *not* for high-throughput, high-batch serving (where the batch already amortizes the weight read and there's no spare compute). The practical rule: enable speculation when your operating batch size leaves the target memory-bound (spare compute); measure the actual speedup at your batch (it can be negative at high batch). Engines may dynamically enable/disable speculation based on batch size for this reason. This batch-regime dependence is the single most misunderstood aspect of speculative decoding — it's not universally beneficial, but specifically beneficial in the memory-bound, spare-compute regime that low-batch decode occupies.

---

## 17. Debugging and Tuning Speculation

Practical signals for speculative decoding (File 11 §17):

- **Acceptance rate** (logged): the key health metric. Low acceptance (<50%) means a poor draft (wrong family, too small, or high draft temperature) — the speedup will be small. Aim for ~60%+ (chat) to ~80% (code).
- **Effective speedup:** measure tokens/sec with and without speculation *at your operating batch* (§16). If it's not faster, the batch is too high (compute-saturated) or acceptance too low.
- **K tuning:** sweep `num_speculative_tokens`; acceptance decays with depth, so find where the marginal token stops being accepted (~K=3–5 typically, File 11 §17.2).
- **Draft cost:** if the draft is too large (cost ratio too high), the denominator in the speedup (§2.2) grows — use a smaller draft or EAGLE/MTP.
- **Memory:** the draft weights + speculation KV add to memory; ensure it fits (it reduces KV-bound concurrency slightly).

The tuning loop: measure acceptance and effective speedup at the operating batch, adjust the draft (size/family/temperature) and K, and verify the net speedup is positive. If speculation doesn't help, the diagnosis is usually batch too high (§16) or acceptance too low (draft quality) — both measurable, both fixable (lower the operating batch via more replicas, or improve the draft). Speculation rewards measurement: its benefit is workload- and batch-specific, so the only way to know it's helping is to measure the effective speedup in your actual operating regime, not to assume the theoretical ~2× applies. That measurement discipline — verify the effective speedup in your operating regime rather than trusting the headline number — is the recurring lesson, and it applies to speculation as much as to every other lever in the inference toolkit (File 11 §30).

