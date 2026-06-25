# SGLang RadixAttention — Algorithm, Data Structures, and Implementation

> **PRIMARY reference file.** RadixAttention is SGLang's foundational contribution: automatic, fine-grained KV cache reuse across requests using a radix tree keyed on token sequences. This file derives the multi-call problem it solves, the radix-tree data structure and its operations, the RadixAttention algorithm, SGLang's KV block management, the scheduler integration, the XGrammar structured-output engine, and the SGLang frontend DSL. Paper: Zheng et al., *"SGLang: Efficient Execution of Structured Language Model Programs"* (arXiv 2312.07104). Prerequisites: File 02 (architecture), File 03 (PagedAttention and vLLM's hash-based prefix caching, the natural comparison point).

---

## Table of Contents

1. The Multi-Call Problem
2. Why Prefix Reuse Is the Central Opportunity
3. The Radix Tree Data Structure
4. Radix Tree Operations: Insert, Match, Evict
5. The RadixAttention Algorithm
6. Multiple Sessions and Tree-Structured Sharing
7. Fork and Join
8. RadixAttention vs Hash-Based Prefix Caching
9. SGLang KV Cache Block Management
10. The SGLang Scheduler and RadixAttention
11. XGrammar — Structured Output Engine
12. The SGLang Frontend Language

---

## 1. The Multi-Call Problem

vLLM and most serving engines were designed around *independent, single-call* requests: a prompt comes in, a completion goes out. But a large and growing fraction of real LLM usage is *multi-call programs* that make several LLM calls sharing context:

- **Multi-turn chat:** each turn resends the system prompt + the entire conversation history; only the newest user message is new.
- **Few-shot prompting:** every request shares the same (often long) set of few-shot examples, differing only in the final query.
- **Chain-of-thought / agentic programs:** a program calls the LLM multiple times, each call building on the previous output (reasoning steps, tool-use loops).
- **RAG (retrieval-augmented generation):** many queries share the same system prompt and, often, the same retrieved documents.
- **Tree-of-thought / parallel exploration:** a program forks multiple continuations from a common context, explores each, and selects the best — the shared root context is identical across branches.
- **Self-consistency / best-of-N:** generate N independent completions from the same prompt and aggregate.

In all of these, **large spans of tokens are shared across calls**, and the KV cache for those shared spans is identical. The naive approach — treat each call as independent and recompute everything — wastes enormous compute and memory recomputing KV that was just computed moments ago for the same tokens.

SGLang was designed *from the start* around this multi-call structure: its frontend DSL (§12) lets programs express these patterns explicitly, and its backend (RadixAttention) automatically reuses the shared KV. This program-level view is the deepest difference between SGLang and vLLM (which sees only independent requests, File 04 §24).

---

## 2. Why Prefix Reuse Is the Central Opportunity

Quantify the opportunity. Consider a 2,048-token system prompt + few-shot examples, followed by a 50-token user query, served to 1,000 requests.

**Without prefix reuse:** each request prefills all `2048 + 50 = 2098` tokens. Total prefill ≈ `1000 × 2098 ≈ 2.1M` token-computations. And each request holds its own copy of the 2,048-token prefix KV in memory.

**With prefix reuse:** the 2,048-token prefix is computed *once* and its KV cached. Each request reuses that cached KV and prefills only its 50-token suffix. Total prefill ≈ `2048 + 1000 × 50 ≈ 52K` token-computations — a **~40× reduction**. And the prefix KV is stored once (shared), not 1,000 times — a massive memory saving that translates into far higher concurrency (File 01 §4, File 03 §31).

For prefix-heavy workloads this is the single largest available optimization, dwarfing most kernel-level tuning. The questions are *how* to detect and reuse shared prefixes (a), at what granularity (b), and how to handle the *tree-structured* sharing that arises when many sessions diverge at different points (c). vLLM's hash-based APC (File 03 §6) answers (a) and (b) at block granularity for linear prefixes. RadixAttention answers all three at *token* granularity with a *tree* structure — which is what makes it more powerful for complex multi-session workloads.

---

## 3. The Radix Tree Data Structure

RadixAttention's core is a **radix tree** (a compressed trie) whose paths represent cached token sequences and whose nodes hold the corresponding KV cache.

### 3.1 Trie → radix tree

A plain trie has one node per token, so a 2,048-token prefix is a 2,048-node chain — wasteful. A **radix tree** (compressed trie / Patricia trie) collapses each chain of single-child nodes into one node whose edge is labeled with the whole token *sequence*. So an unbranched 2,048-token prefix is a single node with a 2,048-token edge label, splitting into children only where different sessions diverge. This compression is what makes the tree compact (proportional to the number of distinct branches, not total tokens) and the operations efficient.

### 3.2 Node structure

A `RadixNode` (conceptually) holds:

- **`children: Dict[token, RadixNode]`** — keyed by the first token of each child's edge, so a lookup can pick the right child in O(1).
- **`parent: RadixNode`** — for upward traversal during eviction.
- **`key: List[token]`** — the token sequence on the edge from the parent to this node (the compressed segment).
- **`value: KV blocks`** — the physical KV cache for the tokens on this edge (the K/V computed for those positions).
- **`ref_count: int`** — how many active requests are currently using this node's KV (for safe eviction).
- **`last_access_time`** — for LRU eviction.
- **`lock` / pinned flag** — prevents eviction while a request is actively using the node.

A path from the root to any node spells out a token sequence, and the concatenated `value` KV blocks along that path are the KV cache for that sequence. The tree thus represents *all* cached prefixes across all sessions simultaneously, with shared prefixes physically shared (one node, referenced by many sessions).

### 3.3 Why a tree (not a hash map)

vLLM's APC uses a hash map (prefix hash → blocks), which efficiently matches a *linear* prefix but doesn't represent the *relationships* between prefixes. The radix tree explicitly encodes the branching structure: when session A and session B share a 2,048-token prefix then diverge, the tree has one node for the shared prefix and two child branches — the sharing is structural and automatic. This matters for tree-of-thought (many branches from one root), multi-turn chat (each turn extends a path), and any workload where the set of cached sequences has a natural tree structure (which is most multi-call workloads). The tree also supports *partial* (token-granularity) matching naturally — a new sequence matches down the tree token by token, splitting an edge if it diverges mid-segment (§4).

---

## 4. Radix Tree Operations: Insert, Match, Evict

### 4.1 Longest-prefix match

`match_prefix(token_sequence) → (matched_length, kv_blocks, node)`: starting at the root, greedily follow the tree matching the input tokens. At each node, look up the child keyed by the next token; if found, match along that child's edge label as far as the tokens agree. The match ends when the tokens diverge from an edge or no matching child exists. Return the length matched, the KV blocks along the matched path, and the node reached. Complexity: **O(matched_prefix_length)** — proportional to how much matched, not the tree size. This is the operation run for every incoming request to find its reusable prefix.

### 4.2 Insert (with edge splitting)

After a request computes KV for its (previously uncached) suffix, that KV is inserted into the tree so future requests can reuse it. `insert(token_sequence, kv_blocks)`: traverse as in match; when the tokens diverge mid-edge, **split** that edge — create an intermediate node at the divergence point (the existing node's KV is split between the new intermediate node and the remainder), then attach the new branch. Edge splitting is what enables token-granularity sharing: if a new sequence shares the first 1,000 tokens of an existing 2,048-token edge then diverges, the edge splits at token 1,000, the shared 1,000-token prefix becomes a parent node (shared), and the two divergent tails become children. Complexity: O(inserted_length).

### 4.3 Eviction

When the KV pool is full and new allocation is needed, evict cached KV. RadixAttention uses **LRU eviction on the tree**:

- Evict the node with the smallest `last_access_time` (least recently used).
- **Cannot evict locked/in-use nodes** (`ref_count > 0` or pinned) — a node whose KV an active request is reading must not be reclaimed.
- **Evict leaf nodes first:** leaves are the least-shared (deepest, most-specific) KV; interior nodes are shared by more sessions and should be retained longer. Eviction propagates upward only when all of a node's children are evicted (an interior node becomes a leaf once its children are gone, then is itself eligible).
- Eviction frees the node's KV blocks back to the pool (File 03's block allocator analog, §9).

This LRU-leaf-first policy keeps the *hot, shared* prefixes (system prompts, common few-shot sets — high in the tree, frequently accessed) resident while reclaiming *cold, specific* tails. It is the tree-structured generalization of vLLM's LRU evictor (File 03 §25.3), exploiting the tree to evict the least valuable (most specific) KV first.

---

## 5. The RadixAttention Algorithm

Putting the operations together, RadixAttention augments generation with automatic cache reuse:

1. **On each request (or LLM call within a program):** run `match_prefix` on the request's tokens against the radix tree. This returns the matched prefix length `p` and its cached KV blocks.
2. **Reuse:** the first `p` tokens need no computation — their KV is the matched cached blocks. Increment the matched nodes' `ref_count` (and lock them) so they aren't evicted while this request uses them. The request's `num_computed_tokens` starts at `p` (File 02 §19).
3. **Compute the suffix:** prefill only the tokens after position `p` (the non-cached suffix), attending over the cached prefix KV plus the new suffix KV. This is the work saved — only the suffix is computed.
4. **Insert:** after computing the suffix KV, `insert` it into the tree so subsequent requests sharing this (now longer) prefix can reuse it.
5. **Generate:** continue autoregressive decoding, appending each new token's KV to the tree (extending the request's path).
6. **On completion:** decrement the `ref_count` of the nodes the request used (unlock them), making them eligible for eviction once no active request needs them.

The elegance is that this is *automatic* — no explicit cache API. Every request's tokens are matched against the tree; whatever prefix is already cached is reused; whatever is new is computed and added. Multi-turn chat, few-shot, RAG, and tree-of-thought all benefit without any workload-specific code, because they all produce token sequences with shared prefixes that the tree naturally captures.

---

## 6. Multiple Sessions and Tree-Structured Sharing

The power of the radix tree shows when many concurrent sessions share prefixes in complex ways.

Consider a server handling: 100 chat sessions all sharing a common system prompt, where each session has its own conversation history; plus a few-shot classification workload sharing a different prompt; plus a tree-of-thought program forking 8 branches from one context. The radix tree represents *all* of this in one structure:

- The common system prompt is a single high node, referenced (ref_count) by all 100 chat sessions.
- Each chat session is a distinct path branching off the system-prompt node, extended turn by turn.
- The few-shot prompt is a separate high node with its own subtree.
- The tree-of-thought root is a node with 8 child branches, each an explored continuation.

Every shared span is physically stored once and referenced by all sessions that share it, via reference counting. A session's KV is the path from root to its current leaf; sessions sharing a prefix share the upper portion of their paths. This is fundamentally a *trie across all concurrent sessions' token sequences*, with the KV cache attached to the edges. No hash-based scheme represents this as naturally — the tree *is* the sharing structure, making both lookup (longest-prefix-match) and eviction (leaf-first LRU) operate directly on the actual sharing relationships.

### 6.1 The multi-turn chat case, traced

A chat session over the radix tree:

- **Turn 1:** system prompt (512 tok) + user Q1 (40 tok). `match_prefix` finds the system prompt already cached (shared with other sessions) → reuse 512 tokens, compute only Q1's 40. Generate A1 (100 tok), inserting Q1+A1 as a branch off the system-prompt node. The session's path is now `[system] → [Q1 A1]`.
- **Turn 2:** the client resends system + Q1 + A1 + Q2. `match_prefix` matches the full `[system][Q1 A1]` path (cached from turn 1) → reuse 652 tokens, compute only Q2. The conversation's KV is reused, not recomputed.
- **Turn N:** each turn matches the entire prior conversation path and computes only the new turn. The per-turn cost is `O(new turn)`, not `O(conversation length)` — the difference between a chat app that stays fast and one that slows every turn (File 03 §25.4 shows vLLM's APC achieving the same for linear histories; RadixAttention additionally handles *branching* conversations, e.g. regenerating a turn creates a branch, naturally).

---

## 7. Fork and Join

SGLang's frontend (§12) exposes `fork` and `join` primitives for parallel exploration, and RadixAttention makes them cheap.

- **`fork(session, n)`:** create `n` child sessions branching from the current point. Each child shares the parent's entire KV path (the tree node is referenced by all `n` children, ref_count += n) — **copy-on-write semantics**: no KV is copied at fork, only the block-table references. The children diverge as they generate, creating `n` branches in the tree (CoW on write, exactly like vLLM's parallel sampling, File 03 §5, but expressed at the program level).
- **`join`:** the program aggregates results from the branches (in application logic — selecting the best, voting, etc.). Join does *not* merge KV; it merges *results*. The branches' KV is freed (ref_count decremented) when no longer needed.

This makes tree-of-thought, beam-search-like exploration, and self-consistency efficient: the shared root context is computed once and shared across all branches, with only the divergent exploration computed per branch. For an 8-way fork from a 5,000-token context, the 5,000-token root KV is shared (computed once), and only the 8 branches' new tokens are computed — versus 8× recomputation without the tree. The frontend's awareness of the fork structure (§12) lets SGLang schedule the branches efficiently, batching their generation.

---

## 8. RadixAttention vs Hash-Based Prefix Caching

A direct comparison with vLLM's Automatic Prefix Caching (File 03 §6):

| Dimension | vLLM APC (hash-based) | SGLang RadixAttention |
|---|---|---|
| Data structure | hash map (prefix hash → blocks) | radix tree (compressed trie) |
| Match granularity | block (e.g. 16 tokens) | token (any length) |
| Partial-block match | no (only full blocks) | yes (edge splitting) |
| Sharing structure | linear prefix | tree (multi-session branching) |
| Lookup cost | O(num_blocks) hash lookups | O(matched_length) tree traversal |
| Eviction | LRU on unreferenced blocks | LRU leaf-first on tree nodes |
| Overhead | low (hash) | higher (tree management) |
| Best for | shared system prompts, simple linear reuse | multi-turn, few-shot, tree-of-thought, RAG, complex sharing |

The trade-off: APC is simpler and lower-overhead, excellent for the common case of a shared linear system prompt. RadixAttention is more powerful — token-granularity matching (no wasted partial blocks), tree-structured sharing (multi-session, branching), and natural fork/join support — at the cost of more complex tree management and slightly higher per-request overhead. For workloads dominated by complex multi-call structure (the workloads SGLang was designed for), RadixAttention's richer model pays off; for simple high-QPS chat with one system prompt, the two are closer. Both rest on the same underlying capability PagedAttention introduced — sub-request KV addressing and sharing via reference counting (File 03 §14); RadixAttention is, in essence, a more sophisticated *policy* (tree-structured, token-granular) on top of a paged block pool.

### 8.1 Token-granularity and the block-size connection

RadixAttention's token-granularity matching is enabled by SGLang's use of fine-grained (token-level) KV blocks (File 03 §13). Where vLLM's 16-token blocks force block-granularity matching (a prefix ending mid-block can't share the partial block), SGLang's token-level granularity lets a prefix of *any* length be shared exactly. The cost is more metadata (a block-table entry per token rather than per 16 tokens), which SGLang manages with the two-level `ReqToTokenPool`/`TokenToKVPool` indirection (§9). This is the concrete memory-design difference between the engines: vLLM trades match granularity for lower metadata overhead; SGLang trades metadata overhead for exact, token-granular sharing — each choice coherent with its prefix-caching philosophy.

---

## 9. SGLang KV Cache Block Management

SGLang's memory management uses a two-level indirection that supports token-granularity blocks and the radix tree.

### 9.1 The two pools

- **`ReqToTokenPool`:** maps a request ID to the sequence of token-buffer indices for that request's tokens. Essentially, for each request, a list of slots (one per token) into the token-level KV buffer. This is the request's "block table" at token granularity.
- **`TokenToKVPool`:** maps token-buffer indices to the actual physical KV cache memory (the K/V tensors). This is the physical KV pool.

The two-level scheme (`request → token indices → physical KV`) decouples the request's logical token sequence from the physical KV layout, allowing flexible reuse: shared prefix tokens (via RadixAttention) point to the *same* physical KV slots across requests, while each request's `ReqToTokenPool` entry records which slots are its tokens. This indirection is how SGLang achieves token-granularity sharing without exploding memory — the physical KV for a shared prefix exists once, and many requests' token-index lists reference it.

### 9.2 The physical block pool

A pre-allocated pool of KV buffers (like vLLM's, File 03 §7), managed with a free list for O(1) allocate/free. SGLang's `--mem-fraction-static` (File 11) controls how much GPU memory is reserved for static allocation (weights, CUDA graph buffers) vs the KV pool. The radix tree's nodes hold references into this pool; eviction (§4.3) frees pool slots.

### 9.3 Reference counting for sharing

As in the tree (§3.2), shared prefix KV has `ref_count > 1`; a node's KV is freed only when `ref_count == 0` and it's selected for eviction. Decode-only new tokens have `ref_count == 1`. The reference counting is the correctness core (like vLLM's, File 03 §5.4) — a bug means premature free (corruption) or leak. The radix tree's ref_count on nodes and the pool's ref_count on physical slots together track sharing across all sessions.

### 9.4 Metadata overhead

Token-granularity means more metadata: a per-token entry in `ReqToTokenPool` and the radix-tree node bookkeeping. The tree itself is compact (one node per branch, not per token, §3.1) — for 1M cached tokens with average segment length 64, ~15K nodes × ~200 bytes ≈ 3 MB of tree metadata, negligible against the KV cache itself (gigabytes). So the token-granularity choice's overhead is manageable, and the radix-tree compression keeps the structural metadata small.

---

## 10. The SGLang Scheduler and RadixAttention

SGLang's scheduler (`sglang/srt/managers/scheduler.py`, full architecture File 09) integrates RadixAttention into continuous batching.

### 10.1 Cache-aware scheduling

Before scheduling each request, the scheduler runs `match_prefix` to compute the request's prefix-hit length — i.e. how much of its prompt is already cached and thus free to "compute." This lets the scheduler:

- **Size the actual prefill work** correctly (only the non-cached suffix counts against the token budget).
- **Optionally prioritize cache-warm requests** — those with large prefix hits are cheap to run (little compute), so scheduling them is high-throughput. SGLang can order admission to favor cache-warm requests, a policy enabled by its explicit prefix-hit-length computation (File 04 §20, §24).

### 10.2 Continuous batching with RadixAttention

The scheduler's loop (detailed in File 09 §continuous batching):
1. Check the waiting queue for new requests.
2. For each, run RadixAttention `match_prefix` → prefix hit length.
3. Allocate KV (token slots) for the *non-cached* suffix only.
4. Merge prefill and decode requests into one batch (varlen, File 02 §20).
5. Enforce the token budget (`max_total_num_tokens`).
6. Build the `ModelWorkerBatch` and dispatch to the executor.

After each step, update each request's state (append new token, check stop conditions, extend its radix-tree path), remove finished requests (decrement ref_counts), and promote from the waiting queue if budget allows.

### 10.3 Chunked prefill and request states

SGLang supports chunked prefill (`--chunked-prefill-size`, File 04 §7), interleaving prefill and decode, with chunks aligned to radix-tree boundaries where possible. Request states (`Waiting`, `Running`-prefill, `Running`-decode, `Finished`) are managed by the scheduler; aborts (client disconnect) free the request's non-shared KV and decrement ref_counts on shared nodes (File 04 §28). The integration is tight: the scheduler is *the* place RadixAttention's match/insert/evict operations are invoked, woven into the per-step batching decisions.

---

## 11. XGrammar — Structured Output Engine

SGLang's second major contribution (alongside RadixAttention) is **XGrammar** (Zheng et al., arXiv 2411.15100), a constrained-decoding engine that makes grammar-constrained generation (JSON, regex, CFG) near-free in latency. Foundations are in File 02 §10; here is the depth.

### 11.1 The problem restated

Constrained decoding masks, at every step, all tokens that would violate the grammar (set their logits to `−∞`). The naive cost is high: vocabularies are 32K–256K tokens, and computing "which tokens are valid in the current grammar state" each step, for each sequence, is expensive — early implementations spent 5–15% of decode latency on it. For production structured output (function calling, JSON APIs, agents), this overhead is unacceptable at scale.

### 11.2 Push-down automata and precomputed masks

XGrammar compiles the grammar to a **push-down automaton (PDA)** — more expressive than a finite-state machine, able to handle the *recursive*, *nested* structures of context-free grammars (e.g. nested JSON objects/arrays, balanced brackets) that a flat FSM cannot. The compilation pipeline: JSON schema → EBNF grammar → LL(1) parser tables → PDA. 

The key optimization is **precomputing per-state allowed-token masks**. For each PDA state, XGrammar precomputes (offline, at compile time) the bitmask over the vocabulary of which tokens are allowed in that state. At inference, masking is then just: look up the current PDA state → fetch its precomputed token mask → apply to the logits. This is **O(1) per decode step** (a mask lookup and apply) instead of O(vocab_size) per-step FSM evaluation. An **adaptive token-mask cache** caches masks for recently-seen grammar states, so even dynamically-encountered states are fast after first computation.

### 11.3 Compilation optimizations

Grammar compilation includes: expanding nullable rules, factoring common prefixes, identifying equivalent states (state minimization), and handling the tokenizer's subword structure (a grammar terminal may span multiple tokens, or a token may straddle grammar terminals — XGrammar reconciles the byte-level grammar with the token vocabulary). Compilation is fast — typically <100 ms for a typical JSON schema — so it can be done per-request for arbitrary schemas without prohibitive latency, and cached for repeated schemas.

### 11.4 Integration with SGLang and batched masking

At inference, a `Grammar` object wraps the compiled grammar and a `GrammarMatcher` tracks the current PDA state for each constrained sequence (advancing one token per step, File 04 §29). Multiple requests with *different* grammars are processed in one batch — each carries its own matcher state, and a **CUDA kernel applies each sequence's mask to its logits in parallel** across the heterogeneous batch. The matcher advance and next-mask computation are CPU work overlapped with the GPU forward pass (File 04 §29), so the structured-decoding overhead stays off the critical path.

### 11.5 Performance

XGrammar achieves **<1% decode-step overhead** for typical schemas — versus 5–15% for naive per-step FSM evaluation. This near-zero overhead is what makes structured output viable in production at scale: function calling and JSON APIs (File 07 §12, File 20 §agents) can be the *default* rather than an expensive option. XGrammar has been adopted beyond SGLang (including by vLLM as a guided-decoding backend), a sign of its impact. It is the structured-output counterpart to RadixAttention's prefix reuse — both take a workload pattern (multi-call sharing; constrained output) that naive engines handle expensively and make it nearly free through a better algorithm and data structure.

### 11.6 Preserving the output distribution

A subtle but important property: well-implemented constrained decoding masks invalid tokens and *renormalizes* over the valid ones, so the model still samples according to its (constrained) probability distribution — it doesn't just greedily pick the first valid token. This preserves output quality within the grammar: the model expresses its preferences among valid continuations. XGrammar applies the mask before the sampler's normalization (File 06 §10), so temperature/top-p still operate on the valid token set. The combination of correctness (always valid output) and quality (sampling within the constraint) is what distinguishes a production-grade grammar engine from a naive "reject invalid tokens" loop.

---

## 12. The SGLang Frontend Language

SGLang's name — *Structured Generation Language* — refers to its **frontend DSL**, a Python-embedded language for expressing multi-call LLM programs, tightly coupled to the RadixAttention backend. This frontend has no vLLM equivalent and is the source of SGLang's program-level optimizations.

### 12.1 The DSL

SGLang programs are Python functions decorated with `@sgl.function`, using primitives:

- **`sgl.gen(name, ...)`:** generate text (optionally constrained by `regex=`, `json_schema=`, `choices=`).
- **`sgl.select(name, choices)`:** force the model to pick from a discrete set of options — implemented efficiently by computing logits only for the first diverging token among the choices (you don't generate full options, just enough to disambiguate).
- **`sgl.fork(n)`:** branch into `n` parallel continuations from the current context (§7).
- **`sgl.image(...)`:** include an image (multimodal, File 14).
- String operations to build prompts, plus normal Python control flow (loops, conditionals) interleaved with LLM calls.

A program reads like ordinary Python but expresses an LLM *computation graph* — the sequence and structure of LLM calls, their shared context, and their branching.

### 12.2 Interpreter and lazy execution

`@sgl.function`-decorated functions compile to a `Program` object; an `Interpreter` executes it on the `Runtime`. Execution is *lazy* in the sense that the runtime sees the program's structure and can optimize across calls: batching multiple `gen()` calls from concurrent sessions, parallelizing independent `fork` branches, and ordering execution for maximum GPU utilization. This is the program-aware scheduling that distinguishes SGLang (File 04 §24) — the runtime exploits structural knowledge a request-level engine can't see.

### 12.3 Structural (ahead-of-time) batching

The headline frontend optimization: SGLang can batch LLM calls from *different users running the same program* at the call level. If two users invoke the same `@sgl.function` with different inputs, the runtime recognizes the shared program structure and batches the corresponding LLM calls together — even though the users are independent. This **structural batching** is unique to SGLang's program-aware design; a request-level engine sees two unrelated requests and can only batch them opportunistically via continuous batching, whereas SGLang knows they have the same structure and can batch them deliberately, sharing the program's common prefixes via RadixAttention. The frontend and backend co-design — the DSL exposes structure, RadixAttention reuses shared KV, the scheduler batches structurally — is the integrated system SGLang's paper describes.

### 12.4 Constrained generation primitives

The frontend's constrained primitives map to XGrammar (§11):
- **`sgl.gen(regex=r"\d{3}-\d{4}")`:** regex-constrained generation.
- **`sgl.gen(json_schema=schema)`:** JSON-schema-constrained — guarantees valid, conforming JSON (function calling, File 07 §12).
- **`sgl.select(choices=["yes","no"])`** / **`sgl.gen(choices=...)`:** discrete choice, efficiently computing only the disambiguating logits.

These make structured programs (extraction, classification, tool use, form-filling) both reliable (always valid output) and efficient (XGrammar's near-zero overhead, `select`'s minimal-computation choice). The frontend thus unifies SGLang's three innovations — RadixAttention (reuse shared context across calls), XGrammar (constrain output cheaply), and program-aware scheduling (batch and parallelize across the program structure) — into a single coherent programming model for structured LLM applications.

---

## 13. A Worked Radix-Tree Trace

To cement the data structure, trace the tree through a sequence of requests. Tokens are shown as letters for clarity; in reality they are token IDs and edges hold thousands.

**State 0 — empty tree:** just a root node.

**Request A: tokens `S Y S Q1`** (system prompt `SYS` + query `Q1`). `match_prefix` finds nothing (empty tree) → compute all of `SYSQ1`, then `insert`. Tree:
```
root ── "SYSQ1" (node A, KV for 5 tokens, ref=1 while A runs)
```
A generates answer `A1`, extending its path: `root ── "SYSQ1A1"` (the generated tokens are appended to A's node/path).

**Request B: tokens `S Y S Q2`** (same system prompt, different query). `match_prefix` walks from root: matches `SYS` along node A's edge, then diverges (A has `Q1`, B has `Q2`). **Edge split** at position 3: node A's `"SYSQ1A1"` edge splits into a parent node `"SYS"` (shared, the system prompt KV) and a child `"Q1A1"` (A's tail). B becomes a second child:
```
root ── "SYS" (shared, ref=2)
          ├── "Q1A1"  (request A's tail)
          └── "Q2"    (request B's tail, newly computed)
```
B reused the 3-token `SYS` prefix (no recomputation) and computed only `Q2`. The split happened automatically because the tokens diverged mid-edge — token-granularity matching in action (§4.2).

**Request C: tokens `S Y S Q1 A1 Q3`** (a follow-up turn after A's conversation). `match_prefix` matches `SYS` then `Q1A1` (both cached, from A) → reuses 5 tokens, computes only `Q3`:
```
root ── "SYS" (ref=3)
          ├── "Q1A1" (ref=2)
          │      └── "Q3"  (request C's new turn)
          └── "Q2"
```
C is a multi-turn continuation reusing the entire prior conversation — the chat case (§6.1). 

**Eviction:** suppose memory fills and `Q2` (request B, now finished, ref=0) is the least-recently-used leaf. It's evicted (its KV freed), leaving its parent. `SYS` (high in the tree, frequently accessed, ref>0) is retained — leaf-first LRU keeps the hot shared prefix (§4.3). 

This trace shows all four operations — match, insert, edge-split, evict — and why the tree captures the sharing structure exactly: `SYS` shared by three requests, `Q1A1` shared by A and C, each computed once.

---

## 14. Distributed RadixAttention

RadixAttention extends to multi-GPU and multi-replica settings, each with considerations.

### 14.1 Across tensor-parallel ranks

Under TP (File 05 §2), the KV cache is sharded by head across ranks, but the radix tree's *structure* (which token sequences are cached, the ref counts, the eviction order) is replicated and kept consistent across ranks — like vLLM's replicated block tables (File 03 §19). All ranks make identical match/insert/evict decisions (driven by the scheduler) so their sharded KV stays consistent; each rank holds its head-shard of every cached node's KV. A divergence would corrupt the cache, so the tree operations are coordinated centrally and applied in lockstep, exactly as block-table mutations are in vLLM.

### 14.2 Across data-parallel replicas

With multiple DP replicas (File 05 §7), each replica has its *own* radix tree (its own KV pool). This means a prefix cached on replica 1 is *not* automatically available on replica 2 — so naive round-robin load balancing scatters a shared prefix across replicas, and each must compute it (low aggregate hit rate). The fix is **prefix-cache-aware routing** (File 05 §7.2, File 04 §20): route requests with the same prefix to the same replica, concentrating that prefix's cache hits on one tree. SGLang deployments use such routing (consistent hashing on the prefix) to make RadixAttention effective across replicas — turning per-replica caching into effective cluster-wide caching via routing. This is the data-parallel analog of the intra-engine cache awareness, and it's essential: without it, the multi-call efficiency RadixAttention provides within a replica is diluted across the fleet.

### 14.3 RadixAttention in disaggregation

In a prefill–decode disaggregated setup (File 04 §10, File 15), prefill workers maintain the radix tree (prefix caching happens during prefill). Routing prefix-sharing requests to the same prefill worker maximizes its tree's hit rate (Mooncake's prefix-aware routing, File 04 §10.4). The decode workers don't need the tree (they continue generation on transferred KV). So RadixAttention lives on the prefill side in disaggregation, and the routing that concentrates prefixes there is what preserves its benefit — again, caching effectiveness depends on routing.

---

## 15. Multimodal RadixAttention (Image Prefix Caching)

RadixAttention is a natural fit for **vision-language** workloads (File 06 §9, File 14). The image's KV (the LLM's KV for the image-feature tokens) is fixed for a given image regardless of the question asked about it. So in multi-turn visual QA — "describe this image," then "what color is the car?", then "how many people?" — all referencing the same image, the image's KV (often hundreds of tokens) is computed once and reused across all questions via the radix tree. The image bytes (or a hash) key the tree path, so the same image with different questions matches the cached image prefix and computes only each question's tokens. For applications doing many queries per image (visual search, document understanding, iterative editing), this is a large saving — and RadixAttention provides it automatically, the same way it caches text prefixes. SGLang's `sgl.image()` primitive (§12.1) and its preprocessing pipeline (File 09) integrate images into the prefix-caching flow. This is one reason SGLang is strong for multimodal serving: the prefix-caching mechanism that helps text multi-call workloads helps image-grounded multi-turn workloads identically.

---

## 16. Benchmark Results and the SGLang Paper

The SGLang paper (arXiv 2312.07104) and subsequent benchmarks demonstrate RadixAttention's impact:

- On workloads with significant prefix sharing (multi-turn chat, few-shot, tree-of-thought, JSON decoding, multi-call agents), SGLang showed **up to several-fold throughput improvements** over baseline systems without automatic fine-grained KV reuse, driven by the cache-hit reduction in prefill compute and memory (§2's ~40× prefill reduction for heavy-sharing cases).
- The combination of RadixAttention + structural batching + XGrammar made structured, multi-call programs dramatically more efficient than running them as independent constrained-decoding requests on a generic engine.
- General-traffic comparisons (File 11 §vLLM vs SGLang): SGLang typically leads by 10–30% on prefix-sharing-heavy workloads (RadixAttention benefit); the two are close, or vLLM leads, on workloads *without* shared prefixes (where RadixAttention's tree overhead is pure cost with no reuse to offset it). Both improve rapidly; the right choice depends on the workload's sharing structure.

The deeper claim of the paper is *co-design*: by exposing program structure in the frontend and exploiting it in the backend (RadixAttention reuse, structural batching, constrained decoding), SGLang serves structured LLM programs far more efficiently than treating each LLM call as an opaque, independent request. For the growing class of agentic, multi-call, structured-output workloads (File 20 §agents), this co-design is increasingly valuable.

---

## 17. Frontend Programs by Example

Concrete SGLang programs illustrate how the DSL expresses structure that the runtime exploits.

### 17.1 A multi-step extraction program

```python
@sgl.function
def extract_info(s, document):
    s += sgl.system("You are an information extraction assistant.")
    s += sgl.user("Document:\n" + document + "\nExtract the fields below.")
    s += sgl.assistant_begin()
    s += "Name: " + sgl.gen("name", stop="\n")
    s += "\nDate: " + sgl.gen("date", regex=r"\d{4}-\d{2}-\d{2}")
    s += "\nCategory: " + sgl.select("category", ["finance", "legal", "medical", "other"])
```

This single program makes several constrained LLM calls sharing the document context. RadixAttention caches the system prompt + document KV (computed once), so each `gen`/`select` extends the shared context cheaply. The `regex` on `date` guarantees a valid date format (XGrammar); the `select` on `category` forces one of four labels, computing only the disambiguating tokens (§17.2). When many documents are processed, the *system prompt* is shared across all (cached once); when the same document is queried repeatedly, the *document* is shared too.

### 17.2 The `select` primitive internals

`sgl.select(name, choices)` forces the model to choose among a fixed set of strings. Naively you'd generate each choice fully and compare likelihoods, but that's wasteful. SGLang's efficient implementation: compute the model's logits and determine which choices remain consistent with the generated tokens so far, advancing only as far as needed to *disambiguate* among the choices. If the choices are `["yes", "no"]`, the very first token usually disambiguates (`yes` vs `no` differ at token 1), so `select` computes essentially one token's logits rather than generating two full words. For choices sharing prefixes (`["cardiology", "cardiovascular"]`), it advances through the shared prefix and disambiguates at the diverging token. This makes discrete classification/choice extremely cheap — a key efficiency for structured programs that do a lot of categorical decisions (routing, classification, multiple-choice). It's a frontend-backend co-design: the frontend knows the choice set, so the backend can compute the minimal logits to decide.

### 17.3 A tree-of-thought program with fork

```python
@sgl.function
def tree_of_thought(s, problem):
    s += sgl.user(problem)
    s += sgl.assistant("Let me consider several approaches.")
    forks = s.fork(3)                      # 3 branches sharing the problem context
    for i, f in enumerate(forks):
        f += sgl.gen(f"approach_{i}", max_tokens=200)   # each explores independently
    # join: aggregate in Python
    best = select_best([f["approach_" + str(i)] for i, f in enumerate(forks)])
    s += sgl.assistant("Best approach: " + best)
```

The `fork(3)` creates three branches sharing the problem + preamble KV (computed once, shared via RadixAttention CoW, §7). The three explorations are computed in parallel (the runtime batches them), each adding only its own tokens. Without RadixAttention, the shared context would be recomputed three times. This is the tree-of-thought pattern (File 15 §parallel reasoning) expressed naturally, with the engine providing the efficiency automatically.

---

## 18. Cache Tuning in SGLang

RadixAttention's effectiveness depends on configuration (full methodology File 11):

- **`--mem-fraction-static`** (default ~0.88): the fraction of GPU memory for static allocation (weights + CUDA graph buffers); the remainder is the KV pool that the radix tree uses. Lower it to give more KV to the cache (more prefix retention, higher hit rate) if weights+graphs leave too little; raise it if you're OOMing on static allocation.
- **`--max-running-requests`:** concurrency cap (like vLLM's `max_num_seqs`, File 04 §6.2); set to match sustainable KV memory to avoid thrash.
- **`--chunked-prefill-size`:** chunked-prefill token budget (File 04 §7), radix-tree-aligned where possible.
- **Cache retention vs eviction pressure:** the larger the KV pool relative to the working set of hot prefixes, the higher the hit rate (the evictor thrashes less, File 03 §31). For workloads with a few large shared prefixes (one system prompt, fixed few-shot set), even a modest pool retains them with high hit rate. For workloads with many distinct prefixes exceeding the pool, the hit rate degrades as the evictor cycles — the same capacity reasoning as any cache.

The tuning principle: size the static fraction so the KV pool comfortably holds the hot-prefix working set plus active-request KV, and cap concurrency below the memory limit. Monitor the prefix-cache hit rate (logged per request, File 11 §SGLang debug) — a low hit rate on a known prefix-heavy workload signals the pool is too small (raise KV share) or routing is scattering prefixes across replicas (fix routing, §14.2).

---

## 19. RadixAttention Overhead Analysis

RadixAttention is not free — its overhead is the price of token-granular, tree-structured reuse, and it's worth understanding when that price is worth paying.

- **Per-request match cost:** `match_prefix` is O(matched_length) tree traversal — cheap, proportional to the prefix found. For a request with no shared prefix, the match fails quickly (no matching child at the root) — minimal cost.
- **Insert/split cost:** O(inserted_length), and edge splits are O(1) structural operations. Inserting a request's suffix is cheap.
- **Eviction cost:** O(log n) or O(n) depending on the LRU structure (a priority queue or sorted access-time structure over evictable leaves). Manageable since evictions are bounded by allocation pressure.
- **Tree metadata memory:** ~200 bytes/node, compact due to radix compression (§9.4) — negligible vs the KV cache.
- **The break-even:** RadixAttention pays off when the prefix-reuse savings (skipped prefill compute + shared KV memory) exceed the tree-management overhead. For prefix-sharing workloads, the savings are enormous (§2's 40×) and dwarf the overhead. For workloads with *no* sharing (every request a unique prompt), there's no reuse to offset the (small but nonzero) tree overhead, so a simpler scheme (or vLLM's lower-overhead APC) may be marginally faster — which is why benchmarks show SGLang leading on prefix-heavy workloads and roughly tied or behind on prefix-free ones (§16, File 11). The engineering judgment: RadixAttention's overhead is small and its upside is large *when sharing exists*, so it's a strong default for the multi-call, agentic, multi-turn workloads SGLang targets, and a wash for purely independent single-shot requests.

---

## 20. XGrammar Compilation, Worked

To make §11 concrete, trace how a JSON schema becomes a runtime mask source. Consider the schema for `{"name": string, "age": integer}`.

1. **JSON schema → EBNF grammar.** The schema is translated to a context-free grammar describing all valid serializations: an object starts with `{`, then the key `"name"` (a string literal), then `:`, then a string value (`"` followed by any non-quote characters followed by `"`), then `,`, then `"age"`, `:`, an integer (a sequence of digits), then `}` — with whitespace allowed between tokens. EBNF rules capture the recursion (strings, numbers) and the structure.
2. **EBNF → LL(1) parse tables → PDA.** The grammar compiles to a push-down automaton. The stack handles nesting (objects within objects, arrays) that a flat FSM couldn't; the states encode "where in the structure are we" (expecting a key, inside a string value, expecting a digit, etc.).
3. **Per-state token masks.** For each PDA state, XGrammar precomputes which vocabulary tokens are valid. In the "expecting the opening brace" state, only tokens that start with `{` (possibly with leading whitespace) are valid — a small mask. In the "inside a string value" state, most tokens are valid (any text), except those containing an unescaped `"` that would close the string prematurely. In the "expecting a digit for age" state, only tokens that are digits (or continue a number) are valid. These masks are bitvectors over the 128K vocabulary, precomputed once.
4. **Runtime.** As the model generates, the `GrammarMatcher` tracks the PDA state. After each token, it advances the PDA (pushing/popping the stack as structure opens/closes) and the next step's mask is the precomputed mask for the new state — an O(1) lookup. The sampler applies it (zeroing invalid tokens) before sampling (§11.6). The result is guaranteed-valid JSON conforming to the schema, at <1% overhead.

The subtlety XGrammar handles well is the **tokenizer-grammar mismatch**: the grammar is defined over characters/bytes, but generation is over subword tokens. A single token might be `","` (comma) or `"name\":` (multiple grammar terminals in one token) or part of a longer string. XGrammar reconciles the byte-level PDA with the token vocabulary so that the per-token masks correctly reflect which *tokens* keep the output grammar-valid — a nontrivial alignment that naive FSM-over-characters approaches get wrong or slow.

---

## 21. Related Prefix-Sharing Systems

RadixAttention is the most general automatic prefix-sharing scheme, but it sits among related ideas worth knowing (some predate or parallel it):

- **Prompt Cache** (Gim et al.): precompute and cache KV for *reusable prompt modules* (e.g. a document, a set of instructions) so they can be assembled into prompts without recomputation. More manual/structured than RadixAttention's automatic matching, but similar spirit.
- **Hydragen** (Juravsky et al., arXiv 2402.05099): optimizes attention *computation* for shared prefixes by decomposing attention into a shared-prefix part (computed once, batched efficiently as a large GEMM) and a per-sequence suffix part. This is complementary to RadixAttention: RadixAttention reuses the shared *KV storage*; Hydragen makes the *attention over* that shared prefix more efficient (turning many sequences' attention-over-the-same-prefix into one batched operation). Combining shared-KV storage with shared-prefix attention computation is the full optimization.
- **ChunkAttention / Cascade Inference:** similar ideas of sharing prefix KV and optimizing the attention over shared vs unique portions.
- **vLLM APC** (File 03 §6): the hash-based, block-granular linear-prefix counterpart (§8).

The conceptual landscape: prefix sharing has two halves — *storage* reuse (don't store/recompute shared KV: RadixAttention, APC, Prompt Cache) and *compute* reuse (don't redundantly compute attention over the shared prefix for each sequence: Hydragen, cascade/chunk attention). RadixAttention is the leading storage-reuse mechanism; the compute-reuse optimizations layer on top. SGLang and vLLM have both explored combining them. Knowing this distinction clarifies that "prefix caching" can mean reusing KV *storage*, reusing attention *compute*, or both — and the biggest wins combine the two.

---

## 22. Correctness Edge Cases

RadixAttention's token-granular, tree-structured sharing introduces edge cases the implementation must handle (analogous to vLLM's, File 03 §35):

- **Position IDs:** a request reusing a `p`-token cached prefix must assign its suffix positions starting at `p` (File 02 §19) — the cached KV was rotated (RoPE) for positions `0..p−1`. Mis-positioning the suffix corrupts attention. The `num_computed_tokens = p` from the match drives this.
- **Sliding-window interaction:** for SWA models (File 02 §2.5), cached prefix KV outside the current window must not be served as a valid match for positions that can't attend to it. The tree and the window logic must agree on validity.
- **LoRA / multimodal keys:** requests using different LoRA adapters or different images must not share a prefix node even if their text tokens match — the adapter/image must be part of the cache key (File 03 §25.2's analog), or one request's KV (adapter-modified) would wrongly serve another.
- **Eviction of in-use nodes:** a node with `ref_count > 0` (an active request reading it) must never be evicted — the lock/pin prevents this. A bug here frees KV mid-use → corruption.
- **Edge split correctness:** splitting an edge must correctly partition the node's KV blocks between the new parent and child, preserving each request's view of its KV. An off-by-one in the split position corrupts the boundary token's KV.
- **Concurrent modification:** the tree is mutated by the scheduler each step (insert, evict, ref-count changes) while requests read it; the operations must be consistent (the scheduler is single-threaded per File 09, which simplifies this — all tree mutations happen in the scheduler's loop, avoiding concurrent-modification races).

These are the RadixAttention analogs of File 03's block-manager invariants: ref-count accuracy, position/cache consistency, key completeness (adapter/image), and safe eviction. Violations manifest as silent KV corruption (wrong tokens, not crashes) — the hardest inference bugs to diagnose, which is why the tree operations are among SGLang's most carefully tested code.

---

## 23. Modeling the Cache-Hit Benefit

Quantify RadixAttention's effect on a concrete few-shot workload to build intuition for when it dominates. Suppose a classification service: a 1,800-token few-shot prompt (instructions + 10 labeled examples) shared by every request, plus a ~60-token item to classify, generating a ~5-token label.

**Without RadixAttention** (independent requests): each request prefills `1800 + 60 = 1860` tokens and decodes 5. At 500 requests, prefill work ≈ `500 × 1860 ≈ 930,000` token-computations.

**With RadixAttention** (few-shot prompt cached after the first request): the 1,800-token few-shot prompt is computed once and cached; each subsequent request reuses it and prefills only its 60-token item. Prefill work ≈ `1800 + 500 × 60 ≈ 31,800` token-computations — a **~29× reduction** in prefill compute. Since prefill is compute-bound (File 01 §3), this frees ~29× of the prefill GPU time for more throughput, and the few-shot KV is stored once (not 500×), freeing memory for more concurrent decodes. The end-to-end throughput gain depends on the prefill/decode mix, but for this prefill-dominated, heavy-sharing workload it is large — easily 2–5× aggregate throughput, consistent with the paper's results (§16).

**Hit-rate sensitivity:** this assumes the few-shot prompt stays cached (high hit rate). If the KV pool is too small to retain it under load (many distinct prefixes competing), the evictor cycles and the hit rate drops, eroding the benefit (§18). The benefit is therefore `≈ hit_rate × (shared_prefix_fraction)` of the prefill — maximized when the hot prefixes fit comfortably in the pool. This is why sizing the KV pool to hold the working set of hot prefixes (§18) is the key tuning lever for prefix-heavy workloads, and why cluster routing must concentrate prefixes on replicas (§14.2) so each replica's pool sees a small enough working set to retain them.

---

## 24. Cache-Aware Scheduling: Benefit and Risk

SGLang's scheduler can use the prefix-hit length to *prioritize* cache-warm requests (§10.1), which has a subtle trade-off.

**Benefit:** cache-warm requests are cheap (little prefill compute), so running them first maximizes throughput — you complete many cheap requests quickly. Sorting the batch by prefix-hit length and favoring high-hit requests can raise aggregate throughput on prefix-sharing workloads.

**Risk — starvation and fairness:** if the scheduler always favors cache-warm requests, a cache-*cold* request (a new, unique prompt with no cached prefix) could be repeatedly deprioritized behind a stream of cache-warm ones, starving it. This is the cache-aware analog of priority starvation (File 04 §8.2). Mitigations: bound how much cache-warmth can reorder admission, age cold requests up in priority over time, or use cache-awareness as a tiebreaker within FCFS rather than as the primary order. The right balance depends on the workload and SLOs — pure throughput maximization favors aggressive cache-aware ordering; latency fairness favors tempering it.

**Interaction with routing:** at the cluster level, prefix-cache-aware routing (§14.2) concentrates a prefix on one replica, which raises that replica's hit rate but can *imbalance load* (the replica holding a hot prefix gets more traffic). So both intra-engine cache-aware scheduling and inter-engine cache-aware routing trade cache locality against load balance and fairness — the same tension at two levels. Production systems tune this balance (consistent hashing with bounded loads for routing, tempered cache-awareness for scheduling) to capture most of the cache benefit without severe imbalance or starvation.

---

## 25. The Runtime Interpreter

The frontend program (§12) is executed by SGLang's interpreter on the runtime. Understanding the execution clarifies the program-awareness.

When a `@sgl.function` runs, the interpreter walks the program, issuing each `gen`/`select`/`fork` as an operation on the runtime. The runtime maintains the program's state (the accumulating prompt `s`) and, crucially, sees the *structure*: it knows when calls share a prefix (the accumulating `s`), when branches fork (independent continuations), and when calls from different program instances have the same structure. This structural knowledge feeds three optimizations:

1. **RadixAttention reuse:** each `gen` call's prompt is matched against the radix tree; the shared accumulating prefix is reused automatically (the interpreter doesn't even need to know about caching — the backend matches the tokens).
2. **Parallelization of independent work:** `fork` branches are independent, so the runtime can compute them concurrently (batched), rather than sequentially.
3. **Structural batching across instances:** multiple users running the same program (e.g. the extraction program of §17.1 on different documents) produce calls with the same structure at the same program points; the runtime batches these across users (§12.3), even though the users are independent requests.

The interpreter thus turns a high-level program into an optimized schedule of batched, cache-reusing LLM calls. This is the concrete realization of "program-aware execution" — the runtime exploits the program graph that the DSL exposes, which a request-level engine (seeing only opaque independent requests) cannot. For a developer, this means writing natural multi-call programs and getting the batching, caching, and parallelization for free; for the system, it means the structure is available to optimize rather than hidden behind an API boundary.

### 25.1 Structural batching, worked

Suppose 50 users each call `extract_info(document_i)` (§17.1) concurrently. Each program does: system prompt → user(document) → `gen(name)` → `gen(date)` → `select(category)`. The runtime sees that all 50 are at the same program point (`gen(name)`) at roughly the same time, with the same system-prompt prefix (shared via RadixAttention) and different documents. It batches the 50 `gen(name)` calls into one forward pass (the system prompt shared, the 50 documents as the varying part), then the 50 `gen(date)` calls, then the 50 `select(category)` calls. This is more efficient than 50 independent requests' opportunistic continuous batching because the runtime *knows* the calls align structurally and can batch them deliberately with maximal prefix sharing. The result: 50 structured extractions run almost as efficiently as one, sharing the system prompt and batching each step — the payoff of frontend-backend co-design.

---

## 26. Hierarchical and Tiered Radix Caching

Just as vLLM extends KV to a CPU swap tier (File 03 §8) and external stores (File 03 §28), RadixAttention can be extended to a memory hierarchy. The radix tree's KV need not all live in GPU HBM: cold-but-valuable prefixes (recently evicted from GPU, or large shared prefixes used periodically) can be **offloaded to CPU RAM** (or SSD/distributed store), keeping the tree structure as the index while the KV data tiers down. A prefix hit on a CPU-resident node triggers a swap-in (PCIe transfer) rather than a recompute — worthwhile when the swap-in is cheaper than recomputing the prefix's KV (the same swap-vs-recompute trade as File 03 §9, applied to cached prefixes). This **hierarchical radix cache** raises the effective cache capacity (more prefixes retained, higher hit rate) at the cost of swap-in latency for cold hits. It's especially valuable for workloads with a large working set of hot prefixes that exceed GPU capacity (many distinct documents in RAG, many system prompts in a multi-tenant deployment) — the GPU holds the hottest, CPU/SSD holds the warm tail, and the tree indexes all of them. SGLang and the broader ecosystem (LMCache-style external KV, File 03 §28) have moved toward such tiered caching; it's the RadixAttention analog of treating KV as a managed, relocatable resource across a memory hierarchy.

---

## 27. RadixAttention and Speculative Decoding

Speculative decoding (File 12) interacts with RadixAttention because both touch the KV cache and the per-step token accounting. The draft model's proposed tokens and the target's verification extend the request's path in the tree; accepted tokens' KV is inserted, rejected tokens' speculative KV is discarded. The prefix reuse still applies — a speculative request reuses its cached prefix like any other, computing/speculating only the suffix. SGLang supports EAGLE-style speculative decoding (File 12) alongside RadixAttention; the combination compounds their benefits (prefix reuse cuts prefill, speculation cuts decode steps). The implementation care is in the tree updates under speculation's variable accepted-token count (File 04 §19) — only accepted tokens' KV becomes part of the permanent cached path; the rest is transient. The two optimizations are largely orthogonal (one reuses shared prefix KV, the other amortizes weight reads across multiple verified tokens), so they stack.

---

## 28. Grammar Engines Compared: XGrammar, Outlines, lm-format-enforcer

XGrammar is one of several constrained-decoding backends; knowing the landscape helps (File 02 §10, File 07 §12):

- **Outlines** (Willard & Louf): compiles regex/JSON-schema to an FSM and precomputes per-state allowed-token masks. Efficient for regular languages; the FSM model is simpler than a PDA but cannot natively express the full recursion of arbitrary CFGs (deeply nested structures). Widely used and integrated into vLLM.
- **lm-format-enforcer:** another schema-enforcement library, token-by-token validity checking with a focus on broad format support.
- **XGrammar:** PDA-based (handles true CFGs, nested/recursive structures), with aggressive mask precomputation and the adaptive mask cache for <1% overhead (§11). Generally the fastest for complex/nested schemas and adopted across engines (SGLang natively, vLLM as a backend option).

The trade-offs: FSM-based engines (Outlines) are simple and fast for regular constraints (regex, flat JSON); PDA-based XGrammar handles the recursive structures (nested objects, balanced grammars) that FSMs approximate awkwardly, at comparable or better speed due to its precomputation. For production JSON/tool-calling, XGrammar's combination of correctness (true CFG support) and speed (near-zero overhead) made it influential. The engine choice is usually exposed as a backend flag (`--guided-decoding-backend`), and for most JSON-schema workloads any of them works, with XGrammar preferred for complex schemas and lowest overhead.

---

## 29. A Deeper vLLM Comparison

Beyond the table in §8, the SGLang/vLLM comparison on prefix handling reflects a philosophical difference (File 17 for the full engine comparison):

- **vLLM** treats requests as independent and adds prefix caching (APC) as an *optimization* on top of a request-level engine — block-granular, hash-based, linear-prefix. It's simple, low-overhead, and excellent for the common shared-system-prompt case, and it integrates cleanly with vLLM's broad model/feature support.
- **SGLang** treats *programs* as the unit and builds the engine *around* fine-grained reuse — token-granular, tree-structured RadixAttention, plus a program-aware frontend that exposes structure for batching and the structured-output engine for cheap constraints. It's a more specialized, more powerful design for multi-call/structured workloads.

Neither is universally better. For high-QPS chat with one system prompt and independent requests, vLLM's APC captures most of the benefit at lower overhead, and vLLM's broader ecosystem (model support, quantization breadth, community) is an advantage. For agentic, multi-turn, few-shot, tree-of-thought, and structured-output-heavy workloads — the multi-call programs SGLang was designed for — RadixAttention's token-granular tree reuse, structural batching, and XGrammar give SGLang a real edge. The two have also converged: vLLM adopted XGrammar as a guided-decoding backend and improved its prefix caching; SGLang broadened its model support and serving features. The choice is increasingly workload-driven rather than capability-driven, which is the healthy result of two strong designs cross-pollinating (File 17, File 20 §competitive dynamics).

---

## 30. Eviction Policy in Depth

The LRU-leaf-first eviction (§4.3) deserves elaboration because it determines the cache's effectiveness under pressure. The policy must answer: when the pool is full and we need `k` blocks, which nodes' KV do we reclaim?

- **Eligibility:** only nodes with `ref_count == 0` (no active request using them) and not pinned. In-use nodes are protected (evicting them would corrupt live requests).
- **Leaf-first:** among eligible nodes, prefer leaves (deepest, most-specific KV). Interior nodes are shared by more descendants/sessions and are more valuable to retain. An interior node becomes eligible only after all its children are evicted (it becomes a leaf).
- **LRU among eligible leaves:** pick the least-recently-accessed eligible leaf. Recency is a good proxy for future reuse (a prefix used recently is likely used again soon — temporal locality).

Implementing this efficiently requires tracking eligible leaves in a structure ordered by access time (a priority queue or an intrusive LRU list of evictable leaves), so the least-recently-used eligible leaf is found in O(log n) or O(1). As leaves are evicted, their parents may become eligible (newly leaves) and are added to the structure. The policy elegantly captures the value hierarchy: hot shared prefixes (high, frequently accessed, often ref>0) are retained; cold specific tails (deep leaves, ref=0, old) are reclaimed first. 

**Alternative policies** are possible — e.g. weighting by subtree size (retain prefixes that, if evicted, would cost more to recompute) or by hit frequency (LFU rather than LRU). LRU-leaf-first is a strong, simple default that matches the temporal locality of most workloads (recently-used prefixes recur). The policy interacts with the workload: for a few stable hot prefixes (one system prompt), almost any policy retains them; for many competing prefixes exceeding capacity, the policy's quality determines the hit rate, and leaf-first LRU's bias toward retaining shared interior nodes is well-suited to the tree-structured sharing RadixAttention targets.

---

## 31. Observability and Tuning Signals

RadixAttention exposes signals an operator monitors (File 11 §SGLang debug):

- **Prefix-cache hit rate:** the fraction of prompt tokens served from the cache (logged per request and aggregated). The headline metric for prefix-heavy workloads — a low hit rate on a known-shared-prefix workload signals an undersized KV pool (§18) or prefix-scattering routing (§14.2).
- **Tree size / node count / cached tokens:** how much is cached; useful for capacity reasoning.
- **Eviction rate:** high eviction means the working set exceeds the pool (the cache is thrashing) — raise the KV share (`--mem-fraction-static` lower) or reduce concurrency.
- **KV pool utilization:** like vLLM's `gpu_cache_usage_perc` (File 03 §22), the occupancy gauge.

The tuning loop: measure the hit rate on the real workload; if it's lower than the workload's inherent sharing would allow, the pool is too small (give RadixAttention more KV memory) or prefixes are scattered across replicas (fix routing). The hit rate directly drives the prefill-compute savings (§23), so it's the metric most worth optimizing for prefix-sharing deployments — the difference between RadixAttention delivering its ~29× prefill reduction and delivering little.

---

## 32. Key Takeaways

1. **RadixAttention solves the multi-call problem:** automatic, fine-grained KV reuse across requests/calls that share prefixes (chat, few-shot, RAG, tree-of-thought, agents), where naive engines wastefully recompute shared KV.
2. **A radix tree (compressed trie) over token sequences** is the data structure: paths are cached token sequences, nodes hold KV, edge-splitting enables token-granularity matching, and reference counting + leaf-first LRU manage sharing and eviction.
3. **The algorithm is automatic:** match the longest cached prefix, reuse its KV, compute only the suffix, insert the result. No explicit cache API — sharing emerges from the tree.
4. **Token-granularity + tree structure** distinguish it from vLLM's block-granular, hash-based, linear APC — more powerful for complex multi-session sharing, at modestly higher overhead (a wash when there's no sharing).
5. **XGrammar** makes structured output (JSON, regex, CFG) near-free via PDA compilation and precomputed per-state token masks — <1% overhead vs 5–15% for naive FSM-per-step — making reliable function calling and JSON APIs the default.
6. **The frontend DSL** exposes program structure (`gen`, `select`, `fork`), enabling program-aware optimizations — RadixAttention reuse, parallelized forks, and structural batching across users running the same program — that request-level engines can't see.
7. **The whole is co-designed:** frontend exposes structure → RadixAttention reuses shared KV → XGrammar constrains output cheaply → the scheduler batches structurally. This integrated design is SGLang's defining characteristic and its edge on multi-call, structured, agentic workloads.

RadixAttention is to SGLang what PagedAttention is to vLLM — the foundational idea the engine is built around. Both rest on sub-request KV addressing and reference-counted sharing (File 03 §14); RadixAttention adds the tree-structured, token-granular policy and the program-aware frontend that exploit it. For the growing universe of multi-call LLM programs, it is the mechanism that turns "many redundant calls" into "shared computation," which is exactly the efficiency that makes agentic and structured LLM applications economical at scale.

---

## 33. Appendix: Source Pointers

- **Paper:** Zheng et al., *"SGLang: Efficient Execution of Structured Language Model Programs,"* arXiv 2312.07104 (RadixAttention, frontend, co-design).
- **XGrammar:** Zheng et al., arXiv 2411.15100 (PDA compilation, precomputed masks).
- **Codebase:** `sglang/srt/mem_cache/` (radix tree, `RadixCache`, token/KV pools); `sglang/srt/managers/scheduler.py` (cache-aware scheduling, continuous batching); `sglang/srt/constrained/` (grammar integration); the frontend DSL under `sglang/lang/` (`@sgl.function`, `gen`, `select`, `fork`, the interpreter).
- **Related:** Hydragen (arXiv 2402.05099, shared-prefix attention compute), Prompt Cache, ChunkAttention/Cascade Inference (prefix-sharing variants), Outlines and lm-format-enforcer (alternative grammar backends).

To trace RadixAttention execution: a request arrives → scheduler runs `RadixCache.match_prefix` → reuse matched KV (ref++), allocate suffix tokens → build batch → forward pass (prefill suffix attending over cached prefix) → generate, extending the tree → on completion, ref-- (eligible for leaf-first LRU eviction). For the frontend: `@sgl.function` → `Program` → `Interpreter` issues `gen`/`select`/`fork` on the `Runtime` → structural batching + RadixAttention reuse + XGrammar masking → optimized execution. Each step maps to a section above; together they realize SGLang's thesis that *structured LLM programs deserve a structured execution engine* — File 09 details that engine's runtime architecture (processes, communication, attention backends, CUDA graphs) that makes RadixAttention and the frontend run fast.

---

## 34. A Worked RAG Example

RAG (retrieval-augmented generation) is a paradigmatic RadixAttention beneficiary, worth tracing. A RAG service has a fixed system prompt, retrieves documents per query, and generates an answer grounded in them.

**Prompt structure:** `[system prompt (500 tok)] [retrieved doc A (1000 tok)] [retrieved doc B (800 tok)] [user question (50 tok)]`.

- The **system prompt** is shared by *every* request → cached once, reused always (a high node in the tree, ref-counted by all sessions).
- **Documents** are shared by requests that retrieve the same document. In RAG, popular documents are retrieved for many queries; the first query to retrieve doc A computes its KV, and subsequent queries retrieving doc A reuse it. The radix tree captures this: `[system][docA]` becomes a shared path for all queries using doc A.
- The **question** is unique per request → computed fresh.

Without RadixAttention, every query recomputes the full `500 + 1800 + 50 = 2350` tokens. With it, a query retrieving already-cached docs computes only the ~50-token question (plus any not-yet-cached doc) — a huge saving when documents recur. The catch: the document *order* matters for caching (attention is causal, so `[docA][docB]` and `[docB][docA]` produce different KV — File 03 §6.2's chained-hash insight applies). To maximize hits, RAG systems can canonicalize document order (e.g. by document ID) so the same set of documents always produces the same prefix. This is a place where understanding RadixAttention informs *application* design: ordering retrieved documents consistently turns them into cacheable shared prefixes. SGLang's token-granular tree handles the partial matches (a query sharing only docA, not docB, with a cached path matches `[system][docA]` and computes from docB onward) more gracefully than block-granular APC. RAG is thus a workload where RadixAttention's design directly shapes both engine efficiency and application prompt-construction strategy.

---

## 35. The SGLang Router and Cluster Deployment

For multi-replica SGLang deployments, the **SGLang router** provides prefix-cache-aware load balancing (the cluster-level complement to per-replica RadixAttention, §14.2). It tracks (approximately) which prefixes are cached on which replicas and routes requests to maximize cache hits while balancing load. The router uses prefix matching (a cluster-level analog of the radix match) to decide affinity: a request sharing a prefix with a recently-routed request goes to the same replica (where that prefix is likely still cached), while load-balancing constraints prevent any one replica from being overloaded. This turns the fleet's collective KV cache into something closer to a shared cache, capturing RadixAttention's benefit across replicas rather than only within one. Combined with the hierarchical caching of §26, a large-scale SGLang deployment can retain a substantial working set of hot prefixes across GPU, CPU, and routing, serving prefix-heavy workloads (agents, multi-turn chat, RAG at scale) with high aggregate cache-hit rates. This cluster architecture — cache-aware router + per-replica RadixAttention + tiered KV — is how SGLang's prefix-reuse advantage scales from a single engine to a production fleet (File 19, File 20).

---

## 36. When to Choose SGLang (Decision Guide)

Synthesizing the comparisons (§8, §29) into practical guidance:

- **Choose SGLang when** the workload is multi-call/structured: agentic loops, multi-turn chat with long histories, few-shot-heavy classification/extraction, RAG with recurring documents, tree-of-thought/parallel reasoning, or heavy structured output (JSON/tool calling at scale). These exercise RadixAttention's token-granular tree reuse, structural batching, and XGrammar — SGLang's strengths.
- **Choose vLLM when** the workload is independent single-shot requests, or when you need its broadest-in-class model support, quantization breadth, or the largest community/ecosystem — and when a single shared system prompt (handled well by APC) is the extent of the sharing.
- **Either works well for** general OpenAI-API chat serving; both have continuous batching, prefix caching, chunked prefill, CUDA graphs, and strong performance, and they've converged substantially. Benchmark on your specific workload (File 11) — the prefix-sharing fraction is the key discriminator.

The honest summary: the engines are both excellent and increasingly similar; the workload's *structure* (independent requests vs multi-call programs with shared context and structured output) is what tips the choice. RadixAttention and the program-aware frontend are SGLang's differentiators, most valuable exactly when that structure is present and most dispensable when it isn't. For the trajectory of LLM applications toward agents and structured, multi-step interactions (File 20 §future), the kind of efficiency RadixAttention provides is becoming more central, which is why its ideas — automatic fine-grained KV reuse and program-aware execution — are influential well beyond SGLang itself.

---

## 37. Inference-Engineer FAQ

**Q: My multi-turn chat gets slower every turn even with SGLang. Why?** Likely a cache miss each turn — the client may be re-sending the conversation in a way that doesn't match the cached path (e.g. reformatting, changing whitespace, or the chat template producing different tokens). Verify the resent prompt's tokens exactly extend the prior path (RadixAttention matches *tokens*, so any reformatting breaks the match). Also check the KV pool is large enough to retain the conversation between turns (§18) and that routing sends the session's turns to the same replica (§14.2/§35).

**Q: RadixAttention isn't helping my workload.** If requests don't share prefixes (every prompt unique), there's nothing to reuse — RadixAttention's overhead is then pure cost (§19), and the workload simply isn't a prefix-sharing one. Confirm with the hit-rate metric (§31): a near-zero hit rate means no sharing to exploit.

**Q: Should I order my RAG documents consistently?** Yes (§34) — because attention is causal, the same documents in different orders produce different KV and thus different cache paths. Canonicalizing document order (by ID) turns recurring document sets into cacheable shared prefixes, dramatically raising the hit rate for RAG.

**Q: How is this different from just enabling vLLM's prefix caching?** vLLM's APC is block-granular, hash-based, and linear-prefix (§8); RadixAttention is token-granular, tree-structured, and handles multi-session branching, fork/join, and partial matches more gracefully. For a single shared system prompt they're similar; for complex multi-call/branching workloads RadixAttention is more powerful. Both help; the difference matters most for tree-of-thought, branching conversations, and partial-prefix sharing.

**Q: Does constrained decoding (XGrammar) hurt quality?** No, when implemented correctly (§11.6): it masks invalid tokens and renormalizes, so the model still samples per its (constrained) distribution — it expresses preferences among *valid* continuations rather than greedily taking the first valid token. The output is guaranteed schema-valid *and* quality-preserving within the grammar.

**Q: Can I use RadixAttention and XGrammar without the frontend DSL?** Yes — RadixAttention works automatically for any requests (including plain OpenAI-API requests) that share prefixes, and XGrammar works via the guided-decoding API parameters (`guided_json`, etc.). The frontend DSL adds program-aware structural batching and explicit fork/join, but the core prefix-reuse and constrained-decoding benefits apply to ordinary API traffic too. This is why SGLang is a strong general-purpose engine, not only a DSL runtime.

**Q: Does RadixAttention increase memory usage?** The tree metadata is negligible (§9.4, ~3 MB for 1M cached tokens). The *cached prefix KV* does occupy the pool — but that KV is *shared* (stored once for all sharing requests) and held only while valuable (evicted leaf-first under pressure, §30). Net, RadixAttention *reduces* memory for prefix-sharing workloads (one shared copy instead of N), increasing effective concurrency; it doesn't add meaningful overhead beyond the tiny tree structure.

**Q: What happens to cached KV when a request finishes?** Its nodes' ref_count is decremented. If a node reaches ref_count 0, it isn't immediately freed — it stays in the tree as an evictable leaf, ready to be re-hit by a future request (a cache hit avoiding recomputation), and is only physically reclaimed when memory pressure triggers eviction (§30). This lingering-while-free behavior is what lets popular prefixes survive across requests and produce high hit rates.

---

## 38. Closing

RadixAttention reframes LLM serving for the multi-call era. Where PagedAttention (File 03) made KV cache a managed, shareable resource, RadixAttention makes *sharing across requests* automatic and fine-grained via a radix tree, capturing the redundancy in the multi-turn, few-shot, RAG, and agentic workloads that increasingly dominate LLM usage. Paired with XGrammar (near-free structured output) and a program-aware frontend (structural batching, fork/join), it forms a co-designed system optimized for *structured LLM programs* rather than opaque independent requests. The mechanism's elegance — a trie over token sequences with reference-counted nodes and leaf-first LRU eviction — belies its impact: for prefix-heavy workloads it delivers order-of-magnitude prefill reductions and corresponding throughput and cost gains, and its token-granular, tree-structured model handles the complex sharing patterns of real applications that simpler caches approximate poorly. Its ideas have propagated across the field (XGrammar adopted broadly, prefix-aware routing now standard), confirming that as LLM applications grow more structured and agentic, automatic fine-grained KV reuse is not a niche optimization but a central one. File 09 turns to the runtime architecture — the multi-process design, zero-copy communication, attention backends, and CUDA-graph machinery — that executes RadixAttention and the frontend efficiently, completing the picture of how SGLang is built.

---

## 39. Origin and Context

RadixAttention and SGLang came out of UC Berkeley and **LMSYS** (the organization behind Chatbot Arena and the earlier FastChat serving framework), led by Lianmin Zheng, Liangsheng Yin, and collaborators. The motivation was directly observed in LMSYS's own workloads: serving Chatbot Arena and structured-generation research surfaced the multi-call, prefix-sharing patterns (§1) that generic engines handled inefficiently. SGLang's thesis — that *structured LLM programs deserve a structured execution engine* — reflects this origin: the team saw real programs making many related LLM calls and built an engine that exploits their structure end to end (frontend DSL → RadixAttention → structural batching → constrained decoding). 

The lineage from FastChat (an early, simpler serving framework) to SGLang mirrors the field's maturation: from "serve a model behind an API" to "efficiently execute structured, multi-call LLM programs at scale." LMSYS's continued role — Chatbot Arena for evaluation, SGLang for serving — gives it a feedback loop between how models are used (the Arena's diverse, often multi-turn traffic) and how they're served (SGLang's optimizations for exactly that traffic). SGLang is Apache 2.0 licensed, has a small but very active core team, and has tended to ship cutting-edge algorithm implementations (RadixAttention, XGrammar, DP attention, aggressive CUDA-graph usage, EAGLE speculative decoding) quickly — often ahead of equivalents elsewhere (File 20 §SGLang ecosystem). Its commercial adopters include hosted-inference providers serving the structured and multi-turn workloads where its design excels.

This origin matters for understanding the engine's character: SGLang is opinionated toward the multi-call, structured, agentic future of LLM applications, betting that the efficiency of fine-grained reuse and program-aware execution will matter more as applications grow more complex. That bet has largely paid off — the patterns RadixAttention targets (agents, RAG, multi-turn, structured output) are exactly the patterns growing fastest in production LLM use (File 20 §future), and the ideas have spread well beyond SGLang. Whether one deploys SGLang or vLLM, understanding RadixAttention is understanding how the field is adapting from single-shot completion to structured, stateful, multi-call LLM computation — the direction the entire inference stack is evolving toward, and the reason this file sits at the center of the database alongside PagedAttention (File 03) as one of the two foundational serving-systems ideas. Together, PagedAttention's managed KV blocks and RadixAttention's automatic tree-structured reuse define how modern engines treat the KV cache: not as a static per-request buffer, but as a shared, relocatable, fine-grained resource whose efficient management is the difference between an economical inference service and a wasteful one. That reconception of the KV cache, more than any single kernel or scheduling trick, is what the two engines contributed to the field — and RadixAttention, by making fine-grained cross-request reuse automatic and tree-structured, pushed that reconception furthest, turning the KV cache from a per-request buffer into a shared, hierarchical, reusable structure spanning every concurrent session and call in the system.








