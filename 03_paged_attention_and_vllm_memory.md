# PagedAttention — Algorithm, Implementation, and Memory Management

> **PRIMARY reference file.** PagedAttention is vLLM's foundational contribution and the single most important systems idea in open-source LLM serving. This file derives the algorithm from first principles, details its CUDA implementation, and covers the full memory-management stack built on top of it: copy-on-write, prefix caching, CPU swapping, preemption, and the block manager. Paper: Kwon et al., *"Efficient Memory Management for Large Language Model Serving with PagedAttention,"* SOSP 2023. Prerequisite: File 02 §6 (KV cache layout) and File 01 §4 (KV cache as the central resource).

---

## Table of Contents

1. The Memory Fragmentation Problem Before PagedAttention
2. The PagedAttention Algorithm
3. The Block Table and Address Translation
4. The PagedAttention CUDA Kernel
5. Copy-on-Write for Parallel Sampling and Beam Search
6. Prefix Caching (Hash-Based Automatic Prefix Caching)
7. Memory Pools and the Block Allocator
8. CPU–GPU Memory Hierarchy and Swapping
9. Preemption: Swap vs Recompute
10. Memory Efficiency Analysis
11. Block Manager Implementation Details
12. Sequence Lifecycle and State Machine
13. Block Size Selection and Tuning
14. Comparison to RadixAttention and Other Systems

---

## 1. The Memory Fragmentation Problem Before PagedAttention

To appreciate PagedAttention you must understand the catastrophe it replaced. Before vLLM, serving systems (early HuggingFace `generate`, FasterTransformer, the first TGI) allocated the KV cache as a **single contiguous tensor per request**, sized to the maximum possible sequence length.

### 1.1 Static contiguous allocation and its waste

When a request arrives, the system does not know how many tokens it will ultimately generate. To be safe, it reserves a contiguous buffer for `max_seq_len` tokens up front:

```
reserved_bytes_per_request = max_seq_len · KV_bytes_per_token
```

For LLaMA-3 70B (`320 KB/token`) with `max_seq_len = 2048`, that is `640 MB` reserved per request *the moment it arrives*, regardless of whether it generates 10 tokens or 2,000. This produces several distinct kinds of waste, which the PagedAttention paper categorizes precisely:

- **Internal fragmentation (reservation waste):** a request that generates only 100 of its 2,048 reserved tokens wastes 95% of its allocation for its entire lifetime. Since output length is unknown at arrival and the system must reserve for the worst case, this is unavoidable under static allocation. The paper measured **60–80% of KV memory wasted** to internal fragmentation in pre-vLLM systems.
- **Reservation waste for not-yet-generated tokens:** even a request that *will* eventually use all 2,048 tokens has, at any given moment, reserved space for tokens it hasn't generated yet. That space sits idle and cannot be lent to another request.
- **External fragmentation:** because each request needs a *contiguous* buffer of a *different* size (different `max_seq_len` or different model), the free memory becomes a patchwork of holes. A new request needing a 640 MB contiguous block may be rejected even when 2 GB of total free memory exists, simply because no single 640 MB hole is available — the classic external fragmentation of any contiguous allocator.

### 1.2 The quantitative consequence

The combined effect was that pre-PagedAttention systems achieved **under 40%, often ~20–30%, effective KV memory utilization**. Since KV memory determines how many requests you can batch (File 01 §4), and batch size determines decode throughput (File 02 §15.2), low memory utilization translates almost directly into low throughput and high cost. A system wasting 70% of its KV memory serves roughly one-third the concurrent requests it could — a 3× cost penalty hiding in plain sight.

### 1.3 Why dynamic contiguous allocation doesn't fix it

The obvious patch — grow the buffer as the sequence grows — fails on GPUs:

- **Reallocation requires copying.** Growing a contiguous tensor means allocating a larger buffer and *copying* the existing KV cache into it. For a multi-gigabyte cache, this copy is expensive and stalls the request, and doing it every few tokens is untenable.
- **It doesn't solve external fragmentation.** Variable-length growing buffers fragment free memory worse, not better.
- **It can't share.** Two requests with an identical prompt prefix each hold their own contiguous copy of the prefix KV — no sharing is possible because sharing requires the shared data to be addressable independently of each request's private suffix.

The root cause of all of this is the **contiguity requirement**. PagedAttention's insight is to *remove* it.

---

## 2. The PagedAttention Algorithm

### 2.1 The core insight: KV cache need not be contiguous

PagedAttention borrows directly from **operating-system virtual memory**. An OS gives each process a contiguous *virtual* address space but backs it with non-contiguous *physical* pages, mapped through a per-process page table. Physical pages can be allocated on demand, shared between processes (copy-on-write), and swapped to disk — all transparently to the process.

PagedAttention applies the identical idea to the KV cache:

- The KV cache of a sequence is *logically* a contiguous sequence of tokens, but is *physically* stored in fixed-size **blocks** scattered anywhere in GPU HBM.
- A per-sequence **block table** maps logical block indices to physical block numbers — exactly an OS page table.
- Physical blocks are allocated on demand (one at a time, as the sequence grows), can be **shared** across sequences via reference counting (copy-on-write), and can be **swapped** to CPU RAM under pressure.

This single structural change eliminates internal fragmentation (you only allocate blocks you actually use, wasting at most a fraction of one block per sequence), eliminates external fragmentation (all blocks are the same size, so any free block fits any need), and enables sharing (the prerequisite for prefix caching and CoW).

### 2.2 Block structure

A **physical KV block** is a fixed-size chunk of HBM holding the K and V for `B` consecutive tokens (the **block size**, typically 16 in vLLM). For one layer, a block stores:

```
block tensor shape = [2, num_kv_heads, block_size, head_dim]
                      ^                  ^
                      K and V            B tokens per block
```

(Layouts vary — some backends use `[2, block_size, num_kv_heads, head_dim]` or split K and V into separate pools — but the principle is identical.) The block size in bytes:

```
block_bytes = 2 · num_kv_heads · block_size · head_dim · bytes_per_element · num_layers
```

For LLaMA-3 70B, `block_size = 16`: `16 tokens · 320 KB/token = 5 MB` per block (across all layers). The entire KV cache is a flat **pool** of such blocks; the number of blocks is fixed at startup (§7).

### 2.3 Logical-to-physical mapping

Each sequence sees a logical KV cache: token `t` lives at logical block `⌊t / B⌋`, offset `t mod B` within that block. The **block table** is an array indexed by logical block number, holding the physical block number where that logical block actually lives:

```
logical block i  →  block_table[i]  →  physical block number
token t          →  block_table[⌊t/B⌋] , offset (t mod B)
```

When the sequence needs a new block (it has filled the current one), the allocator hands it any free physical block and appends that block's number to the block table. The physical blocks need not be adjacent — the block table indirection makes the scattered physical layout look contiguous to the rest of the system.

### 2.4 The allocation lifecycle in one example

A request with a 30-token prompt, `B = 16`:

1. **Prefill allocation:** needs `⌈30/16⌉ = 2` blocks. Allocator pops physical blocks, say `#107` and `#43`, from the free pool. Block table = `[107, 43]`. The prompt's KV is written: tokens 0–15 into block 107, tokens 16–29 into block 43 (slots 0–13; slots 14–15 of block 43 are unused for now — the only internal fragmentation, at most `B−1` tokens).
2. **Decode step 1:** generates token 30, which goes into block 43 slot 14. No new block needed.
3. **Decode step 2:** token 31 → block 43 slot 15. Block 43 now full.
4. **Decode step 3:** token 32 needs logical block 2. Allocator pops, say, `#200`. Block table = `[107, 43, 200]`. Token 32 → block 200 slot 0.
5. **Completion:** when the sequence finishes, all three blocks are returned to the free pool in O(1) each.

At no point was anything copied or reserved beyond immediate need; internal fragmentation is at most the 2 unused slots in the last block. This is the entire mechanism — its power is in what it *enables* (sharing, swapping, near-100% utilization), covered below.

---

## 3. The Block Table and Address Translation

### 3.1 Data structure

A block table is a small, variable-length array of integers (physical block numbers), one entry per logical block. Its maximum length is `⌈max_seq_len / B⌉`. For `max_seq_len = 32768`, `B = 16`, that is 2,048 entries — a few KB per sequence, trivial compared to the multi-megabyte KV cache it indexes. Block tables live in two places:

- **On the GPU:** as a tensor passed to the attention kernel, so the kernel can translate logical positions to physical addresses while reading the cache.
- **On the CPU (host):** in the block manager's Python data structures, which decide allocation/free/share and serialize the table to the GPU each step.

### 3.2 The slot mapping

For *writing* new K/V during a forward pass, vLLM precomputes a **`slot_mapping`**: a flat array mapping each token in the current batch to the absolute physical slot where its K/V should be written. A "slot" is a `(physical_block_number · B + offset)` linear index into the flat KV pool. The model's attention layer, after computing K/V for the new tokens, scatters them into the cache at these slots. The `slot_mapping` is rebuilt every step by the `ModelRunner` (File 06) from the current block tables and is the write-side analog of the block table's read-side translation.

### 3.3 Cost of translation

Address translation is O(1) per token (an array index plus arithmetic). The block table is read once per attention kernel invocation and cached in shared memory. The overhead versus contiguous addressing is negligible in compute but does introduce one real cost: **non-coalesced, scattered memory access** when consecutive logical blocks map to distant physical blocks. The kernel (§4) is engineered to mitigate this, and block size (§13) trades off this scatter cost against internal fragmentation.

---

## 4. The PagedAttention CUDA Kernel

The algorithm requires a custom attention kernel that reads K/V from non-contiguous physical blocks, because standard FlashAttention assumes a contiguous K/V tensor.

### 4.1 Kernel inputs

The decode-phase PagedAttention kernel takes:

- `query`: the current token's query, shape `[num_seqs, num_heads, head_dim]` (one query per sequence in the batch).
- `key_cache`, `value_cache`: the flat physical block pools, shape `[num_blocks, num_kv_heads, head_dim/x, block_size, x]` (vLLM uses a vectorized inner layout with a packing factor `x` for coalesced 128-bit loads).
- `block_tables`: `[num_seqs, max_num_blocks_per_seq]` — each sequence's logical→physical map.
- `context_lens`: `[num_seqs]` — how many tokens each sequence has cached (so the kernel knows how far to read).
- Scalars: `block_size`, `num_kv_heads`, scale, etc.

### 4.2 Parallelization and the two-phase reduction

The kernel assigns each `(sequence, query_head)` pair to a CUDA thread block (or a group of warps). For a given sequence, it iterates over that sequence's **logical blocks**, and for each:

1. **Resolve** the physical block number via `block_tables[seq][logical_idx]`.
2. **Load** the K (and later V) for that block from `key_cache[physical_idx]` into registers/shared memory.
3. **Compute partial scores** `q · Kᵀ` for the `block_size` keys in this block, applying the scale.
4. **Online-softmax accumulate** (the FlashAttention running-max/running-sum trick, File 02 §3.2) the partial scores and the partial `P·V` output across blocks, so the full `[1 × context_len]` score row is never materialized.

Because a single sequence's context may be long and split across many blocks, vLLM's kernel uses a **two-phase reduction** for long contexts: multiple thread blocks each handle a *partition* of the sequence's KV blocks and produce partial outputs plus their local softmax statistics (`m`, `ℓ`); a second reduction kernel combines these partial results into the final attention output using the log-sum-exp combination. This exposes more parallelism than assigning one thread block per sequence when batch size is small but context is long (a common decode regime).

### 4.3 Memory-access pattern and its weakness

The kernel's challenge is that consecutive logical blocks may live at scattered physical addresses, so the loads are not as coalesced as a contiguous read. For **many short sequences** (large batch, short contexts), the access pattern becomes fine-grained and scattered, and the original vLLM kernel underperforms a contiguous FlashAttention. This is precisely why **FlashInfer** (File 10) was adopted: its paged-attention kernels use better block layouts, warp specialization, and GQA-aware grouping to recover coalescing, and they outperform the native vLLM kernel in most regimes. vLLM now defaults to FlashAttention/FlashInfer backends where available, falling back to the native paged kernel otherwise. The *algorithm* (paged KV, block tables) is unchanged; only the *kernel* implementing the gather improved.

### 4.4 Prefill with paging

During prefill the kernel processes many queries (the whole prompt or a chunk) against the cache. With prefix caching (§6), some of the keys come from *cached* (read-only, shared) blocks and the rest from *newly written* blocks. The prefill kernel reads K/V through the same block-table indirection. A subtlety: cached prefix blocks are read-only and shared (refcount > 1), so the kernel must never write to them — only the new suffix blocks are written. This read-only/read-write distinction lets the kernel skip write-coalescing logic for cached blocks (File 05 §prefix-aware attention).

---

## 5. Copy-on-Write for Parallel Sampling and Beam Search

Once KV blocks can be referenced indirectly, *sharing* them becomes possible — and sharing is what makes parallel sampling, beam search, and forking cheap.

### 5.1 The motivation

A request with `n > 1` parallel samples (`SamplingParams(n=4)`) generates four independent continuations *from the same prompt*. The prompt's KV cache is identical for all four. Naively, you would store four copies of the prompt KV — `4×` the memory. For a long prompt this is enormous waste, and it scales with `n`.

### 5.2 The mechanism

PagedAttention shares the prompt blocks across all `n` sequences and copies only when a sequence diverges:

1. **Shared prefill:** all `n` sequences' block tables point to the *same* physical blocks for the prompt. Each shared physical block has a **reference count** equal to the number of sequences pointing at it (`ref_count = n` for the prompt blocks).
2. **Divergence on write:** when a sequence needs to append a new token (write K/V), the block manager checks the target block's refcount:
   - If `ref_count == 1` (the block is private to this sequence), write in place.
   - If `ref_count > 1` (the block is shared), **copy-on-write**: allocate a fresh physical block, copy the existing contents of the shared block into it, decrement the shared block's refcount, point this sequence's block table at the new block, and write the new token there. The other sequences keep sharing the original.

This is exactly OS copy-on-write `fork()`. The cost is one block copy *only at the moment of divergence*, and only for the (usually single) block where divergence happens — the rest of the prompt stays shared forever. For `n` parallel samples of a 1,000-token prompt with `B = 16`, you share all ~63 prompt blocks and copy at most one block per sample when generation begins — a memory saving of nearly `n×` on the prompt.

### 5.3 Beam search with CoW

Beam search (File 02 §8.4, File 04 §beam scheduling) maintains `beam_width` candidate sequences that share a common prefix and branch at each step. CoW makes the shared prefix free and copies only the divergent tail blocks. When a beam is pruned (drops out of the top-`beam_width`), its blocks' refcounts are decremented; blocks reaching `ref_count == 0` return to the free pool. When a beam is forked (one beam expands into multiple candidates), the shared blocks' refcounts are incremented and CoW handles subsequent writes. Reference counting thus cleanly handles the branch-and-prune structure of beam search without any explicit copying except at true divergence points.

### 5.4 Reference counting invariants

The block manager maintains the invariant that a physical block's `ref_count` equals the number of block-table entries (across all sequences) pointing to it. Allocation sets `ref_count = 1`; sharing (fork) increments; freeing a sequence decrements every block in its table; CoW decrements the old block and sets the new block to 1. A block is eligible for the free pool exactly when its `ref_count` hits 0. These invariants are the correctness core of the whole memory manager — a refcount bug manifests as either premature freeing (corruption: a block reused while still referenced) or leaks (blocks never returned). vLLM's `PhysicalTokenBlock.ref_count` and the `BlockAllocator`'s alloc/free/fork operations enforce them.

---

## 6. Prefix Caching (Hash-Based Automatic Prefix Caching)

CoW shares blocks *within* a request (across its samples/beams). **Prefix caching** shares blocks *across different requests* that happen to start with the same tokens.

### 6.1 The opportunity

Enormous fractions of real traffic share prefixes: a fixed system prompt prepended to every chat, few-shot examples reused across queries, a long document shared by many questions (RAG), or the conversation history in a multi-turn session. Without prefix caching, every request recomputes the KV for the shared prefix — wasted prefill compute *and* duplicated KV memory. For a 2,000-token system prompt served to 1,000 requests, that is 1,000 redundant 2,000-token prefills.

### 6.2 vLLM Automatic Prefix Caching (APC)

vLLM's APC (`--enable-prefix-caching`) works at **block granularity** using content hashing:

1. **Hashing:** each *full* block of prompt tokens is assigned a hash computed from its token IDs **and the hash of all preceding blocks** (a chained/rolling hash). The chaining is essential: a block's KV depends on all prior tokens (attention is causal), so two blocks with identical tokens but different prefixes must hash differently. The hash key for block `i` is effectively `hash(prefix_tokens[0 : (i+1)·B])`.
2. **Lookup:** when a new request arrives, vLLM hashes its prompt blocks and checks a hash → physical-block map. For each leading block whose hash matches a cached block, it **reuses** that physical block (incrementing refcount) instead of allocating and recomputing.
3. **Reuse:** the matched blocks are shared exactly like CoW-shared blocks (read-only, refcount-tracked). Only the *non-matching suffix* of the prompt is allocated fresh and computed in prefill. The position IDs for the suffix continue from the matched prefix length (File 02 §19).
4. **Insertion:** after prefill, the newly computed blocks are added to the hash map so future requests can reuse them.
5. **Eviction:** cached blocks that are not currently referenced by any active request (`ref_count == 0`) are kept in the map but are eligible for **LRU eviction** when the free pool runs low. The block is only physically reclaimed (and its hash entry removed) when memory pressure forces it; until then it lingers, ready to be re-hit.

### 6.3 Properties and limits

- **Block-granularity matching** means APC only reuses *complete* blocks; a prefix that ends mid-block (e.g. 2,005 tokens with `B = 16` → 125 full blocks + 5 tokens) reuses the 125 full blocks and recomputes the partial last block. This is APC's main limitation versus SGLang's token-granularity RadixAttention (§14, File 08).
- **Linear (chain) matching** — APC matches a single linear prefix per request efficiently but does not natively represent the *tree* of divergent continuations that RadixAttention's radix tree captures. For simple shared-system-prompt workloads APC is excellent and lower-overhead; for complex multi-session tree-structured sharing, RadixAttention is more powerful.
- **Hash collisions** are handled by also comparing token IDs on a hit (hash is a fast filter, not the sole identity), so a collision causes a miss-and-recompute, never corruption.

### 6.4 Performance impact

For prefix-heavy workloads the win is large: a shared 2,000-token system prompt cached once turns 1,000 requests' prefill from `1000 × 2000` token-computations into `2000 + 1000 × (suffix only)`. The PagedAttention/APC sharing is what makes high-QPS chat with long system prompts economical. File 11 quantifies prefix-cache hit-rate effects on goodput; File 08 contrasts the SGLang radix-tree approach.

---

## 7. Memory Pools and the Block Allocator

### 7.1 Startup: how many blocks exist

vLLM does not malloc KV blocks on demand from the CUDA allocator during serving (that would fragment and stall). Instead, at startup it **pre-allocates the entire KV pool** as one big reservation and carves it into fixed blocks. The number of blocks is computed by a profiling step:

```
1. Load model weights, measure their GPU memory footprint W.
2. Run a profiling forward pass to measure peak activation/workspace memory A.
3. Available for KV = total_GPU_memory · gpu_memory_utilization − W − A
4. num_gpu_blocks = floor(Available_for_KV / block_bytes)
```

`--gpu-memory-utilization` (default 0.9) is the fraction of GPU memory vLLM is allowed to use; the rest is headroom for fragmentation and CUDA context. The resulting `num_gpu_blocks` is fixed for the server's lifetime and printed at startup ("# GPU blocks: N"). Getting `--gpu-memory-utilization` right is one of the highest-leverage deployment knobs (File 11): too low wastes capacity; too high risks OOM on an unexpectedly long sequence or a profiling underestimate.

### 7.2 The BlockAllocator

The allocator manages the free pool with O(1) allocate and free:

- A **free list** (or bitmap) of available physical block numbers.
- `allocate() → PhysicalTokenBlock`: pop a block number from the free list, set `ref_count = 1`, return it. O(1).
- `free(block)`: decrement `ref_count`; if it reaches 0, push the block number back onto the free list. O(1).
- `fork(block)`: increment `ref_count` (used for CoW/prefix sharing). O(1).

No memory is copied on allocate/free — only block *numbers* move between the free list and block tables. This O(1), copy-free allocation is what lets vLLM allocate and free blocks every single decode step for hundreds of sequences without overhead.

### 7.3 Separate GPU and CPU allocators

vLLM maintains two allocators: one for the **GPU block pool** (hot KV cache) and one for a **CPU block pool** (swapped-out KV cache, §8). They have the same block size; swapping moves block *contents* between pools and updates the sequence's block table to point at the new location. The CPU pool size is set by `--swap-space` (default a few GB).

---

## 8. CPU–GPU Memory Hierarchy and Swapping

Even with perfect packing, GPU HBM is finite. When the running batch's KV cache plus incoming requests exceed the GPU block pool, vLLM uses the CPU as a second tier of the memory hierarchy — again mirroring OS virtual memory's swap.

### 8.1 The swap mechanism

When the scheduler decides to **preempt** a running sequence by swapping (§9), it copies that sequence's GPU KV blocks to the CPU block pool over PCIe and frees the GPU blocks for other sequences. The sequence's block table is updated to reference CPU block locations, and its state becomes `SWAPPED`. When GPU memory frees up, the scheduler **swaps in**: copies the blocks back to freshly allocated GPU blocks and resumes the sequence.

### 8.2 Swap cost analysis

Swap bandwidth is limited by PCIe: PCIe 4.0 ×16 ≈ 32 GB/s, PCIe 5.0 ×16 ≈ 64 GB/s. A single block for LLaMA-3 70B (`B = 16`) is 5 MB:

```
swap latency per block ≈ 5 MB / 64 GB/s ≈ 78 µs   (PCIe 5.0)
```

Swapping a sequence with, say, 60 blocks (≈960 tokens) costs `60 · 78 µs ≈ 4.7 ms` each direction. This is acceptable as an *occasional* relief valve but catastrophic if it happens every step — frequent swapping (thrashing) destroys throughput, exactly like an over-committed OS thrashing to disk. vLLM uses pinned (page-locked) CPU memory for the swap pool so the PCIe transfers hit peak bandwidth and can overlap with compute via a separate CUDA stream.

### 8.3 When swapping helps and when it hurts

Swapping is beneficial when memory pressure is transient — a burst of long requests temporarily exceeds capacity, and swapping a few sequences out lets the rest make progress, with the swapped sequences resuming shortly. It is harmful when pressure is sustained: then swapping just adds PCIe latency to every request without relieving the fundamental shortage, and the right answer is to reduce `--max-num-seqs`, add GPUs, or shed load (File 11 §deadlock/thrash debugging). The decision between swapping and the alternative — recompute — is the subject of §9.

---

## 9. Preemption: Swap vs Recompute

When the scheduler must reclaim KV memory from a running sequence (because a higher-priority request needs it, or because the batch grew beyond capacity), it **preempts** the sequence. vLLM offers two preemption modes.

### 9.1 Swap preemption

Move the victim's KV blocks to CPU (§8) and restore them later. Cost: PCIe transfer out and back (`2 ×` the per-direction cost). Preserves all computed KV — no recomputation needed on resume.

### 9.2 Recompute preemption

Simply **discard** the victim's KV blocks entirely (free them, O(1), no copy) and, when the sequence resumes, **recompute** its KV by re-running prefill over its prompt-plus-generated tokens. Cost: a prefill over the sequence's current length on resume; no PCIe traffic.

### 9.3 Choosing between them

The trade-off is **PCIe transfer cost vs recompute cost**:

```
swap cost     ≈ 2 · (num_blocks · block_bytes) / PCIe_bandwidth
recompute cost ≈ prefill_time(current_seq_len)
```

- For **short sequences**, recompute is cheaper: re-running prefill over a few hundred tokens is fast (compute-bound, efficient), and it avoids PCIe entirely. vLLM defaults to **recompute** preemption in many configurations precisely because prefill is efficient and PCIe is slow.
- For **long sequences**, recompute becomes expensive (prefill is `O(n²)` in attention and `O(n)` in everything else), so swapping the already-computed KV can win — *if* PCIe bandwidth and CPU swap space are available.
- vLLM can choose dynamically based on sequence length and CPU memory availability; the policy is configurable. The key insight is that **recompute is not obviously worse than swap** — discarding and recomputing avoids the round-trip PCIe cost, and for the common case of short-to-medium sequences it is the better default, a non-obvious result that the PagedAttention design surfaced.

### 9.4 Preemption order

Which sequence to preempt? Under FCFS, vLLM preempts the *most recently arrived* running sequence (LIFO victim selection), so that earlier requests — which are closer to completion and have waited longest — are protected. Under priority scheduling, the lowest-priority running sequence is preempted. The preempted sequence returns to the waiting (recompute) or swapped queue and is rescheduled when memory permits (File 04 §preemption).

---

## 10. Memory Efficiency Analysis

### 10.1 Internal fragmentation bound

With PagedAttention, the *only* internal fragmentation is the unused slots in each sequence's **last** block: at most `B − 1` tokens. For `B = 16`, that is ≤15 wasted token-slots per sequence. In bytes, for LLaMA-3 70B that's `15 · 320 KB = 4.8 MB` worst case per sequence — versus the *hundreds of MB to GB* wasted per sequence under static allocation. Across many sequences the expected waste is `~B/2` tokens each, vanishingly small relative to the cache. Effective KV utilization rises to **>95%** (versus <40% before).

### 10.2 No external fragmentation

Because all blocks are identical in size, any free block satisfies any allocation request. There are no "holes too small to use" — external fragmentation is eliminated by construction. This is the same reason OS paging eliminated the external fragmentation of segment-based memory.

### 10.3 The throughput result from the paper

The PagedAttention paper (SOSP 2023) demonstrated the payoff empirically. On OPT-13B on a single A100-80GB with a 512-token max, a naive contiguous system could batch ~14 concurrent requests; vLLM with PagedAttention batched **28+** — roughly **2× throughput** at identical memory. The improvement grows with model size and context length, because larger KV caches make fragmentation waste more costly in absolute terms: the bigger the cache, the more PagedAttention saves. Against the strongest prior baselines (Orca variants), vLLM showed 2–4× throughput at the same latency on realistic workloads.

### 10.4 Where the recovered memory goes

The memory PagedAttention reclaims is spent on **larger batches**, which (File 02 §15.2) is exactly what decode throughput needs — more sequences amortizing each weight read. So PagedAttention's memory efficiency converts almost directly into throughput by enabling the batch sizes that the memory-bound decode regime rewards. This is the causal chain that makes it the foundational optimization: contiguity removed → fragmentation eliminated → memory utilization up → achievable batch up → decode throughput up → cost per token down.

---

## 11. Block Manager Implementation Details

### 11.1 Core classes (vLLM codebase)

The memory manager is implemented in `vllm/core/`:

- **`PhysicalTokenBlock`**: represents one physical block — its block number, device (GPU/CPU), and `ref_count`.
- **`BlockTable`**: a sequence's list of `PhysicalTokenBlock`s (its logical→physical map).
- **`BlockAllocator`**: the free-pool manager with `allocate`/`free`/`fork` (one per device).
- **`BlockSpaceManager`** (a.k.a. `BlockManager`): the top-level coordinator that owns the allocators and manages block tables for all active sequences.

### 11.2 BlockManager v1 vs v2

vLLM has had two block-manager generations:

- **v1**: the original single-purpose implementation — solid for basic paging and CoW on a single configuration.
- **v2**: a redesign that cleanly supports **sliding-window attention** (aging out old blocks), **prefix caching** (the hash map and shared-block accounting), and **multi-GPU** consistency. v2 abstracts the block allocation behind interfaces so these features compose. (The V1/V2 engine rearchitecture, File 07, further moves scheduling and block management off the Python critical path.)

### 11.3 Key BlockSpaceManager operations

- **`can_allocate(seq)`**: check whether enough free blocks exist for a new sequence's prompt (used by the scheduler before admitting a request).
- **`allocate(seq)`**: initial block allocation for the prompt at prefill, wiring up the block table (and reusing prefix-cached blocks if APC is on).
- **`append_slots(seq, num_tokens)`**: allocate new block(s) as the sequence generates tokens during decode, triggering CoW if the last block is shared. Returns any copy-on-write operations the executor must perform.
- **`fork(parent_seq, child_seq)`**: create a child sequence sharing the parent's blocks (incrementing refcounts) — for parallel sampling and beam search.
- **`free(seq)`**: release all of a sequence's blocks (decrement refcounts, return freed blocks to the pool) when it finishes or is aborted.
- **`swap_in` / `swap_out`**: move a sequence's blocks between GPU and CPU pools, returning the block-copy operations for the executor.

These operations return *descriptions* of the memory movements (which blocks to copy, swap, etc.); the actual GPU copies are executed by the worker as part of the step, so the block manager stays a pure CPU bookkeeping component.

---

## 12. Sequence Lifecycle and State Machine

Each sequence moves through a well-defined state machine, and block allocation is tied to these states:

```
WAITING ──► RUNNING ──► FINISHED_{STOPPED | LENGTH_CAPPED | ABORTED}
   ▲           │
   │           ▼
   └──────── SWAPPED  (or back to WAITING for recompute preemption)
```

- **WAITING**: admitted to the engine but not yet running; no GPU blocks held (or only prefix-cached shared blocks). Lives in the scheduler's waiting queue.
- **RUNNING**: actively executing prefill or decode; holds GPU blocks; `append_slots` extends its block table each step. Block allocation happens **only** in this state.
- **SWAPPED**: preempted via swap; its blocks live in the CPU pool; awaiting swap-in. (Recompute-preempted sequences instead return to WAITING with their blocks freed.)
- **FINISHED_STOPPED**: hit EOS or a stop string. **FINISHED_LENGTH_CAPPED**: hit `max_tokens`. **FINISHED_ABORTED**: client disconnected or request cancelled. On any finish, `free(seq)` releases all blocks immediately.

The scheduler (File 04) drives these transitions every iteration: promoting WAITING→RUNNING when memory allows, swapping RUNNING→SWAPPED under pressure, restoring SWAPPED→RUNNING when memory frees, and finalizing RUNNING→FINISHED on stop conditions. The tight coupling between state and block ownership is what keeps the free pool accurate.

---

## 13. Block Size Selection and Tuning

Block size `B` is a fundamental trade-off parameter:

- **Small blocks (e.g. 8 tokens):** low internal fragmentation (≤7 wasted tokens/sequence) and fine-grained prefix matching, but **high metadata overhead** (longer block tables, more entries to manage, more scattered memory accesses in the kernel → worse coalescing).
- **Large blocks (e.g. 64 tokens):** low metadata overhead and better-coalesced kernel reads (each block is a longer contiguous run), but **higher internal fragmentation** (≤63 wasted tokens/sequence, which hurts when serving many short sequences) and **coarser prefix matching** (prefix reuse only at 64-token granularity).
- **vLLM default: 16 tokens** — an empirically good balance for typical GQA models on A100/H100, keeping coalescing acceptable while bounding fragmentation.

**SGLang takes the opposite extreme: token-granularity blocks (effectively `B = 1`)** to enable RadixAttention's exact-length prefix matching (File 08). This maximizes sharing granularity at the cost of larger metadata; SGLang's two-level `ReqToTokenPool`/`TokenToKVPool` indirection (File 09) manages the resulting overhead. The block-size philosophy is one of the clearest concrete differences between the two engines' memory designs.

Tuning guidance: keep the default 16 unless profiling shows the attention kernel is bandwidth-starved by scatter (consider larger blocks) or you have a heavily prefix-shared workload that would benefit from finer matching (where SGLang's approach may simply be the better tool). Block size interacts with the chosen attention backend — FlashInfer's page-size expectations should match the configured block size.

---

## 14. Comparison to RadixAttention and Other Systems

| Dimension | vLLM PagedAttention + APC | SGLang RadixAttention |
|---|---|---|
| Block granularity | 16 tokens (default) | 1 token (token-level) |
| Prefix match granularity | full blocks only | any prefix length |
| Sharing structure | linear prefix (hash chain) | tree (radix tree) across sessions |
| Data structure | hash map → blocks | compressed trie of token sequences |
| Overhead | low (hash lookup) | higher (tree traversal/management) |
| Best for | shared system prompts, simple reuse | multi-turn, few-shot, tree-of-thought, RAG |
| Eviction | LRU on unreferenced blocks | LRU on unlocked leaf nodes |

PagedAttention and RadixAttention are not mutually exclusive ideas — RadixAttention is, in effect, a more sophisticated prefix-cache *policy* layered on a paged block pool. vLLM's APC and SGLang's radix tree both rely on the same underlying ability that PagedAttention introduced: to address and share KV at sub-request granularity via indirection. The difference is the matching data structure (hash map vs radix tree) and granularity (block vs token), which determines which workloads each excels at. Beyond these two, the broader systems context — TensorRT-LLM's paged KV (XQA), HuggingFace TGI's adoption of paged attention, and LMDeploy's TurboMind block manager — all descend from or converge on the same paged-KV insight, a testament to how decisively PagedAttention reframed the problem. File 08 covers RadixAttention in full; File 17 compares the engines end to end.

---

## 15. Summary

PagedAttention removes the contiguity requirement from KV cache storage, which:

1. **Eliminates internal fragmentation** (≤`B−1` tokens wasted per sequence) and **external fragmentation** (uniform block size), pushing KV utilization from <40% to >95%.
2. **Enables sharing** via reference-counted blocks — the foundation for **copy-on-write** (parallel sampling, beam search) and **prefix caching** (cross-request reuse).
3. **Enables a CPU swap tier** and the swap-vs-recompute **preemption** choice, giving the scheduler graceful behavior under memory pressure.
4. **Converts reclaimed memory into larger batches**, which the memory-bound decode regime rewards with near-linear throughput gains.

The cost is a custom paged-attention kernel (now largely superseded by FlashInfer's faster paged kernels) and O(1) block-table bookkeeping. Every other vLLM subsystem — the scheduler (File 04), distributed KV sharding (File 05), the model runner's `slot_mapping` (File 06), and prefix-cache-aware routing (Files 05, 11) — is built on top of this block abstraction. It is, justifiably, the idea that defined modern open-source LLM serving.

---

## 16. The Paged-Attention Kernel in Depth: Thread Mapping and the `x` Layout

Section 4 sketched the kernel; here is the detail that explains its performance characteristics and the design choices vLLM made.

### 16.1 The vectorized KV cache layout

vLLM does not store the key cache as the naive `[num_blocks, num_kv_heads, block_size, head_dim]`. It uses a layout that splits `head_dim` into an outer and inner factor `x`:

```
key_cache:   [num_blocks, num_kv_heads, head_dim/x, block_size, x]
value_cache: [num_blocks, num_kv_heads, head_dim, block_size]
```

`x` is chosen so that `x · sizeof(dtype) = 16 bytes` — the width of a 128-bit vectorized load (`x = 8` for FP16/BF16). This layout makes the innermost dimension a 16-byte-aligned vector, so each thread issues a single 128-bit `LDG` instruction that the memory subsystem coalesces. The key cache puts `block_size` as the second-to-last dimension (so consecutive tokens within a block are adjacent for the dot-product accumulation), while the value cache uses a different ordering optimized for the `P·V` aggregation. The asymmetry between the key and value layouts reflects that K participates in `q·Kᵀ` (reduction over `head_dim`) while V participates in `P·V` (reduction over `block_size`) — each is laid out so its reduction dimension is contiguous.

### 16.2 Thread-block assignment

For decode, the kernel launches a grid where each thread block is responsible for one `(sequence, kv_head_group)` and a partition of the sequence's context. Within a thread block:

- Warps cooperatively load the query for the head group into shared memory once (reused across all KV blocks).
- The thread block iterates its assigned logical blocks; for each, it resolves the physical block via the block table (loaded into shared memory), loads K with vectorized 128-bit loads, computes `q·kⱼ` dot products (each thread handles a subset of the `block_size` keys), and updates the running softmax max `m` and sum `ℓ`.
- After the score pass, it loads V and accumulates `Σ pⱼ · vⱼ` into the output, applying the online-softmax rescaling.

### 16.3 The two-phase (split-K) reduction for long contexts

When batch size is small but context is long — a single user with a 50K-token conversation, say — assigning one thread block per sequence leaves most of the GPU's SMs idle (only as many thread blocks as sequences × heads). vLLM's kernel addresses this with a **split along the KV (context) dimension**, analogous to split-K in GEMM:

1. **Phase 1:** partition each sequence's context into `P` chunks; launch `P ×` more thread blocks, each computing the partial attention output and the local softmax statistics (`m_p`, `ℓ_p`) for its chunk. This floods the GPU with parallel work even for batch size 1.
2. **Phase 2:** a reduction kernel combines the `P` partial outputs per sequence using the log-sum-exp rule: rescale each partial output by `exp(m_p − m_global)` weighted by `ℓ_p`, sum, and divide by the global `ℓ`. This is the same numerically-stable combination FlashAttention uses across tiles (File 02 §3.2), here across thread-block partitions.

The number of partitions `P` is chosen at launch based on context length and SM count to keep occupancy high. This split-K decode kernel is one of the trickier pieces of vLLM's CUDA and a frequent target of FlashInfer replacement, which generalizes the same idea with more tuning.

### 16.4 Why FlashInfer often wins

FlashInfer (File 10) improves on the native kernel by: (1) using a page table layout tuned for its wrappers; (2) GQA-aware grouping so a shared KV head is read once per group rather than once per query head (File 02 §28); (3) better autotuned tile sizes and warp specialization; and (4) a unified varlen API that handles mixed prefill+decode batches without separate kernels. The net is 1.5–2× over the native paged kernel in many regimes, which is why both vLLM and SGLang default to it on supported NVIDIA hardware. Crucially, none of this changes the paging *algorithm* — the block table and block pool are identical; only the gather kernel differs.

### 16.5 Why decode kernels are latency-critical and hard to coalesce

The decode paged-attention kernel runs every single step for every sequence, so its efficiency multiplies across the entire serving lifetime. Yet it operates in the worst regime for GPU memory systems: each sequence reads a *different* set of physical blocks (its own scattered KV), the query is tiny (one token), and the arithmetic intensity is ~1. There is little compute to hide memory latency behind. Three techniques mitigate this. First, **block-table prefetch**: the kernel loads the next physical block's address while computing on the current block, hiding the indirection latency. Second, **shared-memory staging of the query and running softmax state**: the single query and the `(m, ℓ, O)` accumulators live in registers/SMEM across the whole KV sweep, so only K and V stream from HBM. Third, **GQA grouping** (§16.4): batching the query heads that share a KV head so that KV head is read once. Even with all three, decode attention rarely exceeds ~70–80% of peak HBM bandwidth because of the scatter — which is precisely why reducing KV *bytes* (GQA, MLA, FP8 KV) attacks the problem from the other side: if you cannot read the cache faster, read less of it. The kernel and the architecture choices (File 02) are two halves of the same memory-bandwidth optimization, and PagedAttention's block layout is the substrate both operate on.

---

## 17. A Fully Worked Memory Budget

To make the startup profiling (§7.1) concrete, work an end-to-end example: **LLaMA-3 70B, FP8 weights, single H100-80GB-equivalent slice under TP=4** (so per-GPU we consider one quarter of weights and KV).

```
Total HBM per GPU:                 80 GB
gpu_memory_utilization = 0.90  →   72 GB usable by vLLM
Model weights (FP8, 70B/4 per GPU): 70e9 · 1 byte / 4 ≈ 17.5 GB
Activation/workspace (profiled):    ~3 GB
CUDA context + fragmentation buffer: in the 10% headroom

Available for KV pool = 72 − 17.5 − 3 = 51.5 GB per GPU

Per-GPU KV block bytes (B=16, TP=4 → 2 of 8 KV heads per GPU):
  per_token_per_gpu = 2 · 80 layers · 2 kv_heads · 128 · 2 bytes(BF16 KV) = 81,920 B ≈ 80 KB
  block_bytes = 16 · 80 KB = 1.28 MB

num_gpu_blocks per GPU = 51.5 GB / 1.28 MB ≈ 40,234 blocks
total token capacity per GPU = 40,234 · 16 ≈ 643,750 tokens
```

So this deployment can hold ~644K tokens of KV per GPU. At 4K context that is ~157 concurrent requests; at 32K context ~20 requests; at 128K context ~5 requests. Switching KV to FP8 (§ File 13) halves `per_token` and roughly *doubles* these numbers. This single calculation — done before deployment — predicts the achievable concurrency and is the backbone of capacity planning (File 11). Note how every architectural choice from File 02 enters: GQA sets `kv_heads`, layer count sets the multiplier, TP divides both weights and KV heads, and KV dtype scales the whole thing.

### 17.1 The danger of over-setting utilization

If you set `--gpu-memory-utilization 0.97` and the profiling under-measured peak activation memory by even 2 GB (easy to do — peak activation depends on the largest batch and longest sequence actually seen, which profiling may not hit), the server will OOM mid-serving when a worst-case batch arrives, killing *all* in-flight requests. This is why production deployments leave a safety margin and why §10's >95% figure refers to *KV utilization within the allocated pool*, not pushing total GPU utilization to the edge.

---

## 18. Sliding-Window Attention and Block Aging

For models with sliding-window attention (File 02 §2.5, e.g. Mistral with `w = 4096`), the KV cache for SWA layers is bounded: a token only needs the last `w` keys. The block manager exploits this by **freeing blocks that fall entirely outside the window**.

As a sequence at position `t` advances, any block whose tokens are all older than `t − w` can never be attended to again and is returned to the free pool. This caps the KV cache per SWA layer at `⌈w / B⌉` blocks regardless of how long the sequence grows — turning unbounded streaming into bounded memory. BlockManager v2 added explicit support for this aging logic, including the interaction with attention sinks (File 02 §29): if sink retention is enabled, the first few blocks are *pinned* (never aged out) while the rest of the window slides. For models that mix SWA and full-attention layers, the block manager tracks per-layer-type cache sizes, since full-attention layers still grow unboundedly while SWA layers stay capped — a complication that makes the per-layer memory accounting non-uniform.

---

## 19. Interaction with Tensor Parallelism

Under tensor parallelism (File 05), the KV cache is **sharded across TP ranks by KV head**: with 8 KV heads and TP=4, each GPU holds the K/V for 2 heads. The crucial design point is that the **block table is logically shared and identical across all TP ranks** — logical block `i` maps to the *same* physical block number on every rank, because all ranks process the same sequences in lockstep and allocate/free in unison. The block manager (a CPU component) makes allocation decisions once (on the driver/rank-0 process) and those decisions apply uniformly; each rank's physical block at number `n` holds *its* shard of that logical block's KV (its 2 heads' worth). 

This synchronization is essential for correctness: if rank 0 freed block 50 but rank 1 didn't, their block tables would diverge and the next allocation would corrupt one rank's cache. vLLM keeps the block managers in sync by having the scheduler/driver broadcast the scheduling decision (which includes block allocation/free/swap operations) to all workers each step; every worker applies the identical block-table mutation. The KV *contents* differ per rank (different heads), but the *bookkeeping* is replicated. This is why §7's block count is *per GPU* but the *number of logical blocks a sequence uses* is the same on every GPU — the sharding reduces per-GPU `block_bytes` (fewer heads), letting each GPU hold the same number of blocks for less memory, which is the concurrency benefit of TP (File 05 §multi-GPU memory).

---

## 20. Interaction with Chunked Prefill

Chunked prefill (File 04) splits a long prompt into chunks processed across multiple scheduler steps so that decode work can interleave. From the block manager's perspective, a chunked prefill is just a sequence of `append_slots` calls: chunk 1 allocates blocks for the first `chunk_size` tokens and writes their KV; chunk 2 allocates more blocks and writes the next `chunk_size` tokens, *attending over chunk 1's already-cached KV*; and so on until the prompt is consumed, after which the sequence transitions to decode. The KV from earlier chunks stays resident (it's needed by later chunks and by decode), so chunked prefill does not reduce peak KV memory — it reduces *latency interference*, not memory. The block allocation is incremental and identical in mechanism to decode-time allocation; the only difference is how many tokens are written per step (a chunk vs one). Position IDs continue across chunks (File 02 §19), and with prefix caching the first chunk may begin with cached blocks.

---

## 21. KV Cache Transfer for Disaggregation

Prefill–decode disaggregation (File 04 §PD, File 15) runs prefill and decode on *different* GPU instances, which means the prefill instance's KV cache must be **transferred** to the decode instance. This is a direct extension of the swap mechanism (§8), but over a network/interconnect rather than to local CPU:

- The prefill worker computes the prompt's KV into its block pool.
- The KV blocks are transferred to the decode worker — over NVLink (intra-node, ~600–900 GB/s), PCIe P2P (~64 GB/s), or RDMA/InfiniBand (inter-node, ~50 GB/s at 400 Gbps).
- The decode worker allocates blocks in its own pool, receives the KV, wires up the block table, and resumes generation.

vLLM exposes this through an experimental `--kv-transfer-config` and a connector framework (with third-party backends like **LMCache**) so the transfer mechanism is pluggable. The transfer size for a 1,000-token prompt of LLaMA-3 70B is `1000 · 320 KB = 320 MB`, ~6.4 ms over 50 GB/s RDMA — small compared to the ~1 s prefill compute, which is what makes disaggregation amortizable (File 15 §transfer analysis). The block abstraction is what makes this clean: KV is already chunked into transferable, independently-addressable units, so transferring "a sequence's KV" is just transferring its list of blocks and rebuilding the table on the far side.

---

## 22. Debugging and Operational Notes

Practical failure modes that trace back to the memory manager:

- **OOM at startup vs at serving:** startup OOM means weights + profiled workspace exceed `gpu_memory_utilization × HBM` — lower the model precision or raise GPU count. Serving-time OOM means the profiling under-estimated peak workspace or `--gpu-memory-utilization` is too aggressive — back it off (§17.1).
- **Throughput collapse under load (thrashing):** if `nvidia-smi` shows GPU utilization dropping while latency spikes, the scheduler is likely preempting/swapping constantly because `--max-num-seqs` admits more concurrency than KV memory supports. Reduce `max-num-seqs`, enable recompute preemption, or add capacity (File 11 §deadlock debugging).
- **"Works for fresh requests, breaks on cache hits":** a prefix-caching or position-ID bug (File 02 §19) — the suffix positions must continue from the matched prefix length.
- **Memory not returning after requests finish:** a refcount leak (§5.4) or cached blocks held by APC that aren't being evicted; `vllm:gpu_cache_usage_perc` should fall as load drops (File 07 §metrics).
- **Block-size mismatch with backend:** FlashInfer expects its page size to match the configured block size; a mismatch surfaces as a kernel error or silent slowdown.

The metric to watch in production is **`vllm:gpu_cache_usage_perc`** (the fraction of the KV block pool in use). Sustained >90% means you are near the preemption cliff and should scale; chronically low means you over-provisioned and can raise `--max-num-seqs` or lower GPU count. This single gauge, derived directly from the block allocator's free-list occupancy, is the operational pulse of a PagedAttention deployment.

---

## 23. Key Takeaways

1. **Contiguity was the enemy.** Static contiguous KV allocation wasted 60–80% of memory to internal and external fragmentation. PagedAttention removes the contiguity requirement via OS-style paging: fixed blocks + per-sequence block tables.
2. **The block table is a page table.** Logical→physical indirection, O(1) allocate/free via a free list, refcounts for sharing — the entire OS virtual-memory toolkit, applied to KV cache.
3. **Sharing is the payoff.** Reference-counted blocks enable copy-on-write (parallel sampling, beam search) and prefix caching (cross-request reuse), neither possible under contiguous allocation.
4. **A custom kernel pays the indirection cost**, with a vectorized `x`-layout and a split-K two-phase reduction for long contexts; FlashInfer now usually supersedes it without changing the algorithm.
5. **The CPU is a swap tier**, and preemption can swap *or* recompute — recompute often wins for short sequences because PCIe is slow and prefill is efficient.
6. **>95% KV utilization → bigger batches → more decode throughput → lower cost.** That causal chain is why PagedAttention roughly doubled serving throughput and became the foundation every other subsystem builds on.

With the memory layer established, File 04 turns to the component that *decides* how to use it: the scheduler that orchestrates continuous batching, chunked prefill, and preemption on top of the paged block pool.

---

## 24. A Worked Copy-on-Write Refcount Trace

To make §5 concrete, trace the refcounts through a parallel-sampling request (`n = 2`) with a 20-token prompt, `B = 16`.

**Prefill.** The prompt is 20 tokens → 2 logical blocks. The SequenceGroup is created with one prompt sequence; prefill allocates blocks `#10` (tokens 0–15) and `#11` (tokens 16–19, slots 0–3). After prefill, the group forks into 2 sequences (`seqA`, `seqB`) for the two samples:

```
fork: seqA and seqB both point to [#10, #11]
ref_count[#10] = 2,  ref_count[#11] = 2
```

**Decode step 1.** Both sequences generate their first token (position 20 → block #11 slot 4). Each must *write* to block #11, but `ref_count[#11] = 2 > 1`, so CoW triggers for whichever writes:

```
seqA appends: #11 is shared → CoW:
   allocate #12, copy #11's contents into #12, ref_count[#11] -= 1 (→1)
   seqA block table: [#10, #12];  write token-20 to #12 slot 4
seqB appends: #11 now has ref_count 1 (private to seqB) → write in place
   seqB block table: [#10, #11];  write token-20 to #11 slot 4
```

Note `#10` (the first 16 prompt tokens) stays shared at `ref_count = 2` forever — both samples have identical first-16-token KV, never written again. Only the *divergence block* (#11) was copied, and only once. The two samples now have private tails (#12 for A, #11 for B) and continue independently, allocating new private blocks as they grow.

**Completion.** When seqA finishes, `free(seqA)` decrements `#10`→1 and frees `#12`→0 (returned to pool). When seqB finishes, `#10`→0 (freed) and `#11`→0 (freed). The invariant — refcount equals the number of block-table references — held throughout, and the total copying was exactly one block despite two full samples. For a 2,000-token prompt this would share ~125 blocks across both samples and copy one, versus naively storing 250 blocks — a ~50% memory saving at `n = 2`, growing toward `(n−1)/n` of the prompt as `n` rises.

---

## 25. Prefix Caching Internals: Computed Tokens, Hashing, and the Evictor

### 25.1 Tracking "computed" tokens

For prefix caching to assign positions and attention correctly, each sequence tracks `num_computed_tokens` — how many of its tokens already have valid cached KV (either from a prior step or from a prefix-cache hit). On a cache hit of `p` tokens, `num_computed_tokens` starts at `p`, and prefill only computes tokens `[p : prompt_len]`. The scheduler uses this to size the prefill work and to set the position offset. Mis-tracking `num_computed_tokens` is the root of the classic "cache-hit corruption" bug (File 02 §19).

### 25.2 The block hash

vLLM computes a block's hash as a function of *all tokens up to and including that block*, not just the block's own tokens, because a block's KV depends causally on the entire prefix. Concretely, the hash for block `i` chains the previous block's hash with the current block's token IDs:

```
block_hash[i] = hash( block_hash[i-1], tokens[i·B : (i+1)·B] )
block_hash[-1] = hash of empty prefix (a sentinel)
```

This chaining guarantees that two blocks with identical 16-token contents but different histories receive different hashes, so they are never incorrectly shared. Only **full** blocks are hashed and cached (a partial trailing block has no stable hash because more tokens may extend it), which is the source of APC's block-granularity limitation (§6.3). LoRA-adapted requests and requests with different multimodal inputs incorporate those into the hash so adapters/images don't cross-contaminate the cache.

### 25.3 The Evictor

Unreferenced cached blocks (refcount 0 but still holding valid, hashed KV) are managed by an **Evictor**. vLLM's default is an LRU evictor: it tracks the last-access time of each cached-but-free block and, when the allocator needs a fresh block and the free list is empty, evicts the least-recently-used cached block (removing its hash entry and returning it to use). This is the mechanism that lets the cache opportunistically retain hot prefixes (popular system prompts stay resident across many requests) while reclaiming cold ones under pressure. The evictor sits between "free but cached" and "truly free": a block hit while still in the evictor is reclaimed *with its KV intact* (a cache hit), whereas a block evicted and reused loses its KV (a miss next time). Tuning here is implicit — the bigger the KV pool relative to the working set of prefixes, the higher the prefix-cache hit rate.

### 25.4 Worked multi-turn chat example

Consider a chat session with a 512-token system prompt, accumulating turns:

- **Turn 1:** user sends system prompt (512 tok) + question 1 (40 tok). Prefill computes all 552 tokens; the system-prompt blocks (32 blocks at B=16) are hashed and cached. Assistant generates a 100-token answer; those blocks are cached too.
- **Turn 2:** the client resends system prompt + Q1 + A1 + question 2. APC matches the cached system-prompt blocks **and** the Q1/A1 blocks (they were cached in turn 1), so prefill only computes question 2's new tokens. The KV for ~650 prior tokens is reused, not recomputed.
- **Turn N:** each turn reuses the entire prior conversation's cached KV, computing only the newest user turn. This turns the per-turn prefill cost from `O(total conversation length)` into `O(new turn length)` — the difference between a chat app that gets slower every turn and one that stays responsive. (SGLang's RadixAttention, File 08, achieves the same with token-granularity matching and explicit session trees, which also naturally handles *branching* conversations.)

---

## 26. The Scheduler ↔ Block Manager Interface

The scheduler (File 04) and block manager interact through a small, well-defined interface every iteration. Understanding it clarifies how memory constraints shape scheduling.

- **`can_allocate(seq_group) → AllocStatus`**: before admitting a waiting request, the scheduler asks whether the block pool has room for its prompt (accounting for prefix-cache hits, which reduce the needed blocks). Returns `OK`, `LATER` (no room now, retry), or `NEVER` (prompt larger than the entire pool — reject the request). This is the admission-control gate.
- **`can_append_slots(seq_group) → bool`**: before running a decode step, check whether the running sequences can get the blocks they'll need this step (a new block per sequence that just filled its last block, plus CoW copies). If not, the scheduler must preempt.
- **`allocate` / `append_slots` / `fork` / `free` / `swap_in` / `swap_out`**: the mutations described in §11.3, returning the concrete block-copy/swap operations the workers must execute.

The scheduler's loop is, in essence, a negotiation with the block manager: admit as many requests as `can_allocate` permits, extend running sequences as `can_append_slots` permits, and when the answer is "no room," preempt (swap or recompute) to free blocks for the requests that matter most. The block manager is the source of truth for "is there memory"; the scheduler is the policy for "who gets it." This separation — mechanism (block manager) vs policy (scheduler) — is clean systems design and mirrors the OS kernel's split between the memory allocator and the process scheduler.

---

## 27. Comparison with Naive Allocation: A Fragmentation Walkthrough

To viscerally appreciate the win, contrast the two schemes on the *same* workload: 10 requests, each with a 100-token prompt, generating a random 50–1,950 tokens, `max_seq_len = 2048`, LLaMA-3 70B (`320 KB/token`), on a hypothetical pool of 20 GB KV memory.

**Naive contiguous:** each request reserves `2048 · 320 KB = 640 MB` on arrival. `20 GB / 640 MB ≈ 31` requests *could* fit by reservation — but most generate far fewer than 2,048 tokens, so the *actual* KV used is a fraction of reserved. If the average sequence is 600 tokens, real usage is `600 · 320 KB = 187 MB` but 640 MB is reserved → **29% utilization**, and you can only admit ~31 requests despite memory for 100+ at actual usage.

**PagedAttention:** each request allocates blocks as it grows. The 100-token prompt takes 7 blocks; a 600-token sequence takes ~38 blocks (`38 · 5 MB = 187 MB` — wait, here block_bytes for the full model is 5 MB, so 600 tokens = 37.5 blocks = ~188 MB). Internal fragmentation is ≤1 block (5 MB) per sequence. Utilization ≈ `188 / (188 + 2.5) ≈ 98.7%`. You can admit `20 GB / 188 MB ≈ 106` average-size requests — **3.4× more concurrency** than naive, purely from eliminating reservation waste. And because you only allocate what's used, a burst of short requests packs tightly instead of each grabbing 640 MB. This is the fragmentation arithmetic behind the paper's 2–4× throughput, made concrete.

---

## 28. External KV Stores and the Future of the Memory Tier

The block abstraction has opened a research and engineering frontier: treating KV cache as a first-class, *movable, storable* object beyond the single GPU.

- **LMCache** and similar layers store KV blocks in CPU RAM, local SSD, or a distributed cache, keyed by prefix hash, so that prefix-cache hits can survive across requests, across engine restarts, and even across *machines*. A long document's KV, computed once, can be reused by any worker that pulls it from the shared store — extending prefix caching from per-GPU to cluster-wide.
- **Disaggregated KV** (§21, File 15) makes KV transfer between prefill and decode pools routine, which requires exactly this movable-block design.
- **Tiered KV** generalizes the GPU→CPU swap (§8) into GPU→CPU→SSD→network hierarchies, each tier larger and slower, with the hottest prefixes in HBM. FlexGen (File 13) explored the extreme of this for single-GPU large-model inference.

All of these rest on PagedAttention's core move: making KV cache addressable and relocatable at block granularity. A contiguous-allocation system simply cannot participate in this ecosystem — you cannot transfer, share, or tier a monolithic per-request buffer. The block is the unit that made KV cache into a *managed resource* rather than a static reservation, and that is the deeper reason the idea reorganized the entire field. File 15 covers the disaggregation and external-store frontier; here the point is that it is all downstream of the humble fixed-size block and its reference count.

---

## 29. The OS Virtual-Memory Analogy, Completed

PagedAttention is most deeply understood as a faithful port of OS virtual memory. The mapping is precise:

| OS virtual memory | PagedAttention |
|---|---|
| Process | Sequence (request) |
| Virtual address space | Logical KV cache (contiguous token positions) |
| Physical page frame | Physical KV block |
| Page (fixed size, e.g. 4 KB) | Block (fixed size, e.g. 16 tokens) |
| Page table | Block table (per sequence) |
| Page table entry | Logical→physical block mapping |
| Free frame list | Block allocator free list |
| `fork()` + copy-on-write | Parallel sampling / beam fork + CoW |
| Shared library pages | Prefix-cached shared blocks |
| Page reference count | Block `ref_count` |
| Swap to disk | Swap to CPU RAM (PCIe) |
| Page fault | Swap-in / recompute on resume |
| Page replacement (LRU) | Evictor (LRU) for cached blocks |
| Internal fragmentation (last page) | Last-block fragmentation (≤B−1 tokens) |
| TLB (address-translation cache) | Block table loaded into kernel shared memory |

The analogy is not merely pedagogical; it imported decades of OS design wisdom. The choice of LRU eviction, the swap-vs-recompute decision (analogous to swap vs re-fault), the copy-on-write fork semantics, and the free-list allocator are all textbook OS techniques, which is why PagedAttention felt simultaneously novel (in ML serving) and obviously correct (to systems engineers). The one place the analogy breaks: OS pages are demand-paged with hardware MMU support and a TLB, whereas PagedAttention's "MMU" is software (the kernel reads the block table explicitly) — there is no hardware address translation for KV blocks, so the block-table read is an explicit shared-memory lookup in the attention kernel (§16.2). This software-MMU cost is the price of the abstraction and the reason kernel quality (FlashInfer) matters.

---

## 30. The Watermark and Preemption Threshold

vLLM does not wait until the free pool is *completely* empty to act — that would risk being unable to even perform the copy operations needed to preempt. Instead it maintains a **watermark**: a small reserve of free blocks (e.g. a configurable fraction, often a few hundred blocks) kept available as headroom. The scheduler's `can_allocate`/`can_append_slots` checks respect the watermark, treating the pool as "full" once free blocks drop to the watermark. This guarantees there is always room for the CoW copies and swap-out staging that preemption itself requires — preventing a deadlock where you need free blocks to free blocks. The watermark is analogous to the OS keeping a minimum free-page reserve so the page-out path (which may need to allocate) can always proceed. Tuning it trades a slightly smaller usable pool for robustness against allocation deadlock under pressure.

---

## 31. Modeling Prefix-Cache Hit Rate and Its Throughput Effect

Prefix caching's benefit is workload-dependent and worth modeling. Let a workload have a shared prefix of length `L_shared` (e.g. a system prompt) present in fraction `f` of requests, and a per-request unique suffix of length `L_unique`. Without prefix caching, average prefill work per request is `L_shared + L_unique` (for the `f` fraction) — but every request recomputes the shared part. With prefix caching, the shared prefix is computed once and then *hit*, so average prefill work drops to:

```
without APC:  prefill_tokens = L_shared + L_unique           (per request, recomputed)
with APC:     prefill_tokens ≈ L_unique + (1−hit_rate)·L_shared
```

For a 2,000-token system prompt (`L_shared = 2000`) with `L_unique = 50` and a 95% hit rate, per-request prefill drops from 2,050 tokens to `50 + 0.05·2000 = 150` tokens — a **13.7× reduction in prefill compute**. Since prefill is compute-bound, this directly frees GPU time for more decode (more concurrency) and slashes TTFT for cache-hit requests. The hit rate itself depends on the prefix working set fitting in the KV pool: if there are `K` distinct hot prefixes each `L_shared` long and the pool can cache `C` tokens, the hit rate approaches 1 when `K · L_shared ≪ C` and degrades as the prefix working set exceeds cache capacity (the evictor thrashes). This is why File 11's benchmarking distinguishes cold-cache and warm-cache measurements — reporting throughput without specifying cache state is meaningless for prefix-heavy workloads.

---

## 32. BlockManager v2 Abstractions in Code-Level Detail

BlockManager v2 (`vllm/core/block/`) introduced cleaner abstractions that make the features above composable:

- **`Block`** (interface): represents a block with its `block_id`, `token_ids` (the tokens it holds), `ref_count`, and a `content_hash` (for prefix caching). Implementations include `NaiveBlock` (no prefix caching) and `PrefixCachingBlock` (hash-aware).
- **`BlockAllocator`** (interface): `allocate_mutable_block` (a fresh writable block), `allocate_immutable_block` (a full block with a content hash, eligible for prefix-cache sharing), `free`, `fork`. Implementations: `NaiveBlockAllocator` and `PrefixCachingBlockAllocator` (which maintains the hash→block map and the evictor).
- **`BlockTable`**: owns a sequence's list of `Block`s and provides `append_token_ids` (which internally allocates/forks/CoWs as needed), `fork`, and `free`. It hides the mutable-vs-immutable transition: a block being written is *mutable*; once full, it becomes *immutable* and acquires a content hash, at which point it can be matched by the prefix cache.
- **`DeviceAwareBlockAllocator`**: wraps GPU and CPU allocators, routing swap operations between them.
- **`Evictor`** (interface): `LRUEvictor` is the default; pluggable for other policies.

The mutable→immutable transition is the elegant core: a block is private and writable while being filled, then "sealed" into an immutable, content-addressed, shareable block when complete. This is what makes prefix caching automatic — every sealed block is a candidate for future reuse without any explicit caching API, and the same machinery serves CoW (forking an immutable block shares it; writing forces CoW back to mutable). The v2 design thus unifies CoW, prefix caching, and swapping under one block-state model, which is why it replaced the more special-cased v1.

---

## 33. Final Synthesis

PagedAttention is the load-bearing wall of vLLM. Strip away the kernels, the scheduler policies, the distributed runtime, and the API server, and what remains — the thing that everything else assumes — is a pool of fixed-size KV blocks, addressed through per-sequence block tables, reference-counted for sharing, and relocatable between GPU, CPU, and the network. From that single abstraction flow: >95% memory utilization (vs <40%), copy-on-write parallel sampling and beam search, automatic prefix caching, graceful preemption via swap or recompute, sliding-window block aging, tensor-parallel KV sharding, chunked-prefill incremental allocation, and disaggregated KV transfer. The throughput it unlocked — 2–4× over prior systems — came not from a faster kernel but from a better *abstraction*, which is the hallmark of a genuine systems contribution. Every subsequent file in this database that touches KV cache is, at bottom, manipulating these blocks. With the memory mechanism fully mapped, we proceed to the policy layer: how vLLM's scheduler decides, iteration by iteration, which sequences run and how the block pool is spent (File 04). The recurring lesson — that a better abstraction beats a faster implementation, and that importing mature OS techniques (paging, CoW, swap, LRU) into a new domain can reorganize an entire field — is one worth carrying into every other system described in the remainder of this database.

---

## 34. Appendix: Memory-Related vLLM Configuration Reference

The flags that directly govern the memory manager, with their effects:

- **`--gpu-memory-utilization`** (default 0.9): fraction of each GPU's HBM vLLM may use. Determines `num_gpu_blocks` (§7.1). Higher → more KV cache, more concurrency, more OOM risk. Lower → safety margin. The single most important memory knob.
- **`--block-size`** (default 16; 8, 16, 32 typical): tokens per block (§13). Smaller → less fragmentation, finer prefix matching, worse coalescing. Must be compatible with the attention backend's page size.
- **`--swap-space`** (default 4 GiB): CPU swap pool size per GPU (§8). Bounds how many sequences can be swapped out simultaneously. Set higher if using swap preemption with long sequences.
- **`--enable-prefix-caching`**: turn on APC (§6). Big win for shared-prefix workloads; small overhead otherwise. On by default in newer versions.
- **`--max-num-seqs`** (default 256): cap on concurrent running sequences. Interacts with KV memory — admitting more sequences than KV supports causes preemption/thrash (§22).
- **`--max-num-batched-tokens`**: cap on tokens processed per step (prefill + decode). Bounds peak activation memory and is central to chunked-prefill tuning (File 04).
- **`--max-model-len`**: maximum sequence length. Sets the maximum block-table size and, with `rope_scaling`, the usable context. Larger values reserve more per-sequence block-table capacity and affect the profiling estimate.
- **`--preemption-mode`** (`recompute` | `swap`): the preemption strategy (§9). `recompute` avoids PCIe and is often the better default for short/medium sequences.
- **`--kv-cache-dtype`** (`auto` | `fp8_e5m2` | `fp8_e4m3` | ...): KV cache precision (File 02 §7, File 13). FP8 halves `block_bytes`, roughly doubling KV capacity on H100+.
- **`--num-gpu-blocks-override`**: manually set block count, bypassing profiling (for debugging or reproducible benchmarks).

These flags are not independent: `--max-num-seqs`, `--max-model-len`, `--gpu-memory-utilization`, and `--kv-cache-dtype` jointly determine whether the block pool can sustain the configured concurrency at the configured context length. File 11 provides a tuning methodology that sets them coherently from a target workload.

---

## 35. Appendix: Numerical and Correctness Edge Cases

A few subtleties that the memory manager must handle correctly, each a real source of bugs:

- **Empty prefix / first block:** the hash chain (§25.2) needs a well-defined sentinel for the "no preceding block" case so the first block of every sequence hashes consistently — otherwise identical prompts wouldn't share their first block.
- **Partial last block on a cache hit:** if a cached prefix ends mid-block, the partial block cannot be shared (it has no stable hash) and must be allocated fresh; the cached full blocks before it are shared. The boundary arithmetic (`num_computed_tokens` rounding down to a block multiple) must be exact.
- **CoW during a multi-token append (chunked prefill):** when a chunk writes several tokens that span a shared block boundary, CoW must trigger before the first write into a shared block, and the copy must capture the block's existing (shared) contents, not partial new state.
- **Swap during fork:** a sequence being swapped while its group forks requires the block manager to keep refcounts consistent across the GPU↔CPU move; v2's device-aware allocator handles the bookkeeping so a shared block swapped out and back retains its sharing.
- **Abort mid-prefill:** a client disconnect during a long chunked prefill must free all allocated blocks (including any CoW copies made) and decrement refcounts on any shared prefix blocks — a leak here slowly starves the pool.
- **Sliding-window + prefix-cache interaction:** aged-out SWA blocks must not be served as prefix-cache hits for positions outside the window; the cache and the window logic must agree on what KV is still valid.
- **TP rank divergence under preemption:** a swap or recompute decision must be applied identically on every tensor-parallel rank (§19); if one rank swapped a sequence out and another did not, their block tables would desynchronize and the next collective would mix valid and stale KV. The driver broadcasts a single scheduling decision so all ranks mutate state in lockstep.
- **FP8 KV scale storage:** quantized KV (FP8/INT8) requires a per-tensor or per-token scale alongside the quantized values; the block layout must reserve space for and correctly associate these scales with their blocks, or dequantization on read produces garbage. The scales are small but must travel with the block through swap and transfer.

These edge cases are why the block manager is one of the most carefully tested components in vLLM, and why the v2 redesign (which unified the state model, §32) reduced their incidence — a single mutable/immutable/shared/swapped state model is easier to get right than the v1 collection of special cases. For an inference engineer extending vLLM (a custom KV transfer backend, a new preemption policy, a novel attention variant), these are the invariants that must be preserved: refcount accuracy, position/`num_computed_tokens` consistency, watermark headroom, and the mutable→immutable sealing discipline. Violate any one and the symptom is silent KV corruption — wrong tokens, not a crash — which is the hardest class of inference bug to diagnose. The disciplined block abstraction is what keeps these invariants tractable, which is the final argument for why PagedAttention's *design*, not just its performance, was the contribution.

---

## 36. Appendix: Source References and Where to Look in the Code

For readers going to the source:

- **Paper:** Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica — *"Efficient Memory Management for Large Language Model Serving with PagedAttention,"* SOSP 2023. The canonical description of the algorithm, CoW, and the OPT-13B throughput results.
- **`vllm/core/scheduler.py`**: the scheduler that drives block-manager calls each step (detailed in File 04).
- **`vllm/core/block_manager.py`** and **`vllm/core/block/`**: the v1 and v2 block managers, allocators, block tables, and evictor (§§11, 32).
- **`vllm/attention/`**: the attention backends, including the native paged kernel and the FlashAttention/FlashInfer integrations (§16, File 10).
- **`vllm/worker/`** and **`vllm/model_executor/`**: where `slot_mapping` is built and KV is written/read during the forward pass (File 06).
- **CUDA kernels** (historically `csrc/attention/`): the paged-attention kernel with the `x`-vectorized layout and split-K reduction (§16).

The conceptual lineage to follow: read the SOSP paper for the *why*, then `block/` for the *bookkeeping*, then the attention backend for the *kernel*, then `scheduler.py` for the *policy* that spends the blocks. Each maps to a section above. Together they constitute the memory subsystem that this file has dissected — and the foundation on which Files 04 through 11 build.

---

*Next: File 04 — the vLLM scheduler, continuous batching, chunked prefill, preemption policies, and prefill–decode disaggregation, all operating on the block pool established here.*




