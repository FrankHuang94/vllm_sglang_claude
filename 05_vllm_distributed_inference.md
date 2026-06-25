# vLLM Distributed Inference — Tensor, Pipeline, Expert, and Sequence Parallelism

> **PRIMARY reference file.** Large models do not fit on one GPU, and even when they do, splitting them can raise throughput or cut latency. This file covers the four parallelism axes used in LLM inference — tensor (TP), pipeline (PP), expert (EP), and sequence (SP) parallelism — plus data parallelism (DP) and multi-instance deployment, the collective-communication patterns that bind them, the CUDA kernel optimizations vLLM uses, and multi-GPU KV cache management. Prerequisites: File 02 (architecture, esp. MoE §4 and attention §2), File 03 (KV cache), File 04 (the scheduler whose plan these workers execute).

---

## Table of Contents

1. Why and When to Distribute
2. Tensor Parallelism (TP)
3. Collective Communication: AllReduce, AllGather, All-to-All
4. Pipeline Parallelism (PP)
5. Expert Parallelism (EP) for MoE
6. Sequence Parallelism (SP) and Ring/Ulysses Attention
7. Data Parallelism and Multi-Instance Deployment
8. Combining Parallelism Axes
9. Multi-GPU KV Cache Management
10. CUDA Kernel Optimizations
11. Multi-Node Inference
12. Tuning and Topology Selection

---

## 1. Why and When to Distribute

There are three distinct reasons to use more than one GPU, and they call for different parallelism strategies:

1. **The model doesn't fit.** LLaMA-3 70B in BF16 is 140 GB — it cannot fit in one 80 GB GPU. You *must* split the weights across GPUs (tensor or pipeline parallelism) or quantize aggressively. This is a hard capacity constraint.
2. **You want lower latency.** Even if a model fits, splitting its compute across GPUs (tensor parallelism) reduces per-step latency — each GPU does a fraction of the matmuls in parallel. This trades communication overhead for compute parallelism and is how you hit aggressive TTFT/TPOT targets.
3. **You want more throughput.** Running multiple independent replicas (data parallelism) scales aggregate throughput linearly with GPU count, with no inter-GPU communication during inference. This is the simplest scaling axis when one replica already fits and meets latency.

These map to the parallelism axes: **TP** splits within a layer (latency + capacity), **PP** splits across layers (capacity), **EP** splits experts (MoE capacity), **SP** splits the sequence (long-context activation memory), and **DP** replicates the whole model (throughput). Real deployments combine them. The art is choosing the combination that meets latency and capacity at minimum cost, which §12 addresses.

The dominant cost of distribution is **communication**: every split that crosses a GPU boundary requires moving data over an interconnect (NVLink, PCIe, InfiniBand), and on the memory-bound decode path that communication can become the bottleneck. So the recurring theme is: split to fit and to parallelize, but keep the communication on the fastest available interconnect and minimize its volume and frequency.

---

## 2. Tensor Parallelism (TP)

Tensor parallelism (Megatron-LM, Shoeybi et al. 2019, arXiv 1909.08053) partitions individual weight *matrices* across GPUs so that each GPU computes part of every layer. It is the workhorse of large-model inference because it reduces both memory *and* per-step latency.

### 2.1 Column-parallel and row-parallel linear layers

The two primitives are partitioning a linear layer `Y = XW` along its output dimension (column-parallel) or input dimension (row-parallel):

**Column-parallel:** split `W` by columns: `W = [W_1 | W_2 | … | W_p]` across `p` GPUs. Each GPU computes `Y_i = X W_i` — a slice of the output columns. No communication is needed to compute (each GPU has the full input `X` and produces its output slice); the outputs are concatenated logically. Used where the output feeds an operation that can work on partitioned features.

**Row-parallel:** split `W` by rows, which requires splitting the input by columns too: `W = [W_1; W_2; …; W_p]`, `X = [X_1 | X_2 | … | X_p]`. Each GPU computes a partial product `X_i W_i`, and the partial results must be **summed** across GPUs (an AllReduce) to form the final `Y = Σ X_i W_i`. Used where the input is already partitioned (e.g. follows a column-parallel layer).

### 2.2 Applying TP to a transformer block

Megatron arranges the attention and FFN so that each needs exactly **one AllReduce**:

**Attention:**
- Q, K, V projections are **column-parallel**: each GPU computes Q/K/V for a subset of attention heads (`num_heads / p` query heads, `num_kv_heads / p` KV heads). Each GPU then runs attention *independently* for its heads — no communication, because each head's attention is self-contained.
- The output projection is **row-parallel**: it takes the concatenated per-head outputs (which are partitioned across GPUs) and projects back to `d_model`, producing partial sums that are **AllReduced** to combine the heads.

**FFN (SwiGLU):**
- The gate and up projections (`W_1`, `W_3`) are **column-parallel**: each GPU computes a slice of the intermediate `d_ff`.
- The down projection (`W_2`) is **row-parallel**: each GPU produces a partial `d_model` output, **AllReduced** to combine.

So a transformer block under TP needs **2 AllReduces** (one after attention output projection, one after FFN down projection). The activations between the column-parallel and row-parallel layers stay partitioned, never communicated — only the final combine of each sublayer is collective.

### 2.3 The head-divisibility constraint

Because Q/K/V are split by heads, the TP degree must divide the head counts: `num_heads % p == 0` and `num_kv_heads % p == 0`. For LLaMA-3 70B (64 query heads, 8 KV heads), TP can be 1, 2, 4, or 8 (keeping ≥1 KV head per GPU). TP=8 puts exactly 1 KV head per GPU (each GPU's query heads all share that one KV head — GQA, File 02 §2.3). TP=16 would require fewer than one KV head per GPU, so it needs KV-head *replication* (each KV head duplicated across a pair of GPUs), which wastes a little memory but is supported for very large TP degrees. This constraint, set by the model architecture (File 02 §17), directly limits deployment topology — you cannot choose an arbitrary TP degree.

### 2.4 The communication cost of TP

The AllReduce volume per operation is `batch · seq_len · d_model · bytes`. For **decode** (the latency-critical path), batch contributes per-sequence but seq_len = 1:

```
decode AllReduce size = num_running_seqs · 1 · d_model · bytes
For d_model = 8192, BF16, single sequence:  8192 · 2 = 16 KB
```

That is tiny in *volume* but happens **2× per layer × num_layers** times per decode step (160 AllReduces for an 80-layer model), and each is **latency-bound** (a 16 KB AllReduce is dominated by latency, not bandwidth). On NVLink (high bandwidth, low latency) these AllReduces cost single-digit microseconds each; over PCIe (no NVLink) they cost several times more, and over them adding up across 160 per step, the difference between NVLink and PCIe TP is dramatic. This is why **TP demands NVLink**: the frequent small AllReduces on the decode critical path are latency-bound, and only NVLink (or NVSwitch) keeps that latency low enough to scale TP to 8.

### 2.5 vLLM TP implementation

vLLM runs one worker process per TP rank (per GPU). The model's linear layers are the Megatron-style `ColumnParallelLinear` and `RowParallelLinear` (`vllm/model_executor/layers/linear.py`), which handle the weight sharding and insert the AllReduce. NCCL provides the collective. Each rank loads only its shard of the weights (File 06 §weight loading). The scheduler (File 04) runs on the driver and broadcasts the batch plan to all ranks, which execute in lockstep — including identical block-table mutations (File 03 §19) so their sharded KV caches stay consistent. The `--tensor-parallel-size` flag sets `p`.

### 2.6 TP scaling efficiency

Throughput scaling under TP is sublinear because of the AllReduce overhead. On A100/H100 with NVLink, TP=8 typically achieves ~90%+ scaling efficiency (8 GPUs deliver ~7.2× a single GPU's throughput for a model that fits on one). Without NVLink (PCIe-only), efficiency drops to ~60–70% at TP=8 because the AllReduce latency dominates. The efficiency loss is the price of latency reduction and capacity — and it's why, when a model *fits* on fewer GPUs, **TP=4 with DP=2 often beats TP=8** for throughput (less AllReduce overhead per replica, §7, §12).

---

## 3. Collective Communication: AllReduce, AllGather, All-to-All

The parallelism axes rest on a handful of collective operations. Understanding their cost models is essential to predicting distributed inference performance.

### 3.1 AllReduce

Every GPU contributes a tensor; all GPUs receive the elementwise sum. Used to combine row-parallel partial sums (§2.2). The standard efficient algorithm is **ring AllReduce**: data flows around a ring of GPUs in `2(p−1)` steps (reduce-scatter then all-gather), achieving bandwidth-optimal `2(p−1)/p · message_size` transfer per GPU. For small messages (decode AllReduces, §2.4) the `2(p−1)` *latency* term dominates; for large messages (prefill, training) the bandwidth term dominates. NCCL picks ring or tree algorithms based on message size and topology.

### 3.2 AllGather and ReduceScatter

**AllGather**: each GPU has a shard; all GPUs end with the full concatenation. Used to gather vocab-parallel logits before sampling (File 02 §14.4) and in sequence parallelism. **ReduceScatter**: the reduce half of an AllReduce — each GPU ends with the sum of one shard. AllReduce = ReduceScatter + AllGather, which is why some optimizations fuse them with adjacent computation.

### 3.3 All-to-All

Each GPU sends a distinct chunk to every other GPU (a transpose of data distribution). This is the communication pattern of **expert parallelism** (§5) and **Ulysses sequence parallelism** (§6): tokens are redistributed so each GPU holds the data it needs to compute. All-to-all volume is `message_size` per GPU pair; for MoE dispatch it is `batch · top_k · d_model · bytes` total. All-to-all is more demanding than AllReduce because it is genuinely all-pairs (no ring optimization reduces the total volume), making it bandwidth-hungry — which is why EP is sensitive to interconnect bandwidth (§5.2).

### 3.4 NCCL and custom all-reduce

vLLM uses **NCCL** (NVIDIA Collective Communications Library; **RCCL** on AMD) for these collectives. NCCL auto-tunes algorithms and topology (NVLink rings, PCIe trees, multi-node InfiniBand). For the small, latency-bound decode AllReduces, vLLM provides a **custom all-reduce** kernel (`--disable-custom-all-reduce` to turn off) that, for tensors under a threshold (~a few MB), bypasses NCCL's overhead with a hand-written kernel using direct peer-to-peer GPU memory access over NVLink — lower latency than NCCL for the small-message regime. It can also fuse the residual add into the all-reduce (§10.2), saving a kernel launch. This optimization specifically targets the decode critical path where 160 tiny AllReduces per step make per-AllReduce latency the bottleneck.

### 3.5 The communication cost model summary

```
AllReduce (ring):     2(p−1)/p · size  bytes/GPU ;  latency ~ 2(p−1) · hop_latency
AllGather:            (p−1)/p · size   bytes/GPU
All-to-All:           size · (p−1)/p   bytes/GPU, but all-pairs (no volume reduction)
Decode TP AllReduce:  16 KB × 2 × num_layers per step → latency-bound
MoE EP All-to-All:    batch·top_k·d_model·bytes × 2 per layer → bandwidth-bound
```

The decisive distinction: TP's collectives are **small and frequent** (latency-bound, love NVLink), while EP's are **large and all-pairs** (bandwidth-bound, need fat interconnect). This is why TP is kept within a NVLink node and EP is the axis that spans nodes (§8).

---

## 4. Pipeline Parallelism (PP)

Pipeline parallelism partitions the model's *layers* (not within-layer matrices) across GPUs: GPU 0 holds layers 0..k, GPU 1 holds layers k+1..2k, etc. It is primarily a *capacity* technique — it lets a model span more GPUs than TP's head-divisibility allows, and it uses cheaper communication than TP.

### 4.1 PP mechanics for inference

Inference PP is simpler than training PP (no backward pass). Each GPU (stage) processes its layers and sends the resulting activations to the next stage via point-to-point send/recv. Stage 0 also does embedding; the last stage does the final norm and LM head. The KV cache is **partitioned by layer**: each stage holds the KV only for its layers (File 03 §19's TP sharding is by head; PP sharding is by layer). So PP reduces per-GPU model memory *and* per-GPU KV memory by the PP degree.

### 4.2 The pipeline bubble

The cost of PP is the **bubble**: at the start and end of processing a batch, some stages are idle waiting for data to flow through the pipeline. For `P` stages processing a single batch, the bubble wastes `(P−1)/P` of the time if you don't pipeline multiple micro-batches. The fix is **micro-batching**: split the batch into `m` micro-batches and pipeline them so all stages stay busy; the bubble fraction shrinks to `(P−1)/(m + P − 1)`. For `P=4`, `m=8`: bubble = `3/11 ≈ 27%`... still significant; larger `m` helps but requires enough concurrent requests to fill the pipeline (File 04 §36).

### 4.3 PP communication cost

PP communicates only the activations between stages: `batch · seq_len · d_model · bytes` per stage boundary, point-to-point, `P−1` times per forward pass. This is **much less frequent** than TP's per-layer AllReduces — only at stage boundaries, not every layer — and point-to-point send/recv is cheaper than a collective. Crucially, **PP tolerates slower interconnects** than TP: because the communication is infrequent and point-to-point, PP works acceptably over PCIe or even InfiniBand between nodes, whereas TP demands NVLink. This makes PP the natural axis for crossing node boundaries.

### 4.4 TP vs PP trade-offs

- **TP** reduces latency (parallel compute within each layer) but needs NVLink (frequent AllReduces) → use *within* a node.
- **PP** reduces memory with cheap, infrequent communication but adds pipeline-bubble latency and needs concurrency to fill the pipe → use *across* nodes or when TP's head-divisibility is exhausted.
- For a model that fits with TP=8 on one NVLink node, pure TP is usually best (lowest latency, high efficiency). For a model needing 16+ GPUs spanning nodes, the standard is **TP=8 within each node, PP across nodes** (§8, §11).

### 4.5 vLLM PP implementation

`--pipeline-parallel-size` sets `P`. vLLM passes activations between stages with `torch.distributed` point-to-point send/recv. The scheduler runs on rank 0 and the batch (micro-batched) flows through the stages; continuous batching integrates by having the scheduler supply enough micro-batches to keep the pipeline full. PP and TP compose: `--tensor-parallel-size 8 --pipeline-parallel-size 2` uses 16 GPUs (8-way TP within each of 2 pipeline stages).

---

## 5. Expert Parallelism (EP) for MoE

For Mixture-of-Experts models (File 02 §4.2, §21), the experts are the bulk of the parameters but only `k` of `E` activate per token. Expert parallelism distributes the experts across GPUs.

### 5.1 EP mechanics

Each GPU (or group) hosts a subset of the experts. A token, after the router selects its top-`k` experts (which may live on different GPUs), must be *sent* to those GPUs, processed by the expert FFN, and the results *returned*. This is two **all-to-all** operations per MoE layer:

- **Dispatch all-to-all:** each GPU sends each of its tokens to the GPU(s) hosting that token's selected experts.
- **Combine all-to-all:** each GPU sends the processed token results back to the originating GPU, which combines them with the router gate weights.

### 5.2 EP communication cost

The all-to-all volume per dispatch is `batch · top_k · d_model · bytes` (each token's hidden state sent to `top_k` experts). Worked for DeepSeek-V3 scale (File 02 §15.4): batch 1024, d_model 7168, top-k dispatch, over 400 Gbps InfiniBand → ~294 µs per all-to-all → ~588 µs per MoE layer of pure communication. With dozens of MoE layers, this communication can dominate, which is why EP is **bandwidth-bound** and sensitive to interconnect. Overlapping communication with computation (DeepSeek's DualPipe; computing one expert's tokens while dispatching the next) is essential to hide it.

### 5.3 EP is orthogonal to TP

EP and TP operate on different parts of the model: TP splits the attention and dense layers (by head / by matrix); EP splits the MoE expert FFNs (by expert). They compose as independent process groups: `EP_degree × TP_degree` GPUs, where attention/dense layers are tensor-parallel and MoE layers are expert-parallel over (potentially different) groupings. A token's journey through one block: attention (TP, AllReduce) → router → dispatch (EP all-to-all) → expert FFN (local) → combine (EP all-to-all) → next block.

### 5.4 DeepSeek-V3 EP128

DeepSeek-V3 (671B total, 37B active, 256 experts, top-8 routing) uses **EP128**: 128 GPUs for expert parallelism, 2 experts per GPU. Each token routes to 8 experts spread across up to 8 GPUs; the all-to-all over InfiniBand+NVLink moves tokens to their experts. DeepSeek's open-source inference code demonstrates production EP128 with communication-computation overlap (DualPipe), FP8 throughout, and careful load balancing (the model was trained with an auxiliary-loss-free balancing strategy so inference routing is naturally balanced). It is the reference implementation for large-scale MoE serving.

### 5.5 Load imbalance at inference

EP's Achilles' heel is **routing imbalance** (File 02 §21.2): if a popular expert receives far more tokens than others, its host GPU becomes a straggler (the all-to-all and the expert GEMM wait for it), and may even OOM. Inference mitigations: capacity factors (drop or defer overflow — quality cost), expert replication (host a hot expert on multiple GPUs), or relying on the model's training-time balancing. The variance also makes MoE step time noisier than dense (File 02 §FAQ), complicating CUDA graphs and SLO estimation.

### 5.6 vLLM EP

vLLM supports MoE via the `FusedMoE` layer (File 02 §4.2, File 06) and `--enable-expert-parallel` for distributed experts, using process groups for the EP all-to-all. It composes with TP (`--tensor-parallel-size` for the dense/attention parts). For models like Mixtral and DeepSeek, vLLM handles the routing, the grouped expert GEMM, and (with EP) the token dispatch/combine.

---

## 6. Sequence Parallelism (SP) and Ring/Ulysses Attention

At very long context (100K–1M+ tokens), a problem arises that TP doesn't solve: **activation memory**. Under TP, the weights are split but the *activations* (the `batch · seq_len · d_model` residual stream and attention intermediates) are **replicated** on every TP rank. At 1M tokens, those activations are enormous and replicating them wastes memory. Sequence parallelism splits the *sequence dimension* across GPUs.

### 6.1 Ulysses sequence parallelism

DeepSpeed-Ulysses (arXiv 2309.14509): partition the sequence across GPUs (each GPU holds a contiguous chunk of tokens). For the attention operation — which needs all tokens to attend to all tokens — perform an **all-to-all** before attention to switch from "split by sequence, all heads" to "split by heads, all sequence": each GPU then computes attention for *all* tokens but only a *subset of heads*. After attention, another all-to-all switches back to sequence-split for the FFN. Cost: 2 all-to-all per attention layer; benefit: activation memory per GPU reduced by the SP degree.

### 6.2 Ring attention

Ring attention (Liu et al., arXiv 2310.01889): distribute Q, K, V by sequence chunks across GPUs arranged in a ring. Each GPU computes attention for its local Q against its local K/V, then passes its K/V to the next GPU in the ring while receiving the previous GPU's K/V, computing the next partial attention — using online softmax (File 02 §3.2) to combine partials across the ring. Memory per GPU is `O(seq_len / SP_degree)`, enabling extremely long sequences (1M+ tokens) that wouldn't fit on any single GPU. The communication is point-to-point ring passing of K/V, overlapped with compute.

### 6.3 When SP matters

SP is a **long-context** technique (File 13). For typical chat (≤32K context), TP suffices and SP adds unnecessary all-to-all overhead. For 128K–1M+ contexts where activations dominate, SP (often combined with TP) is what makes the context length feasible at all. vLLM has experimental SP support (`sequence_parallel_size`) integrated with TP. The decision is driven by context length: short → no SP; very long → SP becomes necessary as activation memory exceeds what TP-replicated layout allows.

---

## 7. Data Parallelism and Multi-Instance Deployment

The simplest scaling axis: run multiple independent model replicas, each serving a subset of requests, with **no inter-replica communication during inference**.

### 7.1 Request-level data parallelism

Each replica is a complete, independent engine (possibly itself using TP/PP internally). A load balancer distributes requests across replicas. Throughput scales linearly with replica count (no communication overhead, unlike TP's AllReduce tax). This is the right axis when one replica already fits the model and meets latency — you scale *out* for throughput rather than *up* for capacity/latency.

```
total GPUs = TP_per_replica × PP_per_replica × DP_replicas
```

vLLM's `--data-parallel-size` runs `DP` replicas, each using `tensor_parallel_size × pipeline_parallel_size` GPUs. Each DP replica is an independent vLLM engine with its own KV cache, scheduler, and prefix cache.

### 7.2 Load balancing strategies

How the router distributes requests across replicas matters:

- **Round-robin:** simple, even distribution; ignores per-replica load and cache state.
- **Least-pending / least-loaded:** route to the replica with the shortest queue — balances load dynamically, good for heterogeneous request sizes.
- **Prefix-cache-aware (consistent hashing):** route requests sharing a prefix to the *same* replica so they hit that replica's prefix cache (File 03 §6). This is increasingly important — naive round-robin scatters a shared system prompt across all replicas, so each must compute it (low hit rate), whereas affinity routing concentrates it (high hit rate). The trade-off is load balance vs cache locality; consistent hashing with bounded loads balances both.

vLLM's ecosystem includes a router (the **vLLM Production Stack** / router projects) that implements prefix-cache-aware routing and KV-aware load balancing — turning prefix caching into a cluster-level optimization. This is the data-parallel analog of the scheduler's intra-engine prefix awareness (File 04 §20).

### 7.3 DP attention (a special case for MoE)

A nuance: for MoE models, "data parallelism" sometimes refers to **DP attention** — replicating the attention computation across ranks while the experts are expert-parallel. Because attention is relatively cheap for MoE models (the FFN/experts dominate), replicating it per rank (rather than tensor-parallelizing it) can be more efficient, with the experts shared via EP. SGLang's `--enable-dp-attention` (File 09) implements this; it's distinct from request-level DP and is a MoE-serving optimization.

---

## 8. Combining Parallelism Axes

Real large-model deployments combine axes. The governing principle: **match each axis to the appropriate interconnect tier**, putting bandwidth-hungry, latency-sensitive communication on the fastest links.

### 8.1 The interconnect hierarchy

```
Intra-GPU:           HBM, ~3 TB/s (H100)
Intra-node (NVLink): ~900 GB/s (H100 NVL), microsecond latency
Intra-node (PCIe):   ~64 GB/s (PCIe 5.0)
Inter-node (IB/RoCE): ~50 GB/s (400 Gbps), higher latency
```

### 8.2 The standard large-model recipe

- **TP within a NVLink node** (TP=8 on an 8-GPU NVLink node): TP's frequent small AllReduces (§2.4) need NVLink's low latency. TP=8 maps perfectly to an 8-GPU NVSwitch node.
- **PP across nodes** (PP=N over InfiniBand): PP's infrequent point-to-point activation transfers (§4.3) tolerate inter-node latency. PP crosses node boundaries.
- **EP for MoE across nodes** (EP=128 over IB+NVLink): EP's all-to-all is bandwidth-bound; spread experts across nodes with fat IB, overlapping communication with compute.
- **DP for throughput** (replicate the whole TP×PP×EP unit): scale out replicas behind a load balancer.

Example: a 405B dense model on 32 GPUs (4 nodes × 8 GPUs) → TP=8 within each node, PP=4 across nodes. A 671B MoE (DeepSeek-V3) → TP for attention within nodes, EP=128 across nodes, as DeepSeek deploys it. A 70B model that fits on one 8-GPU node → TP=8 single replica, or TP=4 × DP=2 for more throughput (§12).

### 8.3 Why the recipe works

Each axis is placed where its communication pattern is cheapest: latency-bound TP on lowest-latency NVLink, bandwidth-bound EP on fat inter-node IB with overlap, infrequent PP across the slowest links, and communication-free DP as the outermost scale-out. Violating this — e.g. TP across nodes — puts the wrong communication on the wrong interconnect (frequent latency-bound AllReduces over high-latency IB) and tanks efficiency. This placement logic is the single most important thing to get right in multi-GPU deployment topology.

---

## 9. Multi-GPU KV Cache Management

### 9.1 KV sharding under TP

Under TP, the KV cache is sharded by KV head (File 03 §19): with 8 KV heads and TP=8, each GPU holds 1 KV head's worth of KV per token. This means each GPU's `block_bytes` is `1/TP` of the full model's, so each GPU can hold the same number of *blocks* for less memory — and the logical block table is identical across ranks (allocate/free in lockstep). The concurrency benefit: TP both fits a larger model *and*, by sharding KV, leaves proportionally more room for KV cache per GPU (each GPU stores fewer heads), supporting more concurrent sequences. The driver makes block decisions and broadcasts them so all ranks' block tables stay synchronized (File 03 §35's "TP rank divergence" edge case).

### 9.2 KV partitioning under PP

Under PP, each pipeline stage holds the KV only for *its* layers (§4.1). Stage 0 caches layers 0..k/P, stage 1 caches k/P..2k/P, etc. The stages manage their KV independently — each can evict/swap its own blocks — and no cross-stage KV sharing is needed because each stage's KV is self-contained for its layers. This is a clean separation: PP partitions KV by layer, TP partitions by head, and they compose (under TP×PP, each GPU holds its layers' KV for its heads).

### 9.3 Coordinated allocation

The key correctness requirement (File 03 §19, §35): all TP ranks must perform identical block allocations/frees/swaps so their sharded caches remain consistent. vLLM achieves this by having the scheduler (on the driver) decide and broadcast the block operations; every worker applies the same logical mutation to its physical shard. A divergence — one rank freeing a block another still uses — would silently corrupt KV (the next collective would mix valid and stale data). This lockstep discipline is why the block manager is a CPU component whose decisions are replicated, not made independently per GPU.

---

## 10. CUDA Kernel Optimizations

Distribution adds communication, but vLLM also optimizes the per-GPU compute kernels, several of which interact with the parallelism.

### 10.1 Attention backends

vLLM has multiple attention backends (`vllm/attention/backends/`): FlashAttention, FlashInfer, XFormers, and Triton. The backend is selected by hardware capability and sequence characteristics (File 10). **FlashInfer** (arXiv 2501.01005) is the default high-performance backend on NVIDIA — it provides paged-KV attention with variable-length batch support, prefix-cache awareness, and GQA-optimized grouping (File 02 §28, File 03 §16.4), outperforming the native paged kernel in most regimes.

### 10.2 Custom all-reduce with fused residual

As noted (§3.4), vLLM's custom all-reduce kernel handles the small, frequent decode AllReduces faster than NCCL by using direct P2P over NVLink, and can **fuse the residual add** into the all-reduce — instead of (AllReduce, then a separate residual-add kernel), one kernel does both, saving a launch and an HBM round-trip on the decode critical path. This matters because those 160 AllReduces per step (§2.4) are latency-bound, and every saved kernel launch counts.

### 10.3 Fused kernels

vLLM fuses several operations to cut HBM traffic (File 02 §13, §24; File 10):
- **RMSNorm + QKV projection**, with RoPE fused into the QKV computation.
- **SiLU + multiply** for SwiGLU (`act(gate) ⊙ up` in one kernel).
- **Quantized matmuls** (GPTQ Marlin, AWQ, FP8) that fuse dequantization into the GEMM.

These fusions are independent of the parallelism axis but compound with it: under TP, each rank runs the fused kernels on its shard, and fewer kernel launches per layer reduces the per-step overhead that TP's many AllReduces already strain.

### 10.4 Prefix-aware attention

When some KV blocks are shared read-only prefix blocks (File 03 §6) and others are freshly written, the attention kernel can skip write-coalescing logic for the read-only blocks and treat the cached prefix efficiently. FlashInfer's wrappers exploit this distinction, which is why prefix caching and the kernel choice are coupled (File 03 §4.4).

---

## 11. Multi-Node Inference

For models too large for one node (405B+, or 671B MoE) and for large-scale throughput, inference spans multiple nodes.

### 11.1 Topology

A node is typically 8 GPUs on an NVSwitch (full-bandwidth NVLink among them). Nodes connect via InfiniBand or RoCE (RDMA over Converged Ethernet), at ~400 Gbps. The recipe (§8.2): TP within nodes (NVLink), PP or EP across nodes (IB). vLLM supports multi-node via Ray or `torch.distributed` with the appropriate `--tensor-parallel-size` / `--pipeline-parallel-size` spanning the node layout, plus environment configuration for the IB network.

### 11.2 The inter-node latency tax

Inter-node communication has both lower bandwidth and higher latency than NVLink. This is why TP must *not* cross nodes (its frequent AllReduces would each pay inter-node latency × 160/step — catastrophic) and why PP/EP, with their infrequent/overlappable communication, are the cross-node axes. Latency sensitivity also means multi-node decode is harder than multi-node prefill (decode's per-step communication is on the critical path; prefill's is amortized over heavy compute) — a motivation for disaggregation (File 04 §10, File 15) that can keep decode on tighter topologies.

### 11.3 Orchestration

vLLM uses **Ray** for multi-node worker orchestration (placement groups pin workers to GPUs across nodes) or a `torchrun`-style launch. The driver coordinates the scheduler and broadcasts plans; NCCL/RCCL handles the collectives over the IB fabric. Production multi-node deployment also needs the fabric configured (NCCL environment variables for IB, GPUDirect RDMA for GPU-to-NIC transfers bypassing CPU) — misconfiguration here (e.g. NCCL falling back to TCP) silently destroys performance.

---

## 12. Tuning and Topology Selection

Choosing the parallelism configuration is a constrained optimization: meet capacity (model + KV fits) and latency (TTFT/TPOT SLOs) at minimum cost. A decision procedure:

### 12.1 Does it fit?

Compute weight memory (`P × bytes`, after quantization) and required KV (File 03 §17). If it fits on one GPU → no model parallelism needed (use DP for throughput). If not → minimum TP (and PP if TP's head-divisibility or single-node GPU count is exhausted) to fit.

### 12.2 TP vs DP for a model that fits on few GPUs

If a model fits on, say, 4 GPUs, you can deploy TP=8 (one replica, 8 GPUs) or TP=4 × DP=2 (two replicas, 8 GPUs). For **throughput**, TP=4 × DP=2 usually wins: each replica has less AllReduce overhead (TP=4 < TP=8 communication), and two independent replicas scale linearly. For **latency** (lowest TTFT/TPOT for a single request), TP=8 wins (more parallel compute per step). So: latency-critical → higher TP; throughput-critical → lower TP × higher DP. This is one of the most common and consequential tuning decisions (File 11 §multi-GPU scaling).

### 12.3 Measuring scaling efficiency

Benchmark throughput at TP=1, 2, 4, 8 and compute efficiency = `(throughput_TP=N / N) / throughput_TP=1`. With NVLink, expect ~90%+ at TP=8; without (PCIe), ~60–70%. If efficiency is poor, check: NVLink present and used (`nvidia-smi topo -m`), custom all-reduce enabled, NCCL not falling back to a slow path. Poor scaling usually means the AllReduce is on a slow interconnect or NCCL is misconfigured.

### 12.4 MoE-specific topology

For MoE, the dense/attention parts (TP) and expert parts (EP) scale differently. Size TP for the attention/dense memory and latency, and EP for the expert capacity, placing EP across nodes with fat IB and overlapping its all-to-all. Watch expert load balance (§5.5) — imbalance, not raw FLOPs, is often the MoE bottleneck.

### 12.5 The cost lens

Ultimately topology is chosen for **$/token at the SLO**. More TP/PP raises capacity and latency but adds communication overhead (lower efficiency = higher cost). More DP scales throughput cheaply but doesn't help if one replica can't meet latency. Disaggregation (File 04 §10, §30, File 15) can cut cost by putting prefill and decode on differently-sized, differently-parallelized pools. File 11 turns this into a measurement-driven methodology; the principle here is to use the *minimum* model parallelism that meets capacity and latency, and scale throughput with DP, keeping each parallelism axis on its appropriate interconnect tier.

---

## 13. The Megatron Partitioning, Derived Carefully

It is worth deriving why the Megatron column-then-row arrangement needs exactly one AllReduce per sublayer, because the reasoning illuminates how to parallelize any new layer type an inference engineer might encounter.

Consider the FFN `Y = W_2 · f(W_1 · X)`, where `f` is a nonlinearity. The naive question is: how do we split this across `p` GPUs so that communication is minimized? The key observation is that the nonlinearity `f` is applied element-wise to the intermediate `H = W_1 · X`. If we partition `W_1` by columns, `W_1 = [W_1^{(1)} | … | W_1^{(p)}]`, then GPU `i` computes `H^{(i)} = W_1^{(i)} · X`, a slice of the intermediate's columns. Because `f` is element-wise, GPU `i` can apply it locally to its slice: `f(H^{(i)})` — no communication needed, because each output element of `f` depends only on the corresponding element of `H`, which lives entirely on GPU `i`. This is the crucial property: a column split followed by an element-wise nonlinearity requires no communication, because the nonlinearity does not mix columns.

Now `W_2` must consume the full intermediate `f(H)`, but `f(H)` is partitioned by columns across the GPUs. If we partition `W_2` by *rows* to match (`W_2 = [W_2^{(1)}; … ; W_2^{(p)}]`), then GPU `i` computes the partial product `W_2^{(i)} · f(H^{(i)})`, and the final output is the *sum* of these partials: `Y = Σ_i W_2^{(i)} · f(H^{(i)})`. That sum is a single AllReduce. So the column-then-row arrangement places the only required communication at the very end of the FFN, after both matmuls and the nonlinearity — one AllReduce. Any other arrangement (e.g. row-then-column) would require communicating the intermediate `H` *before* the nonlinearity, which is both larger and more frequent. The same logic applies to attention: Q/K/V column-parallel (heads don't mix in attention), output projection row-parallel (combine the heads with one AllReduce). The Megatron pattern is, in effect, the unique arrangement that defers all communication to a single end-of-sublayer reduction by exploiting the points where the computation does *not* mix the partitioned dimension. Internalizing this lets you parallelize a novel architecture: find the operations that don't mix the split dimension (element-wise ops, per-head attention), keep them local, and place an AllReduce only where partitioned partials must be summed.

### 13.1 Why activations stay partitioned (and SP's opportunity)

Notice that between the column-parallel and row-parallel matmuls, the intermediate activations live *partitioned* across GPUs — they are never gathered. This is efficient (no communication) but means the *input* `X` and the *output* `Y` of each sublayer are **replicated** on all ranks (each rank needs the full `X` to compute its column slice, and the AllReduce gives every rank the full `Y`). That replication of the `batch · seq_len · d_model` residual stream is exactly the activation memory SP (§6) targets at long context: SP additionally splits `X` and `Y` by sequence so they aren't replicated, at the cost of all-to-all communication around attention. So TP and SP are complementary: TP splits the weights and keeps activations replicated; SP additionally splits the activations. At short context the replication is cheap and TP alone suffices; at long context SP pays for itself.

---

## 14. Worked AllReduce Latency Numbers

To make TP's communication tax tangible, work the decode-path numbers for LLaMA-3 70B (80 layers, `d_model = 8192`, BF16) at TP=8 on an H100 NVL node (NVLink ~900 GB/s, hop latency on the order of ~1–2 µs for a small ring AllReduce).

Per decode step there are `2 × 80 = 160` AllReduces. Each AllReduce, for a single decoding sequence, moves `8192 × 2 = 16 KB`. The ring AllReduce over 8 GPUs does `2(p−1) = 14` communication steps. For such a small message the transfer time is negligible (16 KB / 900 GB/s ≈ 18 ns per hop) and the cost is dominated by per-step latency overhead and kernel launch — call it ~2–4 µs per AllReduce in practice with the custom all-reduce kernel:

```
per-step AllReduce time ≈ 160 × ~3 µs ≈ 480 µs
```

If the decode step's compute is, say, ~5–10 ms for this model at TP=8, the ~0.5 ms of AllReduce is ~5–10% overhead — acceptable, and the reason TP=8 scales well on NVLink. Now imagine the same over PCIe or inter-node IB, where each small AllReduce costs ~15–30 µs instead of ~3 µs: `160 × 20 µs = 3.2 ms` of communication per step — comparable to or exceeding the compute, collapsing efficiency. This single calculation explains the iron rule "TP only within NVLink": the 160-per-step latency-bound AllReduces amplify any per-AllReduce latency increase by two orders of magnitude in aggregate. It also explains why batching helps amortize: at batch 64 the AllReduce volume is 64× larger (`1 MB` instead of 16 KB) but the *latency* per AllReduce barely changes (still latency-bound until the message gets large), so the communication overhead as a *fraction* of the (now larger) compute step shrinks — TP is more efficient at larger batch.

---

## 15. NCCL Ring AllReduce, Mechanically

Understanding the ring algorithm clarifies the cost model and the failure modes. NCCL's ring AllReduce arranges the `p` GPUs in a logical ring and proceeds in two phases:

**Reduce-scatter (p−1 steps):** the data is divided into `p` chunks. In each step, every GPU sends one chunk to its ring-neighbor and receives a chunk from the other neighbor, adding the received chunk to its local copy. After `p−1` steps, each GPU holds the fully-reduced (summed across all GPUs) value for *one* of the `p` chunks.

**All-gather (p−1 steps):** each GPU now broadcasts its fully-reduced chunk around the ring. After `p−1` more steps, every GPU has all `p` fully-reduced chunks — the complete AllReduce result.

Total: `2(p−1)` steps, each transferring `size/p` bytes, so each GPU sends/receives `2(p−1)/p · size` bytes — **bandwidth-optimal** (independent of `p` in the limit, since `2(p−1)/p → 2`). The catch is the `2(p−1)` *step count*: for small messages, the per-step latency (not bandwidth) dominates, so latency scales with `p`. This is precisely why small decode AllReduces (§14) are latency-bound and why NCCL switches to **tree** algorithms (logarithmic step count, `O(log p)`) for latency-sensitive small messages, and why vLLM's custom all-reduce (which uses direct NVLink P2P with even lower overhead) helps further. For large messages (prefill, training gradients), the ring's bandwidth-optimality wins. NCCL auto-selects based on message size and the topology it discovers. A common production failure: NCCL fails to detect NVLink/NVSwitch and falls back to PCIe or even sockets, silently degrading every collective — always verify with NCCL debug logging and `nvidia-smi topo -m` that the expected fast path is used.

---

## 16. A Fully Worked Deployment: 70B on 8×H100

Tie the axes together with a concrete sizing for serving LLaMA-3 70B (GQA, 64 query / 8 KV heads, 80 layers, `d_model 8192`) on a single 8×H100 NVLink node, targeting a chat workload (avg 1K-token prompt, 500-token output, P99 TPOT < 50 ms).

**Capacity check:** BF16 weights are 140 GB > 80 GB, so a single GPU is impossible; TP≥2 is required. At TP=8, weights are 17.5 GB/GPU, leaving ~54 GB/GPU for KV (at 0.9 utilization). KV per token under TP=8 is `2 · 80 · 1 KV-head · 128 · 2 = 40 KB/GPU` (each GPU holds 1 of 8 KV heads), block (B=16) = 640 KB/GPU. So ~84,000 blocks/GPU → ~1.34M tokens of KV → at 1.5K-token average sequences, ~890 concurrent requests' worth of KV. Plenty.

**Latency vs throughput choice:**
- **TP=8, single replica:** lowest TPOT (8-way parallel compute per step), all 8 GPUs cooperating on one model. AllReduce overhead ~5–10% (§14). Best when P99 TPOT is tight and you want maximum single-request speed.
- **TP=4 × DP=2:** two replicas of 4 GPUs each. Each replica has half the GPUs (higher per-step latency than TP=8) but less AllReduce overhead, and two independent replicas roughly double aggregate throughput. Best when throughput/$ matters and TP=4's TPOT still meets the 50 ms SLO.

Suppose benchmarking shows TP=4 yields ~45 ms TPOT (meets SLO) and TP=8 yields ~28 ms (well under SLO but "wasting" headroom). Then **TP=4 × DP=2 is the better choice**: it meets the SLO with margin while delivering more total throughput than a single TP=8 replica, lowering $/token. If instead TP=4 yielded 60 ms TPOT (violates SLO), you'd need TP=8 to meet latency, accepting lower aggregate throughput. This is the latency-headroom-vs-throughput trade (§12.2) made concrete: **use the lowest TP that meets your latency SLO, then scale throughput with DP.** With FP8 weights (70 GB total, ~8.75 GB/GPU at TP=8), even more KV room opens and TP=2 or TP=4 may meet latency, freeing GPUs for more DP replicas.

---

## 17. Vocabulary Parallelism: Embedding and LM Head Under TP

The embedding table and the LM head are large matrices (`vocab × d_model`, e.g. `128000 × 8192 ≈ 2.1 GB` in BF16) that must also be partitioned under TP. They use **vocabulary parallelism**: split the vocabulary dimension across ranks.

**Embedding (input):** each rank holds a slice of the rows of the embedding table (a subset of the vocabulary). When a token ID is looked up, only the rank owning that token's row has the embedding; the result is combined across ranks via an AllReduce (ranks that don't own the token contribute zeros). This is `VocabParallelEmbedding` in vLLM.

**LM head (output):** each rank computes logits for its slice of the vocabulary (`d_model × vocab/p`), producing partitioned logits. Sampling then needs the *full* vocabulary distribution — argmax, top-k, and top-p all require seeing all logits. So there is a cross-rank reduction: for argmax, an AllReduce of the (local max, local argmax) pairs; for top-k, a distributed top-k that combines each rank's local top-k into the global top-k; for full sampling, an AllGather of the logits (or a more efficient distributed sampling). This **sampling collective** sits on the decode critical path — small but latency-sensitive, like the per-layer AllReduces. vLLM optimizes it to avoid gathering the full 128K-wide logit vector when only top-k is needed (gathering each rank's local top-k and merging is far cheaper than all-gathering 128K logits).

The interaction with `tie_word_embeddings` (File 02 §14.3): if the embedding and LM head share weights, the vocab-parallel sharding is shared too, halving the parameter memory. The vocab-parallelism is also why very large vocabularies (multilingual 256K models) cost more in the sampling collective — more to reduce across ranks each step.

---

## 18. Sharded Weight Loading

A practical concern often underestimated: how do you *load* a 140 GB model onto 8 GPUs without a 140 GB CPU spike or loading the whole thing on each rank?

vLLM's weight loader (`vllm/model_executor/model_loader/`, File 06) supports **sharded loading**: each TP rank loads *only its shard* of each weight tensor directly from the checkpoint (safetensors supports efficient slice reads, so a rank can mmap and read just its columns/rows of `W_1` without materializing the full tensor). Under TP=8, each rank reads ~1/8 of the weights, so the per-rank load is `weights/p` and the loads happen in parallel across ranks — both faster and lower-memory than loading the full model and then sharding. For PP, each stage loads only its layers. For very large models, **streaming load** reads and shards weights incrementally to avoid CPU OOM.

This sharded loading interacts with the checkpoint format: safetensors (the modern default) supports lazy/partial reads and is preferred; legacy PyTorch `.bin` pickle files must often be fully loaded then sharded (slower, more memory). GGUF and quantized checkpoints add a dequantization-or-direct-load step (File 06). Load time matters operationally: a 70B model takes minutes to load (File 19 §readiness), and sharded parallel loading is what keeps that to minutes rather than tens of minutes, which in turn sets the cold-start and autoscaling latency (File 11 §autoscaling).

---

## 19. Quantization × Tensor Parallelism

Quantization (File 02 §11, File 06) and TP interact in ways that constrain valid configurations:

- **Per-tensor scales** (one scale per weight tensor) shard cleanly: each rank holds its weight slice and the shared scale.
- **Per-channel / per-group scales** must be sharded *consistently* with the weight partitioning — a column-parallel weight's per-output-channel scales split by column; a row-parallel weight's per-input-channel scales split by row. Getting this sharding wrong corrupts dequantization.
- **Group quantization** (e.g. GPTQ with group size 128) requires that the TP partition boundary align with group boundaries — you cannot split a quantization group across two GPUs, because the group shares a scale. This means `d_model / TP` (or the relevant dimension) must be a multiple of the group size, an extra divisibility constraint on top of head-divisibility (§2.3). A model quantized with group size 128 and `d_model 8192` allows TP up to `8192/128 = 64` from this constraint, but head-divisibility (8 KV heads) caps it at 8 anyway — usually head-divisibility binds first, but for some shapes the group constraint matters.
- **FP8** (per-tensor or per-channel) shards cleanly and is the friendliest for TP, another reason it's the preferred low-precision path on H100 (File 02 §11.4).

The practical upshot: when deploying a quantized model under TP, verify the quantization scheme's scale granularity is compatible with the TP degree, or the engine will error at load (or, worse, silently mis-dequantize). vLLM's quantized linear layers (`GPTQLinear`, `AWQLinear`, FP8 layers — File 06) handle the scale sharding, but the divisibility constraints are real and occasionally force a lower TP degree than head-divisibility alone would allow.

---

## 20. Pipeline Micro-Batch Scheduling for Inference

Section 4.2 introduced the pipeline bubble; here is how inference schedules micro-batches to minimize it. Training uses sophisticated schedules (GPipe, 1F1B, interleaved 1F1B) that interleave forward and backward passes. **Inference has no backward pass**, so the schedule is simpler — just forward micro-batches flowing through the stages — but the bubble still exists at fill and drain.

With `P` stages and `m` micro-batches, the timeline is: micro-batch 1 enters stage 0, then stage 1 (while micro-batch 2 enters stage 0), and so on, filling the pipeline over `P−1` steps; then all stages are busy for the steady-state; then the pipeline drains over `P−1` steps as the last micro-batches exit. The bubble fraction is `(P−1)/(m + P − 1)`. To make this small, `m` must be `≫ P` — but `m` is bounded by the available concurrency (you need `m` independent micro-batches' worth of requests in flight). For continuous-batching inference, the scheduler (File 04 §36) supplies micro-batches from the running set; if there aren't enough concurrent requests, the pipeline can't fill and PP efficiency suffers. This is the fundamental tension of inference PP: it needs *concurrency* to be efficient, but concurrency also raises latency (larger effective batch). For decode specifically, where each micro-batch is tiny (one token per sequence), keeping the pipeline full requires many concurrent sequences — which is usually available at high load but not at low load, making PP's efficiency load-dependent. This load-dependence, plus the inherent bubble latency, is why PP is a *capacity* tool (use it when you must span more GPUs than TP allows) rather than a latency tool (TP is better for latency when it fits).

### 20.1 Why TP is preferred over PP within a node

Given a single 8-GPU node, you would essentially always choose TP=8 over PP=8. TP=8 parallelizes every layer's compute (lower latency, no bubble) at the cost of NVLink AllReduces — which NVLink makes cheap (§14). PP=8 would have a `7/(m+7)` bubble and 8 pipeline stages of point-to-point hops, with no latency benefit (each token still traverses all layers sequentially). PP only earns its place when crossing nodes (where TP's AllReduces would be too expensive over IB) or when a model needs more GPUs than a single NVLink domain provides. This is why the standard recipe (§8.2) is "TP within node, PP across nodes" — each axis where its communication is affordable.

---

## 21. EP Communication-Computation Overlap (DualPipe and Beyond)

Expert parallelism's all-to-all (§5.2) can dominate MoE-layer latency, so hiding it behind computation is essential at scale. DeepSeek-V3's **DualPipe** is the reference technique.

The idea: while the dispatch all-to-all for one chunk of tokens is in flight (moving tokens to their expert GPUs over the network), the GPU computes the expert FFN for a *previous* chunk whose tokens have already arrived. By pipelining dispatch → compute → combine across chunks (and overlapping the forward and the auxiliary computations), the communication time is hidden behind the expert GEMM time rather than added to it. DeepSeek reports that with DualPipe the all-to-all is almost entirely overlapped, so the MoE layer's wall-clock time approaches the compute time alone — turning a communication-bound layer into a compute-bound one. This requires careful scheduling of CUDA streams (one stream for the all-to-all/NCCL, another for the expert GEMM) and enough chunks to pipeline. It also requires the interconnect bandwidth to be *sufficient* to complete each chunk's communication within the time the GPU spends computing the previous chunk — if communication is slower than computation, overlap can hide only part of it and EP becomes communication-bound again (§5.2's bandwidth sensitivity).

vLLM and SGLang implement MoE communication-computation overlap to varying degrees; the general principle — use separate CUDA streams to overlap the EP all-to-all with the expert compute — is the MoE analog of the prefetch/overlap techniques used throughout the stack (the decode kernel's block-table prefetch, File 03 §16.5; async scheduling, File 04 §11.3). Overlap is the recurring answer to "communication is on the critical path": move it off the critical path by doing useful compute during it.

---

## 22. Fault Tolerance in Distributed Inference

Distributing across GPUs and nodes multiplies failure modes, and inference has weaker fault-tolerance than training (no checkpointing to resume from — in-flight requests are lost on failure).

- **GPU/worker failure:** if one TP rank dies, the whole model replica is down (TP is all-or-nothing — every rank holds a shard of every layer). The replica must be restarted (reload weights, ~minutes), and its in-flight requests are lost. DP provides resilience at the *replica* level: with multiple replicas behind a load balancer, one replica's failure removes capacity but the others keep serving (the load balancer health-checks and routes around the dead replica, File 19).
- **Network partition (multi-node):** an IB link failure breaks the collectives; NCCL operations hang or error. Detection and restart of the affected replica is the typical recovery.
- **Graceful degradation:** the system should shed load (return 503) rather than queue indefinitely when capacity drops (File 04 §18, File 19 §circuit breaking).
- **The reliability strategy** is therefore: model parallelism (TP/PP/EP) *within* a replica for capacity/latency, and data parallelism (multiple replicas) *across* for both throughput and fault tolerance. A single giant TP=64 replica is fragile (any of 64 GPUs failing kills it); several smaller replicas degrade gracefully. This is another argument, alongside throughput (§12.2), for preferring more DP and less model parallelism when the model fits — DP is the axis that provides redundancy. File 19 covers the operational side (health checks, draining, rolling updates) in depth.

---

## 23. The Four Axes Compared

| Axis | What it splits | Communication | Interconnect | Reduces | Cost | Use when |
|---|---|---|---|---|---|---|
| **TP** | weight matrices (by head/by matrix) | AllReduce, 2×/layer, small, frequent | NVLink (latency-bound) | model mem + latency | AllReduce overhead, sublinear scaling | within a node; latency-critical; model doesn't fit |
| **PP** | layers (by depth) | point-to-point activations, at stage boundaries, infrequent | PCIe/IB tolerable | model mem | pipeline bubble; needs concurrency | across nodes; TP exhausted; capacity |
| **EP** | experts (MoE) | all-to-all, 2×/MoE-layer, large, all-pairs | fat IB + NVLink (bandwidth-bound) | expert mem | all-to-all volume; load imbalance | MoE models at scale |
| **SP** | sequence/activations | all-to-all (Ulysses) or ring (ring-attn) | depends | activation mem | extra all-to-all | very long context |
| **DP** | nothing (replicate) | none (during inference) | none | nothing (scales throughput) | none | throughput; fault tolerance; model fits |

The table encodes the whole topology-selection logic (§12): DP is free but only scales throughput; TP cuts latency but taxes communication and needs NVLink; PP and EP cross node boundaries with their respective communication patterns; SP is the long-context specialist. A real deployment is a product `DP × PP × EP × TP` (× SP for long context), with each factor placed on the interconnect tier its communication can afford.

---

## 24. The Prefill/Decode Communication Asymmetry

A subtlety that connects distribution back to the prefill/decode duality (File 01 §3): the *same* parallelism has different communication costs in the two phases.

- **Prefill:** processes many tokens at once, so the AllReduce volume (`batch · seq_len · d_model · bytes`) is large — but prefill is compute-bound, the communication is amortized over heavy GEMM work, and the AllReduces happen for a whole prompt's worth of tokens in one shot. TP communication is a small fraction of prefill time.
- **Decode:** processes one token per sequence, so the AllReduce volume is tiny (16 KB, §14) — but the AllReduce is *latency-bound* and sits directly on the critical path of a memory-bound step with little compute to hide it behind. TP communication is a larger *fraction* of the (shorter) decode step.

So TP's overhead is felt most in decode, the latency-critical phase. This asymmetry reinforces disaggregation's appeal (File 04 §10, File 15): a prefill pool can use aggressive TP (communication amortized over compute) while a decode pool might use lower TP plus more DP (minimizing the latency-bound decode AllReduces) — each phase parallelized for its own communication profile. It also explains why batching helps TP efficiency specifically in decode: larger decode batches give the (still latency-bound) AllReduces more compute to hide behind, raising the compute-to-communication ratio.

---

## 25. vLLM's Distributed Implementation

vLLM's distributed machinery (`vllm/distributed/`) sets up the process groups and device mesh that the parallelism axes use:

- **Process groups:** separate `torch.distributed` groups for TP, PP, and (for MoE) EP. A rank belongs to multiple groups simultaneously — e.g. its TP group (the GPUs sharing its layer's matmuls) and its PP group (the stages of its pipeline). Collectives are issued on the appropriate group (TP AllReduce on the TP group, etc.).
- **Device mesh / coordinator:** a `GroupCoordinator` abstraction manages the groups and provides the collective operations (`all_reduce`, `all_gather`, etc.) routed to the right group, with the custom-all-reduce fast path (§3.4, §10.2) for small TP AllReduces.
- **Worker processes:** one process per GPU (`vllm/worker/worker.py`), each running the model's sharded layers. The driver process runs the scheduler and broadcasts the batch plan (File 04 §22) to all workers, which execute in lockstep and apply identical block-table mutations (§9.3).
- **Backend:** NCCL for CUDA, RCCL for ROCm, with `gloo` for CPU-side coordination. Multi-node uses Ray (§26) or a `torchrun` launch to start workers across hosts and wire up the IB fabric.

The clean separation — scheduler/driver decides, workers execute the broadcast plan on their shards — is what makes the same scheduling logic (File 04) work transparently from single-GPU to multi-node: the workers are interchangeable executors of a centrally-decided plan, differing only in which shard of weights and KV they hold.

---

## 26. Ray Orchestration for Multi-Node

vLLM uses **Ray** for multi-node deployments. Ray provides cluster management, placement groups (to pin workers to specific GPUs across hosts), and an actor model for the workers. The flow:

1. A Ray cluster spans the nodes (head node + workers).
2. vLLM creates a placement group reserving the GPUs across nodes per the TP×PP layout.
3. Worker actors are launched on the placed GPUs; each initializes its NCCL communicators for its TP/PP/EP groups over the IB fabric.
4. The driver coordinates the scheduler and broadcasts plans to the worker actors.

Ray handles the cross-node process launch, failure detection (a dead actor is detectable), and resource accounting. The alternative for single-node or simple multi-node is a `torchrun`/`mp.spawn` launch without Ray. For production multi-node, Ray's placement and management are valuable, though they add a dependency and some overhead. The critical configuration is the NCCL/IB setup (GPUDirect RDMA, correct NCCL environment variables for the IB HCAs) — Ray places the workers, but the fabric performance depends on NCCL finding and using the IB path, which must be verified (§15's NCCL-fallback warning).

---

## 27. Common Distributed Pitfalls

A field guide for distributed inference problems:

- **Poor TP scaling (efficiency < 80% at TP=8):** NVLink not detected/used (check `nvidia-smi topo -m` — want `NV#` links, not `PHB`/`SYS`), NCCL falling back to PCIe/sockets (enable `NCCL_DEBUG=INFO`), or custom all-reduce disabled. The 160-AllReduces-per-step decode path is unforgiving of slow interconnect (§14).
- **TP degree rejected at startup:** head-divisibility (§2.3) or quantization-group-divisibility (§19) violated. Choose a TP degree dividing both head counts and aligning with quantization groups.
- **Multi-node much slower than single-node:** TP accidentally spanning nodes (it must not — §8.3, §11.2), or IB not configured (NCCL on TCP). Verify the topology places TP within nodes and PP/EP across.
- **PP efficiency poor at low load:** not enough concurrent requests to fill the pipeline (§20) — the bubble dominates. Either raise load, reduce PP degree, or accept that PP needs concurrency.
- **MoE straggler / OOM on one GPU:** expert load imbalance (§5.5) overloading a hot expert's GPU. Mitigate with capacity factors or expert replication.
- **OOM after switching to higher TP:** counterintuitively, higher TP can raise *per-GPU* peak activation/workspace memory for some operations even as it lowers weight memory; re-profile and adjust `--gpu-memory-utilization`.
- **Inconsistent outputs across identical replicas:** a sharding or seed bug; under correct TP, all ranks of a replica produce bit-identical results for the shared computation (the AllReduce makes them agree), and replicas with the same seed should match.

The meta-lesson: most distributed inference problems are *interconnect placement* problems (wrong axis on wrong tier) or *divisibility constraint* problems (TP degree incompatible with heads/quantization), both diagnosable from the topology and the model config before deployment.

---

## 28. Synthesis and Key Takeaways

1. **Three reasons to distribute** — capacity (model doesn't fit), latency (parallelize compute), throughput (replicate) — map to different axes; pick the axis for the reason.
2. **TP** splits weight matrices via the Megatron column-then-row pattern (§13), needing one AllReduce per sublayer; it cuts latency and memory but its frequent small AllReduces are latency-bound and demand NVLink. Head-divisibility constrains the degree.
3. **PP** splits layers with infrequent point-to-point communication, tolerating slower interconnects (cross-node), at the cost of a pipeline bubble that needs concurrency to amortize.
4. **EP** splits MoE experts with bandwidth-bound all-to-all; overlap it with compute (DualPipe) and watch load balance. **SP** splits the sequence for long-context activation memory.
5. **DP** replicates for throughput and fault tolerance with zero inference communication — prefer it when the model fits and latency is met.
6. **Match each axis to its interconnect tier** (§8): TP on NVLink within a node, PP/EP across nodes, DP outermost. This single principle drives correct topology.
7. **KV cache shards by head (TP) and by layer (PP)**, with allocation decisions broadcast from the driver to keep ranks in lockstep — divergence means silent corruption.
8. **The lowest model parallelism that meets capacity and latency, scaled out with DP**, is usually the best throughput/$ and the most fault-tolerant — quantization (FP8) often reduces the required parallelism, freeing GPUs for more DP.

Distributed inference is where the architecture (File 02), the memory manager (File 03), and the scheduler (File 04) extend across many GPUs, and where communication — not just compute and memory — becomes a first-class resource to budget. The recurring discipline is to keep communication off the critical path: place it on the fastest interconnect, minimize its frequency and volume, and overlap it with computation wherever possible. File 10 dives into the kernels that run on each GPU; File 11 turns the topology choices here into a measured tuning methodology; File 15 explores disaggregation, which reorganizes the parallelism around the prefill/decode asymmetry (§24) for further gains.

---

## 29. Worked MoE Topologies: Mixtral and DeepSeek-V3

### 29.1 Mixtral-8×7B

Mixtral has 8 experts, top-2 routing, ~47B total / ~13B active parameters, GQA attention. It fits comfortably on a single 8-GPU node. Two reasonable topologies:

- **Pure TP=8:** attention and the experts all tensor-parallel. Each GPU holds 1/8 of every expert's weights and 1/8 of attention. The MoE computation is then local (no all-to-all) but each expert's matmul is split across 8 GPUs with AllReduce — works because all experts live (sharded) on all GPUs. Simple, NVLink handles the AllReduces. This is the common Mixtral deployment.
- **TP=4 × EP=2 or DP:** alternatives that trade communication patterns. For Mixtral on one node, pure TP=8 is usually simplest and efficient because the model fits and NVLink makes the AllReduces cheap; EP shines when experts are too numerous/large to replicate-sharded across the TP group, which is DeepSeek's regime, not Mixtral's.

The lesson: at modest expert counts that fit on a node, TP (sharding each expert across GPUs) is simpler than EP (placing whole experts on different GPUs); EP becomes necessary when the expert *count* is large enough that you want whole experts on dedicated GPUs to avoid sharding overhead.

### 29.2 DeepSeek-V3

DeepSeek-V3: 256 routed experts + shared experts, top-8 routing, 671B total / 37B active, MLA attention (File 02 §2.4, File 15), `d_model 7168`, FP8. This does **not** fit on one node and has too many experts to shard each across a TP group efficiently. Its deployment:

- **MLA attention:** the low-rank KV makes the attention KV cache tiny (File 02 §26's ~64× reduction), so the attention is relatively cheap and is handled with TP (and DP-attention variants) within nodes.
- **EP128 for experts:** 256 experts across 128 GPUs (2 experts/GPU), with top-8 dispatch over the all-to-all. The all-to-all spans nodes over IB+NVLink, with DualPipe overlap (§21) hiding the communication behind expert compute.
- **FP8 throughout:** halves weight and activation bytes, easing both memory and the all-to-all volume.
- **MTP heads** (File 02 §22) for built-in speculative decoding, raising decode tokens-per-step.

The all-to-all volume per dispatch (File 02 §15.4): batch 1024, `d_model 7168`, FP8 (1 byte), top-8 — each token's hidden state (7168 bytes in FP8) dispatched to 8 experts → ~`1024 · 7168 · 8 ≈ 59 MB` total dispatch, spread across the 128 GPUs and overlapped with compute. DeepSeek's open-source inference code is the reference for making EP128 efficient: without the overlap, the all-to-all would dominate; with it, the MoE layer approaches compute-bound. This is the canonical example of every principle in this file — TP within nodes for attention, EP across nodes for experts, fat IB for the bandwidth-bound all-to-all, communication-computation overlap to hide it, FP8 to shrink everything, all matched to interconnect tiers.

---

## 30. CUDA Graphs and TP: Capturing Collectives

CUDA graphs (File 09) capture a forward pass to eliminate per-kernel launch overhead — critical for the decode path where launch overhead is significant relative to the short step. Under TP, the captured graph must include the **NCCL AllReduce** operations, which raises a subtlety: NCCL collectives can be captured into a CUDA graph only with CUDA-graph-compatible NCCL (supported in CUDA 11.3+), and all participating GPUs must capture *simultaneously* with their communicators initialized before capture. vLLM and SGLang capture decode-path graphs that include the TP AllReduces, so a replayed graph performs the matmuls *and* the collectives with no per-kernel launch overhead — a substantial win for TP decode, where the 160 AllReduces per step would otherwise each incur launch overhead. This is why TP and CUDA graphs together are especially valuable: graphs amortize the launch cost of exactly the many small kernels (including AllReduces) that TP decode is full of. The requirement that shapes be static (File 04 §31) means graphs are captured per decode batch size, and the AllReduce in the graph is for that fixed batch's tensor size; the custom all-reduce (§3.4) is also graph-capturable. Getting NCCL into the graph correctly (initialization order, simultaneous capture across ranks) is fiddly, which is part of why these engines pre-capture graphs at startup rather than on the fly.

---

## 31. vLLM vs SGLang: Distributed Differences

Both engines support TP, PP, and (for MoE) EP, with broadly similar mechanisms (NCCL collectives, Megatron-style sharding, per-GPU workers). Differences (full comparison in File 17):

- **Process model:** SGLang separates TokenizerManager / Scheduler / ModelExecutor into distinct processes communicating via ZeroMQ from the outset (File 09), with the model executors being the per-TP-rank workers. vLLM's V1 engine converged toward a similar multi-process separation (File 04 §14). Both aim to keep tokenization/scheduling off the GPU workers' critical path.
- **DP-attention for MoE:** SGLang's `--enable-dp-attention` (§7.3) replicates attention across ranks while expert-parallelizing the FFN — a MoE-specific optimization SGLang surfaced prominently, valuable when attention is cheap relative to experts.
- **Overlap scheduling:** SGLang's overlap scheduler (File 09) and vLLM's V1 async scheduling both overlap CPU scheduling with GPU execution, reducing the per-step overhead that compounds with TP's many kernels.
- **Cutting-edge MoE/long-context:** SGLang has often shipped aggressive distributed features (DP attention, certain EP optimizations) quickly; vLLM offers very broad model and hardware coverage. For DeepSeek-scale EP and MLA, both have invested heavily and track the reference implementations.

The convergence is strong: both place TP on NVLink within nodes, use NCCL/RCCL, support EP for MoE, and overlap scheduling with execution. The distributed-systems fundamentals (interconnect tiers, Megatron sharding, collective cost models) are identical across them — what differs is the process architecture and which optimizations landed first, not the underlying parallelism theory this file develops.

---

## 32. Appendix: Distributed Configuration Reference and Source Pointers

Key flags:
- **`--tensor-parallel-size`** (TP degree; must divide head counts and quantization groups).
- **`--pipeline-parallel-size`** (PP degree; for cross-node or TP-exhausted capacity).
- **`--data-parallel-size`** (DP replicas; throughput + fault tolerance).
- **`--enable-expert-parallel`** (EP for MoE).
- **`--distributed-executor-backend`** (`ray` | `mp`; Ray for multi-node).
- **`--disable-custom-all-reduce`** (fall back to NCCL for the small decode AllReduces; usually leave the custom path on).
- **`--gpu-memory-utilization`** (re-profile when changing parallelism — peak workspace shifts).

Source pointers: `vllm/distributed/` (process groups, `GroupCoordinator`, custom all-reduce), `vllm/model_executor/layers/linear.py` (Column/RowParallelLinear), `vllm/model_executor/layers/vocab_parallel_embedding.py` (vocab parallelism), `vllm/worker/worker.py` (per-rank worker), `vllm/executor/` (Ray/mp executors). The Megatron-LM paper (arXiv 1909.08053) is the foundational reference for TP; DeepSpeed-Ulysses (arXiv 2309.14509) and Ring Attention (arXiv 2310.01889) for SP; the DeepSeek-V3 technical report for production EP128 with MLA and DualPipe.

To trace distributed execution: the scheduler (File 04, on the driver) produces a plan → broadcast to workers → each worker's `execute_model` runs its sharded layers, issuing TP AllReduces (custom or NCCL) at each sublayer boundary, PP send/recv at stage boundaries, and EP all-to-all at MoE layers → results gathered (vocab-parallel logits, §17) → sampled → returned. Every collective in that trace has a cost model in §3 and a correct interconnect placement in §8 — which, taken together, is the discipline of distributed LLM inference: knowing what must be communicated, how much, how often, and over which wire, and arranging the topology so that none of it lands on the critical path.

---

## 33. The Logit Reduction Cost, Worked

To quantify the sampling collective (§17), consider TP=8 on a 128K-vocabulary model. The full logit vector for one decoding sequence is `128000 × 2 bytes = 256 KB`, sharded so each rank computes `16000` logits (`32 KB`).

- **Naive (AllGather full logits):** gather all 8 shards → every rank holds 256 KB of logits. Volume `7/8 · 256 KB ≈ 224 KB` per rank, per sequence, per step. For a batch of 256 decoding sequences: `256 · 224 KB ≈ 57 MB` AllGather per step — non-trivial, and growing with batch and vocabulary.
- **Optimized (distributed top-k):** if sampling only needs top-k (e.g. k=50), each rank computes its local top-50, and the ranks exchange only `8 · 50 = 400` (logit, index) pairs per sequence (a few KB), merged to the global top-50. Volume drops by orders of magnitude — `256 · ~6 KB ≈ 1.5 MB` per step instead of 57 MB.

vLLM and SGLang use the optimized path where the sampling parameters permit (top-k/top-p with bounded candidate sets), reserving the full gather for cases that genuinely need the complete distribution (e.g. returning full `logprobs` over the vocabulary). This is another instance of minimizing communication volume on the decode critical path — the same instinct as GQA reducing KV reads or the custom all-reduce shrinking AllReduce latency. The vocabulary size directly scales this cost, which is part of why very-large-vocabulary multilingual models are marginally more expensive to serve under TP: more logits to reduce each step.

---

## 34. Communication-Computation Overlap: The Unifying Principle

Stepping back, a single principle recurs across every axis and indeed across the whole database: **when communication (or memory access) is on the critical path, hide it behind computation.** The instances:

- **TP:** the custom all-reduce fuses the residual add into the AllReduce (§10.2), and CUDA graphs capture the collectives to remove launch overhead (§30) — minimizing the AllReduce's critical-path cost.
- **PP:** micro-batching overlaps stages so the pipeline stays full, hiding the inter-stage transfer behind other micro-batches' compute (§20).
- **EP:** DualPipe overlaps the all-to-all with expert compute (§21), turning a communication-bound MoE layer into a compute-bound one.
- **SP:** ring attention overlaps the K/V ring-passing with local attention compute (§6.2).
- **Scheduling:** async/overlap scheduling plans step `t+1` during step `t`'s GPU execution (File 04 §11.3).
- **Memory:** the decode kernel prefetches the next block's address while computing on the current block (File 03 §16.5); KV swap overlaps PCIe transfer with compute via a separate stream (File 03 §8).

The mechanism is always the same: separate CUDA streams (or a pipeline structure) so that a transfer in flight on one stream is concurrent with useful compute on another, and the wall-clock time becomes `max(compute, communication)` rather than `compute + communication`. Distributed inference is, in large part, the engineering of this overlap — because the parallelism that provides capacity and latency *introduces* communication, and the only way to keep that communication from negating the benefit is to hide it. An inference engineer evaluating a distributed configuration should always ask: *is the communication overlapped, or is it serialized on the critical path?* The answer usually explains the measured scaling efficiency.

---

## 35. Closing

Distribution is what lets LLM inference scale beyond a single GPU's memory and compute — and what introduces communication as a resource to manage alongside memory (File 03) and GPU time (File 04). The four model-parallelism axes (TP, PP, EP, SP) each split a different part of the model with a different communication pattern, and data parallelism replicates for throughput and resilience. The governing discipline is to match each axis to the interconnect tier its communication can afford — latency-bound TP on NVLink, infrequent PP and bandwidth-bound EP across nodes, communication-free DP as the outer scale-out — and to overlap whatever communication remains with computation so it never serializes the critical path. The Megatron column-then-row pattern, the ring-AllReduce cost model, the head- and quantization-divisibility constraints, and the prefill/decode communication asymmetry are the load-bearing details; the worked examples (70B on 8×H100, Mixtral, DeepSeek-V3 EP128) show how they combine into real topologies. With the execution substrate now spanning many GPUs, File 06 returns to what runs *on* each GPU — model execution, weight loading, quantization kernels, and the per-step batch construction — and File 10 to the attention and compute kernels that make each GPU fast. The distributed runtime is the skeleton; those kernels are the muscle, and the scheduler of File 04 is the nervous system directing both.

---

## 36. Appendix: Heterogeneous and Disaggregated Topologies

A frontier worth flagging (developed in File 15): the parallelism axes need not use identical hardware. Disaggregation (File 04 §10) separates prefill and decode pools, and those pools can be *differently parallelized on different GPUs*:

- **Prefill pool:** compute-bound, benefits from high FLOP/s. Use H100/B200 with aggressive TP (communication amortized over heavy compute, §24). Large batches of prompts.
- **Decode pool:** memory-bandwidth-bound, benefits from high HBM bandwidth and capacity. Use high-bandwidth GPUs (MI300X's 5.3 TB/s, H200's 4.8 TB/s) or more numerous cheaper GPUs, with lower TP and more DP to minimize the latency-bound decode AllReduces (§24).

This heterogeneous disaggregation can cut cost 2–3× at equal throughput (File 15 §disaggregation rationale) by spending expensive compute-optimized GPUs only on prefill and bandwidth-optimized (or cheaper) GPUs on the decode that dominates token generation. The KV transfer between pools (File 03 §21) uses the same movable-block design that underlies swapping. It is the logical endpoint of this file's theme: not just splitting *one* model across homogeneous GPUs, but architecting the *whole serving system* so that each phase, each axis, and each piece of communication runs on the hardware and interconnect best suited to it. The parallelism axes are the vocabulary; disaggregation and heterogeneous placement are the grammar for composing them into cost-optimal systems.

### 36.1 A note on the future

As models grow (trillion-parameter MoE) and contexts lengthen (1M+ tokens), the distributed problem intensifies: more EP for more experts, more SP for longer contexts, and more sophisticated overlap to hide the resulting communication. At the same time, faster interconnects (NVLink 5.0 at 1.8 TB/s, larger NVLink domains spanning more GPUs) raise the ceiling on how far latency-bound TP can scale before crossing into slower tiers. The cost models in this file are stable — Megatron sharding, ring AllReduce, all-to-all volume, the interconnect hierarchy — but the *numbers* shift with each hardware generation (File 16), moving the optimal topology. An inference engineer who holds the cost models, not just a memorized recipe, can re-derive the right topology for each new model and each new GPU — which is the durable skill this file aims to build, just as Files 01–04 built the durable cost models for memory, compute, and scheduling. Together, Files 01 through 05 constitute the systems foundation; Files 06 onward specialize it into kernels, the SGLang counterpart, performance tuning, and the application and operational layers.

To restate the one idea to carry forward: distribution buys capacity, latency, and throughput, but it is paid for in communication, and the entire craft is keeping that payment small — minimize volume (GQA, MLA, top-k logit reduction), minimize frequency (defer to one AllReduce per sublayer), place it on the fastest affordable wire (TP on NVLink, EP across fat IB), and overlap whatever remains with compute. Do that, and many GPUs behave almost like one big one; neglect it, and they behave like many slow ones connected by a straw. That distinction — many GPUs as one fast machine versus many slow ones throttled by their interconnect — is decided entirely by how well the communication is minimized, placed, and overlapped, and it is the single most consequential outcome of getting distributed inference right.






