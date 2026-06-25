# vLLM Model Execution — Quantization, Weight Loading, and Model Support

> **Standard reference file.** This file covers the model-execution layer of vLLM: the forward-pass pipeline from scheduler output to sampled tokens, the attention metadata that addresses the paged KV cache, the quantization methods and their kernels, weight loading, the model architecture registry, MoE execution, speculative decoding integration, and vision-language model support. Prerequisites: File 02 (architecture), File 03 (paged KV), File 04 (scheduler), File 05 (distributed).

---

## 1. The Model Execution Pipeline

A single engine step flows through a well-defined call chain:

```
LLMEngine.step()
  → Scheduler.schedule()              # decide the batch (File 04)
  → Worker.execute_model()            # per-GPU worker (File 05)
    → ModelRunner.execute_model()     # build batch tensors, run forward
      → model.forward()               # the transformer forward pass
      → Sampler                        # apply sampling params, draw tokens
  → process outputs, detokenize, stream (File 07)
```

The **`ModelRunner`** is the linchpin between the scheduler's abstract plan and the GPU. It receives the `SchedulerOutputs` (File 04 §22) and constructs the concrete input tensors the model consumes: the packed `input_ids`, the `positions` (continuing from each sequence's `num_computed_tokens`, File 02 §19), the `slot_mapping` (where to write each token's K/V, File 03 §3.2), the per-sequence block tables, and the attention metadata. It then calls `model.forward()`, gets logits, and invokes the sampler. On subsequent steps it does this again with the updated batch — the relentless per-step heartbeat (File 04 §38).

### 1.1 ModelRunner responsibilities

- **Batch preparation:** gather scattered per-sequence state into packed, GPU-ready tensors with the ragged-batch indices (`cu_seqlens`/`seq_lens`) for varlen attention (File 02 §20).
- **CUDA graph management:** for decode batches of captured sizes, replay a pre-captured CUDA graph instead of eager execution (File 09 §CUDA graphs), padding the batch to the nearest captured size.
- **Attention metadata construction:** build the `AttentionMetadata` the kernel needs to address the paged cache correctly for the mixed prefill+decode batch.
- **Sampling invocation:** pass logits and per-sequence sampling params to the sampler.

---

## 2. Attention Metadata

The attention kernel (File 03 §4, File 10) must correctly address a *paged*, *variable-length*, *mixed-phase* batch. The `AttentionMetadata` dataclass carries everything it needs:

- **`num_prefills`, `num_prefill_tokens`, `num_decode_tokens`:** the split between prefill and decode work in this batch, so the kernel (or the two kernels) can handle each phase. Prefill sequences contribute many tokens; decode sequences contribute one each.
- **`slot_mapping`:** for each token in the batch, the absolute physical KV slot (`physical_block × block_size + offset`) to write its K/V into. This is how new K/V lands in the right place in the paged pool (File 03 §3.2).
- **`seq_lens` / `context_lens`:** the KV length each sequence attends over (its full cached context).
- **`block_tables`:** per-sequence logical→physical block maps, so the kernel can gather cached K/V from scattered physical blocks.
- **`query_start_loc` / `cu_seqlens`:** the cumulative query-length index marking each sequence's span in the packed buffer — query length `n` for a prefilling sequence's chunk, 1 for a decoding sequence.
- **Max sequence lengths** and other scalars for kernel launch configuration.

This metadata is the contract between the scheduler/runner and the attention backend. Building it correctly — especially the `slot_mapping` and `block_tables` for a batch mixing fresh prefills, cache-hit prefills, and ongoing decodes — is exactly where the position/cache-consistency bugs of File 02 §19 and File 03 §35 live. Different attention backends (FlashAttention, FlashInfer, Triton) consume slightly different metadata layouts, so vLLM has per-backend metadata builders.

---

## 3. Quantization Support

vLLM supports a broad set of quantization methods (foundations in File 02 §11), each implemented as a custom linear layer that replaces `nn.Linear` and a matching matmul kernel. The `--quantization` flag (or auto-detection from the checkpoint's quantization config) selects the method.

### 3.1 GPTQ (W4A16)

`GPTQLinear` stores 4-bit quantized weights plus per-group scales and zero-points. At inference, the weights are dequantized and multiplied by FP16 activations. Two kernel paths:
- **Marlin kernel** (A100/H100): a highly optimized INT4×FP16 matmul that interleaves dequantization with the GEMM, achieving near-FP16 throughput at 4-bit storage (File 02 §11.2). Marlin is the reason GPTQ W4A16 is fast on modern GPUs.
- **Generic CUDA kernel:** a fallback for older GPUs without Marlin support, slower but correct.

`--quantization gptq`. The group size (e.g. 128) and the TP-divisibility constraint it imposes (File 05 §19) come from the checkpoint's quantization config.

### 3.2 AWQ (W4A16)

`AWQLinear` uses the AWQ algorithm (activation-aware scaling, File 02 §11.2) with its own INT4 matmul kernel. AWQ often matches or beats GPTQ accuracy at the same bit-width because it protects salient (high-activation) channels. `--quantization awq`. Like GPTQ, it's weight-only — activations stay FP16, so it helps memory-bound decode and total footprint, not compute-bound prefill (File 02 §FAQ).

### 3.3 FP8 (W8A8 and weight-only)

On H100/Ada/Blackwell, vLLM supports native **FP8** (E4M3/E5M2). `--quantization fp8`. FP8 W8A8 uses the FP8 tensor cores (~2× BF16 TFLOP/s) for both prefill and decode, with per-tensor or per-channel scaling (static calibrated or dynamic). FP8 KV cache (`--kv-cache-dtype fp8_e5m2`) is a separate, composable knob (File 02 §7, File 13). FP8 is the friendliest quant for TP (clean scale sharding, File 05 §19) and the preferred low-precision path on Hopper. Benchmark: LLaMA-3 70B FP8 ≈ 1.8× BF16 throughput on H100.

### 3.4 INT8 W8A8

SmoothQuant-based INT8 (File 02 §11.3): `--quantization` with an INT8 method, requiring a calibration dataset to compute the activation scaling factors that migrate outlier difficulty to weights. Enables INT8 tensor-core matmul (2× FP16 on A100, which lacks FP8). Per-tensor/per-channel/per-token activation quantization trade accuracy for speed.

### 3.5 Other methods and GGUF

- **SqueezeLLM:** non-uniform (sensitivity-based) quantization with sparse outlier storage.
- **GGUF loading:** vLLM can load community GGUF checkpoints (llama.cpp's format, File 02 §11.5) by dequantizing to FP16 for GPU execution — convenient for models only distributed as GGUF, though it forgoes GGUF's native low-bit compute (which is CPU-oriented anyway).
- **bitsandbytes** (NF4/INT8): supported for QLoRA-style 4-bit base models (File 18 §QLoRA).

### 3.6 Choosing a method (recap)

Per File 02 §11.6: memory-bound large-model decode → W4A16 (AWQ/GPTQ) or FP8 weights; compute-bound prefill on H100 → FP8 W8A8; KV-bound long context → FP8 KV cache on top. Always validate task accuracy — quantization error is data-dependent.

---

## 4. Weight Loading

The weight loader (`vllm/model_executor/model_loader/`) handles the model's parameters from checkpoint to sharded GPU tensors.

- **Formats:** HuggingFace **safetensors** (preferred — supports lazy/partial reads for efficient sharded loading, File 05 §18), legacy PyTorch `.bin` (pickle, must be fully loaded then sharded), and GGUF (with dequantization).
- **Sharded loading for TP/PP:** each rank loads only its shard of each weight (File 05 §18) — under TP, its column/row slice; under PP, its layers. This keeps per-rank load time and memory to `weights / num_ranks`.
- **Streaming load:** for very large models, weights are read and placed incrementally to avoid a CPU memory spike that would OOM the host.
- **Auto device mapping:** distributes weights across the available GPUs per the TP×PP layout.
- **Quantized checkpoints:** the loader reads the quantization config, instantiates the right quantized linear layers (§3), and loads the quantized weights plus scales/zeros, sharding the scales consistently with the weight partitioning (File 05 §19).

Load time is operationally significant: a 70B model in BF16 is 140 GB, which at PCIe/NVMe read speeds takes minutes; sharded parallel loading across ranks is what keeps cold start to minutes rather than tens of minutes (File 19 §readiness, File 11 §autoscaling).

---

## 5. The Model Architecture Registry

vLLM supports each model family via a module in `vllm/model_executor/models/` that implements the architecture using vLLM's parallel layers (Column/RowParallelLinear, VocabParallelEmbedding, the paged attention, FusedMoE, etc.).

- **Autodetection:** the architecture is read from the checkpoint's `config.json` `architectures` field (e.g. `LlamaForCausalLM`), and vLLM dispatches to the matching model module.
- **Required interface:** each model implements `forward()` (the transformer forward pass, wired to use the paged attention and the parallel layers) and `load_weights()` (mapping checkpoint parameter names to the model's sharded parameters, handling fused-QKV packing, gate/up fusion, etc.).
- **Supported families** (a non-exhaustive list): LLaMA (1/2/3), Mistral/Mixtral, Qwen (1/1.5/2/2.5, MoE variants), Phi, Gemma (1/2), Falcon, GPT-2, GPT-NeoX, OPT, BLOOM, Baichuan, ChatGLM, DeepSeek (V2/V3 with MLA), Command-R/Cohere, StableLM, MPT, and many more, plus vision-language models (§8). The breadth of model support is one of vLLM's defining strengths versus engines requiring per-model porting (TensorRT-LLM, File 17).

Adding a new model means writing this module — implementing the architecture with vLLM's parallel/paged primitives and the weight-name mapping. The primitives (parallel layers, paged attention, fused kernels) do the heavy lifting, so a new model that's architecturally similar to an existing one (most are LLaMA-like) is often a modest addition.

---

## 6. MoE Model Execution

Mixture-of-Experts models (File 02 §4.2, §21; File 05 §5) execute through vLLM's **`FusedMoE`** layer.

- **Routing:** the router projection produces expert logits; top-k selection and gate softmax pick and weight each token's experts.
- **Fused grouped GEMM:** the `fused_moe` Triton kernel groups tokens by their selected expert, pads each group to a tensor-core-friendly multiple (e.g. 16), and runs the per-expert FFN as a grouped matmul over stacked expert weights `[num_experts, d_ff, d_model]` and `[num_experts, d_model, d_ff]`. It supports quantized expert weights (GPTQ/AWQ/FP8) by dequantizing per group inside the kernel.
- **Scatter/combine:** processed tokens are scattered back to their positions and combined with the gate weights.
- **Expert parallelism:** `--enable-expert-parallel` distributes experts across GPUs with the dispatch/combine all-to-all (File 05 §5).

The padding-to-16 on imbalanced routing wastes some compute (File 02 §21.3), and the data-dependent routing makes MoE decode latency noisier than dense (File 02 §FAQ). For very large MoE (DeepSeek-V3), vLLM combines `FusedMoE` + EP with the MLA attention path (§7) and FP8.

---

## 7. MLA Execution for DeepSeek Models

DeepSeek-V2/V3's Multi-head Latent Attention (File 02 §2.4, File 15 §MLA) is implemented in `vllm/model_executor/models/deepseek_v2.py` (and V3 variant). Key execution points:

- **Latent KV cache:** the cache stores the low-rank latent `c_KV` (and the decoupled RoPE keys `K_R`), not full K/V — the ~64× KV reduction that makes DeepSeek's long context economical.
- **Weight absorption:** at load time, the up-projection matrices `W_UK`/`W_UV` can be absorbed into the query projection (`W_Q_absorbed = W_Q · W_UKᵀ`), so attention computes directly from `c_KV` without materializing full K/V — eliminating a matmul per layer at inference (File 15 §MLA optimization).
- **FlashAttention compatibility:** a common implementation materializes K/V from the latent before a standard attention kernel (losing some MLA efficiency); a native MLA attention kernel (which attends directly on the latent) is a performance frontier. The block manager must reserve space for the latent plus the RoPE keys per token (File 03 §35's FP8-scale-style consideration generalizes to MLA's latent layout).

MLA is the clearest example of inference-aware architecture (File 02 §22) requiring engine-side support: the engine must know to cache the latent, absorb the up-projections, and handle the decoupled RoPE — none of which a generic attention path provides.

---

## 8. Speculative Decoding Implementation

vLLM's speculative decoding (`vllm/spec_decode/`, full treatment File 12) wraps a draft and a target model.

- **`SpecDecodeWorker`** orchestrates: run the draft model `K` steps to propose `K` tokens (plus the logits needed), then run the target model over all `K+1` positions in a single forward pass (a mini-prefill verification).
- **Rejection sampling:** the `RejectionSampler` accepts/rejects each proposed token per the speculative-sampling rule (accept with probability `min(1, p/q)`, File 12 §1), producing a target-distributed output.
- **`BatchExpansionTop1Scorer`** and related machinery handle scoring the proposals against the target in batch.
- **Flags:** `--speculative-model` (the draft), `--num-speculative-tokens` (`K`), `--speculative-draft-tensor-parallel-size` (the draft can use a different, usually smaller, TP degree).

The scheduler integration (File 04 §19) accounts for the variable accepted-tokens-per-step and the larger per-step KV growth. Draft and target can run on different CUDA streams to overlap (File 12). EAGLE/Medusa variants (feature-level and multi-head drafts) are supported, trading separate-draft memory for in-model draft heads.

---

## 9. Vision-Language Models (VLMs)

vLLM supports multimodal models (full treatment File 14) through a multimodal input path.

- **`MultiModalInputs`** carries pixel values, pre-computed image embeddings, or features alongside the text token IDs.
- **Image token placeholders:** an `IMAGE_TOKEN_ID` marks positions in the token sequence where image features are injected. The visual encoder (ViT/CLIP/SigLIP) processes the image (on GPU, or CPU per configuration) into features, which replace the placeholder positions' embeddings before the LLM forward pass.
- **`MultiModalPlugin`** per modality handles preprocessing and feature extraction.
- **Image prefix caching:** because image features are fixed for a given image regardless of the question, the image's KV can be prefix-cached (File 03 §6, File 14) — a large saving for multi-turn visual QA on the same image. SGLang's RadixAttention (File 08) is a natural fit for this.
- **Supported VLMs:** LLaVA-1.5/NeXT, InternVL-2, Qwen-VL-2, Pixtral, LLaMA-3.2-Vision, and others. The image-token count and injection style vary by model (File 14 §VLM patterns).

---

## 10. The Sampler

After the forward pass produces logits, the **Sampler** applies the per-sequence `SamplingParams` (File 02 §8, File 07 §sampling params): penalties (presence/frequency/repetition), temperature, top-k, top-p, then draws a token with the per-sequence RNG seed. Under TP it operates on vocab-parallel logits with the sampling collective (File 05 §17, §33). It also handles:
- **`n` parallel samples** and **`best_of`** (generate more, return the best by cumulative logprob).
- **`logprobs` / `prompt_logprobs`:** returning token log-probabilities (requires the full distribution, so the more expensive logit gather, File 05 §33).
- **Structured-decoding masks:** applying the grammar matcher's allowed-token mask before sampling (File 02 §10, File 04 §29, File 08 XGrammar).
- **Stop conditions:** EOS, `max_tokens`, stop strings (with incremental detokenization, File 02 §18.3).

The sampler is a fused GPU kernel for efficiency at batch scale (File 02 §8.5), branching on per-sequence parameters.

---

## 11. Synthesis

The model-execution layer is where the scheduler's plan (File 04), the paged KV cache (File 03), the distributed sharding (File 05), and the model architecture (File 02) all converge into a concrete forward pass. The `ModelRunner` builds the batch tensors and attention metadata; the model's parallel/paged/quantized layers run the forward pass; the sampler draws tokens. Quantization (GPTQ/AWQ/FP8/INT8) is realized as custom linear layers with matching kernels (Marlin, FP8 tensor cores) that trade precision for memory and/or compute. The architecture registry makes vLLM's broad model support possible by expressing each model in terms of reusable parallel/paged primitives. MoE, MLA, speculative decoding, and VLMs are specializations requiring engine-side support beyond the generic path. Together this layer turns "a batch of sequences to run" into "logits, then tokens," every step — the muscle that the scheduler (nervous system, File 04), the distributed runtime (skeleton, File 05), and the kernels (File 10) combine to drive. File 07 covers how requests reach this engine (the serving and API layer) and how tokens stream back.

---

## 12. Quantization Trade-offs, Quantified

To make §3's method choice concrete, consider LLaMA-3 70B and the memory/throughput/accuracy trade-offs of each scheme:

| Scheme | Weight memory | Decode speed vs BF16 | Prefill speed vs BF16 | Accuracy impact | Hardware |
|---|---|---|---|---|---|
| BF16 | 140 GB | 1.0× | 1.0× | baseline | all |
| FP8 W8A8 | 70 GB | ~1.6–1.8× | ~1.8× | <0.5% | H100+ |
| INT8 W8A8 | 70 GB | ~1.5× | ~1.7× | ~0.5–1% | A100+ |
| GPTQ/AWQ W4A16 | 35 GB | ~1.5–2× (memory-bound) | ~1.0× (compute unchanged) | ~0.5–2% | A100+ (Marlin) |
| W4 + FP8 KV | 35 GB + half KV | further KV gains | — | additive | H100+ |

The key reasoning, restated: **W4A16 (GPTQ/AWQ)** wins on *memory* (4× weight reduction → fits a 70B on a single 80 GB GPU, or leaves more KV room) and on *decode* (less weight to read, the memory-bound bottleneck), but does *nothing* for prefill (the matmul still runs in FP16 after dequant). **FP8/INT8 W8A8** wins on *both* phases by using low-precision tensor cores, but only halves (not quarters) the weight memory. So the optimal choice depends on the binding constraint: fitting a big model on few GPUs → W4A16; maximizing prefill throughput on H100 → FP8 W8A8; long-context concurrency → add FP8 KV. The "~1.5–2×" decode range for W4A16 reflects how memory-bound the workload is — the more bandwidth-bound (large model, small batch), the closer to the full 4× weight-read reduction the speedup gets.

A worked memory consequence: LLaMA-3 70B in W4A16 is 35 GB, fitting on a single 48 GB GPU (e.g. L40S) with ~13 GB for KV — enabling single-GPU 70B serving that BF16 (140 GB) makes impossible. This is why quantization is not just an optimization but an *enablement*: it changes which models fit which hardware, reshaping the deployment topology (File 05 §12, File 16).

---

## 13. Attention Backend Selection

vLLM chooses among attention backends (`vllm/attention/backends/`) based on hardware, model, and configuration:

- **FlashAttention:** the default on many NVIDIA GPUs for the dense (non-paged-prefill) and standard paths; FA-2/FA-3 depending on the GPU (File 02 §3, File 10).
- **FlashInfer:** the high-performance paged-attention backend (File 03 §16.4, File 10) — paged KV, varlen batches, prefix-cache awareness, GQA-optimized grouping. Often the fastest on H100; enabled via configuration/environment (`VLLM_ATTENTION_BACKEND=FLASHINFER` or auto-selection).
- **XFormers:** a memory-efficient attention fallback for hardware/cases where FlashAttention isn't available.
- **Triton:** pure-Triton attention kernels, used as a fallback (including on AMD ROCm where FlashInfer may be unavailable, File 16).

The selection considers: GPU compute capability (FA-3 needs Hopper; FlashInfer has its own requirements), the attention variant (MLA needs special handling, §7), the dtype (FP8 paths), and whether features like prefix caching or sliding window are active. The backend determines the exact `AttentionMetadata` layout (§2). For production NVIDIA deployments, FlashInfer is generally preferred for its paged-KV performance; the engine picks a correct default but the backend can be overridden for tuning (File 11). A backend mismatch (e.g. a page size incompatible with the configured block size, File 03 §13) surfaces as a kernel error or silent slowdown.

---

## 14. LoRA Integration in the Model Layer

Multi-LoRA serving (full treatment File 18) touches the model-execution layer through specialized layers and kernels:

- **LoRA-aware linear layers:** the base linear layer is wrapped so that, for each request, the appropriate adapter's low-rank `B·A` update is added to the base output: `y = x·W_0 + x·(B·A)`. 
- **Punica / SGMV kernels:** a single batch may contain requests using *different* adapters. The **Punica** `bgmv` (batched grouped matrix-vector) kernel applies each request's adapter in one batched operation without materializing the full `ΔW` per request — `bgmv_shrink` (`A·x`) then `bgmv_expand` (`B··`) — so mixed-adapter batches run efficiently (File 18 §Punica).
- **Adapter management:** `LoRAManager` / `LRUCacheWorkerLoRAManager` keep up to `--max-loras` adapters resident in GPU memory, evicting LRU and loading from CPU/disk on demand. The scheduler accounts for adapter slots as a second resource (File 04 §21).

The model runner threads each request's adapter ID into the forward pass so the LoRA kernels apply the right adapter per request. This makes serving hundreds of fine-tuned adapters on one base model practical (the SaaS multi-tenant pattern, File 18 §multi-LoRA, File 20).

---

## 15. KV Cache Dtype Handling

The model-execution layer must write and read the KV cache in the configured dtype (File 02 §7, File 13):

- **`--kv-cache-dtype auto`:** matches the model dtype (BF16/FP16).
- **`fp8_e5m2` / `fp8_e4m3`:** quantize K/V to FP8 on write, dequantize on read, halving KV memory and bandwidth (H100+). Requires storing per-tensor/per-layer scales alongside the blocks (File 03 §35) and an attention kernel that handles FP8 KV.
- **INT8 KV:** per-token scaling for better accuracy than per-tensor.

The quantization happens in the K/V write path (after the QKV projection, before storing to the paged pool) and the dequantization in the attention kernel's K/V load. The block layout reserves space for the scales. KV dtype composes with weight quantization — e.g. W4A16 weights + FP8 KV — to attack both the weight-read and KV-read bandwidth bottlenecks simultaneously (File 13 §KV quantization).

---

## 16. Appendix: Configuration Reference and Pitfalls

Key model/quantization flags:
- **`--quantization`** (`gptq` | `awq` | `fp8` | `int8` | `bitsandbytes` | ... | auto): the weight quantization method (§3).
- **`--kv-cache-dtype`** (`auto` | `fp8_e5m2` | `fp8_e4m3`): KV cache precision (§15).
- **`--dtype`** (`auto` | `bfloat16` | `float16`): the compute dtype for unquantized parts.
- **`--load-format`** (`auto` | `safetensors` | `pt` | `gguf` | ...): checkpoint format (§4).
- **`--max-model-len`**, **`--rope-scaling`**: context length and RoPE extension (File 02 §5.2) — must match the checkpoint.
- **`--speculative-model`**, **`--num-speculative-tokens`**: speculative decoding (§8, File 12).
- **`--enable-lora`**, **`--max-loras`**, **`--max-lora-rank`**: multi-LoRA (§14, File 18).

Common pitfalls:
- **Quantization-group / TP-degree mismatch:** the group size must align with the TP partition (File 05 §19), or loading errors.
- **RoPE-scaling mismatch:** wrong `rope_scaling` silently degrades long-context quality (File 02 §5.2, §FAQ).
- **Backend / block-size mismatch:** the attention backend's page size must match `--block-size` (File 03 §13).
- **tie_word_embeddings ignored:** loading duplicate or mismatched LM-head weights (File 02 §14.3).
- **GGUF dequant memory:** loading a GGUF model dequantizes to FP16 on GPU, using *more* memory than the GGUF file size suggests — size the GPU accordingly.

The model-execution layer is where many "the model loads but outputs are wrong" bugs live, almost always traceable to a config mismatch (RoPE, quantization scales, tied embeddings, KV dtype) between what the checkpoint expects and what the engine applies. The discipline is to treat `config.json` as the contract (File 02 §27) and verify the engine honors every field — attention variant, head counts, RoPE scaling, vocabulary, tying, and quantization scheme — before trusting the outputs.

---

## 17. Fused Kernels in the Forward Pass

vLLM's model layers use fused kernels (File 02 §13, §24; File 10) to minimize HBM round-trips, and the model module must wire its forward pass to use them:

- **Fused QKV projection:** the separate Q, K, V projections are combined into one matmul (`W_QKV` stacked) so a single GEMM produces all three, with the column-parallel split (File 05 §2.2) applied to the fused weight. The checkpoint's separate `q_proj`, `k_proj`, `v_proj` are merged into one parameter at load time (a `load_weights` responsibility, §5).
- **Fused gate/up projection:** SwiGLU's gate and up projections are stacked into one matmul, with the SiLU⊙multiply fused afterward (`act(gate) ⊙ up` in one kernel).
- **Fused RoPE:** the rotary embedding is applied inside or immediately after the QKV kernel, avoiding a separate pass over Q and K.
- **Fused RMSNorm + residual:** the normalization and the residual add are fused with adjacent matmuls (File 02 §13.2).

These fusions are why the model module's `load_weights` must handle parameter *repacking* — the checkpoint stores `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj` separately, but vLLM packs them into fused `qkv_proj` and `gate_up_proj` tensors for the fused kernels. The weight-name mapping in `load_weights` does this repacking while also applying the TP sharding. A bug here (wrong packing order, wrong shard slice) produces a model that loads without error but generates garbage — one of the trickier model-porting failure modes.

### 17.1 A worked weight-mapping example

For a LLaMA layer, the checkpoint has (per layer) `self_attn.{q,k,v,o}_proj.weight` and `mlp.{gate,up,down}_proj.weight`. vLLM's LLaMA module maps these to its fused, sharded parameters:

```
q_proj, k_proj, v_proj   →  qkv_proj  (concatenated on output dim, column-parallel sharded)
gate_proj, up_proj       →  gate_up_proj (concatenated, column-parallel sharded)
o_proj                   →  o_proj (row-parallel sharded)
down_proj                →  down_proj (row-parallel sharded)
```

For GQA, the QKV concatenation accounts for the different number of K/V heads vs Q heads (the K and V slices are smaller), and the column-parallel sharding must keep each rank's Q heads aligned with its K/V heads (File 05 §2.3). The `load_weights` method iterates the checkpoint tensors, identifies which fused parameter and which shard slice each belongs to, and copies accordingly. This mapping is the bulk of what writing a new model module entails (§5) — the forward pass mostly reuses vLLM's primitives, but the weight names and packing are model-specific and must be exactly right.

---

## 18. Closing

The model-execution layer is the concrete realization of everything the earlier files specified abstractly: the architecture's contract (File 02) instantiated as parallel/paged/quantized/fused layers, addressing the block pool (File 03) via attention metadata, sharded across GPUs (File 05), driven by the scheduler's per-step plan (File 04). Quantization changes which models fit which hardware; the architecture registry makes broad model support tractable; MoE, MLA, speculative decoding, LoRA, and VLMs are the specializations that require engine-side code beyond the generic transformer path. The recurring engineering theme is *fusion and correctness*: fuse operations to cut HBM traffic (the bandwidth bottleneck of decode), and honor the config contract exactly (the source of most silent-wrong-output bugs). With the engine's compute layer understood, File 07 turns to the serving surface — the async engine, the OpenAI-compatible API, streaming, metrics, and deployment — that exposes this machinery to clients, and File 10 to the kernels that make each operation hit the hardware roofline.

