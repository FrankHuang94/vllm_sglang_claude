# Multimodal Inference, Vision-Language Models, and Embedding Serving

> **Standard reference file.** Beyond text generation, inference engines serve vision-language models (VLMs), embedding models, and rerankers. This file covers VLM architecture patterns, image caching, the vLLM and SGLang multimodal implementations, embedding/reranking serving, and audio/video modalities. Prerequisites: File 06 §9 (vLLM VLM support), File 08 §15 (RadixAttention image caching), File 03 §6 (prefix caching).

---

## 1. VLM Architecture Patterns

Vision-language models combine a visual encoder with an LLM. Three integration patterns dominate:

### 1.1 Visual-tokens-in-sequence (LLaVA-style)

The most common pattern: a **visual encoder** (ViT, CLIP, or SigLIP) processes the image into a grid of features, a **projector** (an MLP or small network) maps those features into the LLM's embedding space, and the resulting "image tokens" are **inserted into the LLM's input sequence** at `<image>` placeholder positions. The LLM then processes the combined image+text token sequence normally (attention over both). LLaVA-1.5 uses ViT-L/14 producing a 14×14 = 256-token grid per image (after the projector). The image becomes ~256 tokens in the sequence — a fixed cost per image regardless of the question. Models: LLaVA-1.5/NeXT, Qwen-VL, InternVL.

### 1.2 Cross-attention (Idefics/Flamingo-style)

Instead of inserting image tokens into the sequence, the image features are attended to via *cross-attention* layers interleaved in the LLM — the text tokens attend to the image features without the images occupying sequence positions. This keeps the text sequence shorter but adds cross-attention layers (and a separate, fixed image-feature KV, File 02 §31). Models: Idefics, Flamingo-style.

### 1.3 Integrated encoder (Pixtral-style)

The visual encoder is more tightly integrated into the LLM's architecture (e.g. the ViT shares layers or is processed within the LLM's forward pass). Models: Pixtral and some newer designs.

For inference, the visual-tokens-in-sequence pattern (§1.1) is the most common and the simplest for the engine: the image becomes tokens, and the LLM processes them like text (with the visual encoder as a preprocessing stage). The image-token count (256 for LLaVA, more for high-resolution models that tile the image into multiple crops) determines the image's contribution to the sequence length and KV cache.

---

## 2. Image Caching

A key efficiency: the image's KV (the LLM's KV for the image tokens) is **fixed for a given image regardless of the question** asked about it. So in multi-turn visual QA — "describe this image," then "what's in the background?", then "count the people" — all referencing the same image, the image's KV can be computed once and reused across all questions (File 06 §9, File 08 §15).

- **Mechanism:** hash the image (bytes or features) and treat the image tokens as a cacheable prefix. vLLM's prefix caching (File 03 §6) and SGLang's RadixAttention (File 08 §15) cache the image-token KV, so the same image with different questions hits the cache and computes only the question tokens.
- **Savings:** for a 256-token image (LLaVA) or thousands of tokens (high-res tiled models) queried many times, caching the image KV saves recomputing it each time — large for multi-turn visual QA, document understanding (many questions per document image), and iterative image editing.
- **SGLang fit:** RadixAttention's token-granular tree (File 08 §8) naturally caches image prefixes; the image must be part of the cache key (different images don't share, File 08 §22). SGLang's `sgl.image()` (File 08 §12) integrates images into the prefix-caching flow.

Image caching is the multimodal analog of text prefix caching — the same mechanism (cache the shared prefix KV) applied to the shared image, with large savings for repeated-image workloads.

---

## 3. vLLM Multimodal Implementation

vLLM's multimodal support (File 06 §9):

- **`MultiModalInputs`** carries the image data — pixel values, pre-computed embeddings, or features — alongside the text token IDs. The API accepts images (URLs, base64) in the request.
- **`IMAGE_TOKEN_ID`** marks the placeholder positions in the token sequence where image features are injected. The visual encoder produces features that replace the placeholder embeddings before the LLM forward pass.
- **`MultiModalPlugin`** per modality handles preprocessing (image loading, resizing, normalization) and feature extraction. The visual encoder runs on GPU (or CPU, per configuration).
- **Pipeline:** image preprocessing → visual encoder → projector → inject features at placeholder positions → LLM forward pass (over the combined image+text sequence). The image-feature extraction is a stage before the LLM's text processing.
- **Supported models:** LLaVA-1.5/NeXT, InternVL-2, Qwen-VL-2, Pixtral, LLaMA-3.2-Vision, and others — each with its image-token count and injection style handled by the model module.

The engine treats the image tokens as part of the sequence (after encoding), so the scheduler, KV cache, and attention work as for text (the image tokens occupy sequence positions and KV slots). The added stages are the image preprocessing and the visual encoder (a separate model run before the LLM).

---

## 4. SGLang Multimodal Implementation

SGLang's multimodal support (File 08 §15, File 09 §39):

- **`sgl.image()`** primitive in the frontend DSL (File 08 §12) includes an image in a program, integrating it into the prompt structure the runtime sees.
- **Image preprocessing in the TokenizerManager** (File 09 §39): image loading/resizing/normalization runs in the TokenizerManager process, overlapped with GPU execution like text tokenization. Feature extraction is batched with text processing.
- **RadixAttention image caching** (File 08 §15, §2 above): the image's KV is cached and reused across questions on the same image — the natural fit of RadixAttention for repeated-image workloads.
- **CUDA stream overlap** (File 09 §12): the visual encoding overlaps with the previous step's decode on a separate stream, hiding the encoder cost.

SGLang's architecture (process separation, RadixAttention, stream overlap) applies to multimodal as to text — the visual encoder is an added GPU stage, the image preprocessing is in the TokenizerManager, and the image prefix caching comes "for free" from RadixAttention (File 08 §15). This makes SGLang strong for image-grounded multi-turn workloads (visual search, document QA) where the image caching is valuable.

---

## 5. Embedding Model Serving

Embedding models (E5, BGE, nomic-embed, etc.) produce a fixed-size vector representation of text — used for retrieval, RAG, semantic search, clustering. Serving them differs fundamentally from generation:

- **`--task embed`** (vLLM): run the model as an encoder. The output is a pooled hidden state — mean pooling over the tokens, or the `[CLS]`/last-token hidden state — projected to a fixed-size embedding vector.
- **No autoregression, no KV cache growth:** embedding is a *single forward pass* per input (process all tokens, pool, output the vector) — no token-by-token generation, no growing KV cache, no decode loop. This is a completely different workload from generation.
- **Compute-bound, batch-maximized:** since it's a single forward pass (like prefill), embedding is *compute-bound* (File 01 §3) — throughput is maximized by large batches that saturate the tensor cores (no decode, so no memory-bound regime). Fill the GPU with a large batch of inputs to maximize FLOP utilization.
- **API:** `/v1/embeddings` (OpenAI-compatible). Input a list of texts, get back a list of vectors.
- **Model sizes:** embedding models range from 335M (small BERT-style) to 7B (LLM-based embedders). Per-request cost is `O(seq_len × model_params)` for the single forward pass — vs `O(output_len × model_params)` for generation. Embedding is cheaper per request (no autoregression) but you typically embed many texts (a whole corpus for indexing).

Embedding serving reuses the engine's model-loading, batching, and forward-pass infrastructure but bypasses the generation loop (no sampler, no KV cache, no streaming). It's a compute-bound, batch-throughput workload — the opposite of memory-bound decode — so it tunes differently: maximize batch to saturate compute, no KV-cache concerns. Serving embeddings and generation from the same stack consolidates infrastructure (one engine, multiple tasks).

---

## 6. Reranking and Classification

**Rerankers** (cross-encoders) score the relevance of a (query, document) pair — used in RAG to rerank retrieved candidates for better relevance than the initial embedding-based retrieval:

- The query and document are concatenated as input; the model outputs a scalar relevance score (a classification head).
- **Single forward pass per pair** (like embedding) — compute-bound, no generation. Latency scales with the number of candidate documents per query (you score each candidate).
- vLLM supports classification/scoring tasks; the pattern mirrors Cohere's rerank API.
- **Scale:** RAG reranking scores all candidate documents per query (e.g. rerank the top 100 retrieved) — so the throughput is `queries × candidates_per_query` forward passes. Batch them to saturate compute.

Classification more broadly (sentiment, topic, etc.) uses the same pattern — a forward pass producing a class score, compute-bound, batch-maximized. Like embedding (§5), reranking/classification bypass the generation loop and are compute-bound single-forward-pass workloads, tuned for batch throughput. The RAG pipeline often uses all three: embedding (retrieve candidates), reranking (score candidates), and generation (answer using the top candidates) — all servable from the same engine, each with its own characteristics (embedding/rerank compute-bound batch; generation memory-bound decode).

---

## 7. Audio and Video Modalities

Beyond images, audio and video are emerging modalities:

- **Audio (Whisper-style):** an audio encoder processes the waveform/spectrogram into features, fed to an LLM decoder (cross-attention or in-sequence). Whisper is encoder-decoder (cross-attention, File 02 §31). Audio LLMs (speech-to-text, audio understanding) follow the VLM pattern with an audio encoder replacing the visual one. SGLang and vLLM are adding audio support (audio encoder plugins).
- **Video:** video as a sequence of frames, each encoded (like images) with temporal attention across frames. Video greatly increases the token count (many frames × tokens-per-frame), stressing the sequence length and KV cache (a long-context problem, File 13). Frame sampling (process a subset of frames) and temporal pooling reduce this.
- **Status:** audio and video support is less mature than text/image in vLLM/SGLang (roadmap items), but follows the same architecture (modality encoder → features → LLM) and the same caching opportunity (the audio/video features are fixed for a given input, cacheable like images, §2).

The general multimodal pattern — a modality-specific encoder producing features injected into the LLM, with the features cacheable as a prefix — generalizes across image, audio, and video. The engine's role is to run the encoder (an added GPU stage), inject the features, and process the combined sequence, with the modality features cached for repeated queries. As multimodal models proliferate (File 20 §future), this pattern (encoder + LLM + feature caching) is the common framework, with each modality adding its encoder and preprocessing.

---

## 8. Synthesis

Multimodal and non-generation inference broaden what serving engines do beyond text generation:

- **VLMs** (§§1–4): a visual encoder produces image tokens injected into the LLM sequence (LLaVA-style) or attended via cross-attention (Idefics-style). The image's KV is cacheable (§2), a major saving for repeated-image multi-turn workloads — RadixAttention (SGLang) and prefix caching (vLLM) provide it.
- **Embeddings** (§5): single-forward-pass, compute-bound, batch-maximized — no KV cache, no decode. The opposite of memory-bound generation; tune for batch throughput.
- **Reranking/classification** (§6): single-forward-pass scoring, compute-bound — used in RAG pipelines alongside embedding and generation.
- **Audio/video** (§7): modality encoders following the VLM pattern, with the long-context challenge for video (many frames).

The unifying observations: (1) **non-generation tasks (embedding, reranking) are compute-bound single-forward-passes**, tuned for batch throughput, completely unlike memory-bound decode — the same engine serves both but they have opposite bottlenecks; (2) **multimodal features are cacheable prefixes** — the image/audio/video features are fixed for a given input, so caching them (RadixAttention/prefix caching) saves recomputation for repeated queries, the multimodal extension of prefix caching; (3) **modality encoders are added GPU stages** before the LLM, with preprocessing in the tokenizer/manager process (overlapped). The serving engine generalizes from a text-generation system to a multi-task, multi-modality platform, reusing the core infrastructure (model loading, batching, KV cache, prefix caching) while adding modality encoders and bypassing the generation loop for encoder tasks. This breadth — generation, multimodal, embedding, reranking from one stack — is increasingly expected, consolidating an organization's model-serving onto a single engine (File 19 §multi-model).

---

## 9. Image-Token Cost and High-Resolution Tiling

The number of image tokens drives the VLM's sequence length and KV cost, and it varies widely by model and resolution.

- **Fixed-grid (LLaVA-1.5):** a ViT-L/14 on a 336×336 image produces a 24×24 grid → 576 patches, often pooled/projected to ~256 tokens per image. So an image adds ~256 tokens to the sequence — modest.
- **High-resolution tiling (LLaVA-NeXT, InternVL, Qwen-VL):** to handle high-resolution images and detail, the image is split into multiple **tiles** (crops), each encoded separately, plus a global thumbnail. A 4-tile + thumbnail scheme produces ~5× the tokens (e.g. ~1,280–2,880 tokens for a high-res image). Document-understanding models processing dense pages can produce thousands of image tokens per image.
- **The KV consequence:** at ~2,500 image tokens, an image contributes `2500 × KV/token` to the KV cache — for LLaMA-3-class KV (~320 KB/token BF16), that's ~800 MB of KV *per image*. For multi-image inputs (a document with many pages, a video's frames), this multiplies — pushing VLM serving toward the long-context regime (File 13). The image-token count is thus a first-order cost driver: a fixed-grid model (256 tokens/image) is cheap; a high-res tiled model (thousands of tokens/image) is expensive (long-context-like KV).

This is why image caching (§2) matters so much for high-res/document VLMs: the thousands of image tokens, computed once and cached, are reused across all questions on that image — a large saving. And it's why VLM serving capacity planning must account for the image-token contribution to sequence length and KV (a VLM request is "longer" than its text suggests, by the image-token count). The high-res tiling trade-off — more tokens (cost) for more visual detail (quality) — is a model-design choice the inference engineer must account for in capacity planning.

---

## 10. VLM Scheduling Challenges

VLMs complicate the scheduler (File 04) in specific ways:

- **The visual encoder is an extra stage:** before the LLM forward pass, the image must be encoded (the ViT/CLIP run). This adds latency to the first step (TTFT includes image encoding) and uses GPU compute that competes with the LLM batch. The encoder can run on a separate stream (overlapped, File 09 §12) or as a preprocessing batch.
- **Variable image-token counts:** different requests have different numbers of images and resolutions → different image-token counts → variable prefill sizes. The scheduler's token budget (File 04 §6) must account for the image tokens (a VLM prefill is text + image tokens).
- **Image encoding batching:** encoding multiple images (across requests) can be batched for efficiency (the ViT is a compute-bound GEMM, batch-friendly) — separate from the LLM batching. The engine may batch image encoding independently from text generation.
- **Image prefix caching interaction:** a cache-hit on the image (§2) skips the encoding *and* the image-token prefill — but a cache-miss pays both. So the scheduler's cost estimate for a VLM request depends on whether its image is cached.

These make VLM scheduling more complex than text — the encoder stage, variable image-token counts, and image caching all add dimensions. The engines handle this by treating image encoding as a preprocessing stage (overlapped where possible) and the image tokens as part of the sequence (scheduled like text once encoded). For high-throughput VLM serving, batching the image encoding and caching image KV are the key levers (the encoder can become a bottleneck if many distinct images are processed without caching).

---

## 11. The Visual Encoder as a Potential Bottleneck

For VLM workloads with many *distinct* images (no cache reuse), the visual encoder can become a bottleneck:

- The encoder (ViT/CLIP) is a compute-bound model run per image — for high-res tiled inputs (multiple crops), it's run per tile. A document-QA workload processing many unique pages runs the encoder heavily.
- If the encoder runs on the same GPU as the LLM, it competes for compute; if many images arrive, the encoder work can dominate, starving the LLM.
- **Mitigations:** batch image encoding (compute-bound, batch-friendly), cache image features/KV (§2) for repeated images, run the encoder on a separate GPU/stream, or use a smaller/faster encoder. For repeated-image workloads (same image, many questions), caching makes the encoder cost negligible (run once); for unique-image workloads (each image seen once), the encoder is a real cost that batching and a fast encoder address.

This is a VLM-specific capacity consideration: the encoder is an added compute stage whose cost depends on the image distinctness (cached vs unique) and resolution (tiling). VLM capacity planning must account for both the LLM (text + image-token processing) and the encoder (image processing) — a two-model pipeline. For text-only LLMs this doesn't arise; for VLMs it's a distinct planning dimension, and image caching (when applicable) is the key lever to keep the encoder from becoming the bottleneck.

---

## 12. Embedding Throughput, Worked

Quantify embedding serving's compute-bound throughput (§5). An embedding model (say 7B, single forward pass per input):

- Each input of ~256 tokens: FLOPs ≈ `2 · 7e9 · 256 ≈ 3.6e12 = 3.6 TFLOP` (forward pass, `2·P·tokens`).
- On an H100 (989 TFLOP/s BF16, ~80% util): ~`3.6e12 / 790e12 ≈ 4.5 µs`... but that's per input; the point is to *batch*. At batch 256 (256 inputs × 256 tokens = 65,536 tokens): FLOPs ≈ `2·7e9·65536 ≈ 9.2e14 = 920 TFLOP` → ~1.16 s, producing 256 embeddings → ~4.5 ms/embedding amortized, or ~220 embeddings/sec... the key is throughput scales with batch (compute-bound, File 01 §3).

The embedding workload is **compute-bound and batch-maximized** — unlike generation's memory-bound decode, embedding wants the largest batch that fits to saturate the tensor cores (no KV-cache concern since there's no autoregression, just a single forward pass). Throughput is `tokens/sec` at the compute roofline. So embedding tuning is simple relative to generation: maximize batch (limited by activation memory, not KV), use the compute roofline as the target, and there's no decode/TPOT to worry about. For indexing a large corpus (embed millions of documents), this batch-throughput is the metric, and the engine's batching infrastructure (reused from generation) delivers it. The contrast with generation is instructive: embedding is pure compute-bound batch throughput (like prefill), while generation is memory-bound decode — the same engine, opposite bottlenecks, tuned differently (§5).

---

## 13. Multimodal Prefix Caching, Worked

Quantify image caching (§2) for a document-QA workload: a 2,000-image-token document (high-res tiled, §9), queried by 20 questions.

- **Without caching:** each query encodes the image (ViT runs) + prefills 2,000 image tokens + the question. 20 queries → 20× image encoding + 20× 2,000-token image prefill = 40,000 image-token prefills + 20 encoder runs.
- **With caching (RadixAttention/prefix cache):** the image is encoded once, its 2,000-token KV cached; each subsequent query reuses the cached image KV, prefilling only its question. 20 queries → 1 encoder run + 2,000 image-token prefill (once) + 20× question prefill.

The saving: ~20× reduction in image-token prefill *and* image encoding for this repeated-image workload. For document understanding (many questions per document), visual search (compare against a fixed image), or iterative editing (repeated reference to the same image), this is a major lever — the multimodal analog of text prefix caching's system-prompt reuse (File 03 §31). The catch is the same as text: the image must be the cache key (different images don't share), and the cache must retain the image KV between queries (KV pool sized for the image working set). For VLM workloads with image reuse, enabling image caching is as impactful as enabling text prefix caching for shared-prompt text workloads — and SGLang's RadixAttention provides it naturally (File 08 §15).

---

## 14. Multimodal Deployment Considerations

Deploying VLMs and multimodal models adds operational dimensions:

- **Two models to load:** the visual encoder and the LLM (and the projector) — more weight memory and longer load time (File 07 §19). The encoder is usually small (a ViT is hundreds of MB to a few GB) relative to the LLM.
- **Quantization:** the LLM can be quantized (FP8/W4A16, File 06) as usual; the visual encoder is smaller and often kept in higher precision (its cost is the per-image encoding, less memory-critical). FP8 KV (File 13) helps the (potentially large, §9) image-token KV.
- **Image input handling:** the API accepts images (URLs to fetch, base64 inline) — fetching URLs adds latency and a failure mode (broken URLs); base64 inflates request size. The gateway/API layer handles image input validation and fetching (File 07).
- **Capacity:** account for the image-token contribution to KV (a VLM request is "longer," §9) and the encoder compute (a second model stage, §11). VLM capacity is lower than text-only for the same hardware (images add tokens and encoder work).
- **Multimodal + long context:** document and video models combine multimodal (image tokens) with long context (many tokens) — the union of File 13 and this file's challenges. A multi-page document VLM is both a VLM (encoder, image tokens) and a long-context workload (thousands of image tokens) — apply both toolkits (image caching + FP8 KV + chunked prefill).

VLM deployment is text-LLM deployment plus the encoder stage and the image-token cost — the core infrastructure (engine, KV cache, batching) is shared, with multimodal-specific additions (encoder, image preprocessing, image caching). The same engine serves text and multimodal, consolidating infrastructure, with VLM-specific tuning (image caching, encoder batching, image-token-aware capacity).

---

## 15. FAQ and Synthesis

**Q: My VLM serving is slow / low throughput.** Check the image-token count (high-res tiling can be thousands of tokens/image, §9) inflating sequence length and KV; check the encoder isn't a bottleneck (many unique images, §11) — batch encoding and cache image KV; check image caching is enabled for repeated images (§2, §13).

**Q: Can I cache images across requests?** Yes (§2, §13) — vLLM prefix caching / SGLang RadixAttention cache the image-token KV, reused for the same image with different questions. The image is the cache key (different images don't share). Major saving for repeated-image workloads.

**Q: How do I serve embeddings and generation together?** The same engine serves both (`--task embed` for embedding); they have opposite bottlenecks (embedding compute-bound batch §5, §12; generation memory-bound decode), so consider separate deployments tuned for each, or one engine handling both with awareness of the different profiles (File 19 §multi-model).

**Q: Why is embedding tuned differently from generation?** Embedding is a single forward pass (no autoregression, no KV growth) — compute-bound, batch-maximized (§5, §12). Generation is memory-bound decode. Embedding wants the largest batch to saturate compute; generation balances batch against TPOT. Opposite regimes.

The synthesis: serving engines have evolved from text-generation systems into **multi-task, multi-modality platforms**. The core infrastructure (model loading, batching, KV cache, prefix caching, the attention kernels) is shared, while each task/modality adds specifics: VLMs add the visual encoder and image-token cost (with image KV caching as a key lever); embeddings/rerankers bypass the generation loop and are compute-bound batch workloads; audio/video extend the modality-encoder pattern. The two unifying insights — **non-generation tasks are compute-bound single-forward-passes** (opposite of memory-bound decode) and **multimodal features are cacheable prefixes** (the multimodal extension of prefix caching) — let the inference engineer reason about these workloads using the same cost models (roofline, File 01 §2) and mechanisms (prefix caching, File 03 §6, File 08) as text generation, just applied to the different bottlenecks. As models become natively multimodal (text, image, audio, video in one) and as RAG (embedding + reranking + generation) becomes standard, the engine's role as a unified multi-task platform — serving generation, multimodal, and encoder tasks from one stack — is increasingly central (File 19, File 20), and the techniques here (encoder integration, feature caching, compute-bound batch tuning) are the toolkit for it. The KV cache and prefix caching, central to text generation (File 01 §4, File 03 §6), extend naturally to multimodal (cache the image/audio features) — confirming that the core abstractions of LLM inference generalize across modalities and tasks. An inference engineer who understands text generation deeply — the KV cache, prefix caching, the prefill/decode duality, the roofline — already holds most of what's needed for multimodal and encoder workloads; the additions (modality encoders, feature caching, compute-bound batch tuning) are extensions of the same principles, not a separate discipline. That transferability is why the foundations (Files 01–11) precede the specializations (this file and the others): master the core, and the specializations follow. This is the organizing logic of the entire database, made concrete here: multimodal and embedding serving are not new systems but the familiar text-serving system with a modality encoder bolted on and the generation loop sometimes removed — and recognizing that lets the engineer carry the hard-won text-serving intuition directly into these adjacent domains, adapting only for the modality encoder and the compute-bound encoder-task regime — a small delta on a large shared foundation that the preceding files established in full — which is the reassuring conclusion for any engineer facing a new modality or task: it is mostly the system you already know, with a well-understood delta to learn for each new modality and task.


