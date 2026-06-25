# Inference Framework Comparison — vLLM vs SGLang vs TensorRT-LLM vs TGI

> **Standard reference file.** This file compares the major LLM serving frameworks — vLLM, SGLang, TensorRT-LLM, TGI, DeepSpeed-MII, MLC-LLM, Ollama, LMDeploy — across architecture, performance, model support, and use cases, with a feature-comparison table. It synthesizes the engine-internals files (vLLM 03–07, SGLang 08–09) into a comparative landscape. Prerequisites: Files 03–11 (the mechanisms compared).

---

## 1. TensorRT-LLM (NVIDIA)

NVIDIA's closed-source, ahead-of-time-compiled inference library — the performance benchmark on NVIDIA hardware for supported models.

- **Architecture:** a CUDA kernel library with graph compilation via TensorRT. Models are *compiled* ahead of time (`trtllm-build` CLI) into optimized engines, then served (often via Triton Inference Server).
- **Strengths:** the best raw FLOP utilization on NVIDIA for supported models — highly optimized CUDA kernels (FP8, W4A8, INT4), XQA (cross-attention extension for paged KV), in-flight batching (NVIDIA's term for continuous batching), beam search. For a supported model on NVIDIA, often the fastest.
- **Weaknesses:** **compilation time** (10–30 min per model/config), **no Python extensibility** (you can't easily modify the model code — it's compiled), **limited model support** (each model must be explicitly ported to TensorRT-LLM), and **NVIDIA-only**. The AOT compilation also means config changes require recompilation.
- **Use case:** maximum performance on NVIDIA for a stable, supported model where the compilation overhead and reduced flexibility are acceptable. Common in production NVIDIA deployments prioritizing raw speed over flexibility. Convergence: TensorRT-LLM adopted continuous batching (from the vLLM ecosystem); vLLM/SGLang adopted FP8 and speculative decoding (from TRT-LLM's direction).

---

## 2. TGI (Text Generation Inference, HuggingFace)

HuggingFace's production inference server, deeply integrated with the HF ecosystem.

- **Architecture:** a Rust HTTP server + Python inference backend. Flash Attention 2 via custom Rust bindings; tensor parallelism via safetensors sharding; paged attention; Marlin quantization; continuous batching.
- **Strengths:** the **widest HuggingFace model compatibility** (it's HF's own server), OpenAI API compatibility, deep HF ecosystem integration (Hub, transformers), and a solid Rust HTTP layer.
- **Weaknesses:** historically **slower than vLLM on throughput benchmarks**, and less active development on the core inference *algorithms* (it adopted PagedAttention-style paging but isn't the source of serving-systems innovation the way vLLM/SGLang are).
- **Use case:** HuggingFace-ecosystem deployments wanting broad model support and HF integration, where vLLM's throughput edge isn't decisive. TGI is a solid, well-supported choice especially for HF-centric workflows.

---

## 3. DeepSpeed-MII (Microsoft)

Microsoft's DeepSpeed Inference component.

- **Architecture:** built on DeepSpeed Inference; **ragged batching** (continuous batching variant), **kernel injection** (replace HuggingFace model layers with DeepSpeed-optimized kernels), SmoothQuant/ZeroQuant quantization.
- **Strengths:** good MoE support (ZeRO-Inference for large-model offloading), RLHF training integration (part of the DeepSpeed training ecosystem).
- **Weaknesses:** **less maintained than vLLM**, slower development pace post-2023 — the serving-systems momentum moved to vLLM/SGLang.
- **Use case:** DeepSpeed-ecosystem deployments (especially those also using DeepSpeed for training/RLHF), or large-model offloading via ZeRO-Inference. Less common for greenfield serving than vLLM/SGLang.

---

## 4. MLC-LLM (Machine Learning Compilation)

Apache TVM-based compilation for *universal* hardware portability.

- **Architecture:** compiles models via TVM to optimized code for **any backend** — CUDA, ROCm, Metal, Vulkan, OpenCL, WebGPU.
- **Strengths:** **unmatched hardware portability** — deploy the same model on iOS, Android, web browsers (WebGPU), desktops, and servers. The MLCChat app demonstrates phone/browser LLM inference.
- **Weaknesses:** compilation time (TVM compilation per target), and **not optimized for server-class throughput** (it targets portability, not multi-tenant serving — no advanced continuous batching/prefix caching like vLLM/SGLang).
- **Use case:** cross-platform/edge deployment (the same model on phone, browser, desktop) where portability is paramount. The edge/local regime (File 15 §23, File 16 §7), not server-class serving.

---

## 5. Ollama

Local-first inference with a dead-simple UX.

- **Architecture:** llama.cpp backend, Go HTTP server. `ollama run llama3` — automatic GPU detection, GGUF models, simple model management.
- **Strengths:** **ease of use** (one command), cross-platform (Mac/Windows/Linux), great for local development and single-user inference.
- **Weaknesses:** **not designed for high-throughput serving** — single-user focused, no continuous batching (llama.cpp serves one request efficiently but doesn't batch many).
- **Use case:** local development, single-user local inference, prototyping. The local/edge regime (File 16 §7), not production multi-tenant serving. (For production, you'd use vLLM/SGLang/TGI.)

---

## 6. LMDeploy (Shanghai AI Lab)

A production inference framework with strong quantization.

- **Architecture:** the TurboMind C++ inference backend; W4A16 quantization (AWQ); continuous batching; paged attention. `lmdeploy serve api_server`.
- **Strengths:** fast on NVIDIA GPUs, good quantization support (AWQ W4A16). A solid, performant engine.
- **Weaknesses:** smaller community than vLLM, slower model-support addition (fewer models supported, added more slowly).
- **Use case:** NVIDIA serving where its quantization and TurboMind performance suit, particularly in the Chinese AI ecosystem where it originated. A capable alternative, less broadly adopted than vLLM/SGLang.

---

## 7. vLLM and SGLang Recap

The protagonists (Files 03–11), recapped in the comparative context:

- **vLLM:** PagedAttention (File 03), continuous batching + chunked prefill (File 04), broad model/quantization support, OpenAI API, the largest community (LF AI governance, File 20). The de facto open-source serving default — broad, well-supported, production-hardened.
- **SGLang:** RadixAttention (token-granular tree prefix reuse, File 08), XGrammar (near-free structured output, File 08 §11), the program-aware frontend DSL (File 08 §12), the overlap-scheduler architecture (File 09). Excels on multi-call/structured/agentic workloads; ships cutting-edge algorithms quickly.

Both have converged on the systems architecture (multi-process, FlashInfer, CUDA graphs, File 09 §22). They differ in prefix caching (RadixAttention's token-granular tree vs vLLM's block-hash APC, File 08 §8) and the frontend (SGLang's DSL has no vLLM equivalent). The choice (File 08 §36, File 11 §8): SGLang for prefix-sharing/structured/agentic workloads, vLLM for broad model support and the larger ecosystem; both excellent for general serving, with the workload's prefix-sharing fraction the key discriminator.

---

## 8. Feature Comparison Table

| Feature | vLLM | SGLang | TensorRT-LLM | TGI |
|---|---|---|---|---|
| OpenAI API | ✅ | ✅ | via Triton | ✅ |
| Continuous batching | ✅ | ✅ | ✅ (in-flight) | ✅ |
| Paged attention | ✅ (PagedAttention) | ✅ (token-granular) | ✅ (XQA) | ✅ |
| Prefix caching | ✅ (APC, block) | ✅ (RadixAttention, token) | partial | partial |
| Chunked prefill | ✅ | ✅ | ✅ | partial |
| Speculative decoding | ✅ | ✅ (EAGLE) | ✅ | ✅ |
| Structured output | ✅ (XGrammar/Outlines) | ✅ (XGrammar native) | partial | ✅ |
| LoRA serving | ✅ (multi-LoRA) | ✅ | limited | ✅ |
| FP8 | ✅ | ✅ | ✅ | ✅ |
| INT4 (W4A16) | ✅ (GPTQ/AWQ) | ✅ | ✅ | ✅ (Marlin) |
| MoE support | ✅ | ✅ (DP-attention) | ✅ | ✅ |
| AMD ROCm | ✅ | ✅ | ❌ (NVIDIA only) | ✅ |
| Program DSL | ❌ | ✅ (unique) | ❌ | ❌ |
| Model count | very broad | broad | limited (ported) | very broad (HF) |
| Extensibility | high (Python) | high (Python) | low (compiled) | moderate |
| License | Apache 2.0 | Apache 2.0 | proprietary | Apache 2.0 |
| Community | largest | active | NVIDIA | HF |

The table summarizes the landscape: vLLM and SGLang are the most feature-complete open engines (both ✅ across the board), TensorRT-LLM trades flexibility/portability for peak NVIDIA performance (compiled, NVIDIA-only, limited models), and TGI offers broad HF compatibility. The standout differentiators: SGLang's program DSL (unique), RadixAttention's token-granular prefix caching, and XGrammar native; vLLM's breadth and ecosystem; TRT-LLM's peak NVIDIA performance; TGI's HF integration.

---

## 9. Choosing a Framework

A decision guide synthesizing the comparison:

- **Default open-source serving:** **vLLM** — broadest support, largest ecosystem, production-hardened, OpenAI API. The safe default for most server-class deployments.
- **Prefix-sharing / structured / agentic workloads:** **SGLang** — RadixAttention, XGrammar, the program DSL give it the edge on multi-call, structured-output, agentic workloads (File 08 §36).
- **Maximum NVIDIA performance, stable model:** **TensorRT-LLM** — if the model is supported and the compilation/flexibility trade-off is acceptable, the fastest on NVIDIA.
- **HuggingFace ecosystem:** **TGI** — broadest HF model compatibility and integration.
- **Cross-platform / edge / browser:** **MLC-LLM** — unmatched portability (File 16 §7).
- **Local single-user / development:** **Ollama** (or llama.cpp directly) — simplest local UX.
- **AMD ROCm:** **vLLM or SGLang** (both support ROCm; TRT-LLM doesn't).
- **Strong AWQ quantization, NVIDIA:** **LMDeploy** — TurboMind + AWQ.

The decision drivers: the workload (prefix-sharing → SGLang, broad → vLLM), the hardware (NVIDIA-only → TRT-LLM an option; AMD → vLLM/SGLang), the deployment context (server → vLLM/SGLang/TGI; edge → MLC-LLM/Ollama), the model support needed (broad → vLLM/TGI; specific supported → TRT-LLM), and the performance/flexibility trade-off (peak NVIDIA + stable → TRT-LLM; flexible + broad → vLLM/SGLang). For most open-source server-class serving, the choice is **vLLM vs SGLang** (both Apache 2.0, both feature-complete, both ROCm-capable), decided by the workload's prefix-sharing fraction and whether the program DSL is valuable (File 08 §36, File 11 §8). TensorRT-LLM is the NVIDIA-peak option; TGI the HF option; the rest serve specific niches (edge, local, ecosystem).

---

## 10. Architectural Philosophies Compared

The frameworks embody different design philosophies, worth understanding because they explain the trade-offs:

- **Compiled vs interpreted:** TensorRT-LLM *compiles* models ahead of time into optimized engines — maximizing performance for a fixed model at the cost of flexibility (recompile to change anything) and per-model porting effort. vLLM/SGLang are *interpreted* (Python model definitions, dynamic execution) — flexible (modify model code, add models easily, change config without recompiling) at a small performance cost (mitigated by CUDA graphs, fused kernels, FlashInfer). The compiled-vs-interpreted axis is the core trade-off: TRT-LLM's peak performance vs vLLM/SGLang's flexibility and breadth. For a stable, supported, performance-critical model, compilation wins; for a fast-moving environment (new models, experimentation, broad support), interpretation wins. This is why vLLM/SGLang dominate the open-source ecosystem (flexibility matches the fast-moving open-model landscape) while TRT-LLM serves performance-critical stable deployments.

- **Request-level vs program-level:** vLLM (and TGI, TRT-LLM) treat requests as independent — the engine sees opaque requests and optimizes them (continuous batching, prefix caching). SGLang adds a *program-level* view (the frontend DSL, File 08 §12) — it sees the structure of multi-call programs and optimizes across calls (structural batching, RadixAttention reuse). The program-level view is SGLang's unique architectural bet (File 08 §36): for structured multi-call workloads, knowing the program structure enables optimizations a request-level engine can't see. For independent single-shot requests, the request-level view suffices. This is the deepest architectural difference between SGLang and the others.

- **Portability vs specialization:** MLC-LLM (TVM, compile to any backend) prioritizes portability over peak performance on any one backend. TRT-LLM specializes maximally for NVIDIA. vLLM/SGLang are in between (NVIDIA + AMD via Triton, optimized but not TVM-universal). The portability-vs-specialization axis distinguishes the edge/cross-platform engines (MLC-LLM) from the server-class NVIDIA-focused ones (TRT-LLM) from the multi-vendor server engines (vLLM/SGLang).

These philosophies — compiled/interpreted, request/program-level, portable/specialized — explain the frameworks' positions: TRT-LLM (compiled, request-level, NVIDIA-specialized: peak NVIDIA performance for stable models), vLLM (interpreted, request-level, multi-vendor: flexible broad serving), SGLang (interpreted, program-level, multi-vendor: structured-workload serving), MLC-LLM (compiled, portable: cross-platform/edge), TGI (interpreted, request-level, HF-integrated: HF ecosystem). The philosophy determines the trade-off, and the trade-off determines the use case.

---

## 11. The Convergence and Divergence Trends

The frameworks both converge and diverge over time:

**Convergence (the systems fundamentals are settling):** all serious server engines now have continuous batching, paged KV, chunked prefill, FP8/quantization, and (increasingly) the multi-process overlap architecture (File 09 §22). vLLM and SGLang converged on nearly identical systems architectures (File 09 §22). TRT-LLM adopted continuous batching (in-flight batching) from the vLLM ecosystem; vLLM/SGLang adopted FP8 and speculative decoding (TRT-LLM's strengths). XGrammar (SGLang's, File 08 §11) was adopted by vLLM as a guided-decoding backend. So the *systems fundamentals* are converging — the field agrees on how to build a serving engine (paged KV, continuous batching, chunked prefill, FlashInfer, CUDA graphs, multi-process overlap).

**Divergence (the differentiators):** the frameworks differentiate on the higher-level choices — SGLang's RadixAttention and program DSL (no vLLM equivalent), TRT-LLM's compilation (peak performance, no flexibility), MLC-LLM's portability, TGI's HF integration. These differentiators persist because they reflect different bets (program-level optimization, compilation, portability, ecosystem) that suit different workloads/contexts.

The pattern (File 08 §36, File 09 §36): the systems architecture converges (settled best practices, cross-pollinated), while the differentiating features (prefix-caching policy, frontend, compilation, portability) persist as the frameworks compete on their bets. For the user, this means the *baseline* (continuous batching, paged KV, quantization) is available everywhere, and the *choice* is about the differentiators (SGLang's structured-workload edge, TRT-LLM's NVIDIA peak, etc.). The convergence is healthy — it means the hard systems lessons are widely shared — and the divergence is healthy too — it means the frameworks push different frontiers, cross-pollinating the best ideas (XGrammar adopted, continuous batching spread, FP8 everywhere).

---

## 12. Benchmarking Caveats

Cross-framework benchmark comparisons are fraught (File 11 §8.1), and the caveats matter:

- **Workload-dependent:** a framework's relative performance depends heavily on the workload (prefix-sharing → SGLang; broad models → vLLM; supported stable model → TRT-LLM). A benchmark on one workload doesn't transfer.
- **Version-dependent:** all frameworks improve monthly; a benchmark showing framework A 20% faster can reverse next release. Benchmark numbers age quickly.
- **Config-dependent:** the tuning (File 11) matters — a poorly-tuned vLLM vs a well-tuned SGLang isn't a fair comparison. Both must be tuned for the workload.
- **Cache-state-dependent:** for prefix-sharing workloads, cold vs warm cache hugely affects results (File 11 §6.3, §32) — a cold benchmark misses RadixAttention's benefit.
- **Hardware-dependent:** the framework's kernel maturity on the hardware (NVIDIA most mature; AMD ~85–90%) affects results.

So treat cross-framework benchmarks skeptically — they're snapshots of specific workloads, versions, configs, and hardware. The right approach (File 11 §6, §8.1): benchmark the *candidate frameworks* on *your workload* with *realistic traffic and cache warming*, tuned for each, on your hardware. Don't rely on published comparisons (they may not match your workload/version). The frameworks are close enough on fundamentals that the workload-specific benchmark, not a generic comparison, should decide. This is why the database emphasizes mechanisms over benchmark numbers — understanding *why* SGLang suits prefix-sharing (RadixAttention) or vLLM suits breadth (ecosystem) lets you predict the right choice and verify it on your workload, rather than trusting a benchmark that may not transfer.

---

## 13. Orchestration Layers and Other Engines

Beyond the core engines, the ecosystem has orchestration and packaging layers:

- **Triton Inference Server (NVIDIA):** a general model-serving *orchestration* layer that can front any backend (TensorRT-LLM, vLLM via a backend, ONNX, PyTorch) — handles multi-model serving, request routing, metrics, dynamic batching at the orchestration level. Often used to deploy TensorRT-LLM. It's an orchestration layer, not an inference engine itself — it manages and routes to engines (File 19 §multi-model).
- **OpenLLM (BentoML):** a packaging/ops layer that wraps inference engines (including vLLM) with BentoML's deployment tooling — for productionizing/packaging models.
- **Xinference:** a multi-model serving framework (manage and serve many models) that can use various backends.
- **Ray Serve:** distributed serving orchestration (multi-replica, multi-model, autoscaling) that integrates vLLM (File 07 §8.1).
- **LiteLLM:** a unifying proxy/gateway providing an OpenAI-compatible interface over many providers/engines (routing, fallback, rate limiting — the gateway layer, File 07 §15).

These layers sit *above* the inference engines (vLLM/SGLang/TRT-LLM), handling orchestration (multi-model, routing, scaling, gateway) while the engines do the inference. The distinction: engines (vLLM, SGLang, TRT-LLM, TGI) run the model forward pass with the serving optimizations (Files 03–11); orchestration layers (Triton, Ray Serve, OpenLLM, LiteLLM) manage multiple engines/models, routing, scaling, and the gateway functions (File 07 §15, File 19 §multi-model). A production deployment often combines them: an inference engine (vLLM/SGLang) wrapped in an orchestration layer (Ray Serve, Triton) behind a gateway (LiteLLM/custom) — the engine for fast inference, the orchestration for multi-model/scaling, the gateway for routing/auth/limits. Knowing this layering clarifies the ecosystem: the inference engine is one component (the focus of this database); the orchestration and gateway are complementary layers (Files 07, 19).

---

## 14. Open-Source vs Closed-Source Dynamics

The open-source (vLLM, SGLang, TGI) vs closed-source (TensorRT-LLM) dynamic is a key landscape feature (File 20 §competitive dynamics):

- **TensorRT-LLM's edge:** best raw FLOP utilization on NVIDIA for supported models — NVIDIA's deep hardware knowledge and dedicated optimization.
- **TensorRT-LLM's disadvantages:** NVIDIA-only (no AMD/Intel), limited model support (porting effort), low extensibility (compiled), and NVIDIA's incentive to favor its hardware.
- **Open-source's edge:** AMD/Intel support (vLLM/SGLang on ROCm), rapid new-model support (community adds models fast), extensibility (Python, modify freely), community contributions (hundreds of contributors), and hardware independence.
- **The convergence:** open-source (vLLM/SGLang) adopts TRT-LLM's ideas (FP8, speculation); TRT-LLM adopts the open-source ecosystem's (continuous batching). The gap in raw performance narrows as open-source kernels (FlashInfer, Marlin) mature.

The trajectory (File 20): open-source serving (vLLM/SGLang) has won the breadth, extensibility, and hardware-flexibility dimensions, while TRT-LLM retains a raw-performance edge for supported NVIDIA models. For most users — who value broad model support, hardware flexibility (AMD option), extensibility, and the community ecosystem — open-source (vLLM/SGLang) is the choice; for NVIDIA-locked, performance-critical, stable-model deployments, TRT-LLM's peak performance can justify its constraints. The broader dynamic (File 20 §moat): open-source inference is a competitive moat for companies building on open-weight models — mastery of vLLM/SGLang (this database) is what makes open-model serving cost-competitive, and the open-source engines' rapid improvement (driven by the community and the hyperscalers contributing) keeps them competitive with (and often ahead of, on flexibility) the closed alternatives. The open-source serving stack is central to the open-weight-model ecosystem (File 20).

---

## 15. Synthesis

The LLM serving framework landscape spans server-class engines (vLLM, SGLang, TensorRT-LLM, TGI, DeepSpeed-MII, LMDeploy), portability/compilation stacks (MLC-LLM), local/edge runtimes (Ollama, llama.cpp), and orchestration layers (Triton, Ray Serve). The server-class open-source engines — **vLLM** (broad, ecosystem-leading) and **SGLang** (structured-workload-optimized, cutting-edge) — dominate open-source serving, having converged on the systems fundamentals (continuous batching, paged KV, chunked prefill, FlashInfer, CUDA graphs, multi-process overlap) while differentiating on prefix caching (RadixAttention vs APC) and the frontend (SGLang's DSL). **TensorRT-LLM** offers peak NVIDIA performance (compiled, NVIDIA-only, limited models); **TGI** offers HF integration; the rest serve niches (MLC-LLM portability, Ollama local, LMDeploy AWQ). The choice is workload-, hardware-, and context-driven (§9): vLLM as the broad default, SGLang for prefix-sharing/structured/agentic workloads, TRT-LLM for NVIDIA-peak stable models, the others for their niches. The convergence (systems fundamentals) and divergence (differentiating bets) pattern (§11), the architectural philosophies (compiled/interpreted, request/program-level, portable/specialized, §10), and the open-source-vs-closed dynamics (§14) explain the landscape. For the inference engineer, knowing the frameworks' strengths and trade-offs enables choosing the right one for a workload — and the deep understanding of the *mechanisms* (Files 03–11) is what makes the choice informed (predicting which framework's bets suit the workload) and the deployment effective (tuning the chosen engine, Files 11). The framework is the vehicle; the mechanisms (and the tuning methodology) are what make it deliver — and this database's focus on vLLM and SGLang reflects their dominance of the open-source server-class serving that most LLM applications are built on.

---

## 16. The Bottom Line

If forced to a single recommendation for most open-source server-class serving: **start with vLLM** (broadest support, largest ecosystem, production-hardened, the safe default), and **consider SGLang** if your workload is prefix-sharing-heavy, structured-output-heavy, or agentic (where RadixAttention + XGrammar + the program DSL give it a real edge). Benchmark both on your workload (File 11 §6, §8) — they're close on fundamentals, and the workload decides. Reach for **TensorRT-LLM** only if you need peak NVIDIA performance for a stable supported model and can accept the compilation/flexibility/lock-in trade-offs. Use **TGI** for HF-ecosystem integration, **MLC-LLM/Ollama** for edge/local, and orchestration layers (**Triton/Ray Serve**) above whichever engine for multi-model/scaling. This recommendation reflects the landscape's reality: vLLM and SGLang are the two dominant open-source server engines (the focus of this database), excellent and converging, differentiated by workload fit; the others serve specific needs. The deep mechanism knowledge (Files 03–15) is what lets you make and execute this choice well — understanding *why* each framework suits what it suits, and tuning the chosen one to the workload's goodput-per-dollar optimum (File 11). The framework choice is important but secondary to the mechanism mastery that makes any chosen framework perform — which is why this database invests most in the mechanisms (Files 03–11) and treats the framework comparison (this file) as the application of that understanding to the selection decision.

---

## 17. A Worked Framework Selection

Apply the decision guide (§9) to three scenarios:

**Scenario A — a startup serving a chatbot on open models (LLaMA/Qwen), moderate scale, AMD or NVIDIA GPUs.** Choice: **vLLM** (broad model support, ROCm option, OpenAI API, large ecosystem for support) — or SGLang if the chatbot has a long shared system prompt and multi-turn history (RadixAttention's edge). Both work; vLLM is the safe default, SGLang if prefix-sharing is significant. Benchmark both (File 11 §8).

**Scenario B — an agentic platform (many tool-using LLM calls, structured JSON output, multi-step).** Choice: **SGLang** — this is exactly its sweet spot (RadixAttention for the shared growing context across steps, XGrammar for the tool-call JSON, the program DSL for the multi-call structure, File 15 §20). The agentic, structured, multi-call workload is where SGLang's differentiators shine (File 08 §36).

**Scenario C — a large enterprise serving a fixed, stable model at maximum performance on a dedicated NVIDIA cluster.** Choice: **TensorRT-LLM** (compiled for peak NVIDIA performance, the model is stable so the compilation overhead is amortized, NVIDIA-locked is acceptable) — possibly behind Triton for orchestration. The peak-performance, stable-model, NVIDIA-dedicated context is where TRT-LLM's trade-offs (compilation, NVIDIA-only, limited flexibility) are acceptable for the performance gain.

These scenarios illustrate the decision drivers in action: the workload (chatbot/agentic/stable), the hardware (AMD option, NVIDIA-dedicated), the priorities (breadth, structured-workload performance, peak performance), and the context (startup flexibility vs enterprise stability). The same model could be served by different frameworks depending on these factors — there's no universal "best framework," only the best for the workload, hardware, and context. The mechanism understanding (Files 03–15) is what lets you map the scenario to the right framework (predicting which framework's bets suit) and then tune it (File 11). This worked selection is the framework-comparison analog of the tuning playbook (File 11 §12): characterize the situation, match it to the framework whose design suits, and validate by benchmark — the same disciplined, situation-driven approach applied to framework selection rather than parameter tuning. Framework choice, like every other inference decision, is made by matching the design to the workload and validating by measurement, grounded in understanding the mechanisms.

---

## 18. Closing

The framework landscape, surveyed here, places vLLM and SGLang as the dominant open-source server-class engines — the focus of this database — among a broader ecosystem of compiled (TensorRT-LLM), HF-integrated (TGI), portable (MLC-LLM), local (Ollama), and orchestration (Triton, Ray Serve) options. The frameworks converge on the systems fundamentals (continuous batching, paged KV, FlashInfer, CUDA graphs) and diverge on their differentiating bets (RadixAttention, compilation, portability, ecosystem), embodying different philosophies (compiled/interpreted, request/program-level, portable/specialized). The selection is workload-, hardware-, and context-driven, decided by matching the framework's design to the situation and validating by benchmark — vLLM the broad default, SGLang for structured/prefix-sharing/agentic workloads, the others for their niches. The deep mechanism understanding built across Files 03–15 is what makes the framework choice informed and the deployment effective: it lets you predict which framework suits a workload, choose accordingly, and tune the chosen engine to its optimum. The framework is the vehicle for the mechanisms; this comparison helps choose the vehicle, but the mechanisms (and the tuning methodology, File 11) are what make it perform. With the frameworks compared, the remaining files turn to LoRA serving (File 18), production operations (File 19), and the business/ecosystem context (File 20) — completing the practical picture of LLM inference engineering from mechanisms through frameworks to operations and business. The choice of framework matters, but it is the understanding beneath it — of why each engine works as it does — that distinguishes an engineer who can merely pick a framework from one who can wield it to its full potential, extend it, or, when the field moves, recognize and adopt whatever supersedes it. That depth — understanding the engines well enough to choose, tune, extend, and evolve with them — is the capability this database builds, and the framework comparison is one facet of it: not a verdict to memorize, but a structured way to reason about which tool fits which job, informed by the mechanisms that make each tool what it is — the same mechanisms this database has detailed at length, here applied to the practical act of choosing among the tools that implement them for a given workload, hardware, and organizational context — a decision that, made well, rests entirely on the mechanism understanding the rest of the database provides, and which therefore repays the investment in that understanding many times over across the lifetime of any serious inference deployment, from the initial framework choice through every subsequent tuning, scaling, and migration decision it informs over the deployment's evolving life as models, traffic, hardware, and the frameworks themselves continually change around it year after year.


