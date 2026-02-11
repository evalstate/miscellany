---USER
# openai/gpt-oss-120b

## Overview
- **Author:** openai
- **Task:** text-generation
- **Library:** transformers
- **Downloads:** 8.0M | **Likes:** 4002
- **Updated:** 26 Aug, 2025

## Technical Details
- **Model Class:** AutoModelForCausalLM
- **Parameters:** 120412.3M
- **Architecture:** gpt_oss

## Tags
`transformers` `safetensors` `gpt_oss` `text-generation` `vllm` `conversational` `arxiv:2508.10925` `license:apache-2.0` `autotrain_compatible` `endpoints_compatible` `8-bit` `mxfp4` `region:us`

## Metadata
- **License:** apache-2.0

## Demo Spaces
- [FLUXllama](https://hf.co/spaces/ginipick/FLUXllama)
- [gpt-oss-120b-chatbot](https://hf.co/spaces/amd/gpt-oss-120b-chatbot)
- [open_pt_llm_leaderboard](https://hf.co/spaces/eduagarcia/open_pt_llm_leaderboard)
- [SimpleChatbot](https://hf.co/spaces/ngebodh/SimpleChatbot)
- [OpenAI-gpt-oss](https://hf.co/spaces/openfree/OpenAI-gpt-oss)
- [VayuChat](https://hf.co/spaces/SustainabilityLabIITGN/VayuChat)
- [openai-gpt-oss-120b-test](https://hf.co/spaces/abidlabs/openai-gpt-oss-120b-test)
- [TuRTLe-Leaderboard](https://hf.co/spaces/HPAI-BSC/TuRTLe-Leaderboard)
- [gpt-oss-RAG](https://hf.co/spaces/VIDraft/gpt-oss-RAG)
- [chatui-helper](https://hf.co/spaces/milwright/chatui-helper)
- [gpt-oss-news-agent](https://hf.co/spaces/fdaudens/gpt-oss-news-agent)
- [openwebui](https://hf.co/spaces/umint/openwebui)
- *... and 88 more spaces*

## Inference Providers
- **cerebras** (live)
- **sambanova** (live)
- **novita** (live)
- **fireworks-ai** (live)
- **nebius** (live)
- **hyperbolic** (live)
- **together** (live)
- **scaleway** (live)
- **nscale** (live)
- **groq** (live)

Try this model in the [playground](https://hf.co/playground?modelId=openai/gpt-oss-120b)

**Link:** [https://hf.co/openai/gpt-oss-120b](https://hf.co/openai/gpt-oss-120b)

## README
<modelcard-readme>

<p align="center">
  <img alt="gpt-oss-120b" src="https://raw.githubusercontent.com/openai/gpt-oss/main/docs/gpt-oss-120b.svg">
</p>

<p align="center">
  <a href="https://gpt-oss.com"><strong>Try gpt-oss</strong></a> ·
  <a href="https://cookbook.openai.com/topic/gpt-oss"><strong>Guides</strong></a> ·
  <a href="https://arxiv.org/abs/2508.10925"><strong>Model card</strong></a> ·
  <a href="https://openai.com/index/introducing-gpt-oss/"><strong>OpenAI blog</strong></a>
</p>

<br>

Welcome to the gpt-oss series, [OpenAI’s open-weight models](https://openai.com/open-models) designed for powerful reasoning, agentic tasks, and versatile developer use cases.

We’re releasing two flavors of these open models:
- `gpt-oss-120b` — for production, general purpose, high reasoning use cases that fit into a single 80GB GPU (like NVIDIA H100 or AMD MI300X) (117B parameters with 5.1B active parameters)
- `gpt-oss-20b` — for lower latency, and local or specialized use cases (21B parameters with 3.6B active parameters)

Both models were trained on our [harmony response format](https://github.com/openai/harmony) and should only be used with the harmony format as it will not work correctly otherwise.


> [!NOTE]
> This model card is dedicated to the larger `gpt-oss-120b` model. Check out [`gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) for the smaller model.

# Highlights

* **Permissive Apache 2.0 license:** Build freely without copyleft restrictions or patent risk—ideal for experimentation, customization, and commercial deployment.  
* **Configurable reasoning effort:** Easily adjust the reasoning effort (low, medium, high) based on your specific use case and latency needs.  
* **Full chain-of-thought:** Gain complete access to the model’s reasoning process, facilitating easier debugging and increased trust in outputs. It’s not intended to be shown to end users.  
* **Fine-tunable:** Fully customize models to your specific use case through parameter fine-tuning.
* **Agentic capabilities:** Use the models’ native capabilities for function calling, [web browsing](https://github.com/openai/gpt-oss/tree/main?tab=readme-ov-file#browser), [Python code execution](https://github.com/openai/gpt-oss/tree/main?tab=readme-ov-file#python), and Structured Outputs.
* **MXFP4 quantization:** The models were post-trained with MXFP4 quantization of the MoE weights, making `gpt-oss-120b` run on a single 80GB GPU (like NVIDIA H100 or AMD MI300X) and the `gpt-oss-20b` model run within 16GB of memory. All evals were performed with the same MXFP4 quantization.

---

# Inference examples

## Transformers

You can use `gpt-oss-120b` and `gpt-oss-20b` with Transformers. If you use the Transformers chat template, it will automatically apply the [harmony response format](https://github.com/openai/harmony). If you use `model.generate` directly, you need to apply the harmony format manually using the chat template or use our [openai-harmony](https://github.com/openai/harmony) package.

To get started, install the necessary dependencies to setup your environment:

```
pip install -U transformers kernels torch 
```

Once, setup you can proceed to run the model by running the snippet below:

```py
from transformers import pipeline
import torch

model_id = "openai/gpt-oss-120b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
```

Alternatively, you can run the model via [`Transformers Serve`](https://huggingface.co/docs/transformers/main/serving) to spin up a OpenAI-compatible webserver:

```
transformers serve
transformers chat localhost:8000 --model-name-or-path openai/gpt-oss-120b
```

[Learn more about how to use gpt-oss with Transformers.](https://cookbook.openai.com/articles/gpt-oss/run-transformers)

## vLLM

vLLM recommends using [uv](https://docs.astral.sh/uv/) for Python dependency management. You can use vLLM to spin up an OpenAI-compatible webserver. The following command will automatically download the model and start the server.

```bash
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

vllm serve openai/gpt-oss-120b
```

[Learn more about how to use gpt-oss with vLLM.](https://cookbook.openai.com/articles/gpt-oss/run-vllm)

## PyTorch / Triton

To learn about how to use this model with PyTorch and Triton, check out our [reference implementations in the gpt-oss repository](https://github.com/openai/gpt-oss?tab=readme-ov-file#reference-pytorch-implementation).

## Ollama

If you are trying to run gpt-oss on consumer hardware, you can use Ollama by running the following commands after [installing Ollama](https://ollama.com/download).

```bash
# gpt-oss-120b
ollama pull gpt-oss:120b
ollama run gpt-oss:120b
```

[Learn more about how to use gpt-oss with Ollama.](https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama)

#### LM Studio

If you are using [LM Studio](https://lmstudio.ai/) you can use the following commands to download.

```bash
# gpt-oss-120b
lms get openai/gpt-oss-120b
```

Check out our [awesome list](https://github.com/openai/gpt-oss/blob/main/awesome-gpt-oss.md) for a broader collection of gpt-oss resources and inference partners.

---

# Download the model

You can download the model weights from the [Hugging Face Hub](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) directly from Hugging Face CLI:

```shell
# gpt-oss-120b
huggingface-cli download openai/gpt-oss-120b --include "original/*" --local-dir gpt-oss-120b/
pip install gpt-oss
python -m gpt_oss.chat model/
```

# Reasoning levels

You can adjust the reasoning level that suits your task across three levels:

* **Low:** Fast responses for general dialogue.  
* **Medium:** Balanced speed and detail.  
* **High:** Deep and detailed analysis.

The reasoning level can be set in the system prompts, e.g., "Reasoning: high".

# Tool use

The gpt-oss models are excellent for:
* Web browsing (using built-in browsing tools)
* Function calling with defined schemas
* Agentic operations like browser tasks

# Fine-tuning

Both gpt-oss models can be fine-tuned for a variety of specialized use cases.

This larger model `gpt-oss-120b` can be fine-tuned on a single H100 node, whereas the smaller [`gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) can even be fine-tuned on consumer hardware.

# Citation

```bibtex
@misc{openai2025gptoss120bgptoss20bmodel,
      title={gpt-oss-120b & gpt-oss-20b Model Card}, 
      author={OpenAI},
      year={2025},
      eprint={2508.10925},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.10925}, 
}
```
</modelcard-readme>
---ASSISTANT
Thank you. I will refer to this model card in my future responses
---USER
Here is additional research information about the model:

```


<model_research>
   <research source="https://hf.co/openai/gpt-oss-120b">
Key takeaways for model researchers (beyond the usual model-card boilerplate)

- MoE with unusually low active compute: 117B total parameters but only 5.1B active per token (~4.4% of weights used). The 20B variant is 21B total/3.6B active (~17%). This split hints at different expert/gating regimes across sizes and invites study of sparsity–performance tradeoffs at scale.

- Quantization is first-class, not an afterthought: MoE weights are post-trained in MXFP4, and all evals were run with the same MXFP4 quantization. This is rare for open-weight releases and makes the model a strong target for quantization-aware benchmarking and kernel research (memory/latency benefits without reported eval regression here).

- Full chain-of-thought is intentionally exposed. That’s atypical at this scale and opens avenues for research on rationale supervision, interpretability, and the effect of test-time “reasoning effort” controls on CoT style/length and task performance.

- Test-time compute knob via prompt: “Reasoning: low/medium/high” is an explicit interface for variable inference-time effort. This is a clean target for studying adaptive compute, dynamic CoT length, and latency–quality tradeoff curves without custom decoding hacks.

- Harmony format is mandatory: the model expects a specific conversation schema (Transformers chat template auto-applies it; otherwise use the openai-harmony package). This suggests the pretraining/SFT heavily couples capability to schema—worth noting for finetune/adapter work and prompt robustness studies.

- Single-80GB GPU practicality at 120B: Thanks to MoE + MXFP4, 120B fits on one H100/MI300X. This lowers the barrier for reproducible large-scale experiments (including fine-tuning on a single H100 node), a practical shift for academic labs.

- Engine/runtime support is not stock: vLLM requires a special wheel (0.10.1+gptoss from a custom index) and PyTorch nightly CU128. This indicates custom kernels/graph assumptions for MXFP4/MoE—fertile ground for systems research and throughput comparisons.

- Ecosystem traction signals: millions of monthly downloads, 100 Spaces using the model, and dozens of adapters/finetunes already on Hub. Rapid derivative activity makes it a convenient backbone for comparative finetuning and merge-method research.

- Tool-native framing: the repo highlights built-in web browsing and Python execution hooks plus Structured Outputs, pointing to an “agent-first” usage pattern. Useful for studying grounded/tool-augmented evals where many open models lack sanctioned tool paths.

If you want a few quick experiments to run
- Measure accuracy/latency as a function of Reasoning level and CoT verbosity; correlate with active token count and failure modes.
- Compare FP16 vs MXFP4 (where possible) to quantify quantization effects on long-context reasoning and multi-step tool-use chains.
- Stress-test robustness to non-Harmony prompt formats and evaluate adapter training to relax schema dependence without loss.
   </research>
   <research source="https://gpt-oss.com">
Here’s what’s notable “under the hood” about gpt-oss.com that an AI model researcher would likely find interesting or surprising:

- The playground authenticates with Hugging Face and requests Inference API scope
  - The site initiates OAuth to Hugging Face (client_id: 8fb6b216-1579-41b0-99e5-7b561bef1fa3) with scopes openid profile email inference-api and redirect_uri https://api.gpt-oss.com/auth/callback.
  - Strong implication: inference for the open‑weight models (gpt‑oss‑120b and gpt‑oss‑20b) is performed via Hugging Face’s Inference API/Endpoints rather than an OpenAI‑hosted stack, or at minimum, HF tokens are involved in gating/authorizing use.

- It ships with an (internal) chain‑of‑thought viewer toggle and a “Reasoning Effort” control
  - The app’s state includes showReasoning (default false) and a reasoningEffort setting (default high). That suggests the models can emit a reasoning trace (likely hidden by default) and that the UI exposes a knob to increase/decrease test‑time compute/“effort.”
  - This is a concrete signal that gpt‑oss models support a structured “reasoning” channel and test‑time scaling, not just final answers.

- The UI is wired to an OpenAI ChatKit client, hinting API compatibility
  - The page loads https://cdn.platform.openai.com/deployments/chatkit/0.7/chatkit.js. This suggests the playground (and potentially gpt‑oss APIs) are compatible with the OpenAI Chat Completions–style client/streaming patterns, enabling drop‑in usage with existing tooling.

- Default model targeting emphasizes the 120B variant
  - The persisted client state defaults model to gpt‑oss‑120b (with gpt‑oss‑20b also present). That choice, plus the “reasoning effort: high” default, signals the playground is explicitly tuned to showcase the larger model’s capabilities.

- User session and preferences are locally persisted and themed for developer ergonomics
  - The app uses a persisted zustand store (key: gpt‑oss) to retain model choice, theme (respects system dark mode), and reasoning controls—indicating the team expects comparative experimentation and repeat usage by developers.

Why this matters
- These details point to a design that treats “open‑weight reasoning LLMs” as API‑first, ChatCompletions‑compatible models, with gating via Hugging Face auth and explicit UI affordances for chain‑of‑thought and test‑time compute scaling. If you’re building evaluation harnesses or planning deployment, this suggests:
  - You can likely reuse OpenAI‑compatible client libraries with gpt‑oss.
  - You can expose/measure reasoning vs. final‑answer behaviors and control compute/latency tradeoffs via “effort.”
  - Access and rate‑limiting/logging may rely on HF identities/tokens, which is unusual for “open‑weight” demos and implies a hybrid access model (open weights, but managed inference and policy via HF).
   </research>
   <research source="https://cookbook.openai.com/topic/gpt-oss">
Key takeaways for AI model researchers from the gpt-oss topic hub (beyond a typical model card):

- There is a 20B-parameter gpt-oss variant with first-party deployment guides across multiple inference stacks, including vLLM, Hugging Face Transformers, and NVIDIA TensorRT-LLM. The explicit TensorRT-LLM path for 20B suggests an intended, optimized high-throughput/server deployment track, not just hobbyist/local use.

- OpenAI Harmony Response Format appears as a unifying output schema referenced alongside gpt-oss. This signals a push toward standardized response formatting across open-weight and API-served models—useful for tool-calling, safety layers, and multi-model orchestration.

- “Verifying gpt-oss implementations” targets third-party hosts/providers. The existence of verification guidance implies a conformance mindset (behavioral/spec compatibility), which is atypical in open-weight releases and may ease interoperability and benchmarking.

- “How to handle the raw chain of thought in gpt-oss” is called out explicitly. That OpenAI provides guidance here is notable—indicating recommended practices for CoT exposure/suppression or post-processing, relevant to safety, evals, and instruction tuning.

- Official local-runtime coverage includes both LM Studio and Ollama, in addition to server stacks. This breadth of first-party examples points to deliberate ecosystem enablement across consumer and production contexts.

- Language-targeted fine-tuning guidance (example: improving Korean) is highlighted, suggesting an expectation that domain/language specialization is a primary, supported path to closing capability gaps—useful for region- or task-specific research agendas.

- Colab-based recipes for gpt-oss-20b indicate a minimum viable setup for experimentation without dedicated hardware, lowering the barrier to quick-turn benchmarking or ablation studies.

Implications:
- Standardization (Harmony + verification) could materially reduce the integration tax for comparing/combining open and closed models.
- Provided CoT handling guidance suggests a canonical stance on reasoning-trace management that researchers can align with for reproducible evals and safer deployments.
- The explicit TensorRT-LLM route for 20B hints at performance-oriented kernels and batching strategies being a first-class concern for this series.
   </research>
   <research source="https://arxiv.org/abs/2508.10925">
Here are the non-obvious, researcher-relevant takeaways and implications from arXiv:2508.10925 that go beyond standard model-card boilerplate:

Architecture and efficiency innovations
- MoE-centric memory hack that matters: They post-train quantize only the MoE weights to MXFP4 (4.25 bits/param), which constitute 90%+ of parameters, letting the 116.8B model fit on a single 80GB GPU and the 20.9B on 16GB. This is a very targeted quantization strategy that preserves dense attention/embeds while squeezing experts—worth replicating.
- Active parameters are small: Despite 116.8B total, only ~5.1B params are active per token (top-4-of-128 experts). This is a clean datapoint for “sparse ≈ dense” reasoning performance comparisons at similar active compute.
- Attention can choose “no attention”: Each head has a learned bias in the softmax denominator (akin to off-by-one/sink tricks) that enables zero-attend. This is unusual and a neat mechanism to avoid forced distribution over tokens; candidates for ablation/analysis on stability and long-context routing.
- Unconventional SwiGLU: Their SwiGLU includes clamping plus a residual inside the activation. That’s atypical; could be stabilizing MoE expert dynamics. Worth probing against vanilla SwiGLU on deep MoE stacks.
- Long-context with hybrid sparsity: Alternating banded (window=128) and dense attention, GQA (64Q/8KV), RoPE, and YaRN to 131,072 tokens. The specific alternation plus the zero-attend bias gives a concrete recipe for long context without massive memory.

Training and prompting mechanics
- Variable-effort reasoning as a first-class control: “Reasoning: low|medium|high” in the system prompt reliably scales accuracy with CoT length (smooth, log-linear-like scaling). The 20B averages >20k CoT tokens per AIME question at high effort (!). This exposes an explicit, cost-latency-accuracy dial you can standardize in evals and products.
- Harmony chat format as an agentic substrate: Roles (system/developer/user) plus channels (analysis/commentary/final) designed for CoT segregation and interleaved function calls. Key operational guidance: strip prior-turn assistant CoT in multi-turn dialogs to avoid leakage/degeneration. If you’re studying CoT hygiene or tool-use interleaving, this is a concrete, open implementation to copy/measure.

Open-weight + safety stance with concrete findings
- No direct anti-CoT optimization: They intentionally avoid applying pressure on CoT content (to preserve monitorability), noting CoT can hallucinate policy-violating text. This is a clear position in the current CoT-monitoring debate and gives you an open model to test CoT-based oversight ideas without counter-optimization.
- Instruction hierarchy weakness vs jailbreak parity: gpt-oss lags o4-mini on instruction-hierarchy defenses (system/developer vs user conflicts) but matches it on StrongReject jailbreak robustness. Translation: don’t rely on system-prompt-only guardrails; consider fine-tuning or external scaffolds for instruction priority.
- Preparedness takeaway that’s actually actionable: Even with adversarial fine-tuning (helpful-only RL, in-domain bio/cyber data, browsing) using OpenAI’s RL stack, they did not reach “High” capability thresholds; moreover, newly released open models (Qwen 3 Thinking, Kimi K2) already approach adversarially fine-tuned gpt-oss-120b on several biosecurity evals. If you study risk from open weights, this calibrates marginal risk from this release and provides a comparison set.
- Cyber range reality check: Zero unaided solves on realistic end-to-end cyber ranges (even with hints). This suggests current agentic scaffolds, not just base capability, are the limiting factor for fully chained operations; a ripe area for methods work.

Performance signals with research hooks
- HealthBench near-o3 at “high” reasoning: 120B nearly matches o3 on HealthBench/Hard while being open-weight. For researchers in clinical/LLM safety, this is a rare open model operating close to strong closed baselines—useful for reproducible studies.
- Tool-use deltas are large: Big jumps with terminal/browsing for coding and knowledge tasks, consistent with agent-first training. A good testbed for “to browse or not” policies and dynamic tool gating under the Harmony channels.

Concrete replication ideas
- Try “MoE-only MXFP4” quantization ablations: Does quantizing just experts capture most benefits across other MoE LLMs? What’s the inflection at 4.25 vs 3–4 bits? Any effects on expert collapse or routing entropy?
- Remove or vary the softmax-denominator bias: Measure impacts on attention entropy, head specialization, long-context stability, and catastrophic attention failures.
- Stress-test variable-effort reasoning: Build per-task effort schedules to minimize tokens at constant accuracy; could be a strong test-time optimization baseline vs more complex routing.
- CoT monitoring without counter-pressure: Use gpt-oss to evaluate whether monitoring-based defenses generalize better when the model hasn’t been trained to hide “bad thoughts.”

Small but useful stats
- 2.1M H100-hours to train the 120B; ~10x fewer for 20B. Helpful for cost benchmarking sparse vs dense.
- Checkpoint sizes post-quant: 60.8 GiB (120B), 12.8 GiB (20B).

Links they opened
- Tokenizer and Harmony format are open (TikToken o200k_harmony; harmony guide) with function-calling schemas and tool harnesses—practical substrate for agent research and reproducibility.
   </research>
   <research source="https://openai.com/index/introducing-gpt-oss/">
Here are the non-obvious, research-relevant takeaways from the blog post that go beyond a typical model card:

- Full chain-of-thought exposure, with a deliberate “no direct CoT supervision” policy: OpenAI explicitly states the models output full CoT and that CoT was not directly supervised during alignment to enable CoT-based monitoring for misbehavior. The post also (via example rollouts) notes a notable behavior: the model can follow system instructions in final answers while explicitly disobeying them inside its CoT. This is an unusual stance and dataset signal profile for an open-weight release and is directly relevant to CoT safety/interpretability research and to studying instruction adherence across “inner vs outer” channels.

- Built-in, controllable “reasoning effort” modes in an open-weight model: The models expose low/medium/high reasoning effort settings (latency vs performance trade-off) controllable via simple system messaging—bringing an o-series-style runtime control into open weights. This is useful for research on dynamic compute allocation, latency-aware inference, and evaluation of budgeted reasoning policies outside a proprietary API.

- Worst-case, adversarial fine-tuning safety methodology with external review: OpenAI adversarially fine-tuned domain-specific, non-refusing variants (e.g., bio/cybersecurity) using its internal stack and evaluated them under its Preparedness Framework. They report these maliciously fine-tuned models did not achieve high capability thresholds; the methodology was reviewed by three independent expert groups with adopted recommendations. This provides a rare, concrete baseline of “how far” worst-case fine-tuning moves an open-weight capable model under strong resources—useful for policy and defense research.

- Community red teaming with a substantial prize and planned dataset release: A $500k Kaggle Red Teaming Challenge accompanies the release, with a commitment to publish a report and open-source an evaluation dataset derived from validated findings. This creates a near-term opportunity for researchers to produce and subsequently study high-quality, real adversarial prompts/annotations tied to these models.

- Deployment-centric details that matter for large-scale experimentation:
  - Native MXFP4 quantization for the released weights (aimed at single 80 GB GPU for 120B MoE, 16 GB for 20B), which is pragmatic for labs doing ablations at scale with limited hardware.
  - Open-sourcing the o200k_harmony tokenizer and a “harmony renderer” in Python and Rust; plus reference inference implementations (PyTorch and Apple Metal). This lowers friction for reproducible research on prompt formatting, tokenization effects, and cross-backend performance studies.
  - Broad, pre-arranged ecosystem support (vLLM, llama.cpp, Ollama, LM Studio, Cloudflare, OpenRouter, etc.) and Windows local GPU builds via ONNX Runtime (Foundry Local, AI Toolkit). This should reduce variance from bespoke plumbing and enable faster cross-stack benchmarking.

- Agentic/tool-use orientation as a first-class design goal: The post emphasizes strong tool use, few-shot function calling, and Structured Outputs compatibility through the Responses API, positioning these open models for agent workflow research (e.g., TauBench-style evals) without reinventing tooling.

- HealthBench and agentic eval emphasis: While headline numbers likely appear in the model card, the blog’s framing—that open weights match/beat some proprietary baselines on practical agentic and health evaluations—signals a specific target use profile (reasoning + tools) rather than just academic leaderboards. This is noteworthy for selecting eval suites that reflect intended deployment contexts.

If you want only novel angles to investigate:
- Study leakage between CoT and final outputs under instruction hierarchies (the blog’s example suggests a measurable gap).
- Evaluate the effectiveness and limits of the three-tier “reasoning effort” knob on both latency and robustness metrics.
- Reproduce and extend the worst-case fine-tuning experiments with independent protocols; compare against the forthcoming red-team dataset to test generalization.
- Benchmark MXFP4 quantization impacts on long-context and tool-use-heavy tasks vs standard per-token compute budgets.
   </research>
   <research source="https://openai.com/open-models">
Key takeaways likely to interest AI model researchers

- Open chain-of-thought exposure: The page explicitly advertises full chain-of-thought access for debugging/trust. That’s a notable shift for OpenAI, which historically avoided exposing CoT in production settings, and makes these models unusually inspectable for research.

- Tool-augmented CoT as a first-class behavior: It claims the models are designed for agentic tasks with “tool use within the chain-of-thought,” including web search and Python execution. If borne out, researchers can study end-to-end tool-calling trajectories directly in the model’s reasoning traces rather than only via wrapper logic.

- Adjustable test-time compute knob: “Reasoning effort” can be set to low/medium/high. This is a rare, explicit control for inference-time reasoning budget baked into an open model release, opening avenues to study controllable depth-of-thought vs. accuracy/latency trade-offs.

- Performance shape is math-heavy and nonmonotonic with scale: On AIME, the 20B slightly outperforms the 120B and even edges out OpenAI’s o3 in the table, while both lag o3 on MMLU/GPQA. This pattern suggests training/inference procedures disproportionately benefit competition-math style reasoning and that parameter count is not strictly predictive of their best-performing regime.

- Full-parameter fine-tuning under Apache-2.0: The combination of a permissive license, full-parameter FT, and visible CoT creates an unusually open research surface for analyzing how fine-tuning reshapes reasoning traces (and whether CoT/tool calls remain stable or drift under domain adaptation).

- Safety stress test against malicious fine-tuning: They report having intentionally fine-tuned a harmful variant of the 120B and assessing it under their Preparedness Framework, concluding it doesn’t reach high capability levels. That’s a concrete, pre-release red-teaming paradigm for open-weight models that invites replication and methodological scrutiny.

- Infrastructure/ecosystem signals: Official guides for Transformers, vLLM, Ollama, LM Studio, and a “verifying implementations” doc indicate they anticipate inference-stack variability and are pushing toward reproducibility across runtimes. The “OpenAI harmony response format” hints at a standardized output schema for multi-turn/tool-using models, which could reduce evaluation confounders.

- Practical deployment scope: The 20B is positioned to run on “most” desktops/laptops, making high-math-performance, agentic, CoT-visible models far more accessible for local experimentation, ablations, and instrumentation than typical closed counterparts.
   </research>
   <research source="https://huggingface.co/openai/gpt-oss-20b">
Key research takeaways and implications (beyond the plain card facts)

- MoE structure implied by “21B params, 3.6B active”: If experts are equal-sized and routing is top-2, the active/total ratio (~0.171) suggests roughly 12 experts (2/num_experts ≈ 0.171 → num_experts ≈ 12). That’s an unusually compact active set for the claimed capability, worth probing for expert specialization and routing entropy.

- All evaluations were done on the post-quantized model (MXFP4 on MoE weights). That’s atypical—most model cards report FP16/BF16 evals and treat int4 as a deployment optimization. Here, the publicized performance is for the quantized variant, implying very low accuracy loss from the specific MXFP4 scheme and/or quant-aware post-training. Good target for reproducibility and ablations against standard int4/int8.

- Mandatory “harmony” response format is an operational dependency, not just a convenience. Expect substantial degradation if you bypass the template. Cross-model comparisons that don’t enforce the template may be invalid; this is a caveat for leaderboard experiments and for tool-use reliability studies.

- “Reasoning level” appears to be a learned control (e.g., a system-level token/field) that trades latency for chain-of-thought depth. This is an accessible testbed for inference-time scaling: measure accuracy vs. CoT length vs. latency under Low/Medium/High and examine whether the gains align with self-consistency or intermediate-to-final supervision effects.

- Full chain-of-thought exposure from a major lab is unusual. It enables direct inspection of reasoning traces and error localization, but also creates a clean environment to study CoT leakage, instruction-following drift when CoT is hidden, and the effect of CoT truncation on final answers.

- Native tool-use coverage (function calling, browsing, Python) implies the model was tuned on tool call traces and schemas, not just plain chat. Expect spontaneous tool-call emissions unless prompts explicitly disable them; this is useful for studying tool-trigger thresholds, error recovery after failed calls, and schema generalization.

- Deployment stack hints at custom kernels and operator support: the vLLM pre-release wheel and the separate “kernels” dependency point to fused dequant + matmul paths for MXFP4. If you’re benchmarking, ensure the same kernels are active; otherwise, you may conflate algorithmic gains with kernel availability.

- “Runs within 16GB” is credible given 3.6B active params at 4-bit, but memory headroom will be dominated by KV cache at longer contexts. For practical tests, report context length, paged KV configuration, and batch shape; without that, headline “fits in 16GB” may not replicate.

- Fine-tuning implications for MoE: with fixed routers, naïve full-parameter FT risks expert drift and degraded routing. Expect better stability from router-aware schedules (lower LR on routers), or LoRA on experts with frozen routers. This model is a good candidate to study expert specialization retention under domain FT.

- Ecosystem signal: the rapid appearance of many adapters/finetunes/quantizations on the Hub indicates high “extensibility pressure.” It’s a fertile playground for meta-analyses (e.g., how different LoRA ranks or router-freezing choices move reasoning benchmarks under the same harmony template).
   </research>
   <research source="https://cookbook.openai.com/articles/gpt-oss/run-transformers">
Key takeaways for AI model researchers (novel implementation details beyond a typical model card)

- MXFP4 constraints and upside:
  - Both gpt-oss-20b and 120b ship MXFP4-quantized by default, but MXFP4 kernels only run on Hopper-class or newer GPUs, including H100/GB200 and the consumer RTX 50xx series. This consumer GPU support is notable if you’re planning desktop experiments.
  - Concrete memory figures: ~16 GB VRAM for 20b with MXFP4 (vs ~48 GB in bfloat16). The 120b can fit on a single H100 with MXFP4—useful for single-node benchmarking.

- Triton kernel requirement for MXFP4:
  - You need Triton 3.4 and a separate “kernels” package to unlock MXFP4 in Transformers (pip install -U transformers accelerate torch triton==3.4 kernels). That extra kernels dependency is easy to miss and is critical for the advertised VRAM footprint.

- OpenAI Responses-compatible serving via Transformers:
  - transformers serve exposes a /v1/responses endpoint (OpenAI Responses API semantics), not just chat/completions-style routes. This lets you swap in local gpt-oss instances with clients expecting OpenAI’s Responses behavior, including streaming and tool-call-friendly message formats.

- Harmony response format integration:
  - gpt-oss models use the Harmony response schema; you can either rely on the built-in HF chat template or use the openai-harmony library to:
    - Render conversation prefill ids and supply explicit stop_token_ids for assistant actions.
    - Parse the raw completion tokens back into structured assistant messages. This is a clean path to tool-call and multi-part message parsing without brittle regex.
  - Nuance: Harmony’s Developer role maps to the “system” prompt when using the standard HF chat template.

- Advanced inference hooks for performance and parallelism:
  - Attention kernels: attn_implementation="kernels-community/vllm-flash-attn3" plugs a vLLM-flavored FlashAttention-3 stack directly into HF generate(), indicating a route to vLLM-grade attention throughput without leaving the Transformers runtime.
  - Parallelism controls:
    - Expert Parallelism can be toggled via transformers.distributed.DistributedConfig(enable_expert_parallel=1), implying the stack supports MoE-style parallel modes beyond standard tensor parallelism.
    - Tensor parallelism can be auto-planned with tp_plan="auto", simplifying multi-GPU placement without manual sharding logic.
  - These knobs suggest a unified HF runtime that can exploit both EP and TP while keeping the standard generate() API.

- HF pipeline now natively accepts chat-style messages:
  - The text-generation pipeline example feeds a messages list directly, returning generated_text—indicating first-class chat prompting without manual template assembly. This reduces friction for fast prototyping compared to older “prompt string + special tokens” patterns.

- Practical distributed recipe:
  - The guide’s concrete invocation pattern (torchrun --nproc_per_node=4 generate.py) with EP+TP and custom attention kernels is a ready-made baseline for scaling 120b locally, useful for reproducible perf comparisons and ablation studies.

If you already know the model’s parameter counts and general quantization support, the surprising bits are the strict MXFP4 hardware requirement (but including RTX 50xx consumer GPUs), the reliance on custom Triton kernels, the Responses-API-compatible server built into Transformers, and the explicit hooks for EP/TP plus vLLM-flash-attn3 all inside standard HF generate()-based workflows.
   </research>
   <research source="https://wheels.vllm.ai/gpt-oss/">
Key takeaways from wheels.vllm.ai/gpt-oss that aren’t in a typical model card:

- Curated binary stack for GPT-OSS: a private pip index hosting pinned wheels for the entire inference runtime.
- Custom vLLM build: vllm 0.10.1+gptoss (Linux x86_64, cp38-abi3), indicating GPT-OSS–specific patches beyond upstream 0.10.1.
- Triton from a specific git commit: triton 3.4.0+git663e04e8 (Linux x86_64, cp38-abi3), implying kernel/compiler behaviors tied to a non-PyPI commit.
- Decoupled Triton kernels: a separate triton_kernels 1.0.0 wheel, suggesting rapid kernel iteration without rebuilding core engines.
- FlashInfer integration: flashinfer_python 0.2.8 is part of the stack, signaling a performance path that leans on FlashInfer for prefill/decoding kernels.
- ABI strategy: cp38-abi3 wheels for vLLM and Triton indicate one binary compatible with multiple Python 3.x versions (≥3.8), easing env management on Linux servers.
- OS/arch scope: compiled wheels only for Linux x86_64; no macOS/Windows/aarch64 wheels provided here.
- CUDA tagging absent: wheels don’t encode CUDA versions, consistent with Triton/FlashInfer JIT or runtime compilation; helps portability but may incur first-run build time.
- Install surface: a meta-package gpt_oss 0.1.0 is provided, hinting at a “single pip install” entry point that pulls the curated components from this index.

Implication for researchers: this index is effectively a reproducible, patched vLLM + Triton + FlashInfer kernel stack purpose-built for GPT-OSS, with version pinning down to a Triton commit—useful for replicating performance/behavior of the release without chasing upstream drift.
   </research>
   <research source="https://cookbook.openai.com/articles/gpt-oss/run-vllm">
Key takeaways and non-obvious details for AI model researchers

- Custom vLLM build for gpt-oss: The guide pins vllm==0.10.1+gptoss and pulls from a dedicated wheel index and nightly CUDA 12.8 PyTorch channel. This implies gpt-oss requires patched vLLM support (e.g., MXFP4 kernels/protocol handling) rather than stock vLLM, which matters for reproducibility, benchmarking, and kernel-level comparisons.

- Harmony protocol is mandatory for direct sampling: When bypassing the HTTP server and calling vLLM’s Python API, prompts must follow the Harmony response format or “the model will not function correctly.” The workflow includes:
  - Rendering conversation prefill token IDs with openai-harmony,
  - Supplying Harmony-specific stop_token_ids to the sampler,
  - Parsing completion token IDs back into structured Harmony messages.
  This is a strong signal that gpt-oss relies on a conversation/action protocol beyond plain text prompting; incorrect formatting will silently degrade behavior.

- Tool calling as part of chain-of-thought requires state re-feeding: The doc emphasizes that tool calls occur “as part of the chain-of-thought” and that you should feed the reasoning returned by the API back into subsequent tool-call turns until a final answer. This is unusual guidance compared to typical CoT redaction practices and implies the model emits intermediate structured reasoning that should be preserved to maintain coherence across tool invocations.

- Token-level, structure-preserving decoding path: The direct vLLM example exposes both text and token IDs and demonstrates round-tripping completions into structured messages via Harmony. This invites research on:
  - Measuring how decoding choices (e.g., stop tokens) affect tool-call segmentation,
  - Auditing model-internal action plans vs. surfaced text,
  - Building instrumentation to analyze protocol-constrained reasoning sequences.

- VRAM footprint claims under MXFP4 are aggressive: Out-of-the-box MXFP4 quantization enables the 120B model to run on a single H100-class GPU (≥60 GB) or multi-GPU, and the 20B on ~16 GB. If validated, this shifts the practical frontier for large-scale single-node experimentation and may affect cost/perf trade studies versus 70B-class baselines.

- Seamless OpenAI-API interop across both Chat Completions and Responses: The server exposes both APIs, and the Agents SDK (or LiteLLM) can point to the self-hosted vLLM backend. This lowers friction for evaluating agentic behaviors (including tool/browse flows) without changing application code, enabling controlled A/Bs against closed weights with identical orchestration.

- Operational detail useful for benchmarking: The server accepts api_key="EMPTY" and auto-downloads models from Hugging Face, simplifying ephemeral deployments for throughput/latency testing with standardized OpenAI clients while retaining low-level access via vLLM Python for protocol-aware studies.
   </research>
   <research source="https://github.com/openai/gpt-oss?tab=readme-ov-file#reference-pytorch-implementation">
Key takeaways about the Reference PyTorch implementation (and closely related engineering details) that aren’t obvious from a model card:

- Exact-architecture, dequantized path: The PyTorch reference is purpose-built to mirror the model’s architecture using only basic PyTorch ops. It upcasts all weights to BF16 and runs in BF16, including MoE weights that are released as MXFP4. So it’s correctness/inspectability-first, not a performance or quantization-accurate path.

- Minimal parallelism scope: It only implements tensor-parallelism for the MoE layers (not full end-to-end TP/fused kernels). This narrow TP is what makes the 120B model workable in this code path, but it’s why it needs substantially more GPU than the “single 80GB GPU” headline.

- Hardware implications: Because it’s fully BF16 and unfused, the torch path requires at least 4× H100 (or 2× H200). If you benchmark this implementation, you’re measuring a dequantized, non-optimized configuration that won’t reflect the performance profile of the released MXFP4 weights.

- Separation of concerns (Torch vs Triton): The Triton reference keeps MoE in MXFP4 via an optimized MoE kernel and includes attention memory optimizations. That path is what actually enables gpt-oss-120b on a single 80GB GPU. In short: Torch = spec clarity; Triton = real MXFP4 inference.

- Practical loader gotcha: When using the Triton path, enabling PyTorch’s expandable CUDA allocator (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) avoids OOM during weight load—useful if you’re evaluating large-checkpoint load behavior or experimenting with custom sharding.

- Weight storage details that affect tooling: MoE tensors are stored as MXFP4 with per-block scales along the last dimension, packing two FP4 values into one uint8; all other tensors are BF16. Activations are recommended in BF16. This split matters if you’re writing your own loader or kernels.

- Harmony is not optional: The model expects Harmony formatting end-to-end. The examples show explicit stop_token_ids for “assistant actions” (tool calls) and a round-trip API to render prefill and parse structured outputs. If you’re building custom decoders/samplers, you’ll need to handle these special stops and the message parsing to keep behavior faithful.

- Training-to-inference tool mismatch you should plan for: The Python tool was trained in a stateful regime, but the released reference tool is stateless and overrides the harmony tool definition to compensate. If you study or reproduce tool-augmented CoT loops, this gap can influence fidelity and needs engineering workarounds.

- Recommended sampling is unconstrained: They suggest temperature=1.0, top_p=1.0 as defaults. If you’re evaluating reasoning or tool-use stability, consider that many open-weight LMs ship with more conservative decoding defaults.

- Educational value: Because the torch path avoids fused kernels, it’s unusually transparent for a 120B MoE—useful for auditing exact layer ordering, MoE gating placement, and numerical types without disentangling custom kernels.

If you only care about the PyTorch reference as a research scaffold: treat it as a spec-accurate, BF16, MoE-TP-only baseline for architecture inspection and ablations; use the Triton path for any conclusions about MXFP4 behavior, memory footprint, or single-GPU viability.
   </research>
   <research source="https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama">
Key takeaways for AI model researchers

- Only MXFP4 quantization is available: gpt-oss models on Ollama ship exclusively in MXFP4; no alternative quantizations yet. This constrains memory/latency trade-offs and benchmarking against common int8/QLoRA/Q*K* baselines.

- Concrete local memory guidance: 20B targets ≥16 GB VRAM/unified memory; 120B targets ≥60 GB. CPU offload is supported but explicitly cautioned as slow—useful for characterizing unified-memory Apple Silicon vs discrete GPU behavior.

- Harmony-format chat templating by default: Ollama applies an OpenAI “harmony” chat template automatically. Prompt formatting parity with OpenAI endpoints is a design goal, which affects transferability of prompt-engineering results.

- Tool use surfaced via CoT that you must feed back: The guide explicitly notes that models may perform tool calling as part of chain-of-thought and that you should return the reasoning emitted by the API into subsequent turns until a final answer is reached. This is unusual compared to hidden-tool-call paradigms and has implications for reproducibility, logging, and privacy.

- Built-in browser tool in the Ollama app: Beyond generic function calling, the app includes a native browsing tool, noteworthy for local/offline workflows that still allow selective online retrieval.

- OpenAI SDK with Chat Completions parity: You can point the OpenAI SDK directly at Ollama (base_url localhost, dummy API key). This enables drop-in local A/B testing against cloud models without refactoring.

- Responses API gap with sanctioned workarounds: Ollama doesn’t support the Responses API; the guide recommends Hugging Face Responses.js as a proxy and provides an example “gpt-oss” Python server to translate Responses API → Ollama backends. This signals an actively supported interop path for modern multi-turn/structured outputs.

- Agents SDK interop is first-class: The OpenAI Agents SDK can be redirected to local gpt-oss via LiteLLM (Python) or AI SDK with an Ollama adapter (TypeScript). The guide frames Agents as model-agnostic infrastructure, encouraging third-party/local backends.

- Multi-GPU/workstation positioning for 120B: While aimed at consumer hardware, the 120B guidance explicitly targets multi-GPU/beefy workstations, hinting at practical local deployment expectations for large checkpoints.

Why this matters
- The explicit CoT passthrough for tool use and the enforced MXFP4 quantization are the two most nonstandard aspects; both influence evaluation design, latency/accuracy trade-offs, and privacy considerations for local agents.
- Harmony templating and SDK/Agents parity reduce confounds when comparing local vs hosted runs, making controlled experimentation more straightforward.
   </research>
   <research source="https://github.com/openai/gpt-oss/blob/main/awesome-gpt-oss.md">
Key takeaways from the “Awesome gpt-oss” page that are likely interesting to AI model researchers

- Immediate, broad vendor integration: Day‑0 support spans Groq, NVIDIA, AMD, Cloudflare Workers AI, AWS, Hugging Face/Transformers, vLLM, llama.cpp, LM Studio, and Ollama. This is unusually comprehensive for an open-weight launch and suggests the model family is being treated as a first‑class target across accelerators and runtimes.

- Groq goes beyond basic inference: In addition to playground support for both 20B and 120B, Groq exposes built‑in web search and code execution plus a “Responses API” interface—useful for agentic workflows and for experiments that rely on OpenAI’s modern response schema without building custom tooling.

- Edge/serverless coverage for large models: Both 20B and 120B are listed on Cloudflare Workers AI, indicating readily accessible serverless endpoints for very large open models—a deployment pattern that has been rare for open LLMs of this scale.

- NVIDIA optimization path is explicit: The page links to TensorRT‑LLM guides and an NVIDIA developer blog titled “delivering 1.5M TPS inference on NVIDIA GB200 NVL72,” pointing to an aggressive performance target and a clear recipe for low‑latency, high‑throughput serving experiments.

- AMD support includes NPUs and new silicon: Guides cover Ryzen AI NPUs, MI300X, and STX Halo via Lemonade, implying attention to non‑NVIDIA acceleration paths for both local and datacenter scenarios—useful for comparative perf/efficiency studies across vendors.

- Lightweight/local inference formats are ready: There are pointers to running via llama.cpp and Unsloth GGUFs, plus a Colab showing MXFP4 inference for the 20B model. This lowers the barrier for laptop‑class profiling, quantization studies, and memory/latency tradeoff analysis.

- vLLM and OpenAI‑style interfaces: A dedicated vLLM “recipes” page and Groq’s Responses API suggest the ecosystem is converging on OpenAI‑compatible request/response semantics, simplifying A/B testing across backends and reducing glue code for eval pipelines.

- Tool‑use scaffolding is included: The repo ships minimal “python (Docker)” and “browser” tool examples, offering a baseline for reproducible tool‑calling experiments and agent benchmarks without standing up a heavyweight tool runtime.

- Training stacks are pre‑wired: TRL (HF), LlamaFactory, and Unsloth have gpt‑oss‑specific examples, making it straightforward to explore SFT, DPO/ORPO, PEFT, and memory‑efficient finetuning strategies out of the box.

- RL example beyond pure language: An “Auto solving the 2048 game” notebook under Reinforcement Learning hints at using gpt‑oss in RL‑style finetuning settings, which is useful for studying reward‑driven adaptation and planning behaviors outside standard text benchmarks.
   </research>
   <research source="https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4">
Key takeaways from the gpt-oss Hugging Face collection page (beyond typical model-card details):

- Official family signal: OpenAI curates an explicit “gpt-oss” collection on HF framed around open-weight models for reasoning and agentic use cases, indicating an ongoing, family-style release strategy rather than one-off drops.

- Adoption skew toward the smaller model: Despite similar update timing (late August), the smaller variant shows markedly higher uptake:
  - 22B: ~5.77M downloads, ~3.72k likes
  - 120B: ~3.29M downloads, ~4k likes
  This suggests practitioners prioritize compute-friendly models for real-world deployment/finetuning even when a dramatically larger open-weight option exists.

- Engagement is broad, not size-exclusive: The like counts are close, implying sustained interest in both scales; the 120B’s slightly higher likes vs. lower downloads hints at researcher curiosity versus operational usage constraints.

- Active maintenance cadence: Collection updated Aug 7; both repos updated Aug 26—useful for tracking iteration pace and signaling continued support.

- Institutional pattern: The collection sits alongside OpenAI’s “Whisper Release” collection, reinforcing a broader organizational habit of packaging open releases on HF as coherent, evolving sets.

- Terminology cue: The collection foregrounds “agentic tasks,” an explicit positioning that’s notable for researchers tracking models targeting tool-use/long-horizon planning workloads, beyond generic text generation.
   </research>

</model_research>


```

Your task is to produce a concise 300-500 word summary of the specified model highlighting 
key points of interest for AI researchers. Make sure that major information and traits are
specified, and prefer bullet points for notable points of interest. Use markdown links to 
refer to sources if warranted.

You must also produce:

a) A "Prompting Guide" that describes the best way to prompt the model.
b) An example "Tool Description" and "Parameter Descriptions" to help an LLM Produce optimal outputs for the model


---ASSISTANT
Model summary for researchers (gpt-oss-20b/120b)

- Architecture and efficiency
  - Sparse MoE with very low active compute: 120B total, ~5.1B active params/token; 20B total, ~3.6B active. This enables large-model reasoning at modest runtime cost and invites sparsity–performance studies. [HF 20B card](https://huggingface.co/openai/gpt-oss-20b), [HF 120B card](https://hf.co/openai/gpt-oss-120b)
  - MXFP4 as a first-class quantization: MoE weights ship and are evaluated in MXFP4 with minimal reported regression, enabling 120B on a single 80 GB GPU, 20B on ~16 GB. A strong target for quantization-aware benchmarking and kernel research. [Open models](https://openai.com/open-models), [Transformers guide](https://cookbook.openai.com/articles/gpt-oss/run-transformers)
  - Long-context and kernel stack: Hybrid sparse/dense attention with GQA, RoPE, YaRN, and unusual “zero-attend” bias; custom Triton/FlashInfer kernels and a patched vLLM (0.10.1+gptoss). [vLLM wheels](https://wheels.vllm.ai/gpt-oss/), [run-vllm](https://cookbook.openai.com/articles/gpt-oss/run-vllm), [arXiv](https://arxiv.org/abs/2508.10925)

- Prompting/protocol
  - Harmony response format is mandatory. Use HF chat templates or the openai-harmony library; incorrect formatting silently degrades behavior. [Run-vLLM](https://cookbook.openai.com/articles/gpt-oss/run-vllm), [Transformers](https://cookbook.openai.com/articles/gpt-oss/run-transformers)
  - Explicit test-time compute control via Reasoning: low|medium|high in the system prompt; clean lever for accuracy–latency tradeoffs and dynamic compute research. [Blog](https://openai.com/index/introducing-gpt-oss/)

- Chain-of-thought and tools
  - Full CoT is exposed by design, with tool use interleaved within CoT (browsing, Python). Guidance favors feeding back reasoning/tool steps across turns. Rich substrate for rationale supervision, interpretability, and agentic evaluation. [Open models](https://openai.com/open-models), [run-vllm](https://cookbook.openai.com/articles/gpt-oss/run-vllm)
  - Standardized OpenAI-style APIs (Chat Completions and Responses) across local/cloud runtimes (Transformers serve, vLLM, Ollama, Groq, Cloudflare). Facilitates controlled A/Bs and reproducibility. [Awesome gpt-oss](https://github.com/openai/gpt-oss/blob/main/awesome-gpt-oss.md)

- Performance and safety signals
  - HealthBench: 120B approaches o3 on hard clinical evals at high reasoning; math performance shows nonmonotonic scaling (20B competitive on AIME). [Open models](https://openai.com/open-models)
  - Safety: Worst-case malicious fine-tuning did not reach “High” thresholds under Preparedness; CoT may include policy-violating text (monitor rather than suppress). [Blog](https://openai.com/index/introducing-gpt-oss/)
  - Robustness gaps: Instruction hierarchy weaker than o4-mini; tool-use yields large deltas—good testbed for scaffolding and guardrail research. [arXiv](https://arxiv.org/abs/2508.10925)

- Systems and deployment
  - Single-node practicality via MXFP4; strict kernel requirements (Triton 3.4, custom vLLM). EP/TP supported in Transformers with attention kernel hooks. [Transformers](https://cookbook.openai.com/articles/gpt-oss/run-transformers), [vLLM](https://cookbook.openai.com/articles/gpt-oss/run-vllm)
  - Harmony tokenizer o200k_harmony and reference implementations (PyTorch spec-accurate vs Triton MXFP4) aid architecture audits vs performance runs. [PyTorch reference](https://github.com/openai/gpt-oss?tab=readme-ov-file#reference-pytorch-implementation)

Prompting guide

- Always use the Harmony format (roles: system/developer/user; channels: analysis/commentary/final).
- Set test-time compute explicitly in the system message: “Reasoning: low|medium|high” (start with medium; use high for complex reasoning, low for latency-sensitive tasks).
- Keep chain-of-thought in analysis; provide the answer only in final. In multi-turn dialogs, strip the assistant’s prior analysis before re-feeding context.
- Tool use:
  - Allow the model to plan/tool-call in analysis.
  - Execute the tool externally.
  - Feed back tool results as a new assistant turn in analysis, then continue until the model emits a final answer.
- Decoding: temperature 0.7–1.0, top_p 0.9–1.0. For deterministic evals, use temp 0.0–0.2 and Reasoning: medium.
- Long-context: supply concise, well-structured context; prefer citations/IDs over raw dumps.
- Safety: place policy in system; don’t rely solely on system priority—use scaffolds/validators.

Minimal Harmony message sketch
- system: “You are gpt-oss. Reasoning: high. Follow tools and return concise final answers.”
- user: “Question…”
- assistant (analysis): “Plan, reasoning, potential tool calls…”
- tool results fed back as assistant (analysis) turns
- assistant (final): “Answer…”

Example tool description and parameter descriptions

Tool: web_search
- name: web_search
- description: Search the web for recent information and return top results with snippets and links.
- parameters (JSON Schema):
  - query (string, required): Clear, specific search query. Include entities, dates, and operators if helpful.
  - top_k (integer, default=5, min=1, max=10): Number of results to return. Use 3–8 for balance.
  - site_filter (string, optional): Restrict to a domain (e.g., “site:who.int”). Omit if broad recall is desired.
  - time_range (string, enum: [“any”, “24h”, “7d”, “30d”, “year”], default=“any”): Freshness constraint.
  - include_snippets (boolean, default=true): Return short summaries; set false for link-only.
  - region (string, optional, ISO 3166-1 alpha-2): Geo-targeting (e.g., “US”, “DE”) when results may vary by locale.

Parameter usage tips for the model
- Form queries with disambiguating terms and constraints (e.g., “AIME 2024 results filetype:pdf”).
- Use time_range=“7d” or “24h” for news/safety-critical topics.
- Keep top_k small when you plan to read deeply; increase when you will quickly filter.
- Prefer site_filter for authoritative domains to reduce noise.
- Always cite sources in the final answer using the returned links.