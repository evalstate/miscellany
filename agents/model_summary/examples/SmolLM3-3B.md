---USER

Here is the model card for HuggingFaceTB/SmolLM3-3B:

```
---
library_name: transformers
license: apache-2.0
language:
- en
- fr
- es
- it
- pt
- zh
- ar
- ru
base_model:
- HuggingFaceTB/SmolLM3-3B-Base
---


# SmolLM3


![image/png](https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/zy0dqTCCt5IHmuzwoqtJ9.png)


##  Table of Contents

1. [Model Summary](#model-summary)
2. [How to use](#how-to-use)
3. [Evaluation](#evaluation)
4. [Training](#training)
5. [Limitations](#limitations)
6. [License](#license)

## Model Summary

SmolLM3 is a 3B parameter language model designed to push the boundaries of small models. It supports dual mode reasoning, 6 languages and long context. SmolLM3 is a fully open model that offers strong performance at the 3B–4B scale.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6200d0a443eb0913fa2df7cc/db3az7eGzs-Sb-8yUj-ff.png)

The model is a decoder-only transformer using GQA and NoPE (with 3:1 ratio), it was pretrained on 11.2T tokens with a staged curriculum of web, code, math and reasoning data. Post-training included midtraining on 140B reasoning tokens followed by supervised fine-tuning and alignment via Anchored Preference Optimization (APO).

### Key features
- Instruct model optimized for **hybrid reasoning**
- **Fully open model**: open weights + full training details including public data mixture and training configs
- **Long context:** Trained on 64k context and suppots up to **128k tokens** using YARN extrapolation
- **Multilingual**: 6 natively supported (English, French, Spanish, German, Italian, and Portuguese)

For more details refer to our blog post: https://hf.co/blog/smollm3

## How to use

The modeling code for SmolLM3 is available in transformers `v4.53.0`, so make sure to upgrade your transformers version. You can also load the model with the latest `vllm` which uses transformers as a backend.
```bash
pip install -U transformers
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM3-3B"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
).to(device)

# prepare the model input
prompt = "Give me a brief explanation of gravity in simple terms."
messages_think = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages_think,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate the output
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)

# Get and decode the output
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

>[!TIP]
> We recommend setting `temperature=0.6` and `top_p=0.95` in the sampling parameters.

### Long context processing

The current `config.json` is set for context length up to 65,536 tokens. To handle longer inputs (128k or 256k), we utilize YaRN you can change the `max_position_embeddings` and rope_scaling` to:
```
{
  ...,
  "rope_scaling": {
    "factor": 2.0, #2x65536=131 072 
    "original_max_position_embeddings": 65536,
    "type": "yarn"
  }
}
```


### Enabling and Disabling Extended Thinking Mode

We enable extended thinking by default, so the example above generates the output with a reasoning trace. For choosing between enabling, you can provide the `/think` and `/no_think` flags through the system prompt as shown in the snippet below for extended thinking disabled. The code for generating the response with extended thinking would be the same except that the system prompt should have `/think` instead of `/no_think`.

```python
prompt = "Give me a brief explanation of gravity in simple terms."
messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

We also provide the option of specifying the whether to use extended thinking through the `enable_thinking` kwarg as in the example below. You do not need to set the `/no_think` or `/think` flags through the system prompt if using the kwarg, but keep in mind that the flag in the system prompt overwrites the setting in the kwarg.

```python
prompt = "Give me a brief explanation of gravity in simple terms."
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
```

### Agentic Usage

SmolLM3 supports tool calling!
Just pass your list of tools:
- Under the argument `xml_tools` for standard tool-calling: these tools will be called as JSON blobs within XML tags, like `<tool_call>{"name": "get_weather", "arguments": {"city": "Copenhagen"}}</tool_call>`
- Or under `python_tools`: then the model will call tools like python functions in a `<code>` snippet, like `<code>get_weather(city="Copenhagen")</code>`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM3-3B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

tools = [
    {
        "name": "get_weather",
        "description": "Get the weather in a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city to get the weather for"}}}}
]

messages = [
    {
        "role": "user",
        "content": "Hello! How is the weather today in Copenhagen?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    enable_thinking=False, # True works as well, your choice!
    xml_tools=tools,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
)

outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

### Using Custom System Instructions. 

You can specify custom instruction through the system prompt while controlling whether to use extended thinking. For example, the snippet below shows how to make the model speak like a pirate while enabling extended thinking.

```python
prompt = "Give me a brief explanation of gravity in simple terms."
messages = [
    {"role": "system", "content": "Speak like a pirate./think"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

For local inference, you can use `llama.cpp`, `ONNX`, `MLX` and `MLC`. You can find quantized checkpoints in this collection (https://huggingface.co/collections/HuggingFaceTB/smollm3-686d33c1fdffe8e635317e23)

### vLLM and SGLang

You can use vLLM and SGLang to deploy the model in an API compatible with OpenAI format.

#### SGLang

```bash
python -m sglang.launch_server --model-path HuggingFaceTB/SmolLM3-3B
```

#### vLLM

```bash
vllm serve HuggingFaceTB/SmolLM3-3B --enable-auto-tool-choice --tool-call-parser=hermes
```

#### Setting `chat_template_kwargs`

You can specify `chat_template_kwargs` such as `enable_thinking` to a deployed model by passing the `chat_template_kwargs` parameter in the API request.

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "HuggingFaceTB/SmolLM3-3B",
  "messages": [
    {"role": "user", "content": "Give me a brief explanation of gravity in simple terms."}
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "max_tokens": 16384,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

## Evaluation

In this section, we report the evaluation results of SmolLM3 model. All evaluations are zero-shot unless stated otherwise, and we use [lighteval](https://github.com/huggingface/lighteval) to run them. 

We highlight the best score in bold and underline the second-best score.

### Instruction Model

#### No Extended Thinking
Evaluation results of non reasoning models and reasoning models in no thinking mode. We highlight the best and second-best scores in bold.
| Category | Metric | SmoLLM3-3B | Qwen2.5-3B | Llama3.1-3B | Qwen3-1.7B | Qwen3-4B |
|---------|--------|------------|------------|-------------|------------|----------|
| High school math competition | AIME 2025 | <u>9.3</u> | 2.9 | 0.3 | 8.0 | **17.1** |
| Math problem-solving | GSM-Plus | 72.8 | <u>74.1</u> | 59.2 | 68.3 | **82.1** |
| Competitive programming | LiveCodeBench v4 | <u>15.2</u> | 10.5 | 3.4 | 15.0 | **24.9** |
| Graduate-level reasoning | GPQA Diamond | <u>35.7</u> | 32.2 | 29.4 | 31.8 | **44.4** |
| Instruction following | IFEval | **76.7** | 65.6 | 71.6 | <u>74.0</u> | 68.9 |
| Alignment | MixEval Hard | 26.9 | <u>27.6</u> | 24.9 | 24.3 | **31.6** |
| Tool Calling | BFCL| <u>92.3</u> | - | <u>92.3</u> * | 89.5  | **95.0** |
| Multilingual Q&A | Global MMLU | <u>53.5</u> | 50.54 | 46.8 | 49.5 | **65.1** |

(*): this is a tool calling finetune

#### Extended Thinking
Evaluation results in reasoning mode for SmolLM3 and Qwen3 models: 
| Category | Metric | SmoLLM3-3B | Qwen3-1.7B | Qwen3-4B |
|---------|--------|------------|------------|----------|
| High school math competition | AIME 2025 | <u>36.7</u> | 30.7 | **58.8** |
| Math problem-solving | GSM-Plus | <u>83.4</u> | 79.4 | **88.2** |
| Competitive programming | LiveCodeBench v4 | 30.0 | <u>34.4</u> | **52.9** |
| Graduate-level reasoning | GPQA Diamond | <u>41.7</u> | 39.9 | **55.3** |
| Instruction following | IFEval | 71.2 | <u>74.2</u> | **85.4** |
| Alignment | MixEval Hard | 30.8 | <u>33.9</u> | **38.0** |
| Tool Calling | BFCL | <u>88.8</u> | <u>88.8</u> | **95.5** |
| Multilingual Q&A | Global MMLU | <u>64.1</u> | 62.3 | **73.3** |


### Base Pre-Trained Model

#### English benchmarks
Note: All evaluations are zero-shot unless stated otherwise. For Ruler 64k evaluation, we apply YaRN to the Qwen models with 32k context to extrapolate the context length.

| Category | Metric | SmolLM3-3B | Qwen2.5-3B | Llama3-3.2B | Qwen3-1.7B-Base | Qwen3-4B-Base |
|---------|--------|---------------------|------------|--------------|------------------|---------------|
| Reasoning & Commonsense| HellaSwag | **76.15** | 74.19 |<u>75.52</u> | 60.52 | 74.37 |
| | ARC-CF (Average) | **65.61** | 59.81 | 58.58 | 55.88 | <u>62.11</u> |
| | Winogrande | 58.88 | **61.41** | 58.72 | 57.06 | <u>59.59</u> |
| | CommonsenseQA | <u>55.28</u> | 49.14 | **60.60** | 48.98 | 52.99 |
| Knowledge & Understanding | MMLU-CF (Average) | <u>44.13</u> | 42.93 | 41.32 | 39.11 | **47.65** | 
| | MMLU Pro CF | <u>19.61</u> | 16.66 | 16.42 | 18.04 | **24.92** |
| | MMLU Pro MCF | <u>32.70</u> | 31.32 | 25.07 | 30.39 | **41.07** |
| | PIQA | **78.89** | 78.35 | <u>78.51</u> | 75.35 | 77.58 |
| | OpenBookQA | 40.60 | 40.20 | <u>42.00</u> | 36.40 | **42.40** |
| | BoolQ | **78.99** | 73.61 | <u>75.33</u> | 74.46 | 74.28 | 
| **Math & Code** |  |  |  |  |  |  | 
| Coding & math | HumanEval+ | 30.48 | 34.14| 25.00 | <u>43.29</u>| **54.87** |
| | MBPP+ | 52.91 | 52.11 | 38.88| <u>59.25</u> | **63.75** | 
| | MATH (4-shot) | <u>46.10</u> | 40.10 | 7.44 | 41.64 | **51.20** |
| | GSM8k (5-shot) | 67.63 | <u>70.13</u> | 25.92 | 65.88 | **74.14** | 
| **Long context** |  |  |  |  |  |  | 
| | Ruler 32k | 76.35 | 75.93 | <u>77.58</u> | 70.63 | **83.98** | 
| | Ruler 64k | <u>67.85</u> | 64.90 | **72.93** | 57.18 | 60.29 | 
| | Ruler 128k | 61.03 | <u>62.23</u> | **71.30** | 43.03 | 47.23 | 

#### Multilingual benchmarks


| Category | Metric | SmolLM3 3B Base | Qwen2.5-3B | Llama3.2 3B | Qwen3 1.7B Base | Qwen3 4B Base |
|---------|--------|---------------------|------------|--------------|------------------|---------------|
| Main supported languages |  |  |  |  |  |  |  |
| French| MLMM Hellaswag | **63.94** | 57.47 | 57.66 | 51.26 | <u>61.00</u> |
| | Belebele | 51.00 | <u>51.55</u> | 49.22 |49.44| **55.00** |
| | Global MMLU (CF) | <u>38.37</u> | 34.22  | 33.71 | 34.94  |**41.80** |
| | Flores-200 (5-shot) | 62.85| 61.38| <u>62.89</u> | 58.68 | **65.76** |
| Spanish| MLMM Hellaswag | **65.85** | 58.25 | 59.39 | 52.40 | <u>61.85</u> |
| | Belebele | 47.00 | <u>48.88</u> | 47.00 | 47.56 | **50.33** |
| | Global MMLU (CF) | <u>38.51</u> | 35.84  | 35.60 | 34.79  |**41.22** |
| | Flores-200 (5-shot) | <u>48.25</u>| 50.00| 44.45 | 46.93 | **50.16** |
| German| MLMM Hellaswag | **59.56** | 49.99|  53.19|46.10| <u>56.43</u>|
| | Belebele | <u>48.44</u> | 47.88 | 46.22 | 48.00 | **53.44**|
| | Global MMLU (CF) | <u>35.10</u> | 33.19  | 32.60 | 32.73  |**38.70** |
| | Flores-200 (5-shot) | **56.60**| 50.63| <u>54.95</u> | 52.58 | 50.48 |
| Italian| MLMM Hellaswag | **62.49** | 53.21 | 54.96 | 48.72 | <u>58.76</u> |
| | Belebele | <u>46.44</u> | 44.77 | 43.88 | 44.00 | **48.78** | 44.88 |
| | Global MMLU (CF) | <u>36.99</u> | 33.91  | 32.79 | 35.37  |**39.26** |
| | Flores-200 (5-shot) | <u>52.65<u/>| **54.87**| 48.83 | 48.37 | 49.11 |
| Portuguese| MLMM Hellaswag | **63.22** | 57.38 | 56.84 | 50.73 | <u>59.89</u> |
| | Belebele | 47.67 | **49.22** | 45.00 | 44.00 | 50.00 | <u>49.00</U> |
| | Global MMLU (CF) | <u>36.88</u> | 34.72  | 33.05 | 35.26  |**40.66** |
| | Flores-200 (5-shot) | <u>60.93</u> |57.68| 54.28 | 56.58 | **63.43** |

The model has also been trained on Arabic (standard), Chinese and Russian data, but has seen fewer tokens in these languages compared to the 6 above. We report the performance on these langages for information.
| Category | Metric | SmolLM3 3B Base | Qwen2.5-3B | Llama3.2 3B | Qwen3 1.7B Base | Qwen3 4B Base |
|---------|--------|---------------------|------------|--------------|------------------|---------------|
| Other supported languages |  |  |  |  |  |  |  |
| Arabic| Belebele | 40.22 | 44.22 | <u>45.33</u> | 42.33 | **51.78** |
| | Global MMLU (CF) | 28.57 | 28.81 | 27.67 | <u>29.37</u> | **31.85** |
| | Flores-200 (5-shot) | <u>40.22</u> | 39.44 | **44.43** | 35.82 | 39.76 |
| Chinese| Belebele | 43.78 | 44.56 | <u>49.56</u> | 48.78 | **53.22** |
| | Global MMLU (CF) | 36.16 | 33.79 | <u>39.57</u> | 38.56 | **44.55** |
| | Flores-200 (5-shot) | 29.17 | **33.21** | 31.89 | 25.70 | <u>32.50</u> |
| Russian| Belebele | <u>47.44</u> | 45.89 | <u>47.44</u> | 45.22 | **51.44** |
| | Global MMLU (CF) | <u>36.51</u> | 32.47 | 34.52 | 34.83 | **38.80** |
| | Flores-200 (5-shot) | 47.13 | 48.74 | 50.74 | <u>54.70</u> | **60.53** |

## Training

### Model

- **Architecture:** Transformer decoder
- **Pretraining tokens:** 11T
- **Precision:** bfloat16

### Software & hardware

- **GPUs:** 384 H100
- **Training Framework:** [nanotron](https://github.com/huggingface/nanotron/tree/smollm3)
- **Data processing framework:** [datatrove](https://github.com/huggingface/datatrove)
- **Evaluation framework:** [lighteval](https://github.com/huggingface/lighteval)
- **Post-training Framework:** [TRL](https://github.com/huggingface/trl)

### Open resources
Here is an infographic with all the training details 
- The datasets used for pretraining can be found in this [collection](https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets-685a7353fdc01aecde51b1d9) and those used in mid-training and post-training will be uploaded later 
- The training and evaluation configs and code can be found in the [huggingface/smollm](https://github.com/huggingface/smollm) repository.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/651e96991b97c9f33d26bde6/qiE5ZYr9SD1CIAtfEfuC8.png)

## Limitations

SmolLM3 can produce text on a variety of topics, but the generated content may not always be factually accurate, logically consistent, or free from biases present in the training data. These models should be used as assistive tools rather than definitive sources of information. Users should always verify important information and critically evaluate any generated content.


## License
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
```


---ASSISTANT
Thank you. I will refer to this model card in my future responses.
---ASSISTANT
Thank you. I will refer to this model card in my future responses.
---USER
Here is additional research information about the model:

$```
<model_research>
   <research source="https://hf.co/blog/smollm3">
Here are key points and surprising technical insights from the HuggingFace blog post on SmolLM3 that would be of special interest to AI model researchers, focusing on new or emphasized content beyond the Model Card:

---

**1. Ablation-Driven Architectural Innovations:**
- SmolLM3 introduced Grouped Query Attention (GQA) (4 groups) to dramatically reduce KV cache size without hurting performance, confirmed by ablations on 3B-scale models over 100B tokens.
- A novel adaptation of NoPE (from Yang et al., 2025) is used: rotary position embeddings are selectively removed from every fourth layer, enhancing long-context modeling while keeping short-context abilities intact—again validated by targeted ablations.

**2. Rigorous Data Mixture Tuning:**
- The three-stage curriculum for 11.2T tokens was derived via ablations over mixtures and proportions, especially to balance math/code/web across stages for generalization and reasoning.
- The final decay stage aggressively upsamples both high-quality code and math data, pinpointed as crucial for strong performance in those domains.

**3. Long-Context Specialization via Sequential Context Extension:**
- After base pretraining, a unique mid-training process steadily extends context window (from 4k→32k, then 32k→64k) by retraining for 100B tokens, increasing RoPE “theta” at each step.
- Contrary to expectations, specifically upsampling “long” data (e.g., repositories, books) showed little benefit beyond simply using longer input sequences and the new positional encoding method—improving compute efficiency.

**4. “Dual-Mode” Reasoning Pipeline—An Open Recipe:**
- The smolLM3 dual-mode (reasoning / non-reasoning) capability is not just a system prompt hack; the training plan uses:
   - Mid-training for general reasoning (140B tokens),
   - Supervised fine-tuning with a balanced mixture (including substantial synthetic reasoning data from Qwen3-32B prompts to augment areas lacking rich traces, like multi-turn multilingual chat),
   - Final alignment with Anchored Preference Optimization (APO, a more stable new variant of DPO).

**5. Synthetic Data Generation for Reasoning Traces:**
- Researchers noted a deficit of high-quality reasoning-trace data (especially outside math and code); they filled this gap by prompting Qwen3-32B to generate synthetic reasoning traces using prompts from non-reasoning-focused datasets—demonstrating a practical approach to broadening reasoning capabilities across domains.

**6. Direct Insight into Alignment and Trade-offs:**
- The post details an internal discovery: APO alignment and reasoning mid-training, while improving most downstream tasks, temporarily degraded long-context performance (on benchmarks like RULER), traced to a bias toward reasoning-focused data with shorter sequence lengths.
- This was counteracted by a final model merging step: linearly blending a “strong long-context” mid-trained checkpoint (10%) with the aligned APO “model soup” (90%), restoring RULER scores up to 128k tokens.

**7. Full Transparency and Blueprint Open-Sourcing:**
- Every element—architecture, training configs, data mixtures, and even code (data mixture weights, training logs, and intermediate checkpoints) is being made public, aimed specifically at enabling external researchers to reproduce and extend the engineering pathway.

---

**In summary:** The blog is a goldmine of engineering best practices for small LLMs, especially:
- Extensive, empirical ablation at scale,
- Stepwise curriculum learning and context extension,
- Filling data mode gaps with large model-driven synthetic traces,
- Candidly reporting on performance/efficiency trade-offs and how to remediate them through model merging.

These points collectively offer a rare, fully open recipe—and lessons—for constructing competitive small models with dual reasoning modes and very long context windows.
   </research>
   <research source="https://github.com/huggingface/lighteval">
Lighteval offers several notable features and design choices that may be of particular interest to an AI model researcher:

- Multi-Backend Flexibility: Lighteval can evaluate LLMs across a range of backends—including Hugging Face Transformers, Text Generation Inference (TGI), vLLM, nanotron, and even the OpenAI API. This makes it highly attractive for benchmarking models in a reproducible and backend-agnostic way, lowering friction in comparing research or industry APIs.

- Modular, Extensible Tasks and Metrics: Lighteval enables rapid development and integration of custom evaluation tasks or metrics—researchers can define their own datasets or measurement scripts, tailoring benchmarks precisely to experimental needs rather than being limited to a fixed suite.

- Sample-by-Sample Debugging: Beyond aggregate metrics, Lighteval saves and allows exploration of individual sample results. This “deep dive” feature can be invaluable for error analysis, prompt leak detection, and surfacing subtle benchmark weaknesses or model artifacts that gross accuracy might hide.

- Remote & Local Result Management: Evaluation results can be stored and shared seamlessly via the Hugging Face Hub, on S3, or locally, facilitating collaboration and reproducibility for research groups working in distributed settings.

- Community-Driven: Lighteval originated as a reimagining of EleutherAI’s LM Evaluation Harness (which powers the Open LLM Leaderboard) but has since evolved as a standalone project with significant inspiration drawn from Stanford’s HELM framework. Contributions of new tasks and metrics are actively encouraged, helping the tool to keep pace with evolving research interests and benchmark requirements.

- Command-Line and Python API: Both interfaces are supported, making it easy to integrate Lighteval into existing evaluation pipelines, CI systems, or interactive experiments.

- Not Windows-Compatible: As of now, Lighteval is not tested or officially supported on Windows—worth noting for cross-platform research groups.

Surprising Point: The commitment to both backwards compatibility with Leaderboard infrastructure and forward momentum on extensibility means that Lighteval sits at the intersection of rigorous benchmarking and research-driven customization—ideal for methodically tracking progress on new architectures or fine-tuning strategies emerging in the LLM research community.
   </research>
   <research source="https://github.com/huggingface/nanotron/tree/smollm3">
Here are several notable and potentially surprising points from the HuggingFace nanotron repository (smollm3 branch) relevant to AI model researchers, not already covered in the provided SmolLM3 model card:

1. **Open, Minimalistic LLM Training Framework**:
   - Nanotron offers a minimalistic design for large language model (LLM) pretraining, emphasizing simplicity without sacrificing performance or flexibility. This is unusual compared to more complex frameworks like DeepSpeed or Megatron-LM.

2. **3D Parallelism + Expert Parallelism**:
   - Full support for 3D parallelism (data, tensor, and pipeline parallelism) and expert parallelism specifically targeting Mixture-of-Experts (MoE) models. This framework enables explicit control of Tensor Parallelism (TP) and Pipeline Parallelism (PP)—making debugging at this level easier.

3. **Advanced Training Features**:
   - Features support for both AFAB (All-Finish-All-Batch) and classic 1F1B (1-Forward-1-Backward) schedules for pipeline parallelism.
   - Built-in ZeRO-1 optimizer, parameter tying/sharding, and explicit FP32 gradient accumulation for large model optimization.
   - Spectral µTransfer parametrization is natively supported, facilitating scaling laws research.

4. **Model and Architecture Support Beyond Transformers**:
   - Nanotron includes a worked example for training Mamba-based architectures, making it useful for experimenting with new LLM architectures beyond pure Transformer models.

5. **Comprehensive Benchmarks and “Ultrascale Playbook”**:
   - Nanotron provides detailed and openly available benchmarks, including configurations and logs for a range of hardware/node counts, with the goal of maximizing Model FLOPS Utilization (MFU) and memory throughput.
   - Their “Ultrascale Playbook” (hosted as a HuggingFace space) provides actionable advice and templates for scaling LLM training efficiently at various scales—providing practical field-tested guidance.

6. **Custom Data Loading, Flexible Examples, & S3 Integration**:
   - Included examples for custom dataloaders, integration with the “datatrove” library, Mixture-of-Experts, DoReMi (data selection), automatic checkpoint upload to S3, and more.
   - Users are encouraged to submit their own examples, highlighting ongoing community engagement.

7. **Debugging and Developer UX**:
   - Supports robust debugging with VSCode and includes recommendations for multi-GPU debugging, which is not commonly emphasized in LLM training frameworks.

8. **Roadmap & Future Directions**:
   - FP8 training, ZeRO-3 optimization (FSDP), `torch.compile` compatibility, ring attention, and interleaved pipeline scheduling are all on the roadmap, showing active ongoing innovation.

9. **Licensing and Accessibility**:
   - Fully Apache 2.0-licensed, emphasizing open science and accessibility—lowering the barrier for research and industrial experimentation with state-of-the-art LLM training at scale.

These details make Nanotron a distinctive project in the LLM training ecosystem: researchers can experiment with advanced and novel model architectures, optimization techniques, and efficient large-scale training—using a framework still far simpler and more transparent than industry behemoths.

In summary, Nanotron's combination of ease of use, full support for advanced parallelism paradigms, open benchmarking, and a willingness to push the boundaries with new architectures/components (e.g., Mamba, spectral µTransfer, AFAB, MoE) make it an unusually open and experimental platform for cutting-edge LLM research.
   </research>
   <research source="https://github.com/huggingface/datatrove">
Summary of https://github.com/huggingface/datatrove (highlighting interesting/surprising points for AI Model Researchers):

- DataTrove is designed for extremely large-scale text data processing, focusing on efficient filtering, extraction, and especially deduplication—important for minimizing contamination between training, validation, and test sets for LLMs.
- It is platform-agnostic: the same pipeline can run unchanged locally, on a slurm cluster, or with Ray for distributed compute, making it highly adaptable to both academic labs and industry-scale hardware.
- A unique aspect is the ability to resume failed runs: Datatrove tracks individual “tasks” and only reruns incomplete ones, minimizing wasted compute—very useful when processing terabytes of data.
- It supports advanced deduplication out-of-the-box, including minhash, sentence-level, and even exact substring deduplication (with integration to Google Research’s deduplicate-text-datasets tools)—addressing subtle data leakage and redundancy issues that are hard to handle otherwise.
- There’s native support for reading/writing to local, remote, and cloud file systems (including s3), enabling direct integration with cloud storage/data lakes.
- By leveraging standardized “Document” objects (with text, id, and metadata), you can build highly modular and composable pipelines, including custom filters, extractors, and stats collectors.
- Notably, Datatrove can read/write a wide variety of formats out-of-the-box, including WARC/ARC/WET (web crawl formats), Arrow/Parquet, and more, handling modern large-scale textual datasets without custom code.
- For multi-node parallelism, partial pipelines can be run independently on different machines and later merged, enabling true horizontal scale-out for preprocessing.

These make DataTrove much more robust and scalable than naïve data processing scripts and very attractive for anyone preparing data for new LLM training runs, especially when curating or deduplicating massive multilingual or web-scale datasets. The modular architecture makes incorporating new quality, filtering, or deduplication methods straightforward for researchers.
   </research>
   <research source="https://github.com/huggingface/trl">
Here is a summary of unique, noteworthy points from the HuggingFace TRL (Transformer Reinforcement Learning) repository that would interest an AI model researcher:

1. **Breadth of Post-Training Algorithms**: TRL offers not only the well-known PPO (Proximal Policy Optimization) but also advanced and recently popularized techniques such as Direct Preference Optimization (DPO, as used in Llama 3), Group Relative Policy Optimization (GRPO, as used in DeepSeek’s R1), and Supervised Fine-Tuning (SFT), each with a dedicated trainer class.

2. **Memory and Hardware Efficiency**:
   - TRL natively supports scaling from single GPU to large multi-node clusters, leveraging Accelerate, DeepSpeed, and DDP.
   - It integrates PEFT (Parameter-Efficient Fine-Tuning) methods like LoRA and QLoRA, making it feasible to train large models (e.g., Llama, Qwen) on commodity hardware.
   - TRL incorporates Unsloth for highly optimized training kernels, further reducing resource requirements for fine-tuning.

3. **Model and Modality Agnosticism**:
   - While tightly integrated with HuggingFace Transformers, TRL’s architecture allows it to support a broad variety of model architectures and modalities beyond basic text generation.

4. **CLI for Rapid Experimentation**: 
   - Researchers can experiment with SFT and DPO through a simple command-line interface, enabling quick iteration without writing boilerplate code.

5. **Reward Modeling**:
   - TRL includes a dedicated trainer for reward model training (RewardTrainer), facilitating RLHF (Reinforcement Learning from Human Feedback) pipelines and experiments with custom reward functions.

6. **Community & Scale**:
   - The project is under active development with over 373 contributors, 14.7k stars, and regular releases—a strong indicator of a healthy ecosystem and rapid improvement/adoption.

7. **Cutting-Edge Usage Examples**: 
   - Notable models, such as DeepSeek R1 (using GRPO) and Llama 3 (using DPO), leveraged TRL’s trainers, making the library a reference implementation for state-of-the-art techniques in preference-based LLM optimization.

8. **Direct Access to Labeled Datasets**: 
   - Example scripts and default integrations show TRL can work seamlessly with datasets such as Capybara, UltraFeedback, and TLDR for preference and supervised learning.

9. **Open and Extensible**:
   - All trainers are light wrappers around transformers’ trainers, making TRL extensible and easy to integrate into existing pipelines.

10. **Support for Quantization and Adapter Training**:
    - Out-of-the-box, TRL lets you combine quantization methods and adapter-based training (e.g., QLoRA), a key enabler for low-cost and large-scale training scenarios.

These capabilities make TRL not just a “PPO for transformers” library, but a comprehensive, production-ready framework for the full spectrum of LLM post-training techniques—highlighting trends toward openness, efficiency, and rapid research translation into large-scale real-world models.
   </research>
   <research source="https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets-685a7353fdc01aecde51b1d9">
Interesting/surprising points for an AI model researcher from the SmolLM3 pretraining dataset collection page:

- The collection draws on a diverse and up-to-date mix of new and specialized datasets that go far beyond a generic web scrape. In particular, it includes high-quality educational, coding, and mathematical datasets like:
  - HuggingFaceFW/fineweb-edu (for web-based educational data at massive scale),
  - fine-tuned math and code datasets such as HuggingFaceTB/finemath, LLM360/MegaMath, nvidia/OpenMathReasoning, nvidia/OpenCodeReasoning,
  - kaggle notebook issues (HuggingFaceTB/issues-kaggle-notebooks), indicating a direct dataset of code troubleshooting and problem-solving,
  - facebook/natural_reasoning and EPFL's FineWeb2-HQ for advanced language and reasoning capability.

- There is clear staging of dataset usage: Stage 2 and Stage 3 (decay) introduce new, possibly more domain-specific or reasoning-heavy data. This staged approach, with later "decay" datasets, demonstrates a curriculum or changing mixture during pretraining, which is not always explicitly tracked or shared in open models.

- The presence of curated datasets like "HuggingFaceTB/stack-edu", "allenai/dolmino-mix-1124", "HuggingFaceTB/smollm-corpus", and FineWeb2-HQ suggests intentional filtering and quality focus, rather than broad web scraping.

- The inclusion of "issues-kaggle-notebooks" hints at training the model on real-world problem-solving conversations and debugging sessions, a non-trivial source for functional code intelligence.

- The combination of proprietary, third party, and community-maintained resources (such as bigcode/the-stack-v2, nvidia, facebook, allenai datasets) points to an unusually rich cross-pollination of dataset origins.

In summary: The dataset composition for SmolLM3's pretraining reveals a staged, multi-source, quality-aware approach with surprising emphasis on math, code, educational, and real-world troubleshooting data – and explicit stages/decay mixtures, all of which may contribute to its advanced reasoning and coding performance relative to its scale.
   </research>
   <research source="https://github.com/huggingface/smollm">
Summary of https://github.com/huggingface/smollm — Key Points for AI Model Researchers:

- The repository covers both the SmolLM* (text) and SmolVLM* (vision-language) model families, with resources for local inference and lightweight AI-powered tool integration.

- SmolVLM, the vision-language model, is a compact, on-device multimodal model with support for visual question answering, image description, visual storytelling, and processing multiple images within a single conversation—fitting the “smol” (small) ethos but rarely mentioned in broader summaries.

- The repository structure is modular, cleanly separating code for text, vision, and shared inference tooling. Dedicated folders for local inference suggest an emphasis on accessibility and efficient deployment.

- Several open-source datasets were created or curated specifically for SmolLM: 
    - “SmolTalk” (instruction tuning)
    - “FineMath” (mathematics pretraining)
    - “FineWeb-Edu” (educational pretraining)
  These highlight a willingness to share not just models, but also original fine-tuning and pretraining corpora—uncommon for many model releases.

- The codebase heavily uses Python, but also includes shell scripts and Jupyter notebooks, indicating support for reproducible research and easy experimentation.

- There is visible transparency in model development: contributors (18+) are public, with almost 3,000 GitHub stars, and all model training details, data mixtures, and configs are made available for deep inspection and reuse.

- The Smol Models project is actively maintaining a vision (VLM) branch alongside LLM work, suggesting an integrated roadmap for compact, fully open multimodal AI.

(Points related to model scale, training tokens, dual mode reasoning, and performance are in the model card and therefore are not repeated.)
   </research>
   <research source="https://www.apache.org/licenses/LICENSE-2.0">
Summary of Apache License 2.0 (with focus on unique/research-relevant points for AI model researchers):

- Patent Grant (and Termination): The license grants a royalty-free, perpetual, worldwide patent license for contributions—this is crucial when distributing or using models that may incorporate patented methods. However, this right is immediately terminated if you initiate patent litigation over the work. This "patent peace" clause can be particularly important for AI model researchers working in potentially patent-heavy domains.

- Contribution Definition: “Contribution” is defined as any work submitted for inclusion, including code, documentation, and configuration files, unless marked as “Not a Contribution.” This is broader than some other open source licenses and encourages clarity about intent, which matters when sharing training code, model weights, or dataset scripts.

- Trademark Restriction: The license does not grant any rights to the project’s trademarks, service marks, or product names except for basic descriptive use. Using the name of a popular model (or dataset) in branding or research output may require separate permission.

- Warranty and Liability: All contributions and redistributions are strictly “AS IS”—there’s no warranty or liability. For AI researchers, this means there’s no legal recourse if a model causes harm, bias, or other negative outcomes.

- Notice File Mechanism: If a NOTICE file is distributed, all derivative works must include the NOTICE file contents. This can result in propagation of attributions (including third-party references) through the lineage of derivative models or datasets, which may be missed in scientific reporting.

- Freedom to Add License Terms: You may distribute your own modifications/derivative works with additional terms—as long as you also comply with Apache 2.0 for upstream code. This modularity allows derived AI models to add further restrictions, if desired, potentially making the downstream licensing landscape more complex.

- Boilerplate Application: To apply the license to model code or model weights, the license text should be included with specific attribution. If distributing models with shared weights or in non-code formats, there's room for ambiguity about what it means to “attach the license,” so AI researchers should take care to follow best practices for license notices.

Interesting or surprising aspect: The patent clause and its automatic termination in the event of litigation is one of the most legally significant points and distinguishes Apache 2.0 from more permissive licenses like MIT, especially relevant as AI models frequently risk infringing process or functional patents.

Overall, the Apache 2.0 license’s provisions affect not only the sharing of code, but also of datasets, weights, and other research outputs—any reuse or distribution pathway found in modern AI research.
   </research>
</model_research>
```

Your task is to produce a concise 300-500 word summary of the specified model highlighting 
key points of interest for AI researchers. Make sure that major information and traits are
specified, and prefer bullet points for notable points of interest. Use markdown links to 
refer to sources if warranted.


---ASSISTANT
Here is a summary of HuggingFaceTB/SmolLM3-3B with key points tailored for AI model researchers, focusing on unique and impactful technical aspects:

---

## SmolLM3 (3B) — Researcher-Focused Summary

SmolLM3-3B stands out as a fully open, small-scale (3B parameter) language model that innovates in dual-mode reasoning, long-context scaling, and transparent engineering practices. Its design and training pipeline are valuable sources of learning and experimentation for researchers pursuing efficient, high-performance LLMs.

---

### Architectural and Training Innovations

- **Grouped Query Attention (GQA)**: Uses 4-group GQA, significantly reducing KV cache requirements *without* degrading performance, as validated by ablation studies over 100B tokens [source](https://hf.co/blog/smollm3).
- **NoPE RoPE Adaptation**: Adopts Yang et al.'s NoPE, removing rotary position embeddings in every fourth transformer layer. This improves long-context modeling while preserving short-context performance.
- **Curriculum & Long-Context Specialization**:
  - **Staged pretraining**: A three-stage, ablation-guided data mixture (math, code, web, reasoning), with late-stage *decay* aggressively upsampling code and math.
  - **Sequential context extension**: Mid-training lengthens context (4k → 32k → 64k) by retraining with gradually increased RoPE scaling, enabling context windows up to 128k tokens.
  - **Model merging for trade-off recovery**: Found that reasoning/alignment steps degrade long-context ability; this is remedied via a *model soup* blending strong long-context and aligned checkpoints.

---

### Dual-Mode Reasoning

- **Dual-Mode Pipeline**: Supports both reasoning-rich and fast response modes—not merely via system prompts, but through targeted mid-training (140B reasoning tokens), SFT with substantial synthetic reasoning traces, and final Anchored Preference Optimization (APO) alignment.
- **Synthetic Trace Augmentation**: Fills data mode gaps (especially for non-math/code reasoning) by prompting large models (e.g., Qwen3-32B) on non-reasoning datasets, broadening domain reasoning capabilities.

---

### Data, Evaluation & Transparency

- **Data Mixture Detail**:
  - Draws from a staged, multi-source curation: e.g., FineWeb-Edu, fine-math/code sets, “issues-kaggle-notebooks” (code troubleshooting), and domain-specific collections [source](https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets-685a7353fdc01aecde51b1d9).
  - Data deduplication, contamination checks, and distributed data prep via [DataTrove](https://github.com/huggingface/datatrove), supporting minhash and sentence-level dedup on terabyte-scale corpora.

- **Evaluation and Tool Use**:
  - Benchmarked with [lighteval](https://github.com/huggingface/lighteval), supporting plug-and-play tasks, error analysis, and backend-agnostic comparison.
  - Native agentic/tool use: supports robust tool-calling schemas and code snippets for practical AI assistant pipelines.

- **Frameworks**:
  - Trained using [nanotron](https://github.com/huggingface/nanotron/tree/smollm3) (minimal, flexible LLM training with 3D+Expert parallelism), and [TRL](https://github.com/huggingface/trl) (DPO, APO, PPO)—all fully open-source under Apache 2.0.

---

### Open Science & Reproducibility

- **Blueprint Transparency**: Full publication of model weights, intermediate checkpoints, training configs, code, custom datasets, and pretraining logs. All development and evaluation details are public ([code](https://github.com/huggingface/smollm)).
- **License**: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), with patent peace and minimal redistribution constraints.

---

**In summary:**  
SmolLM3-3B is a rare example of a research-grade, small LLM with dual-mode reasoning, long context, transparent engineering, synthetic-data augmentation, open recipes, and truly reproducible pipelines—uniquely positioned as both a benchmark and a building block for LLM research.