---USER

Here is the model card for Qwen/Qwen3-Coder-480B-A35B-Instruct:

```
---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/blob/main/LICENSE
pipeline_tag: text-generation
---

# Qwen3-Coder-480B-A35B-Instruct
<a href="https://chat.qwen.ai/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/%F0%9F%92%9C%EF%B8%8F%20Qwen%20Chat%20-536af5" style="display: inline-block; vertical-align: middle;"/>
</a>

## Highlights

Today, we're announcing **Qwen3-Coder**, our most agentic code model to date. **Qwen3-Coder** is available in multiple sizes, but we're excited to introduce its most powerful variant first: **Qwen3-Coder-480B-A35B-Instruct**. featuring the following key enhancements:  

- **Significant Performance** among open models on **Agentic Coding**, **Agentic Browser-Use**, and other foundational coding tasks, achieving results comparable to Claude Sonnet.
- **Long-context Capabilities** with native support for **256K** tokens, extendable up to **1M** tokens using Yarn, optimized for repository-scale understanding.
- **Agentic Coding** supporting for most platform such as **Qwen Code**, **CLINE**, featuring a specially designed function call format.

![image/jpeg](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Coder/qwen3-coder-main.jpg)

## Model Overview

**Qwen3-480B-A35B-Instruct** has the following features:
- Type: Causal Language Models
- Training Stage: Pretraining & Post-training
- Number of Parameters: 480B in total and 35B activated
- Number of Layers: 62
- Number of Attention Heads (GQA): 96 for Q and 8 for KV
- Number of Experts: 160
- Number of Activated Experts: 8
- Context Length: **262,144 natively**. 

**NOTE: This model supports only non-thinking mode and does not generate ``<think></think>`` blocks in its output. Meanwhile, specifying `enable_thinking=False` is no longer required.**

For more details, including benchmark evaluation, hardware requirements, and inference performance, please refer to our [blog](https://qwenlm.github.io/blog/qwen3-coder/), [GitHub](https://github.com/QwenLM/Qwen3-Coder), and [Documentation](https://qwen.readthedocs.io/en/latest/).


## Quickstart

We advise you to use the latest version of `transformers`.

With `transformers<4.51.0`, you will encounter the following error:
```
KeyError: 'qwen3_moe'
```

The following contains a code snippet illustrating how to use the model generate content based on given inputs. 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-480B-A35B-Instruct"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Write a quick sort algorithm."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=65536
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)
```

**Note: If you encounter out-of-memory (OOM) issues, consider reducing the context length to a shorter value, such as `32,768`.**

For local use, applications such as Ollama, LMStudio, MLX-LM, llama.cpp, and KTransformers have also supported Qwen3.

## Agentic Coding

Qwen3-Coder excels in tool calling capabilities. 

You can simply define or use any tools as following example.
```python
# Your tool implementation
def square_the_number(num: float) -> dict:
    return num ** 2

# Define Tools
tools=[
    {
        "type":"function",
        "function":{
            "name": "square_the_number",
            "description": "output the square of the number.",
            "parameters": {
                "type": "object",
                "required": ["input_num"],
                "properties": {
                    'input_num': {
                        'type': 'number', 
                        'description': 'input_num is a number that will be squared'
                        }
                },
            }
        }
    }
]

import OpenAI
# Define LLM
client = OpenAI(
    # Use a custom endpoint compatible with OpenAI API
    base_url='http://localhost:8000/v1',  # api_base
    api_key="EMPTY"
)
 
messages = [{'role': 'user', 'content': 'square the number 1024'}]

completion = client.chat.completions.create(
    messages=messages,
    model="Qwen3-480B-A35B-Instruct",
    max_tokens=65536,
    tools=tools,
)

print(completion.choice[0])
```

## Best Practices

To achieve optimal performance, we recommend the following settings:

1. **Sampling Parameters**:
   - We suggest using `temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.05`.

2. **Adequate Output Length**: We recommend using an output length of 65,536 tokens for most queries, which is adequate for instruct models.


### Citation

If you find our work helpful, feel free to give us a cite.

```
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report}, 
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09388}, 
}
```
```


---ASSISTANT
Thank you. I will refer to this model card in my future responses.
---ASSISTANT
Thank you. I will refer to this model card in my future responses.
---USER
Here is additional research information about the model:

```


<model_research>
   <research source="https://qwenlm.github.io/blog/qwen3-coder/">
Here are the most interesting and potentially surprising points from the Qwen3-Coder blog, relevant for AI model researchers (without repeating obvious information from the model card):

- Large-Scale Code Reinforcement Learning (Code RL): Qwen3-Coder applies execution-centric reinforcement learning at scale by generating high-quality, diverse training instances through automated test case generation on a wide set of real-world coding tasks. This approach is distinct from the community's usual focus on competitive code generation and leads to significant gains in code execution success and other task performances. The authors specifically highlight "hard-to-solve, easy-to-verify" coding tasks as an open field for further large-scale RL innovations.

- Long-Horizon Reinforcement Learning for Agentic Coding: The post-training phase introduces long-horizon multi-turn RL (Agent RL), enabling the model to use tools, plan, and interact within complex environments comparable to real-world software engineering (e.g., SWE-Bench). Notably, they built an infrastructure capable of running 20,000 environments in parallel on Alibaba Cloud to provide the scale and feedback loop needed for this work.

- Data Pipeline Innovations: The team used a preceding model (Qwen2.5-Coder) to clean and rewrite noisy data in their 7.5T-token (70% code) training set, showing commitment to synthetic data quality as a cornerstone for downstream performance.

- Open-Source Agentic CLI Tools: Qwen Code, adapted from Gemini Code, is open-source and designed specifically for agentic code interactions with the model. The blog also describes out-of-the-box compatibility with Claude Code and the ability to customize model routing via npm packages, offering unusually broad toolchain integration for agentic coding models.

- Ambition for “Self-Improving Agents”: The authors hint they are actively researching whether Qwen3-Coder’s agentic coding abilities can be directed towards self-improvement, explicitly outlining this as a bold, long-term goal not often stated by similar projects.

- Impressive Real-World Demos: The blog features video demos spanning from physics-based simulations and game creation to web development and novel interaction tasks—all done agentically. The breadth and nature of these demos signal the model's practical multi-domain competence beyond code completion.

These aspects—particularly the scale of RL infrastructure, focus on agentic, multi-turn coding tasks, and the vision for self-improving agents—make Qwen3-Coder stand out in the current open-model landscape and may inform future research directions in code LLMs and agentic tool-use.
   </research>
   <research source="https://github.com/QwenLM/Qwen3-Coder">
Here are noteworthy points for AI model researchers from the Qwen3-Coder GitHub repository that go beyond what is on the model card:

- Qwen3-Coder claims support for an impressive 358 coding languages, which extends well beyond the typical major languages usually supported by similar models. The explicit list contains many rarely-supported and domain-specific languages, including 'Brainfuck', 'LOLCODE', 'BitBake', 'EmberScript', 'SuperCollider', 'YANG', and 'Thrift', plus various markup and configuration languages (e.g., Nginx, TOML, Org). The extent of effective support for so many languages may inspire further investigation.

- The repository references a unique tool parser, qwen3coder_tool_parser.py, which is essential for the model's function calling capabilities, hinting at a bespoke, possibly extensible approach to tool-use not visible in the main model card.

- Fill-in-the-middle (FIM) prompting is directly supported, compatible with the approach in “Efficient Training of Language Models to Fill in the Middle”, including dedicated FIM special tokens like <|fim_prefix|>, <|fim_suffix|>, and <|fim_middle|>. This explicit FIM support is not commonly present in all large code models and may be of methodological interest.

- The codebase itself is multilingual, with a substantial portion of the codebase written in Go (~23% Go vs. ~72% Python), which is unusual for open-source LLM-related repositories that are generally Python-dominated. This could indicate interesting architectural or performance choices.

- Community engagement looks active: the repository boasts over 6,500 stars, 465 forks, 320 commits, and contributions from over 26 collaborators, indicating strong and growing interest, which may affect ecosystem development.

- There are two instruct model variants available: Qwen3-Coder-480B-A35B-Instruct (standard) and Qwen3-Coder-480B-A35B-Instruct-FP8 (presumably in 8-bit floating point for efficiency), signaling attention to both state-of-the-art performance and deployment efficiency.

- Releases are not published as GitHub Releases, so researchers must source checkpoints from Hugging Face or ModelScope per instructions.

- The repository encourages use of the “ChatML” template and has its own conventions for prompt formatting, which can impact benchmarking or prompt engineering strategies.

- Tokenizer special tokens and IDs have been updated specifically for Qwen3 compatibility, which can introduce subtle issues for those attempting to swap in other tokenizers, a detail sometimes overlooked.

- The project maintains a direct line to the research team via Discord and WeChat for researcher interaction.

These details could provide new angles of investigation and experimentation for AI researchers working on large language models, especially in code generation, tool-use, and multilingual support.
   </research>
   <research source="https://qwen.readthedocs.io/en/latest/">
Summary of Unique or Notable Points from Qwen Documentation (Interesting to AI Model Researchers):

- Qwen3 models (including both language and multimodal) seamlessly support both dense and Mixture-of-Experts (MoE) architectures, with released sizes including 0.6B, 1.7B, 4B, 8B, 14B, 32B, as well as MoE-specific 30B-A3B, 235B-A22B. This suggests a wide-ranging, scalable experimentation across dense and MoE paradigms within a consistent framework.

- Unique feature: Qwen3 supports “seamless switching” between “thinking mode” (complex reasoning, including math and coding) and “non-thinking mode” (efficient, general-purpose chat) within a single model. This dynamic, mode-based operation is distinctive and potentially impactful for creating more adaptable, multi-modal AI agents.

- Qwen3’s reasoning capabilities (in both thinking and non-thinking modes) are significantly improved compared to prior Qwen releases (notably "QwQ" and Qwen2.5-Instruct models).

- Qwen3 claims strong alignment with human preference specifically for creative writing, role play, dialogue, and instruction following, moving toward more natural and immersive AI-human interaction.

- Advanced tool-use and agent capabilities are emphasized: Qwen3 is described as achieving leading performance in complex agent-based tasks among open-source models, explicitly supporting “precise integration with external tools” in both modes.

- Multilingual and dialect support: Qwen3 supports 100+ languages and dialects, with notable strength in multilingual instruction following and translation, indicating a design goal for robust global applicability and accessibility.

These points highlight Qwen3's unique dual-mode operation, broad MoE/dense architecture support, and a strong focus on tool/agent integration—features that are both technically ambitious and unusual among current open-source LLM offerings.
   </research>

</model_research>


```

Your task is to produce a concise 300-500 word summary of the specified model highlighting 
key points of interest for AI researchers. Make sure that major information and traits are
specified, and prefer bullet points for notable points of interest. Use markdown links to 
refer to sources if warranted.


---ASSISTANT
# Qwen3-Coder-480B-A35B-Instruct: Researcher-Relevant Summary

Qwen3-Coder-480B-A35B-Instruct is a cutting-edge open-source code-focused language model developed by the Qwen team, offering unprecedented performance, extensibility, and research opportunities in agentic coding, tool-use, and multilingual support.

## Major Features and Architecture

- **Scale & Capacity**: 480B total parameters (35B activated, Mixture-of-Experts), 62 layers, 96+8 GQA heads, 160 experts with 8 active per input. Long context support: 262,144 tokens natively (upgradable to 1M with Yarn).
- **Model Flexibility**: Supports both dense and MoE architectures. The framework spans smaller models (0.6B–32B) up to heavyweight MoE variants (including a unique FP8 instruct version for efficiency) ([doc](https://qwen.readthedocs.io/en/latest/), [repo](https://github.com/QwenLM/Qwen3-Coder)).

## Distinctive Innovations

- **Large-Scale Code Reinforcement Learning (RL)**:
    - Utilizes execution-centric RL at scale, focusing on practical code execution success instead of only competitive programming tasks.
    - Introduces automated test case generation pipeline and applies RL to “hard-to-solve, easy-to-verify” tasks ([blog](https://qwenlm.github.io/blog/qwen3-coder/)).
- **Agentic Coding & Multi-Turn RL**:
    - Advanced tool-calling and planning capabilities, including long-horizon “Agent RL” for realistic, multi-step software engineering tasks (e.g., SWE-Bench).
    - Infrastructure can run 20,000 agentic environments in parallel (see [blog](https://qwenlm.github.io/blog/qwen3-coder/)).

- **Agentic Tool Use & Integration**:
    - Supports seamless function/tool calling, custom tool parser (qwen3coder_tool_parser.py), and out-of-the-box integration with major agentic coding toolchains (Qwen Code, Claude Code, customizable npm routing).
    - Fill-in-the-middle (FIM) prompting support with explicit special tokens.

- **Multilingual/Dialect & Language Support**:
    - Official support for 358 programming languages (including rare ones like Brainfuck, SuperCollider, and config/markup dialects) and strong performance on natural languages (over 100+ for chat/translation).

- **Data Curation and Synthetic Data Quality**:
    - Uses a self-cleaning pipeline where Qwen2.5-Coder helps rewrite and clean large datasets, demonstrating a high-quality synthetic data strategy.

## Methodological & Community Notes

- **Mode Switching**: Supports dynamic switching between “thinking” (reasoning-intensive) and “non-thinking” (general chat) modes, though 480B-A35B-Instruct is strictly non-thinking ([docs](https://qwen.readthedocs.io/en/latest/)).
- **Active Community & Support**: Over 6,500 GitHub stars, accessible research team channels (Discord, WeChat), and multilingual documentation.
- **Prompt Engineering**: Uses its own ChatML template and carefully tuned tokenization, requiring attention for cross-model compatibility.

## Research Implications

Qwen3-Coder sets new research directions for:
- Execution-centered RL for code generation/agent tasks at scale,
- Highly agentic/self-improving coding models,
- Multilingual/dialectal and multi-language code support,
- Efficient MoE deployment and tool parsing architectures.

**References:** [Blog](https://qwenlm.github.io/blog/qwen3-coder/), [GitHub](https://github.com/QwenLM/Qwen3-Coder), [Docs](https://qwen.readthedocs.io/en/latest/)