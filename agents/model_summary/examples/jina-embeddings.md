---USER

Here is the model card for jinaai/jina-embeddings-v4:

```
---
license: cc-by-nc-4.0
tags:
- vidore
- colpali
- multimodal-embedding
- multilingual-embedding
- Text-to-Visual Document (T→VD) retrieval
- feature-extraction
- sentence-similarity
- mteb
- sentence-transformers
language:
- multilingual
inference: false
library_name: transformers
pipeline_tag: visual-document-retrieval
---
<br><br>

<p align="center">
<img src="https://huggingface.co/datasets/jinaai/documentation-images/resolve/main/logo.webp" alt="Jina AI: Your Search Foundation, Supercharged!" width="150px">
</p>


<p align="center">
<b>The embedding model trained by <a href="https://jina.ai/"><b>Jina AI</b></a>.</b>
</p>

# Jina Embeddings v4: Universal Embeddings for Multimodal Multilingual Retrieval


## Quick Start

[Blog](https://jina.ai/news/jina-embeddings-v4-universal-embeddings-for-multimodal-multilingual-retrieval) | [Technical Report](https://arxiv.org/abs/2506.18902) | [API](https://jina.ai/embeddings)


## Intended Usage & Model Info
`jina-embeddings-v4` is a universal embedding model for multimodal and multilingual retrieval. 
The model is specially designed for complex document retrieval, including visually rich documents with charts, tables, and illustrations. 


Built on [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), `jina-embeddings-v4` features:

- **Unified embeddings** for text, images, and visual documents, supporting both dense (single-vector) and late-interaction (multi-vector) retrieval.
- **Multilingual support** (30+ languages) and compatibility with a wide range of domains, including technical and visually complex documents.
- **Task-specific adapters** for retrieval, text matching, and code-related tasks, which can be selected at inference time.
- **Flexible embedding size**: dense embeddings are 2048 dimensions by default but can be truncated to as low as 128 with minimal performance loss.


Summary of features:

| Feature   | Jina Embeddings V4   |
|------------|------------|
| Base Model | Qwen2.5-VL-3B-Instruct |
| Supported Tasks | `retrieval`, `text-matching`, `code` |
| Model DType | BFloat 16 |
| Max Sequence Length | 32768 |
| Single-Vector Dimension | 2048 |
| Multi-Vector Dimension | 128 |
| Matryoshka dimensions | 128, 256, 512, 1024, 2048 |
| Pooling Strategy | Mean pooling |
| Attention Mechanism | FlashAttention2 |



## Training & Evaluation

Please refer to our [technical report of jina-embeddings-v4](https://arxiv.org/abs/2506.18902) for training details and benchmarks.


## Usage

<details>
  <summary>Requirements</a></summary>
  
The following Python packages are required:

- `transformers>=4.52.0`
- `torch>=2.6.0`
- `peft>=0.15.2`
- `torchvision`
- `pillow`
  
### Optional / Recommended
- **flash-attention**: Installing [flash-attention](https://github.com/Dao-AILab/flash-attention) is recommended for improved inference speed and efficiency, but not mandatory.
- **sentence-transformers**: If you want to use the model via the `sentence-transformers` interface, install this package as well.

</details>


<details>
  <summary>via <a href="https://jina.ai/embeddings/">Jina AI Embeddings API</a></summary>


```bash
curl https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JINA_AI_API_TOKEN" \
  -d @- <<EOFEOF
  {
    "model": "jina-embeddings-v4",
    "task": "text-matching",
    "input": [
        {
            "text": "غروب جميل على الشاطئ"
        },
        {
            "text": "海滩上美丽的日落"
        },
        {
            "text": "A beautiful sunset over the beach"
        },
        {
            "text": "Un beau coucher de soleil sur la plage"
        },
        {
            "text": "Ein wunderschöner Sonnenuntergang am Strand"
        },
        {
            "text": "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία"
        },
        {
            "text": "समुद्र तट पर एक खूबसूरत सूर्यास्त"
        },
        {
            "text": "Un bellissimo tramonto sulla spiaggia"
        },
        {
            "text": "浜辺に沈む美しい夕日"
        },
        {
            "text": "해변 위로 아름다운 일몰"
        },
        {
            "image": "https://i.ibb.co/nQNGqL0/beach1.jpg"
        },
        {
            "image": "https://i.ibb.co/r5w8hG8/beach2.jpg"
        }
    ]
  }
EOFEOF
```

</details>

<details>
  <summary>via <a href="https://huggingface.co/docs/transformers/en/index">transformers</a></summary>

```python
# !pip install transformers>=4.52.0 torch>=2.6.0 peft>=0.15.2 torchvision pillow
# !pip install
from transformers import AutoModel
import torch

# Initialize the model
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True, torch_dtype=torch.float16)

model.to("cuda")

# ========================
# 1. Retrieval Task
# ========================
# Configure truncate_dim, max_length (for texts), max_pixels (for images), vector_type, batch_size in the encode function if needed

# Encode query
query_embeddings = model.encode_text(
    texts=["Overview of climate change impacts on coastal cities"],
    task="retrieval",
    prompt_name="query",
)

# Encode passage (text)
passage_embeddings = model.encode_text(
    texts=[
        "Climate change has led to rising sea levels, increased frequency of extreme weather events..."
    ],
    task="retrieval",
    prompt_name="passage",
)

# Encode image/document
image_embeddings = model.encode_image(
    images=["https://i.ibb.co/nQNGqL0/beach1.jpg"],
    task="retrieval",
)

# ========================
# 2. Text Matching Task
# ========================
texts = [
    "غروب جميل على الشاطئ",  # Arabic
    "海滩上美丽的日落",  # Chinese
    "Un beau coucher de soleil sur la plage",  # French
    "Ein wunderschöner Sonnenuntergang am Strand",  # German
    "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία",  # Greek
    "समुद्र तट पर एक खूबसूरत सूर्यास्त",  # Hindi
    "Un bellissimo tramonto sulla spiaggia",  # Italian
    "浜辺に沈む美しい夕日",  # Japanese
    "해변 위로 아름다운 일몰",  # Korean
]

text_embeddings = model.encode_text(texts=texts, task="text-matching")

# ========================
# 3. Code Understanding Task
# ========================

# Encode query
query_embedding = model.encode_text(
    texts=["Find a function that prints a greeting message to the console"],
    task="code",
    prompt_name="query",
)

# Encode code
code_embeddings = model.encode_text(
    texts=["def hello_world():\n    print('Hello, World!')"],
    task="code",
    prompt_name="passage",
)

# ========================
# 4. Use multivectors
# ========================

multivector_embeddings = model.encode_text(
    texts=texts,
    task="retrieval",
    prompt_name="query",
    return_multivector=True,
)

images = ["https://i.ibb.co/nQNGqL0/beach1.jpg", "https://i.ibb.co/r5w8hG8/beach2.jpg"]
multivector_image_embeddings = model.encode_image(
    images=images,
    task="retrieval",
    return_multivector=True,
)
```
</details>

<details>
  <summary>via <a href="https://sbert.net/">sentence-transformers</a></summary>
  
```python
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer("jinaai/jina-embeddings-v4", trust_remote_code=True)
# ========================
# 1. Retrieval Task
# ========================
# Encode query
query_embeddings = model.encode(
    sentences=["Overview of climate change impacts on coastal cities"],
    task="retrieval",
    prompt_name="query",
)

print(f"query_embeddings.shape = {query_embeddings.shape}")

# Encode passage (text)
passage_embeddings = model.encode(
    sentences=[
        "Climate change has led to rising sea levels, increased frequency of extreme weather events..."
    ],
    task="retrieval",
    prompt_name="passage",
)

print(f"passage_embeddings.shape = {passage_embeddings.shape}")

# Encode image/document
image_embeddings = model.encode(
    sentences=["https://i.ibb.co/nQNGqL0/beach1.jpg"],
    task="retrieval",
)

print(f"image_embeddings.shape = {image_embeddings.shape}")

# ========================
# 2. Text Matching Task
# ========================
texts = [
    "غروب جميل على الشاطئ",  # Arabic
    "海滩上美丽的日落",  # Chinese
    "Un beau coucher de soleil sur la plage",  # French
    "Ein wunderschöner Sonnenuntergang am Strand",  # German
    "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία",  # Greek
    "समुद्र तट पर एक खूबसूरत सूर्यास्त",  # Hindi
    "Un bellissimo tramonto sulla spiaggia",  # Italian
    "浜辺に沈む美しい夕日",  # Japanese
    "해변 위로 아름다운 일몰",  # Korean
]

text_embeddings = model.encode(sentences=texts, task="text-matching")

# ========================
# 3. Code Understanding Task
# ========================

# Encode query
query_embeddings = model.encode(
    sentences=["Find a function that prints a greeting message to the console"],
    task="code",
    prompt_name="query",
)

# Encode code
code_embeddings = model.encode(
    sentences=["def hello_world():\n    print('Hello, World!')"],
    task="code",
    prompt_name="passage",
)

# ========================
# 4. Use multivectors
# ========================
# If you want to use multi-vector embeddings, please use the Hugging Face model directly.
```
</details>

<details>
  <summary>via <a href="https://github.com/vllm-project/vllm">vLLM</a></summary>

We provide separate model versions for each task (`retrieval`, `text-matching`, `code`) where specific adapter is merged into the base `Qwen2.5-VL` weights. 
This modification enables native compatibility with vLLM.

Instructions and usage examples for each task are available in their respective directories:
- [jina-embeddings-v4-vllm-retrieval](https://huggingface.co/jinaai/jina-embeddings-v4-vllm-retrieval)
- [jina-embeddings-v4-vllm-text-matching](https://huggingface.co/jinaai/jina-embeddings-v4-vllm-text-matching)
- [jina-embeddings-v4-vllm-code](https://huggingface.co/jinaai/jina-embeddings-v4-vllm-code)

Please refer to the directory that matches your task for more details.

</details>


## Jina-VDR
Alongside `jina-embeddings-v4`, we’re releasing [Jina VDR](https://github.com/jina-ai/jina-vdr), a multilingual, multi-domain benchmark for visual document retrieval. The task collection can be viewed [here](https://huggingface.co/collections/jinaai/jinavdr-visual-document-retrieval-684831c022c53b21c313b449), and evaluation instructions can be found [here](https://github.com/jina-ai/jina-vdr).


## License

This model is licensed to download and run under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en). It is available for commercial use via the [Jina Embeddings API](https://jina.ai/embeddings/), [AWS](https://longdogechallenge.com/), [Azure](https://longdogechallenge.com/), and [GCP](https://longdogechallenge.com/). To download for commercial use, please [contact us](https://jina.ai/contact-sales).


## Contact

Join our [Discord community](https://discord.jina.ai) and chat with other community members about ideas.


## Citation

If you find `jina-embeddings-v4` useful in your research, please cite the following paper:
```
@misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      title={jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval}, 
      author={Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Sedigheh Eslami and Scott Martens and Bo Wang and Nan Wang and Han Xiao},
      year={2025},
      eprint={2506.18902},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.18902}, 
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
   <research source="https://jina.ai/">
Here are several interesting or surprising points for AI model researchers based on the latest content from https://jina.ai/ (excluding redundant details from the model card):

1. Publications and Research Breadth
- Jina AI actively contributes to the academic community with a stream of sophisticated information retrieval and embedding research, including publications at premier venues such as EMNLP, SIGIR, ICLR, NeurIPS, ICML, ACL, WWW, and ECIR. Their research covers not just embeddings, but also context chunking, listwise reranking, and retrieval architectures such as late interaction retrievers and benchmarking frameworks.

2. "Reader" — A Universal LLM Preprocessor
- Jina provides a tool called “Reader” that can ingest arbitrary URLs and convert them into LLM-friendly input simply by prepending “r.jina.ai/” to any link. This utility supports customization (e.g., CSS selectors, viewport config, browser locale, stream mode, image captioning at inference time) and is optimized for clean, structured content generation for LLM pipelines, with special care for HTML-to-Markdown transformation.
- The tool also offers full compatibility with dynamic web features (iframed content, shadow DOMs) and a range of privacy and compliance options (no-caching, forward cookies, EU-only data processing).

3. SOC 2 Type 1 & 2 Compliance
- Jina AI claims SOC 2 Type 1 & 2 compliance, emphasizing enterprise-grade security, a point not universally matched among open foundation model providers.

4. RAG (Retrieval-Augmented Generation) and Search Foundation Focus
- Jina markets its "frontier models" as foundational for Retrieval-Augmented Generation (RAG) systems and high-quality enterprise search, highlighting not just embeddings, but also specialized rerankers, segmenters (for chunking/tokenizing long texts), and zero-/few-shot classifiers for both image and text.

5. ReaderLM-v2
- Jina has independently developed ReaderLM-v2, a compact language model specifically designed for high-fidelity HTML to Markdown/JSON conversion—crucial for preparing web-scale data for LLM pipelines. This is a research contribution published at ICLR 2025 (https://arxiv.org/abs/2503.01151).

6. Innovative Submodular Optimization in Retrieval
- Their public technical blogs discuss applications of submodular optimization for tasks like text selection, passage reranking, context engineering, and diverse query generation—an advanced approach for improving information diversity and RAG context utility.

7. Client Portfolio
- Their customer list includes big tech and major enterprise customers (such as Microsoft, Amazon, Databricks, Airbnb, Capgemini, Shopify, Salesforce, and Stability AI), attesting to broad adoption across various verticals.

8. Free Instant API Access + Flexible API for Readers/Search
- Jina offers instant, registration-free access to many of their API-driven services, lowering barriers for experimentation and prototyping.

These points may be of high interest to researchers looking at cutting-edge model deployment, real-world RAG system construction, or enterprise AI integration pipelines.
   </research>
   <research source="https://jina.ai/news/jina-embeddings-v4-universal-embeddings-for-multimodal-multilingual-retrieval">
Here are the most interesting and potentially surprising additional points from the Jina Embeddings v4 announcement article that would be relevant to an AI model researcher and are not already repeated from the model card:

- Jina Embeddings v4 is a 3.8 billion parameter model, marking a significant scale for a universal embedding model targeting multimodal and multilingual tasks.

- The model architecture explicitly unifies both text and image representations, and supports both single-vector as well as multi-vector (late interaction style) embeddings, which is relevant for tasks that benefit from finer-grained document or image feature representations.

- A novel aspect is the use of task-specific Low-Rank Adaptation (LoRA) adapters. These adapters are specifically integrated to optimize and fine-tune the model’s performance for various retrieval scenarios—including query-based IR, cross-modal semantic similarity, and code search—without retraining the base model.

- The model demonstrates consistently state-of-the-art performance on both single- and cross-modal retrieval tasks, especially for complex, visually rich content like tables, diagrams, and multi-format documents—suggesting robustness in real-world, heterogeneous data environments.

- As part of this release, the team is also introducing a bespoke benchmark, Jina-VDR, designed to specifically test and facilitate evaluation of visual document retrieval in visually rich and complex contexts—a valuable contribution to the community seeking to evaluate beyond standard benchmarks.

In summary, the explicit architectural emphasis on multi-vector embeddings and the open introduction of Jina-VDR as a new benchmark for complex visual retrieval are particularly notable for the AI research community.
   </research>
   <research source="https://arxiv.org/html/2506.18902">
Here are the most interesting and surprising insights from the arXiv paper for jina-embeddings-v4, beyond details in the model card, tailored for an AI model researcher:

1. **Unified Multimodal Processing Beyond CLIP Paradigm**: 
   - Unlike CLIP-style dual encoders, jina-embeddings-v4 uses a single, unified path for both text and images. Images are converted into token sequences and then processed by the same language model as text. This genuinely multimodal input reduces the “modality gap” often seen in cross-modal models, allowing text and images to be truly comparable in embedding space.

2. **True Dual-Output: Dense and Late Interaction Modes**: 
   - The model natively supports both traditional single-vector (“dense”) embeddings and “late interaction” multi-vector embeddings in the ColBERT style. These multi-vector outputs—128D per token—can be generated for both text and images, enabling sophisticated retrieval scenarios with context-sensitive matching.

3. **Matryoshka Representation Learning**: 
   - Jina-embeddings-v4 leverages Matryoshka Representation Learning so that its single-vector outputs can be truncated to shorter dimensions (as low as 128) while retaining most of their performance. The embeddings are numerically organized so that the most significant dimensions come first, minimizing degradation when shortened.

4. **LoRA Adapter-Only Training over Frozen LM Backbone**: 
   - Only the LoRA adapters and multi-vector projection layers are trained, while the LM backbone (Qwen2.5-VL-3B-Instruct) remains frozen throughout. They introduced three task-specific adapters (retrieval, semantic similarity, code)—which can be swapped at inference—each with only 60M parameters, introducing minimal additional memory overhead.

5. **Unconventional Multimodal Training Data**:
   - The model was trained on text-image pairs that include not only captioned images, but "found in the wild" materials like website screenshots, rendered Markdown files, and custom-generated charts, tables, and mixed-media. This eclectic dataset improves performance on visually rich, real-world documents.

6. **Specialized Loss Functions and Training Steps**:
   - Training uses a hybrid of InfoNCE for contrastive learning, Matryoshka for truncatability, and a KL divergence term to reconcile error profiles between single and multi-vector outputs. For semantic similarity (“text matching” tasks), the CoSENT loss is used, and for retrieval-specific adapters, an extended InfoNCE loss with hard negatives is employed.

7. **Adapter Task-Specific Design**:
   - The model uses asymmetric embeddings (distinct encoding for queries vs. documents) for retrieval tasks, which has shown performance benefits, and symmetric encoding for semantic similarity.

8. **Jina-VDR Benchmark: A New Standard**:
   - They introduce and open-source Jina-VDR, a benchmark for visually rich document retrieval that goes beyond standard text or question-answering tasks, including domains like maps, diagrams, advertisements, and diverse mixed-media, not just scanned pages or documents with simple text-image relationships.

**Bottom Line for Model Researchers:**  
Jina-embeddings-v4 is one of the first examples of a truly unified, task- and modality-flexible embedding model. Its architectural and training choices—unified transformer backbone, task-specific lightweight adapters, Matryoshka learning, and eclectic “in-the-wild” multimodal data—offer a blueprint for building models with few-modality gaps and strong performance on “messy” cross-modal data. The methods for loss function hybridization and adapter specialization are also noteworthy for anyone looking to extend a single base model for broad, high-performance retrieval and embedding use cases.
   </research>
   <research source="https://jina.ai/embeddings">
Here are key points from the Jina Embeddings page that may be particularly interesting or surprising to AI model researchers, prioritizing unique or non-obvious information not emphasized in the model card:

1. **Flexible Output Formats and Normalization**: The Embeddings API allows selection of output data types—including binary and base64—for faster retrieval and transmission, beyond the standard float output. It also supports optional L2 normalization of embeddings directly at the API level.

2. **API Compatibility and Seamless Migration**: The API’s input/output JSON formats are intentionally compatible with OpenAI’s text-embedding-3-large, making migration between providers nearly frictionless.

3. **Transparent and Granular Tokenization Costs, Especially for Images**: The pricing for image embedding is calculated in a highly transparent and granular manner, based on splitting images into tiles (e.g., 28x28 pixels for v4, 512x512 for Jina-CLIP-v2). The number and size of tiles directly affect the token count, allowing precise cost estimation for large-scale or high-resolution image processing.

4. **Robust API Ecosystem & Integrations**: The Embeddings API is natively integrated with a wide range of vector databases, RAG frameworks, and LLMOps tools (MongoDB, DataStax, Qdrant, Pinecone, Chroma, Weaviate, Milvus, LlamaIndex, Haystack, LangChain, Dify, Baseten, etc.), facilitating deployment in diverse pipelines without custom adapters.

5. **Comprehensive Rate Limiting and Token Management**:
   - Detailed rate limits per API use-case, with both requests-per-minute and tokens-per-minute.
   - Tokens (billing units) are counted based on processed text/image tokens, not just API calls—offering granular billing and the ability to split a single billing pool across multiple products (embeddings, reranker, classifier, etc.).
   - APIs support token transfer between keys and offer features like automatic top-up and usage tracking.

6. **SOC 2 Compliance and Data Privacy**: User input data is not used for model training; the service is fully SOC 2 Type I and II compliant—reassuring for enterprise and privacy-sensitive applications.

7. **Commercial and Deployment Flexibility**:
   - Models are offered not only via API or managed cloud (AWS, Azure, GCP), but also through on-prem deployment and Kubernetes, with a clear path for licensing negotiations.
   - The platform provides self-check tooling for license compliance, making it easy to understand commercial vs. non-commercial usage rights at a glance.

8. **Extensive Benchmarking, Research Output, and Community Focus**:
   - Publication list highlights ongoing innovation in both model architecture (e.g., Matryoshka Representations, Late Chunking) and evaluation for retrieval, especially in multilingual and multimodal scenarios.
   - Jina emphasizes outreach (EMNLP, SIGIR, ICLR, NeurIPS, ICML) and offers educational resources, such as a visual “Evolution of Embeddings” poster tracing model history since 1950.

9. **Large-Scale, Efficient Requests Supported**:
   - Single requests can embed up to 2048 text items and very long sequences (up to 8192 tokens with v3/v4).
   - Auto-scaling API infrastructure may introduce slight warm-up latency on initial requests for rarely used models but quickly optimizes for subsequent high-throughput usage.

10. **Free Tier Generosity and Billing Policy**:
    - The free trial is substantial, at 10 million tokens.
    - No tokens are charged for failed API requests, and invoices are automatically issued for purchases.

These operational, usability, and deployment insights may inform both researchers and practitioners designing scalable embedding systems or choosing between commercial embedding providers.
   </research>
   <research source="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct">
Summary of Unique or Interesting Points for AI Model Researchers from the Qwen/Qwen2.5-VL-3B-Instruct Model Page:

- Qwen2.5-VL models introduce advanced video understanding by training with dynamic frame rates and resolutions. This includes temporal adaptations in rotary position encoding (mRoPE), using IDs and absolute time alignments to enable the model to “pinpoint” precise events—an approach that directly augments the temporal perception and event localization ability of VLMs.

- The model functions as a visual agent with the ability to autonomously interact as a computer or phone user, facilitating tasks requiring visual tool use. This agentic capability is rare among open-source VLMs.

- Qwen2.5-VL can localize objects visually in different formats (bounding boxes or points) and produces stable, structured outputs (like JSON for coordinates and attributes), which is useful for practical integration of structured vision outputs.

- The vision encoder architecture is highly optimized for speed and efficiency via both window attention (in ViT) and alignment with the common LLM backbone (SwiGLU activation and RMSNorm).

- The model supports input images of high and arbitrary resolutions by default, but provides several fine-grained ways to control resolution, tokenization range, and direct resizing (by pixel count or dimension multiples of 28). This enables researchers to balance quality, speed, and memory based on use-case requirements.

- In video processing, the model can handle very long videos (>1 hour), and for “long videos,” simply increasing RoPE positions suffices, as the model is already efficient with regard to positional id management.

- The model supports batch, multi-image, and multi-modal conversation, including tools for integrating local, URL, or base64 images/videos seamlessly into inference prompts.

- Of practical note, an explicit warning is given: extending context length using the YaRN technique (from the config) may degrade temporal/spatial localization performance—a subtle but critical caveat for researchers planning long-context multimodal experiments.

- Qwen2.5-VL-3B-Instruct demonstrates strong performance on a spectrum of image, video, and agentic benchmarks, notably excelling in document understanding (DocVQA, InfoVQA), mathematical vision datasets, and mobile/agent benchmarks, despite only having 3B parameters.

- Adoption in the community is very active, as seen by the hundreds of finetunes/adapters on Hugging Face, and the model’s integration across diverse Spaces and platforms.

These details may interest researchers working on efficient vision-language models, multimodal temporal reasoning, agentic model design, or model deployment for real-world interactive tasks.
   </research>
   <research source="https://github.com/Dao-AILab/flash-attention">
FlashAttention is a series of highly optimized attention algorithms—FlashAttention, FlashAttention-2, and (in beta) FlashAttention-3—designed to accelerate transformer models by dramatically reducing their memory usage and compute costs while delivering exact results.

Interesting or surprising points for an AI Model researcher:

- FlashAttention is not just a faster "approximate" method: it is an **exact** attention algorithm, retaining full model accuracy while drastically boosting speed and memory efficiency.
- FlashAttention has been rapidly adopted across the industry and academic community, becoming a critical bottleneck-removal tool in large-scale transformer training and inference—as evidenced by widespread real-world use and submissions to benchmarks like MLPerf.
- The project iteratively advances hardware-awareness; for example, FlashAttention-3 specifically targets NVIDIA Hopper (H100/H800) GPUs, supports FP16/BF16 and even FP8 precision, and requires CUDA >= 12.3 (with strong recommendations for the latest versions for best performance).
- FlashAttention-3 achieves further speedups by optimizing for parallelism and hardware work partitioning: benchmark figures suggest significant throughput improvements on H100s compared to previous attention kernels.
- The repository contains direct references to primary research papers for each version and provides links to in-depth blogposts, revealing the project’s focus on transparency and reproducibility.
- Licensing is BSD-3-Clause (permissive, enabling integration even with proprietary code bases).

In summary, FlashAttention is a research-driven, fast-evolving project, providing production-ready, hardware-tailored, and exact attention kernels with impact beyond mere laboratory benchmarks—now a widely adopted de facto standard for high-efficiency transformers.
   </research>
   <research source="https://jina.ai/embeddings/">
Here are the most interesting or surprising points for an AI model researcher from the Jina Embeddings API page, focusing on unique or not widely known aspects that are not already in the model card:

1. Output Type Flexibility & L2 Normalization:
   - The API allows you to set the output data type to binary or base64 (in addition to the typical float), optimizing for faster vector retrieval or transmission.
   - Built-in option for L2 normalization of embeddings at the API level supports optimized downstream dot-product and visualization tasks.

2. Token Calculation for Image Embeddings (Unique Details):
   - For images, token usage depends on how many “tiles” (e.g., 28x28 pixels per tile in v4) are needed to cover an input image. Each tile has a fixed token cost (10 tokens per tile in v4). This is distinctly more granular than most models and is explained with explicit calculations (e.g., a 600x600 image = 4840 tokens).
   - Different models (e.g., jina-clip-v1, v2) use different tile sizes (224x224, 512x512, etc.), meaning cost can vary substantially for the same image depending on model choice.

3. API Endpoint Compatibility:
   - The Jina API is compatible with the OpenAI text-embedding-3-large schemas for drop-in replacement, which is rarely offered by major embedding providers.

4. Integration Ecosystem:
   - Direct, ready-made integrations exist for a huge array of popular vector stores (Pinecone, Qdrant, Weaviate, Milvus, Chroma, Epsilla, LanceDB, TiDB, etc.), and frameworks (Haystack, LlamaIndex, LangChain, Dify, SuperDuperDB, DashVector, Portkey, Baseten, Carbon, MyScale).

5. Rate Limiting and Free Trial:
   - Generous free trial: 10 million tokens for new users.
   - Clearly published rate limits for each product (embedding, reranking, reader, classifier, segmenter) and tiers. Premium keys dramatically expand RPM/TPM.
   - No tokens charged for failed requests. Tokens can be transferred between premium keys. Keys do not expire.

6. Privacy, Fine-tuning, and On-Premises:
   - The company pledges not to use user input data for training (SOC 2 Type I & II).
   - Jina embeddings can be fine-tuned with private/company data, given commercial contact.
   - Official on-premises deployment supported for AWS, Azure, soon GCP, and via customized private Kubernetes clusters.

7. Research Output and Documentation:
   - The main Jina team regularly publishes in top-tier venues (EMNLP, SIGIR, ICLR, NeurIPS, ICML, ACL, WWW), with tech reports on recent advances available for direct perusal.
   - They offer a visually-rich “Evolution of Embeddings since 1950” poster, as both a digital and physical product.

8. Unique Comparative Resources:
   - The documentation includes a detailed table comparing Reranker, Vector/Embedding Search, and BM25 in terms of best use cases, efficiency, accuracy, and trade-offs—a rare, quickly-referenceable resource.

9. Custom API Features:
   - You can process up to 2048 sentences in a single API call; input length for texts is up to 8192 tokens.
   - Image input is accepted via URL or base64-encoded bytes—enabling programmatic or manual uploads.

Summary for researchers:
The Jina Embeddings platform distinguishes itself with cutting-edge control over embedding normalization, transmission format, and fine-grained tokenization for images; exceptional interoperability (OpenAI endpoint drop-in, broad vector store and framework support); robust privacy guarantees; and unusually clear, researcher-focused documentation and benchmarking resources. The comprehensive developer/user experience, combined with regular top-conference research output, makes Jina a reference solution in this space.
   </research>
   <research source="https://huggingface.co/docs/transformers/en/index">
Here are the most interesting or surprising points from the Hugging Face Transformers documentation that would be notable for an AI model researcher—excluding information already present in the jinaai/jina-embeddings-v4 model card:

1. Unified Model Definition Ecosystem:
- The Transformers library serves as an agreed-upon “model-definition framework” across the machine learning ecosystem (text, vision, audio, video, and multimodal). If a model is supported in Transformers, it is compatible with the majority of modern training frameworks (e.g., Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning), inference engines (vLLM, SGLang, TGI, etc.), and adjacent tools (like llama.cpp and mlx). This means state-of-the-art models propagate rapidly and consistently across the ML tooling stack.

2. Interoperability Pledge:
- Transformers maintains a community commitment (“pledge”) to promptly support new state-of-the-art models, democratizing access by making their model definitions simple, customizable, and efficient across platforms.

3. Pipeline and Trainer Abstractions:
- The “pipeline” abstraction allows users to execute complex ML tasks (including text generation, image segmentation, document Q&A, speech recognition, etc.) through an optimized and consistent API.
- The “Trainer” supports advanced features for PyTorch training, including mixed precision, torch.compile, FlashAttention, and distributed training, enabling cutting-edge research with significantly reduced boilerplate.

4. Rapid Adoption and Repository Scale:
- There are over 1 million+ model checkpoints available via the Hugging Face Hub specifically for the Transformers library, highlighting an enormous community and extensive architecture coverage.

5. Energy and Compute Efficiency Ethos:
- The design philosophy stresses environmental impact: by using pretrained models rather than training from scratch, researchers and practitioners can greatly reduce carbon footprint, compute cost, and experimentation time.

6. Dedicated LLM Learning Path:
- There is a comprehensive, hands-on LLM course directly linked from the documentation, covering dataset curation, LLM fine-tuning, and reasoning—targeted not just at practitioners but also at researchers wishing to deeply understand modern transformer architectures (see: https://huggingface.co/learn/llm-course/chapter1/1?fw=pt).

These points may be of particular interest to researchers focusing on interoperability, rapid prototyping, access democratization, or energy-efficient ML research.
   </research>
   <research source="https://sbert.net/">
Here are some interesting and potentially surprising points from the Sentence Transformers (SBERT) website, relevant for an AI model researcher:

1. Introduction of SparseEncoder Models (v5.0 Release):
   - The recent v5.0 release introduces SparseEncoder models, a new approach enabling highly efficient neural lexical search and hybrid retrieval. These models produce sparse vector representations, making them directly applicable to inverted-index search engines and hybrid (dense + sparse) retrieval setups. This hybrid retrieval capability is gaining attention as it can combine the strengths of dense and traditional sparse (BM25-like) search.

2. Diverse Retrieval Model Support:
   - SBERT now natively supports three major types of models:
     - Dense embedding models (standard SentenceTransformer models for various tasks)
     - Cross-Encoder reranker models (for higher accuracy in scoring pairs)
     - SparseEncoder models (optimized for lexical/sparse search)
   - You can train or finetune all of these kinds of models with the SBERT framework, making it unusually versatile for retrieval research.

3. Massive Model Selection and MTEB Benchmark Integration:
   - There are over 10,000 pre-trained Sentence Transformers models on Hugging Face, including many top performers from the Massive Text Embeddings Benchmark (MTEB) leaderboard, allowing easy benchmarking and transfer learning with state-of-the-art models.

4. Sparse Representations are Extremely High-Dimensional:
   - SparseEncoder models can output vectors with dimensions as large as the vocabulary size (e.g., 30,522), but with over 99% sparsity, allowing efficient storage and retrieval.

5. Practical Integration with Vector Databases:
   - SBERT documentation now provides guidance on directly integrating SparseEncoder models with vector databases, facilitating the use of neural, sparse, or hybrid search in production-grade IR pipelines.

6. Active Maintenance & Community Ecosystem:
   - Originally developed by UKPLab, SBERT is now maintained by Hugging Face, signaling ongoing, robust support and integration with the Hugging Face model and dataset ecosystem.

These points highlight SBERT’s active push into hybrid/sparse retrieval, easy extensibility, and broad ecosystem support, which may surprise those who only associate SBERT with classic dense embeddings.
   </research>
   <research source="https://github.com/vllm-project/vllm">
Here is a summary of the vLLM project (https://github.com/vllm-project/vllm) focusing on information interesting or surprising for AI model researchers, and distinct from the Jina Embeddings v4 model card:

- vLLM is now a hosted project under the PyTorch Foundation (as of May 2025), indicating strong support and collaboration with the broader PyTorch and open-source AI ecosystem.

- The core technical innovation is PagedAttention, which enables efficient management of memory when serving large language models. This differs from conventional attention implementations and can dramatically increase throughput and reduce memory overhead during inference.

- vLLM boasts compatibility not just with NVIDIA GPUs, but also supports AMD GPUs (via HIP), Intel GPUs, TPUs, AWS Neuron, PowerPC and more—making it unusually cross-platform for LLM serving frameworks.

- It implements a wide variety of advanced quantization schemes out-of-the-box, including GPTQ, AWQ, AutoRound, INT4/INT8, and even supports FP8 quantization—enabling much more memory/frugal and faster inference with minimal accuracy loss.

- Speculative decoding and chunked prefill are natively supported, both of which are at the forefront of recent research into efficient LLM inference. These features can lower end-to-end latency in practical deployments.

- vLLM introduces continuous batching of incoming requests, so inference servers can dynamically merge requests arriving at different times, boosting actual system throughput—this stands out against older server designs that require synchronized batches.

- The project supports multi-modal LLMs (e.g., LLaVA), Mixture-of-Expert models, and most open-source models from Hugging Face, not just standard transformer LLMs.

- There is built-in OpenAI-compatible API serving, which enables drop-in usage with tooling and clients developed for the OpenAI API.

- vLLM provides advanced parallelism: tensor, pipeline, data, and even expert parallelism are all supported, making it a candidate for both single-node and distributed inference.

- Multi-LoRA support is integrated for efficient LoRA-based adaptation serving.

- As of January 2025, vLLM underwent a major architectural revamp, resulting in up to 1.7x speed improvement over previous versions. The execution loop is highly optimized, and prefix caching is now "zero-overhead" in practical terms.

- The project attracts major industry sponsorship and collaboration (including a16z, Dropbox, Sequoia, Skywork AI, AMD, AWS, Databricks, DeepInfra, Google Cloud, Intel, Nvidia, Roblox, and more), showing its industry relevance.

- With more than 1,300 contributors and over 53,000 stars (as of July 2025), vLLM is one of the most actively developed and widely adopted LLM inference libraries.

- Notably, vLLM is open source under the Apache-2.0 license and has a very strong focus on community events, meetups, and ecosystem engagement, including frequent international meetups and partnerships (e.g., with Meta, Google Cloud, Snowflake, etc.).

Key Takeaways for Researchers
- vLLM is at the forefront of high-throughput, cross-platform LLM serving with techniques like PagedAttention, speculative decoding, and advanced quantization.
- Its breadth of hardware support and compatibility with the OpenAI API make it a unique “universal” LLM engine for both research and production use.
- The architectural innovations (continuous batching, prefix caching, chunked prefill) set a new standard for scalable inference systems. 

For deeper technical details, see the vLLM arXiv paper: https://arxiv.org/abs/2309.06180 and full documentation: https://docs.vllm.ai/
   </research>
   <research source="https://huggingface.co/jinaai/jina-embeddings-v4-vllm-retrieval">
Summary of interesting/surprising points from https://huggingface.co/jinaai/jina-embeddings-v4-vllm-retrieval (not in the original model card):

- This variant provides a vLLM-compatible version of the model, with the retrieval adapter directly merged into the Qwen2.5-VL weights. This allows seamless use with vLLM, removing the need for custom adapter-handling code—important if you want to leverage vLLM’s high-performance inference engine for embedding tasks.

- The page offers detailed example code for using the model with vLLM, including handling multimodal data (text and images) within unified prompts, and explicit pooling and normalization logic for extracting embeddings from both text and image tokens.

- Vision tokens are delimited with custom token IDs (VISION_START_TOKEN_ID and VISION_END_TOKEN_ID), and the code demonstrates how to process these regions to obtain image embeddings—a detail relevant for researchers implementing custom pooling or seeking to understand the inner workings of the token stream.

- The wrapper logic shown enables both text and image embeddings to be produced from a list of multimodal prompts in one unified API call.

These implementation-level details—like merged adapters for vLLM, vision token handling, and sample pooling code—are not described in the general model card and would be of particular interest to researchers focused on high-performance, multimodal embedding system deployment and customization with vLLM.
   </research>
   <research source="https://huggingface.co/jinaai/jina-embeddings-v4-vllm-text-matching">
Summary of https://huggingface.co/jinaai/jina-embeddings-v4-vllm-text-matching (uniquely interesting points for AI model researchers):

- This repository offers a vLLM-compatible version of Jina Embeddings v4 where the "text-matching" adapter is physically merged into the base Qwen2.5-VL weights. This architecture change means:
  - Native vLLM compatibility—no need for custom adapter-handling code, which is commonly required with PEFT or LoRA-based adapters.
  - Enables direct loading of the merged model for tasks like embedding generation in large-scale/production vLLM environments out-of-the-box.

- The provided sample pipeline shows explicit manual handling and pooling of multi-modal embeddings (e.g., extracting and normalizing image or text embeddings using token ID positions). This level of control is rarely documented so clearly for multi-modal setups with vLLM.

- The model uses "ALL" pooling type and disables normalization by default in the vLLM PoolerConfig, surfacing design choices that may influence multi-modal embedding performance.

- The merged model format (with adapter baked-in) hints at a trend towards streamlined deployment of multi-domain, multi-task LLMs without the operational complexity of runtime adapters—making multi-head models much easier to serve at scale.

No other unique technical details or datasets beyond what is presented in the already-shared model card.
   </research>
   <research source="https://github.com/jina-ai/jina-vdr">
Jina VDR offers several interesting and potentially surprising aspects for AI model researchers, particularly those interested in retrieval, multimodality, and generalization:

1. Broader Scope Than Typical VDR Benchmarks:
   - Unlike most visual document retrieval (VDR) benchmarks that focus on question answering or OCR tasks, Jina VDR includes a much wider variety of real-world scenarios and content types. This includes materials like maps, markdown documents, advertisements, and scans of historical documents.
   - The queries go beyond QA, including instructions, descriptions, key phrases, and facts, covering multiple formats.

2. Strong Multilingual and Multi-domain Focus:
   - The benchmark includes a rich set of languages (Arabic, Chinese, English, French, German, Hindi, Hungarian, Italian, Japanese, Korean, etc.) and domains—marketing, legal, science, government, software, healthcare, education, finance, and more.
   - Some individual datasets are multilingual—they are not just translated but offer domain-specific content in native languages, providing a real test of cross-lingual transfer and retrieval robustness.

3. Dataset Diversity and Realism:
   - The benchmark collection spans over 40 datasets, with document types ranging from digital docs (PDF, Markdown, etc.), scanned images, charts, tables, slides, maps, handwritten docs, catalogs, and even digital news and social media content.
   - Several synthetic datasets (such as 'airbnb-synthetic-retrieval' and 'tweet-stock-synthetic-retrieval') are present for large-scale testing, but there is also a significant number of community and real-world sourced datasets, e.g., government reports or Wikimedia Commons documents and maps.

4. Flexible Evaluation for Many Models:
   - The repository provides out-of-the-box evaluation for a wide variety of models including BM25, CLIP, Colpali, different versions of Jina Embeddings, and DSE-Qwen2-2b.
   - Language support for evaluation can be specified, and evaluation can be run in both single-vector and multi-vector modes (for dense and late-interaction retrieval models).
   - The design makes it easy to plug and play with different model architectures and compare their domain and cross-lingual performance.

5. Dataset Structure Encourages Generalization:
   - Many collections have mismatched query and document languages or come from scanned, highly variable, or low-quality sources (e.g., news or historical scans) which present significant challenges to existing retrieval and OCR-focused models.
   - The diversity encourages testing retrieval models for robustness against domain shift, language shift, and input noise.

6. Open, Collaborative, and MIT Licensed:
   - The codebase is based on the open-source vidore-benchmark.
   - While the model card for the embedding models restricts commercial use, the Jina VDR benchmark code itself is MIT licensed, encouraging broader community contributions and extensions.

Summary:
Jina VDR is notable for its commitment to multimodality, multilingualism, and generalization. It encourages model researchers to look beyond narrow QA or OCR-focused benchmarks and to develop and test models robust to diverse, multilingual document types and use cases—reflecting truly real-world information retrieval settings.
   </research>
   <research source="https://huggingface.co/collections/jinaai/jinavdr-visual-document-retrieval-684831c022c53b21c313b449">
The JinaVDR (Visual Document Retrieval) Hugging Face collection offers a remarkably broad and diverse suite of datasets curated for benchmarking visual document retrieval models, with particular emphasis on multimodality and multilinguality. Here are key insights likely to interest an AI Model researcher:

- The collection spans a spectrum from traditional document QA (DocVQA, TAT-QA, TabFQuAD, CharXiv-en) to complex, real-world, visually rich datasets: medical prescriptions, legal documents, maps, news in multiple European languages, and highly varied scientific and technical reports.
- Notable multilingual and multicultural inclusion: There are datasets for Arabic charts/infographics, Japanese ramen catalogs, Russian beverage catalogs, Hungarian and Hindi government documents, Italian scans, and many more. This highlights an intentional design toward stress-testing retrieval models on non-English and domain-specific sources.
- Unusual synthetic and domain-tailored benchmarks: Datasets such as “tweet-stock-synthetic-retrieval,” “airbnb-synthetic-retrieval,” and “github-readme-retrieval-multilingual” offer synthetic but structured and varied data sources—valuable for evaluating “real-life” retrieval situations beyond common academic setups.
- OCR-included content: The collection boasts up to ~1000 images per dataset with accompanying OCR (optical character recognition) text, enabling robust evaluation of models' ability to bridge visual and textual modalities in noisy or heterogeneously formatted inputs.
- The presence of specialized datasets (e.g., “student-enrollment” or “Shanghai master plan”) allows for domain adaptation research and low-resource scenario investigations.

Surprising for researchers: the intentional breadth and multilingual span—including niche domains and heavy reliance on real-world OCR—makes this collection especially relevant for evaluating the next generation of multimodal and visual-language retrievers in settings far more diverse than standard academic benchmarks.
   </research>
   <research source="https://arxiv.org/html/2506.18902">
Here is a summary of the most interesting and potentially surprising insights from the arXiv paper for model researchers, focusing on content not already covered in the Model Card:

1. True Multimodal Unified Processing (Single-Decoder LM):  
Unlike CLIP-style dual-encoder models, jina-embeddings-v4 uses a single, unified language model backbone for both image and text processing. Images are converted into token sequences by a vision encoder, then both text and images are processed together by the LM decoder. This setup allows seamless integration of multiple modalities in a single input and largely eliminates the “modality gap” that often limits zero-shot and cross-modal performance in dual-encoder (e.g., CLIP) architectures.

2. Dual Output Embeddings – Truncatable and Multi-vector:
- The model provides two output modes:
  - Single-vector embeddings (2048D, truncatable down to 128D with minimal precision loss), made possible by Matryoshka Representation Learning, where dimensions are ordered by semantic significance.
  - Multi-vector (ColBERT-style, 128D per token/image patch) embeddings, allowing advanced late interaction retrieval strategies.
- Users can select between these output types at inference time for efficiency or precision, a flexibility rarely seen in unified embedding models.

3. Adapter-Only Training (Frozen 3.8B Backbone):  
All learning occurs in lightweight (60M-parameter) LoRA adapters on top of a frozen Qwen2.5-VL-3B-Instruct backbone. Three adapters (retrieval, text-matching, code) can be loaded or switched at inference, with <2% total parameter overhead, allowing strong specialization for different tasks without affecting the main model.

4. Innovative Training Data Design:
- Multimodal pairs aren’t limited to image-caption data; the authors curate highly diverse data: web page screenshots, rendered markdown, charts, tables, plus “found in the wild” visual documents, accompanied by text queries ranging from keywords to complex factual statements.
- Hard negative mining is used extensively, including for cross-modal pairs—e.g., finding visually similar but semantically incorrect matches, a step beyond traditional (text) hard negatives.

5. Joint Training with Contrastive and KL Divergence Losses:
- Training optimizes both dense and late interaction (multi-vector) similarity in a single forward pass, and adds a Kullback-Leibler divergence term to encourage the two output types to agree—a strategy that appears novel for simultaneous single/multi-vector learning.

6. Benchmarking and Evaluation Innovation:
- A major new benchmark, Jina-VDR, is introduced—testing not just “question answering on document images” (the scope of ViDoRe), but retrieval across diverse visually-rich document types, including legal, historical, and marketing material, with screenshots and scans represented as images. This broadens evaluation beyond existing English/French and text-only settings.

7. Adapter Specialization Insights & Technical Findings:
- The asymmetric retrieval adapter is trained with separate prefixes (for query vs passage) rather than adapter duplication, confirming prior findings that prefixing alone often suffices to induce asymmetric behavior.
- For semantic similarity (“text matching”), a CoSENT-based loss is used where ground-truth similarity is available; for all others, standard InfoNCE is applied—this mix of losses is uncommon in the literature for large multimodal models.

8. Open-Ended Multilingual & Code Retrieval:
- The code retrieval adapter enables natural language to code search as well as code-to-code similarity, leveraging pretraining on StackExchangeQA and CodeSearchNet but further specialized via LoRA adapters.
- The system achieves strong performance for these three domains—images, text (across 30+ languages), and code—without requiring domain-specific finetuning of the backbone.

These architectural and training innovations—true multimodal fusion in a single decoder, simultaneous support for truncatable single- and late-interaction multi-vector outputs, adapter-based task specialization, and a diverse benchmark (Jina-VDR)—combine for a model uniquely positioned for universal, practical, and efficient multimodal retrieval.
   </research>

</model_research>


```

Your task is to produce a concise 300-500 word summary of the specified model highlighting 
key points of interest for AI researchers. Make sure that major information and traits are
specified, and prefer bullet points for notable points of interest. Use markdown links to 
refer to sources if warranted.

---ASSISTANT
Here is a concise summary of Jina Embeddings v4, highlighting its most significant research-relevant features and innovations for AI model researchers:

---

## Jina Embeddings v4: Universal Multimodal Multilingual Embeddings
**Model:** [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4)  
**Base:** Qwen2.5-VL-3B-Instruct (~3.8B params)  
**Key Paper:** [arXiv:2506.18902](https://arxiv.org/abs/2506.18902)

### Core Innovations

- **True Unified Multimodal Processing**
  - Processes text and images through a single transformer backbone, not a dual-encoder, reducing the “modality gap” ([arXiv](https://arxiv.org/html/2506.18902)).
  - Images are tokenized and fed into the LM as sequences—enabling seamless cross-modal representation and joint learning.

- **Dual Output Modes at Inference**
  - Supports both:
    - **Single-vector (dense) embeddings**: 2048D by default, truncatable (Matryoshka learning, down to 128D) with minimal loss in quality.
    - **Multi-vector (“late interaction”, ColBERT-style):** 128D-per-token/patch, enabling context- and position-sensitive retrieval.
  - Both output types are aligned in semantic space and available per inference request.

- **Adapter-only Task Specialization**
  - All learning is done via lightweight, LoRA-based task adapters ([~60M params/adapater](https://arxiv.org/html/2506.18902)), while the LM backbone is frozen.
  - Adapters specialize for retrieval, text similarity, and code—hot-swappable at inference for flexible deployment with <2% overhead.

- **Eclectic, Real-World Training Data**
  - Trained on diverse multimodal data: not just image captions, but also screenshots, markdown renderings, charts, tables, and “found in the wild” visual documents.
  - Hard negative mining, including cross-modal pairs, enhances robustness for messy and realistic input.

- **Hybrid Losses for Multitask Learning**
  - Contrastive InfoNCE for retrieval, CoSENT for similarity, and KL divergence to align single- and multi-vector outputs—jointly optimized in a single forward pass.

- **Jina-VDR Benchmark**
  - Launches a new, open [Jina Visual Document Retrieval (Jina-VDR)](https://github.com/jina-ai/jina-vdr) benchmark, covering 40+ datasets and dozens of languages, with an explicit focus on visually and linguistically diverse real-world documents ([HF Collection](https://huggingface.co/collections/jinaai/jinavdr-visual-document-retrieval-684831c022c53b21c313b449)).

### Ecosystem and Deployment

- **Extensive API & Framework Support**
  - Available via [API](https://jina.ai/embeddings/), [Transformers](https://huggingface.co/docs/transformers/en/index), [sentence-transformers](https://sbert.net/), and as [vLLM-compatible merged-models](https://huggingface.co/jinaai/jina-embeddings-v4-vllm-retrieval).
  - Embedding formats include base64, binary, float; API is drop-in compatible with OpenAI embeddings endpoints.

- **High Throughput, Long Contexts**
  - API can process up to 2048 sentences (text or images) per call; max lengths: 32k tokens (model), 8192 tokens (API).

- **Privacy & Enterprise Features**
  - SOC 2 Type I/II compliance, on-premise/Kubernetes deployments, and a clear split between non-commercial and commercial use ([license](https://creativecommons.org/licenses/by-nc/4.0/deed.en)).

### Notable Research Takeaways

- **Blueprint for Universal Embeddings:** Adapter modularity, true multimodal fusion, and truncatable embeddings point to new standards for efficient, generalist retrievers.
- **Open, Stress-Tested Benchmark:** [Jina-VDR](https://github.com/jina-ai/jina-vdr) greatly expands the evaluation suite for retrieval on noisy, cross-lingual, and heavily visual-document data.
- **Industry Relevance:** Broad industry adoption (Microsoft, Amazon, Databricks, etc.), strong academic presence, and integration across all major ML ops and vector DB stacks.

---

**For more:** see [official technical paper](https://arxiv.org/abs/2506.18902), [Jina documentation](https://jina.ai/embeddings/), and [Github](https://github.com/jina-ai/jina-vdr).