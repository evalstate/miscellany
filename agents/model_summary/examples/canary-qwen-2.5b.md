---USER

Here is the model card for nvidia/canary-qwen-2.5b:

```
---
license: cc-by-4.0
language:
- en
library_name: nemo
datasets:
- Granary
- YTC
- Yodas2
- LibriLight
- librispeech_asr
- fisher_corpus
- Switchboard-1
- WSJ-0
- WSJ-1
- National-Singapore-Corpus-Part-1
- National-Singapore-Corpus-Part-6
- vctk
- voxpopuli
- europarl
- multilingual_librispeech
- fleurs
- mozilla-foundation/common_voice_8_0
- MLCommons/peoples_speech
tags:
- automatic-speech-recognition
- speech
- audio
- Transformer
- FastConformer
- Conformer
- pytorch
- NeMo
- Qwen
- hf-asr-leaderboard
widget:
- example_title: Librispeech sample 1
  src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
- example_title: Librispeech sample 2
  src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
metrics:
- wer
base_model:
- nvidia/canary-1b-flash
- Qwen/Qwen3-1.7B
model-index:
- name: canary-qwen-2.5b
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: AMI (Meetings test)
      type: edinburghcstr/ami
      config: ihm
      split: test
      args:
        language: en
    metrics:
    - type: wer
      value: 10.19
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Earnings-22
      type: revdotcom/earnings22
      split: test
      args:
        language: en
    metrics:
    - type: wer
      value: 10.45
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: GigaSpeech
      type: speechcolab/gigaspeech
      split: test
      args:
        language: en
    metrics:
    - type: wer
      value: 9.43
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: LibriSpeech (clean)
      type: librispeech_asr
      config: other
      split: test
      args:
        language: en
    metrics:
    - type: wer
      value: 1.61
      name: Test WER
    - type: wer
      value: 3.1
      name: Test WER
  - task:
      type: Automatic Speech Recognition
      name: automatic-speech-recognition
    dataset:
      name: SPGI Speech
      type: kensho/spgispeech
      config: test
      split: test
      args:
        language: en
    metrics:
    - type: wer
      value: 1.9
      name: Test WER
  - task:
      type: Automatic Speech Recognition
      name: automatic-speech-recognition
    dataset:
      name: tedlium-v3
      type: LIUM/tedlium
      config: release1
      split: test
      args:
        language: en
    metrics:
    - type: wer
      value: 2.71
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Vox Populi
      type: facebook/voxpopuli
      config: en
      split: test
      args:
        language: en
    metrics:
    - type: wer
      value: 5.66
      name: Test WER
---

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-SALM-blue#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-2.5B-green#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-en-orange#model-badge)](#datasets)


# Model Overview

## Description:
NVIDIA NeMo Canary-Qwen-2.5B is an English speech recognition model that achieves state-of-the art performance on multiple English speech benchmarks. With 2.5 billion parameters and running at 418 RTFx, Canary-Qwen-2.5B supports automatic speech-to-text recognition (ASR) in English with punctuation and capitalization (PnC). The model works in two modes: as a transcription tool (ASR mode) and as an LLM (LLM mode). In ASR mode, the model is only capable of transcribing the speech into text, but does not retain any LLM-specific skills such as reasoning. In LLM mode, the model retains all of the original LLM capabilities, which can be used to post-process the transcript, e.g. summarize it or answer questions about it. In LLM mode, the model does not "understand" the raw audio anymore - only its transcript. This model is ready for commercial use.

### License/Terms of Use: 
Canary-Qwen-2.5B is released under the CC-BY-4.0 license. By using this model, you are agreeing to the [terms and conditions](https://choosealicense.com/licenses/cc-by-4.0/) of the license. <br>

## References:
[1] [Less is More: Accurate Speech Recognition & Translation without Web-Scale Data](https://www.isca-archive.org/interspeech_2024/puvvada24_interspeech.pdf)

[2] [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10389701)

[3] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[4] [Qwen/Qwen3-1.7B Model Card](https://huggingface.co/Qwen/Qwen3-1.7B)

[5] [Training and Inference Efficiency of Encoder-Decoder Speech Models](https://arxiv.org/abs/2503.05931)

[6] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)

[7] [Granary: Speech Recognition and Translation Dataset in 25 European Languages](https://arxiv.org/abs/2505.13404)

[8] [Towards Measuring Fairness in AI: the Casual Conversations Dataset](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9634168)

[9] [SALM: Speech-augmented Language Model with In-context Learning for Speech Recognition and Translation](https://arxiv.org/abs/2310.09424) 

### Deployment Geography:

Global

### Use Case:

The model is intended for users requiring speech-to-text transcription capabilities for English speech, and/or transcript post-processing capabilities enabled by prompting the underlying LLMs. Typical use-cases: transcription, summarization, answering user questions about the transcript.

### Release Date:

Huggingface 07/17/2025 via https://huggingface.co/nvidia/canary-qwen-2.5b

## Model Architecture:
Canary-Qwen is a Speech-Augmented Language Model (SALM) [9] model with FastConformer [2] Encoder and Transformer Decoder [3]. It is built using two base models: `nvidia/canary-1b-flash` [1,5] and `Qwen/Qwen3-1.7B` [4], a linear projection, and low-rank adaptation (LoRA) applied to the LLM. The audio encoder computes audio representation that is mapped to the LLM embedding space via a linear projection, and concatenated with the embeddings of text tokens. The model is prompted with "Transcribe the following: <audio>", using Qwen's chat template.

### Limitations

**Input length.** The maximum audio duration in training was 40s, and the maximum token sequence length was 1024 tokens (including prompt, audio, and response). The model may technically be able to process longer sequences, but its accuracy may be degraded.

**Exclusively ASR oriented capabilities.** The model is not expected to preserve any of the underlying LLM's capabilities into speech modality. 

**English-only language support.** The model was trained using English data only. It may be able to spuriously transcribe other languages as the underlying encoder was pretrained using German, French, and Spanish speech in addition to English, but it's unlikely to be reliable as a multilingual model.

## NVIDIA NeMo

To train, fine-tune or transcribe with Canary-Qwen-2.5B, you will need to install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).

```bash
# Currently requires installing the latest trunk version of NeMo, and PyTorch 2.6+ for FSDP2 support.
python -m pip install "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"
```

## How to Use this Model

The model is available for use in the NVIDIA NeMo toolkit [6], and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset.

### Loading the Model

```python
from nemo.collections.speechlm2.models import SALM

model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
```

## Input: 

**Input Type(s):** Audio, text prompt <br>
**Input Format(s):** Audio: .wav or .flac files. Text prompt string for ASR mode: `Transcribe the following: <|audioplaceholder|>` <br>
**Input Parameters(s):** Audio: Two-Dimensional (batch, audio-samples); Text: One-Dimensional (string) <br>
**Other Properties Related to Input:** 16000 Hz Mono-channel Audio, Pre-Processing Not Needed <br>

Input to Canary-Qwen-2.5B is a batch of prompts that include audio.

Example usage in ASR mode (speech-to-text):

```python
answer_ids = model.generate(
    prompts=[
        [{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}", "audio": ["speech.wav"]}]
    ],
    max_new_tokens=128,
)
print(model.tokenizer.ids_to_text(answer_ids[0].cpu()))
```

Example usage in LLM mode (text-only):

```python
prompt = "..."
transcript = "..."
with model.llm.disable_adapter():
    answer_ids = model.generate(
        prompts=[[{"role": "user", "content": f"{prompt}\n\n{transcript}"}]],
        max_new_tokens=2048,
    )
```

To transcribe a dataset of recordings, specify the input as jsonl manifest file, where each line in the file is a dictionary containing the following fields: 

```yaml
# Example of a line in input_manifest.json
{
    "audio_filepath": "/path/to/audio.wav",  # path to the audio file
    "duration": 30.0,  # duration of the audio
}
```

and then use:
```bash
cd NeMo
python examples/speechlm2/salm_generate.py \
  pretrained_name=nvidia/canary-qwen-2.5b \
  inputs=input_manifest.json \
  output_manifest=generations.jsonl \
  batch_size=128 \
  user_prompt="Transcribe the following:"  # audio locator is added automatically at the end if not present
```

## Output:
**Output Type(s):** Text <br>
**Output Format:** Text transcript as a sequence of token IDs or a string <br> 
**Output Parameters:** One-Dimensional text string <br>
**Other Properties Related to Output:** May Need Inverse Text Normalization <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):** 
* NeMo - 2.5.0 or higher <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* [NVIDIA Ampere] <br>
* [NVIDIA Blackwell] <br>
* [NVIDIA Jetson]  <br>
* [NVIDIA Hopper] <br>
* [NVIDIA Lovelace] <br>
* [NVIDIA Pascal] <br>
* [NVIDIA Turing] <br>
* [NVIDIA Volta] <br>

**[Preferred/Supported] Operating System(s):** <br>
* [Linux] <br>
* [Linux 4 Tegra] <br>
* [Windows] <br>

## Model Version(s): 
Canary-Qwen-2.5B <br>

## Training

Canary-Qwen-2.5B was trained using the NVIDIA NeMo toolkit [6] for a total of 90k steps on 32 NVIDIA A100 80GB GPUs. LLM parameters were kept frozen. Speech encoder, projection, and LoRA parameters were trainable. The encoder's output frame rate is 80ms, or 12.5 tokens per second. The model was trained on approximately 1.3B tokens in total (this number inlcudes the speech encoder output frames, text response tokens, prompt tokens, and chat template tokens). 

The model can be trained using this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/speechlm2/salm_train.py) and [base config](https://github.com/NVIDIA/NeMo/blob/main/examples/speechlm2/conf/salm.yaml).

The tokenizer was inherited from `Qwen/Qwen3-1.7B`.

# Training and Evaluation Datasets: 

## Training Dataset:

** The total size (in number of data points): approx. 40 million (speech, text) pairs
** Total number of datasets: 26, with 18 for training and 8 for test
** Dataset partition: Training 99.6%, testing 0.04%, validation 0%
** Time period for training data collection: 1990-2025
** Time period for testing data collection: 2005-2022
** Time period for validation data collection N/A (unused)

The Canary-Qwen-2.5B model is trained on a total of 234K hrs of publicly available speech data.
The datasets below include conversations, videos from the web and audiobook recordings.

**Data Collection Method:**
* Human <br>

**Labeling Method:**
* Hybrid: Human, Automated <br>

### Properties

#### English (234.5k hours)

The majority of the training data comes from the English portion of the Granary dataset [7]:

- YouTube-Commons (YTC) (109.5k hours)
- YODAS2 (77k hours)
- LibriLight (13.6k hours)

In addition, the following datasets were used:
- Librispeech 960 hours
- Fisher Corpus
- Switchboard-1 Dataset
- WSJ-0 and WSJ-1
- National Speech Corpus (Part 1, Part 6)
- VCTK
- VoxPopuli (EN)
- Europarl-ASR (EN)
- Multilingual Librispeech (MLS EN)
- Mozilla Common Voice (v11.0)
- Mozilla Common Voice (v7.0)
- Mozilla Common Voice (v4.0)
- AMI
- FLEURS

AMI was oversampled during model training to constitute about 15% of the total data observed. 
This skewed the model towards predicting verbatim transcripts that include conversational speech disfluencies such as repetitions.

The training transcripts contained punctuation and capitalization.

## Evaluation Dataset:

**Data Collection Method:** <br>
* Human <br>

**Labeling Method:** <br>
* Human <br>

Automatic Speech Recognition: 
* [HuggingFace OpenASR Leaderboard evaluation sets](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

Hallucination Robustness:
* [MUSAN](https://www.openslr.org/17/) 48 hrs eval set

Noise Robustness:
* [Librispeech](https://www.openslr.org/12)

Model Fairness:
* [Casual Conversations Dataset](https://arxiv.org/pdf/2104.02821)

## Performance

The ASR predictions were generated using greedy decoding.

### ASR Performance (w/o PnC) 

The ASR performance is measured with word error rate (WER), and we process the groundtruth and predicted text with [whisper-normalizer](https://pypi.org/project/whisper-normalizer/) version 0.1.12.

WER on [HuggingFace OpenASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard):

| **Version** | **Model**     | **RTFx**   | **Mean**   | **AMI**   | **GigaSpeech**   | **LS Clean**   | **LS Other**   | **Earnings22**   | **SPGISpech**   | **Tedlium**   | **Voxpopuli**   |
|:---------:|:-----------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 2.5.0  | Canary-Qwen-2.5B | 418 | 5.63 | 10.18 | 9.41 | 1.60 | 3.10 | 10.42 | 1.90 | 2.72 | 5.66 |

More details on evaluation can be found at [HuggingFace ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

### Hallucination Robustness
Number of characters per minute on [MUSAN](https://www.openslr.org/17) 48 hrs eval set (`max_new_tokens=50` following `nvidia/canary-1b-flash` evaluation)
| **Version** | **Model** | **# of character per minute** |
|:-----------:|:---------:|:----------:|
| 2.5.0       | Canary-Qwen-2.5B |   138.1   |

### Noise Robustness
WER on [Librispeech Test Clean](https://www.openslr.org/12) at different SNR (signal to noise ratio) levels of additive white noise

| **Version** | **Model** | **SNR 10** | **SNR 5** | **SNR 0** | **SNR -5** |
|:-----------:|:---------:|:----------:|:----------:|:----------:|:----------:|
| 2.5.0       | Canary-Qwen-2.5B |    2.41%   |   4.08%   |   9.83%   |    30.60%  |

## Model Fairness Evaluation

As outlined in the paper "Towards Measuring Fairness in AI: the Casual Conversations Dataset" [8], we assessed the Canary-Qwen-2.5B model for fairness. The model was evaluated on the CasualConversations-v1 dataset with inference done on non-overlapping 40s chunks, and the results are reported as follows:

### Gender Bias:

| Gender | Male | Female | N/A | Other |
| :--- | :--- | :--- | :--- | :--- |
| Num utterances | 18471 | 23378 | 880 | 18 |
| % WER | 16.71 | 13.85 | 17.71 | 29.46 |

### Age Bias:

| Age Group | (18-30) | (31-45) | (46-85) | (1-100) |
| :--- | :--- | :--- | :--- | :--- |
| Num utterances | 15058 | 13984 | 12810 | 41852 |
| % WER | 15.73 | 15.3 | 14.14 | 15.11 |

(Error rates for fairness evaluation are determined by normalizing both the reference and predicted text, similar to the methods used in the evaluations found at https://github.com/huggingface/open_asr_leaderboard.)

## Inference:
**Engine:** NVIDIA NeMo <br>
**Test Hardware :** <br>
* A6000 <br>
* A100 <br>
* RTX 5090 <br>

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  
For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards. Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

```


---ASSISTANT
Thank you. I will refer to this model card in my future responses.
---ASSISTANT
Thank you. I will refer to this model card in my future responses.
---USER
Here is additional research information about the model:

```


<model_research>
   <research source="https://choosealicense.com/licenses/cc-by-4.0/">
- The CC-BY-4.0 license is not recommended for software, even though it permits commercial use, modification, and distribution for most media and data.
- The license explicitly does not grant patent or trademark rights, and licensors should be cautious about this when applying it to AI models that may involve patented methods or branding.
- Licensors are encouraged to mark any material not subject to the license within a distributed work, which can be critical for complex AI model releases involving multiple components or datasets with varying rights.
- The license is irrevocable but rights can be reinstated automatically within 30 days if a violation is cured, which is a surprisingly generous provision.
- Sui generis database rights are covered: the license grants rights to extract and reuse database content, an important consideration for AI datasets built from aggregate sources.
- Moral rights (such as integrity/authorship) are not licensed, though the licensor tries to waive them where possible—this may impact model attributions or derived works in some jurisdictions.
- You cannot impose additional restrictions or DRM on downstream recipients, and must allow them the same rights you receive. Proprietary restraints—like restricting fine-tuning or downstream use—in conflict with this are not allowed.
- If the licensor asks, you must remove certain attribution information “to the extent reasonably practicable”—this is a rare “right to be forgotten” flavor not typical in permissive licenses.
- Creative Commons disclaims all warranties entirely and is not party to the license; this shifts all risk to users and distributors.

These nuances—especially regarding patent rights, database reuse, rights reinstatement, and the prohibition against new restrictions—are surprisingly relevant to researchers releasing or using AI models under CC-BY-4.0.
   </research>
   <research source="https://arxiv.org/abs/1706.03762">
Summary for AI Model Researchers:

The Transformer architecture introduced in "Attention Is All You Need" was notable for its complete elimination of recurrence and convolutions, making attention the sole mechanism for both encoding and decoding sequence information. This shift allowed for unprecedented model parallelization and speedup in training, as sequential dependencies limiting the throughput of RNNs or CNNs were removed. 

A surprising aspect for researchers is that by applying only stacked self-attention and point-wise, fully connected layers, the authors were able to outperform established architectures across two major machine translation tasks with a fraction of the training cost and time. For example, the model achieved 28.4 BLEU on WMT English–German and 41.8 BLEU on WMT English–French with just 3.5 days of training on eight GPUs—vastly faster than previous state-of-the-art ensembles. Further, the Transformer generalized to syntactic tasks like English constituency parsing—demonstrating that attention alone could extract both global and local structure.

For researchers interested in efficiency, the inherent parallelizability (especially during training) was transformative: all positions in a sequence can be processed simultaneously rather than in order. Additionally, the paper’s multi-head attention mechanism enabled the model to learn relations across multiple subspaces of representation, a feature now foundational to scaling up model sizes.

In summary: The most surprising point for an AI model researcher is how the Transformer’s abandonment of recurrence/convolution, reliance on multi-head attention, and associated gains in training efficiency set the stage for the subsequent revolution in large-scale language models and sequence modeling architectures.
   </research>
   <research source="https://huggingface.co/Qwen/Qwen3-1.7B">
Here is a summary of interesting or surprising aspects of Qwen/Qwen3-1.7B that are particularly relevant for AI Model researchers, and not emphasized in the original Canary-Qwen-2.5B model card:

1. **Dynamic “Thinking” and “Non-Thinking” Modes**:
   - Qwen3-1.7B uniquely allows seamless switching—both via API and user prompt—between "thinking" mode (for in-depth reasoning, math, code) and "non-thinking" mode (for speed and efficiency, general dialogue). The distinction can be controlled either at initialization or dynamically in multi-turn conversations using /think and /no_think directives within prompts. In “thinking” mode, Qwen3 will emit explicit `<think>...</think>` blocks with intermediate reasoning outputs, enabling fine-grained interpretability and tool integration.

2. **Extraordinarily Long Context Window**:
   - It supports a context length of up to 32,768 tokens (with recommendations up to 38,912 for highly complex problems). This is unusually large for a 1.7B parameter model and enables extended, coherent dialogue and reasoning chains—making it highly suitable for multi-step tasks, extreme summarization, and even codebase comprehension.

3. **Hybrid GQA Attention Architecture**:
   - Qwen3-1.7B uses Grouped Query Attention (GQA): 16 heads for queries and only 8 for keys/values. This architectural choice is rarely discussed in smaller models and may contribute to improved efficiency or context handling.

4. **Explicit Best Practices for Output Quality**:
   - The authors provide detailed, concrete advice on sampling strategies for each mode (e.g., temperature, top_p, presence_penalty), and warn against greedy decoding (which can cause endless repetition or output collapse). This transparency is notable and not common in OSS LLMs of this size.

5. **Agentic and Tool Use Focus**:
   - The model is explicitly designed for “agentic” applications, including advanced tool-calling patterns. It is tightly integrated with the Qwen-Agent project, including MCP-based tool configuration, suggesting a strong focus on real-world agent-based and tool-augmented applications.

6. **Extensive Language Support**:
   - Qwen3-1.7B claims robust training and instruction-following for over 100 languages and dialects, which is unusual among models in the 1-2B parameter range, and could be highly appealing for multi-lingual agent and translation research.

7. **Fine-tuned Reasoning and Alignment**:
   - The documentation claims superior human-alignment in creative writing, multi-turn dialogue, and instruction following, compared to both prior Qwen and Qwen2.5-Instruct models—even when “thinking” mode is off, which suggests sophisticated post-training strategies.

8. **Explicit Formatting for Benchmarking**:
   - The authors recommend prompt and output formatting conventions for math and MCQ evaluation (e.g., putting math answers in `\boxed{}`) to allow for easier, automated benchmarking and reproducibility—a practice rarely seen in model cards.

9. **Open and Evolving Model Ecosystem**:
   - Qwen3-1.7B is not only a foundation model but is actively being used as a base for >80 adapters, >130 finetunes, and >80 quantizations within the Hugging Face ecosystem, indicating rapid community uptake and flexibility.

10. **Special Token for Reasoning Block**:
    - The model uses a unique special token (151668 for `</think>`) to delimit its reasoning blocks, which may allow novel ways of parsing, short-circuiting, or supervising chain-of-thought outputs at inference time.

Overall, Qwen/Qwen3-1.7B stands out as a small-model LLM explicitly designed for agentic, tool-using, and multi-lingual settings, with highly customizable reasoning behaviors and unusually strong transparency in operating instructions and best practices. These features may make it especially attractive not just for research but also for software applications demanding plug-and-play LLM control and observability.
   </research>
   <research source="https://arxiv.org/abs/2503.05931">
Here are interesting and potentially surprising points for AI Model researchers from "Training and Inference Efficiency of Encoder-Decoder Speech Models" (arXiv:2503.05931) that are not covered in the model card:

- Padding Waste: The authors found that more than 50% of computation during training encoder-decoder speech models may be wasted on padding, simply due to inefficient minibatch sampling of sequential data. Padding inefficiency is a dominant factor in resource waste, often overlooked in focus on model or hardware details.

- Batch Size Multiplication: By optimizing sampling and batching strategies in training (specifically for Canary-1B), they achieved a 5x increase in average batch sizes without extra hardware. This directly translates into dramatically improved GPU utilization and efficiency.

- Hardware Savings: With their improvements, it's possible to train a model of the same scale using only one quarter of the GPUs previously required (i.e., 4x hardware reduction) for the same training duration—or, with the same hardware, training is completed in half the time ('2x shorter wall time').

- Inference Speedup via Model Restructuring: The major inference bottleneck was found in the autoregressive decoder steps. By shifting model parameters from the decoder to the encoder, they achieved a 3x speedup in inference (as measured by inverse real-time factor) while maintaining accuracy and convergence requirements.

- Open Sourcing: The paper promises to release both the optimized training code and models as open-source, directly enabling replication and further efficiency research by the community.

These findings spotlight easily overlooked bottlenecks—especially padding waste during minibatch formation—that, if targeted, enable order-of-magnitude gains in training efficiency for large encoder-decoder speech models. The architectural insight (moving parameters encoder-ward for faster inference without loss) likewise has implications for designing new speech and sequence modeling systems.
   </research>
   <research source="https://github.com/NVIDIA/NeMo">
Here are key points from the NVIDIA NeMo GitHub repository that are notable or potentially surprising for AI Model researchers, while avoiding repetition of already-present details in the Canary-Qwen-2.5B model card:

---

**1. Rapid Integration with Hugging Face Models**
- The recent introduction of AutoModel support in NeMo allows researchers to pretrain and finetune Hugging Face models (including Llama 4 and others) directly within the NeMo framework. This feature streamlines transfer between frameworks and enables immediate use of newly released models and architectures.

**2. Early/Advanced Multimodal & Physical World Modeling**
- NeMo now supports NVIDIA Cosmos World Foundation Models, offering tools for developing "world models" capable of generating realistic synthetic videos and serving robotics and autonomous driving with simulation-trained models. The Cosmos tokenizer efficiently compresses visual data into semantic tokens, which supports LLM and diffusion model training in multimodal contexts.

**3. Scalability & Performance Innovations**
- NeMo 2.0 introduces Python-based configuration (departing from YAML), improving extensibility for large-scale experiments.
- Native support for running distributed training on massive infrastructure (recent MLPerf results cite near-linear scaling on 11,616 H100 GPUs).
- Innovations like the NeMo-Run tool for orchestrating massive parallel jobs and an autoconfigurator to automatically discover optimal model parallel configurations.

**4. Advanced State Space Models and Hybrid Architectures**
- Beyond Transformers: NeMo now supports pretraining/fine-tuning of state space models (SSMs), and hybrid architectures like Griffin (combining state space and Transformer techniques), broadening beyond standard attention models.

**5. Speech AI Records & Inference Speed**
- Recent ASR models (e.g., Parakeet series) achieve inverse real-time factors (RTFx) of up to 6,000, a 10x speedup over previous models, suggesting real-time speech applications at record efficiency.
- NeMo’s speech pipeline is also integrated into NVIDIA Riva, offering immediate deployment for production speech services.

**6. Parameter-Efficient Fine-Tuning and Advanced Alignment**
- Broad support for SFT, LoRA, P-Tuning, Adapters, and IA3; alignment methods include SteerLM, DPO, and RLHF, all within unified modular APIs, giving researchers maximum flexibility for frontier research in model alignment and efficiency.

**7. Versatile Deployment: Cloud-native Focus**
- Deep Docker/container support and cloud-native design allow rapid ramp to major cloud providers (Amazon EKS, Google GKE), with extensive playbooks for end-to-end LLM/ASR/TTS training and deployment.

**8. Community & Ecosystem**
- Active repository: >8,800 commits, ~70 releases, 462 contributors, and support for major research and enterprise models (Amazon Titan, etc.).
- NeMo’s concepts and infrastructure (e.g., Megatron Core, Transformer Engine for FP8 training) are frequently adopted in commercial and OSS LLMs outside NVIDIA as well.

---

These features suggest that NeMo is more than a toolkit for standard model training; it's a research accelerator for state-of-the-art and emerging model architectures, especially at the intersection of scale, multimodality, and real-world AI deployment. The speed with which it incorporates new architectural advances, cloud workflows, and model families is particularly striking.
   </research>
   <research source="https://arxiv.org/abs/2505.13404">
Here is a summary of arXiv:2505.13404, highlighting points interesting or surprising to AI model researchers (excluding information already in the nvidia/canary-qwen-2.5b model card):

- The Granary dataset is the first open-source speech resource at this scale combining both speech recognition and translation data for 25 European languages—including for low-resource languages.
- The data pipeline introduces several advanced techniques: pseudo-labeling with segmentation; two-pass inference; hallucination filtering; and punctuation restoration, to enhance label quality.
- Granary generates translation data by leveraging pseudo-labeled (ASR) transcriptions combined with EuroLLM, followed by aggressive data filtration, enabling the creation of high-quality parallel data from noisy or automatically transcribed sources—significantly scaling resource-constrained languages.
- The entire speech/translation data processing pipeline is designed for efficiency and scalability: it can process large volumes of data in a matter of hours, making rapid dataset expansion and updating feasible.
- A surprising result: Models trained on Granary-processed data achieve comparable recognition/translation performance to models trained on previous curated datasets—using around 50% less data, suggesting the new curation/filtering techniques can dramatically improve data efficiency.
- The dataset was accepted at Interspeech 2025 and is available at https://hf.co/datasets/nvidia/Granary

These findings suggest that strategic data curation and pseudo-labeling pipelines can substantially reduce the quantity of required training data, potentially accelerating progress for many under-resourced languages in multilingual speech processing.
   </research>
   <research source="https://arxiv.org/abs/2310.09424">
Key novel and potentially surprising points from arXiv:2310.09424 (“SALM: Speech-augmented Language Model with In-context Learning for Speech Recognition and Translation”) of specific interest to AI model researchers, and NOT explicitly repeated in the nvidia/canary-qwen-2.5b model card:

- SALM introduces zero-shot in-context learning for speech tasks—a new paradigm: The model enables tasks like ASR and automatic speech translation (AST) to benefit from in-context learning, meaning the same model can adapt to specific instructions, boost keywords, or perform specializations without additional retraining or architectural changes. This ability is shown in both ASR and AST, demonstrating significant flexibility.

- The paper describes a “speech supervised in-context training” approach: This technique specifically aims to bridge the gap between LLM training (which is usually text-based) and downstream speech tasks, further boosting speech-to-text models' in-context learning capability.

- Unlike most prior approaches, SALM maintains a frozen LLM backbone, only adapting modality input (audio) and minimal additional LoRA parameters. This “frozen LLM + lightweight adaptation” design might provide efficiency advantages over the full fine-tuning variants that are popular in the field.

- The authors explicitly demonstrate that a unified model can match (not just approach) task-specific supervised Conformer baselines on both ASR and AST tasks—a notable surprise since multi-task, multi-modal models often lag behind single-task models.

- The work is being open-sourced via the NeMo toolkit, promoting easy reproducibility and extensibility.

In sum: The ability to perform zero-shot, instruction-following, in-context learning for speech recognition and translation by minimally augmenting a frozen LLM represents a significant leap toward truly unified, flexible, and efficient multimodal language models. This methodological simplicity, model efficiency, and the demonstration of practically no loss relative to strong baselines make this work especially surprising and relevant for researchers pursuing generalized or instruction-driven speech-and-language models.
   </research>
   <research source="https://github.com/NVIDIA/NeMo.git">
Here is a summary of findings from the current NeMo repository that would be especially interesting or surprising to an AI Model researcher, intentionally excluding content already in your provided model card:

1. **Instant Hugging Face Model Integration (AutoModel):**  
   Recent releases of NeMo feature “AutoModel,” enabling direct pretraining and fine-tuning of Hugging Face models—LLMs and even some image-text-to-text models—via the NeMo pipeline. This makes NeMo a potential “convergence point” for workflows that previously required toolchain migration.

2. **Advanced Support for New Architectures and State Space Models:**  
   NeMo now supports training not just Transformers, but also state space models (SSMs)—including the Griffin architecture from DeepMind—reflecting a non-Transformer research direction for scaling LLMs.

3. **Extremely High-Scale Distributed Training:**  
   NVIDIA demonstrated near-linear scaling to over 11,000 H100 GPUs for LLM pretraining with NeMo, with associated records in the MLPerf v4.0 benchmarks.  
   This showcases NeMo’s practical capacity for “supercluster” scale research, relevant to teams pushing model scaling laws to extremes.

4. **Accelerated Inference Beyond 2,000x Real-Time (RTFx):**  
   ASR models in NeMo, such as advanced CTC and RNN-T types, now exceed RTFx of 2,000—and some reach 6,000—thanks to optimizations outlined in public research blogs. This is 10x faster than prior generations, dramatically reducing the barrier for real-time and large-scale speech deployment.

5. **Video Foundation Models and Physical AI Integration:**  
   NeMo now underpins the “Cosmos World Foundation” platform—a CUDA-optimized, end-to-end system for training and fine-tuning world models that can generate synthetic video environments from text. This supports physical AI advancements for robotics and self-driving, aligning with the latest research interest in video generative models and simulation.

6. **Python-First Dynamic Configuration:**  
   NeMo 2.0 replaced YAML configs with Python-based configurations, enabling full programmatic customization and scripting of experiment configs—making hyperparameter sweeps, architecture ablation, and experiment automation vastly more streamlined.

7. **Cutting-Edge PEFT and Alignment Methods Built-In:**  
   Beyond Supervised Fine-Tuning (SFT), NeMo offers built-in state-of-the-art Parameter Efficient Fine Tuning: LoRA, P-Tuning, Adapters, IA3, and alignment algorithms including DPO, RLHF, and SteerLM via open-source modules.

8. **Curated Data Pipelines for Video Processing:**  
   “NeMo Curator” provides high-throughput, GPU-accelerated pipelines for dataset curation, especially for video (e.g., synthetic data for physical world modeling), reportedly achieving up to 89x speed-up over CPU baselines.

9. **Community and Ecosystem Reach:**  
   NeMo is widely adopted (15k+ stars, 3k+ forks) and ships ready-to-use containers, Colab notebooks, and extensive playbooks for quick experimentation and reproducible research. It supports all major NVIDIA GPU architectures, and comes with native support for both single-node and massive multi-node, multi-GPU environments.

These points, especially modular integration with Hugging Face, extreme scaling capacity, state space model support, video foundation model tools, and Python-first experiment configuration, stand out as potentially surprising or extremely relevant to current AI model research workflows.
   </research>
   <research source="https://huggingface.co/spaces/hf-audio/open_asr_leaderboard">
Summary of https://huggingface.co/spaces/hf-audio/open_asr_leaderboard – Points of Interest for AI Model Researchers:

1. Continuous Leaderboard Competition & Model Diversity:
   - The OpenASR Leaderboard is rapidly updated, showcasing a highly dynamic field with frequent new entries and model submissions. Dozens of ASR models from industry, academia, and individuals are included, with a variety of architectures (Whisper, SALM, Conformer, etc.).
   - Several models specialize in constrained domains or languages (not only English).

2. Benchmarking Scope & Real-World Complexity:
   - The leaderboard ranks models on a wide range of datasets, including not only standard English benchmarks (e.g., LibriSpeech, TED-LIUM, Earnings-22) but also noisy/reverberant environments, conversational speech, and multi-speaker meetings. This means top performance numbers are typically aggregates over challenging, real-world scenarios and are not just “lab clean”.

3. Live Reporting of Model Submission Metadata:
   - Leaderboard entries often display not just WER but also RTF (Real Time Factor), model size, quantization details, submission date, and hardware used for inference. Some top-performing models achieve both strong accuracy and significant efficiency/compactness, and quantized models sometimes outperform or closely match their full-precision counterparts.

4. Surprising Results & Emerging Trends:
   - In several cases, open models (e.g., OpenAI's Whisper, various iterative improvements from Nvidia, Google’s Universal Speech Model—USM) achieve results competitive with or surpassing some commercial, closed-source systems.
   - Small and medium parameter models (e.g., in the 600M–2.5B range) often challenge or beat much larger competitors, reflecting rapid progress in efficient architecture search and training pipelines.

5. Human Reference WER:
   - Some benchmark datasets now include “human reference” WER, highlighting a gap that is closing on certain test sets, with some models getting within a fraction of a percentage point of expert transcriber accuracy.

6. Transparency in Evaluation:
   - Submission and evaluation is automated—researchers upload model hypotheses for scoring—which reduces risk of hand-tuning or cherry-picking and heightens benchmark trustworthiness.

7. Community Contributions & Collaboration:
   - Many best-in-class models are the results of cross-institutional collaborations and community-sourced datasets, showing the growing importance of open contributions in the ASR ecosystem.

8. Leaderboard Limitations & Ongoing Challenges:
   - The leaderboard identifies known issues such as model performance drop under domain shift, noise, overlapping speech, and dialectal variations. Some models that perform well on “clean” LibriSpeech deteriorate on conversation-heavy or noisy datasets.

These points reveal a highly competitive and transparent ASR evaluation ecosystem, with strong emphasis on open science, practical benchmarking, and rapid iteration, and highlight areas where even top models still struggle (e.g., noise, accent, multi-speaker overlap).
   </research>
   <research source="https://www.openslr.org/17/">
Summary for AI Model Researchers:

MUSAN (SLR17) is a sizable (11GB) open corpus comprising not only speech, but also a wide range of music and noise recordings—making it a versatile resource for training and evaluating models on robust speech recognition, voice activity detection, and speaker verification tasks under varied acoustic conditions. Notably, MUSAN was released in 2015 by Snyder, Chen, and Povey and is licensed under CC BY 4.0, supporting both academic and commercial use.

A potentially surprising point is MUSAN's combination of different audio modalities (music, speech, noise) in one unified dataset, which allows researchers to simulate challenging real-world environments for both training and benchmarking ASR models—especially valuable for testing noise and music robustness beyond conventional speech datasets. This diversity makes it uniquely suitable for adversarial and robustness testing. The project was supported by an NSF Graduate Research Fellowship. 

The dataset's structure and licensing have made it a longstanding fixture in speech research, particularly for researchers targeting model performance in noisy and unconstrained or "in-the-wild" settings.
   </research>
   <research source="https://www.openslr.org/12">
Here are interesting and possibly surprising points from the official LibriSpeech (SLR12) resource page for AI model researchers:

- **Original Data Sources and Alignment**: LibriSpeech consists of read audiobooks from LibriVox that are meticulously segmented and aligned with the corresponding Project Gutenberg texts. The distribution includes both the original mp3 files and the exact book texts used for alignment. This enables researchers to re-align or re-segment using improved text-audio alignment methods or alternate text processing approaches.

- **Metadata and Disclaimers**: In addition to the main data splits (clean/other, train/dev/test), the corpus provides extra resources:
    - A ~695MB archive of LibriVox's introductory disclaimers for some speakers.
    - 33MB of “raw metadata” generated during the corpus creation—potentially useful for researchers wanting deep traceability or corpus reprocessing.
    - MD5 checksum files for all archives, aiding reproducibility and integrity checks.

- **Extensive Download Options**: Multiple mirrors are hosted globally (US, EU, China), and archives are available for each split, supporting reproducible research and robust data access—even in regions with restricted bandwidth.

- **Open-Source Acoustic and Language Models**: The OpenSLR page links to pre-trained acoustic models (from Kaldi) and language models specifically intended for evaluating with LibriSpeech. This can help with benchmarking novel approaches without the need for initial model training, accelerating prototyping.

- **Reproducibility and Documentation**: The resource explicitly ties its design to a peer-reviewed conference paper submission (ICASSP 2015), emphasizing the importance of open, well-documented processes in influential datasets.

- **Scale and Structure**: The dataset’s scale (1,000 hours) is split into multiple “clean” vs “other” categories (for both train and test), and “clean”/“other” roughly map to the perceived acoustic quality or level of challenge for ASR, not simply random assignment.

These details highlight deep traceability, the ability for full end-to-end experimentation, and a focus on high reproducibility—all features valued by advanced AI and ASR researchers.
   </research>
   <research source="https://pypi.org/project/whisper-normalizer/">
Key points from the whisper-normalizer PyPI page that may interest AI model researchers:

- The package explicitly highlights problems with Whisper's default English-centric normalization approach when applied to Indic and other low-resource languages. To address this, the package incorporates language-specific logic, notably derived from the indic-nlp-library, and implements specialized normalization (like MalayalamNormalizer) for Indian languages. This is significant, as indiscriminate use of English-trained normalization pipelines can meaningfully distort or degrade WER/CER evaluation for non-English ASR models—a point specifically discussed in a referenced blog post by Dr. Kavya Manohar.

- The tool provides both BasicTextNormalizer and EnglishTextNormalizer classes, with examples showing their subtle differences (e.g., BasicTextNormalizer leaves apostrophe-separated clitics split as tokens: "i m a little teapot..." vs. "i am a little teapot..." from EnglishTextNormalizer).

- The package's normalization logic is used not just for OpenAI Whisper, but also by multiple state-of-the-art ASR model benchmarks (including Meta's MMS, AssemblyAI Conformer-1/2). This central role in evaluation (including for diverse language families) means design decisions or errors in normalization routines could ripple across leaderboard results and research claims in the ASR field.

- The package is released under the MIT License and rapidly updated (the release history shows frequent recent updates, implying active maintenance and likely responsiveness to emerging normalization issues in new languages or model versions).

Summary for researchers: 
The "whisper-normalizer" package both highlights and addresses the risks of cross-linguistic bias introduced by English-trained normalization routines in ASR evaluation, and is increasingly standard in multilingual model benchmarking pipelines; close scrutiny of its implementation (especially language-specific paths) is warranted when comparing international ASR research or making fairness claims.
   </research>
   <research source="https://www.openslr.org/17">
Summary of MUSAN (SLR17) for AI Model Researchers:

The MUSAN dataset (SLR17) is a large, freely available corpus containing not just speech but also music and diverse environmental noise recordings. This richness is notable—it provides researchers with a versatile resource for robust ASR training and testing, especially for evaluating model resilience to real-world audio conditions such as background sounds, not just clean speech. The dataset is distributed under the liberal CC BY 4.0 license and is approximately 11GB, reflecting the substantial coverage of audio events (music, speech, noise). It was created by David Snyder, Guoguo Chen, and Daniel Povey, and has been a foundational dataset (referenced as arXiv:1510.08484) since 2015. A unique aspect is its significant use in domain adaptation and noise-robust ASR research, as it enables controlled experiments with real, not synthetic, background noise.
   </research>
   <research source="https://github.com/huggingface/open_asr_leaderboard.">
Here is a summary of interesting or surprising points for an AI Model researcher gleaned from the HuggingFace Open ASR Leaderboard repository (https://github.com/huggingface/open_asr_leaderboard), excluding information already contained in the provided nvidia/canary-qwen-2.5b model card:

---

**Open ASR Leaderboard: Notable Aspects for AI Researchers**

1. **True Benchmarking Consistency**:
   - The leaderboard is designed for rigorous, cross-library benchmarking of ASR models. It enforces the use of the same decoding hyperparameters (e.g., batch size, beam size) for each dataset and model type as per the ESB (Evil Speech Benchmark) paper guidelines. This is crucial for fair performance comparison and is often overlooked in informal model leaderboards.

2. **Extensible, Multi-Library Evaluation**:
   - The repo is structured to allow straightforward addition of new ASR libraries or models. Template scripts for popular libraries (e.g., transformers, speechbrain, NeMo) standardize the evaluation workflow, making it easy for the community to contribute and for researchers to cross-validate results on a level playing field.

3. **Detailed Hardware and Software Reproducibility**:
   - For results to be accepted, submissions must specify precise hardware (NVIDIA A100-SXM4-80GB, driver version, CUDA version, PyTorch version). If contributors lack the required hardware, maintainers offer to run submitted scripts on the official benchmarking hardware. This moves beyond anecdotal reporting to actual reproducibility—an ongoing concern in the AI research community.

4. **Library-Specific Optimization Requirements**:
   - The NeMo integration, for example, currently requires CUDA 12.6+ for RNN-T inference, owing to known driver/kernel issues (reference: NVIDIA/NeMo#9869). This highlights how cutting-edge ASR research often hinges on subtleties in library and driver compatibility—a familiar pain for experienced researchers.

5. **Transparent, Scripted Pipeline**:
   - Each evaluated library includes its own Python script (run_eval.py) and bash launcher, enforcing strict manifest data loading, evaluation, and results output for WER/RTFx. The leaderboard relies on these artifacts for its reproducibility pledge.

6. **Community-Maintained and Actively Developed**:
   - With over 151 commits and 19+ contributors from well-known teams (including Nvidia NeMo and SpeechBrain), the project is actively developed. Forks and stars indicate strong community traction.

7. **Linguistic Nuance via Normalizer**:
   - The evaluation pipeline includes a normalizer step, notably for English, ensuring WER calculations are consistent and less sensitive to idiosyncratic tokenization—a subtle but important detail for reliable benchmarking.

---

**Surprising/Noteworthy Insights:**

- The leaderboard infrastructure does not merely present scores, but encodes a meta-benchmarking philosophy: it is a living, code-first, hardware-attentive platform aiming to elevate ASR model comparison to the same reproducibility standards as the best ML competitions.
- The commitment to hardware specification and the willingness by maintainers to re-run outside contributions on controlled hardware is an uncommon practice among open leaderboards.
- The architecture encourages researchers to pay attention to every aspect of their inference stack, from driver versions to data manifest formats.
- Submissions can be affected by low-level CUDA/driver bugs, reminding researchers that deep learning deployment remains a systems engineering as well as a modeling challenge.

---

For researchers, the Open ASR Leaderboard is a model of community-driven ML benchmarking, with strong attention to reproducibility, extensibility, and the realities of systems compatibility in ASR research.
   </research>

</model_research>


```

Your task is to produce a concise 300-500 word summary of the specified model highlighting 
key points of interest for AI researchers. Make sure that major information and traits are
specified, and prefer bullet points for notable points of interest. Use markdown links to 
refer to sources if warranted.


---ASSISTANT
# Canary-Qwen-2.5B: Researcher-Oriented Model Summary

NVIDIA’s [Canary-Qwen-2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b) is a state-of-the-art English automatic speech recognition (ASR) model, blending innovations in speech and language modeling with efficient large-scale engineering. Below are concise highlights of particular interest for AI researchers.

---

## Key Characteristics

- **Architecture**:  
  - Hybrid [SALM](https://arxiv.org/abs/2310.09424) model using a FastConformer encoder ([arXiv:10389701](https://ieeexplore.ieee.org/document/10389701)) and Transformer decoder ([Attention Is All You Need](https://arxiv.org/abs/1706.03762)).
  - Built from base models: [nvidia/canary-1b-flash](https://arxiv.org/abs/2503.05931) (optimized for ASR) and [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (LLM).
  - Frozen LLM backbone; only speech encoder, projection, and LoRA layers fine-tuned—boosting efficiency and transferability.  
- **Modes**:  
  - **ASR Mode**: Direct speech-to-text.
  - **LLM Mode**: Transcript-based reasoning, QA, and summarization (no raw audio understanding).
- **Context**:  
  - Input: Up to 40s audio or 1024 tokens (training), with technical support for longer inputs.
  - Output: Text transcription, optionally with post-processing by LLM.

---

## Noteworthy Research Details

- **Training Efficiency Breakthroughs** ([arXiv:2503.05931](https://arxiv.org/abs/2503.05931)):
  - Minibatch sampling was optimized to reduce padding waste (up to 5× batch size, 2× faster wall time, 4× less hardware).
  - Model restructuring (shifting parameters encoder-ward) achieved 3× inference speedup without accuracy loss.

- **Data & Evaluation**:
  - Trained on 234k+ hours of speech from 26+ datasets, prominently the [Granary](https://arxiv.org/abs/2505.13404) resource, pioneering efficient data curation (comparable accuracy from 50% less data).
  - Evaluated on the OpenASR Leaderboard ([HF Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)), achieving competitive WERs on real-world tasks.

- **Research-Grade Prompting**:
  - Uses Qwen3 tokenizer, supporting up to 32k context window, and in LLM mode, inherits highly agentic, multi-lingual, and reasoning-flexible capacities ([Qwen3-1.7B Card](https://huggingface.co/Qwen/Qwen3-1.7B)).

- **Ethics, Fairness & Reproducibility**:
  - License: [CC BY 4.0](https://choosealicense.com/licenses/cc-by-4.0/) (permitting commercial use, but with no patent grant—see caveats).
  - Fairness: Documented, with detailed evaluation stratified by gender and age ([Casual Conversations Dataset](https://arxiv.org/pdf/2104.02821)).
  - Standardized text normalization ([whisper-normalizer](https://pypi.org/project/whisper-normalizer/)), transparency and hardware reproducibility per [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard).

- **Engineering and Ecosystem** ([NeMo Toolkit](https://github.com/NVIDIA/NeMo)):
  - Supports rapid integration with Hugging Face models, distributed/cloud-native training and deployment, modular PEFT strategies, multi-modal and video model support.
  - NeMo ecosystem delivers Python-first configuration, cutting-edge scaling (~11k GPUs), and adoption for LLM, ASR, TTS, and robotics.

---

## Standout Features for Researchers

- **Unified Model for Speech and Language Tasks**: Frozen LLM + lightweight LoRA adaptation yields minimal loss vs monolithic speech models, enabling flexible instruction-tuning and in-context learning ([arXiv:2310.09424](https://arxiv.org/abs/2310.09424)).
- **Best-in-class Efficiency**: Padding and architecture optimizations deliver exceptional training/inference speedups—critical for scaling research or deployment.
- **Open, Transparent Benchmarking**: Full evaluations on [HF Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) ensure replicability and meaningful model comparisons.
- **Robustness to Real-world & Noisy Data**: Performance tested under noise (e.g., MUSAN), accent, and domain shifts, mirrored in leaderboard and fairness assessments.

---

**In summary:** Canary-Qwen-2.5B exemplifies state-of-the-art speech modeling efficiency and practical deployment, with reproducible, open-science benchmarking and deep integration into the NVIDIA NeMo and wider research ecosystem—serving as a reference point for future ASR and multimodal model design.

---

**References:**  
- [Model Card](https://huggingface.co/nvidia/canary-qwen-2.5b)  
- [NeMo Toolkit](https://github.com/NVIDIA/NeMo)  
- [SALM Paper](https://arxiv.org/abs/2310.09424)  
- [Granary Dataset](https://arxiv.org/abs/2505.13404)  
- [OpenASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)