---USER

Here is the model card for black-forest-labs/FLUX.1-Krea-dev:

```
---
language:
- en
license: other
license_name: flux-1-dev-non-commercial-license
license_link: LICENSE.md
extra_gated_prompt: By clicking "Agree", you agree to the [FluxDev Non-Commercial
  License Agreement](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
  and acknowledge the [Acceptable Use Policy](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/POLICY.md).
tags:
- text-to-image
- image-generation
- flux
- diffusers
pipeline_tag: text-to-image
base_model:
- black-forest-labs/FLUX.1-dev
library_name: diffusers
---

![FLUX.1 Krea [dev] Grid](./teaser.png)


`FLUX.1 Krea [dev]` is a 12 billion parameter rectified flow transformer capable of generating images from text descriptions.
For more information, please read our [blog post](https://bfl.ai/announcements/flux-1-krea-dev) and [Krea's blog post](https://www.krea.ai/blog/flux-krea-open-source-release).


# Key Features
1. Cutting-edge output quality, with a focus on aesthetic photography.
2. Competitive prompt following, matching the performance of closed source alternatives.
3. Trained using guidance distillation, making `FLUX.1 Krea [dev]` more efficient.
4. Open weights to drive new scientific research, and empower artists to develop innovative workflows.
5. Generated outputs can be used for personal, scientific, and commercial purposes, as described in the [flux-1-dev-non-commercial-license](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev).

# Usage
`FLUX.1 Krea [dev]` can be used as a drop-in replacement in every system that supports the original `FLUX.1 [dev]`.
A reference implementation of `FLUX.1 [dev]` is in our dedicated [github repository](https://github.com/black-forest-labs/flux).
Developers and creatives looking to build on top of `FLUX.1 [dev]` are encouraged to use this as a starting point.

`FLUX.1 Krea [dev]` is also available in both [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [diffusers](https://github.com/huggingface/diffusers).

## ComfyUI
To use `FLUX.1 Krea [dev]` in [Comfy UI](https://github.com/comfyanonymous/ComfyUI) download the `*.safetensors` weights [here](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev/resolve/main/flux1-krea-dev.safetensors)

## 🧨 diffusers
To use `FLUX.1 Krea [dev]` in [diffusers](https://github.com/huggingface/diffusers), first install or upgrade diffusers

```shell
pip install -U diffusers
```

Then you can use `FluxPipeline` to run the model

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU VRAM

prompt = "A frog holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
).images[0]
image.save("flux-krea-dev.png")
```

To learn more check out the [diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux) documentation

---

# Limitations
- This model is not intended or able to provide factual information.
- As a statistical model this checkpoint might amplify existing societal biases.
- The model may fail to generate output that matches the prompts.
- Prompt following is heavily influenced by the prompting-style.

---

# Out-of-Scope Use
The model and its derivatives may not be used

- In any way that violates any applicable national, federal, state, local or international law or regulation.
- For the purpose of exploiting, harming or attempting to exploit or harm minors in any way; including but not limited to the solicitation, creation, acquisition, or dissemination of child exploitative content.
- To generate or disseminate verifiably false information and/or content with the purpose of harming others.
- To generate or disseminate personal identifiable information that can be used to harm an individual.
- To harass, abuse, threaten, stalk, or bully individuals or groups of individuals.
- To create non-consensual nudity or illegal pornographic content.
- For fully automated decision making that adversely impacts an individual's legal rights or otherwise creates or modifies a binding, enforceable obligation.
- Generating or facilitating large-scale disinformation campaigns.
- Please reference our [content filters](https://github.com/black-forest-labs/flux/blob/main/src/flux/content_filters.py) to avoid such generations.

---

# Risks 

Black Forest Labs (BFL) and Krea are committed to the responsible development of generative AI technology. Prior to releasing FLUX.1 Krea [dev], BFL and Krea collaboratively evaluated and mitigated a number of risks in the FLUX.1 Krea [dev]  model and services, including the generation of unlawful content. We implemented a series of pre-release mitigations to help prevent misuse by third parties, with additional post-release mitigations to help address residual risks:
1. **Pre-training mitigation.** BFL filtered pre-training data for multiple categories of “not safe for work” (NSFW) and unlawful content to help prevent a user generating unlawful content in response to text prompts or uploaded images.
2. **Post-training mitigation.** BFL has partnered with the Internet Watch Foundation, an independent nonprofit organization dedicated to preventing online abuse, to filter known child sexual abuse material (CSAM) from post-training data. Subsequently, BFL and Krea undertook multiple rounds of targeted fine-tuning to provide additional mitigation against potential abuse. By inhibiting certain behaviors and concepts in the trained model, these techniques can help to prevent a user generating synthetic CSAM or nonconsensual intimate imagery (NCII) from a text prompt.
3. **Pre-release evaluation.** Throughout this process, BFL conducted internal and external third-party evaluations of model checkpoints to identify further opportunities for improvement. The third-party evaluations focused on eliciting CSAM and NCII through adversarial testing of the text-to-image model with text-only prompts. We also conducted internal evaluations of the proposed release checkpoints, comparing the model with other leading openly-available generative image models from other companies. The final FLUX.1 Krea [dev] open-weight model checkpoint demonstrated very high resilience against violative inputs, demonstrating higher resilience than other similar open-weight models across these risk categories.  Based on these findings, we approved the release of the FLUX.1 Krea [dev] model as openly-available weights under a non-commercial license to support third-party research and development.
4. **Inference filters.** The BFL Github repository for the open FLUX.1 Krea [dev] model includes filters for illegal or infringing content. Filters or manual review must be used with the model under the terms of the FLUX.1 [dev] Non-Commercial License. We may approach known deployers of the FLUX.1 Krea [dev] model at random to verify that filters or manual review processes are in place.
5. **Policies.** Our FLUX.1 [dev] Non-Commercial License prohibits the generation of unlawful content or the use of generated content for unlawful, defamatory, or abusive purposes. Developers and users must consent to these conditions to access the FLUX.1 Krea [dev] model.
6. **Monitoring.** BFL is monitoring for patterns of violative use after release, and may ban developers who we detect intentionally and repeatedly violate our policies. Additionally, BFL provides a dedicated email address (safety@blackforestlabs.ai) to solicit feedback from the community. BFL maintains a reporting relationship with organizations such as the Internet Watch Foundation and the National Center for Missing and Exploited Children, and BFL welcomes ongoing engagement with authorities, developers, and researchers to share intelligence about emerging risks and develop effective mitigations.

---

# License
This model falls under the [`FLUX.1 [dev]` Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
```


---ASSISTANT
Thank you. I will refer to this model card in my future responses.
---ASSISTANT
Thank you. I will refer to this model card in my future responses.
---USER
Here is additional research information about the model:

```


<model_research>
   <research source="https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md">
Here are the key insights from the FLUX.1 [dev] license that would be particularly interesting or surprising to AI model researchers:

## Surprising Commercial Output Rights Despite Non-Commercial Model License

The license contains an unusual asymmetry: while the **model itself** can only be used non-commercially, **outputs generated by the model can be used commercially** without restriction. Section 2(d) explicitly states: "You may use Output for any purpose (including for commercial purposes), except as expressly prohibited herein." This creates an interesting precedent where the generation tool is restricted but not its creative products.

## Mandatory Content Filtering or Human Review Requirement

The license mandates that users either implement content filters OR conduct human review of all outputs before distribution (Section 2(e)). This is legally binding, not just a recommendation, and Black Forest Labs explicitly reserves the right to "approach known deployers of the FLUX.1 [dev] model at random to verify that filters or manual review processes are in place."

## Competitive Model Training Prohibition

Section 2(d) includes a specific restriction that outputs cannot be used "to train, fine-tune or distill a model that is competitive with the FLUX.1 [dev] Model or the FLUX.1 Kontext [dev] Model." This creates an interesting legal framework around competitive moats in the open-source AI space.

## Patent Litigation Termination Clause

Section 8(c) includes an automatic license termination if users initiate IP litigation against Black Forest Labs - a defensive patent clause that protects the company from legal challenges while providing the model under permissive terms.

## Comprehensive Export Control Compliance

The license includes unusually detailed export control provisions (Section 4(e-f)) covering sanctioned jurisdictions, restricted parties lists, and weapons applications - suggesting awareness of potential dual-use applications of text-to-image models.

These provisions suggest Black Forest Labs is pioneering a new licensing model that balances open research access with commercial protection and regulatory compliance.
   </research>
   <research source="https://bfl.ai/announcements/flux-1-krea-dev">
**Summary for AI Model Researchers:**

The most significant findings from this blog post that would interest AI researchers are:

1. **The "Opinionated" Model Philosophy**: FLUX.1 Krea [dev] was specifically trained to counter the "oversaturated AI look" problem - a deliberate aesthetic bias designed to produce more realistic, diverse images that avoid the typical artificial appearance of AI-generated content. This represents an interesting departure from neutrality-focused training approaches.

2. **Competitive Performance Claims**: The model allegedly matches FLUX1.1 [pro] (a closed-source model) in human preference evaluations while being open-weights, which would be a significant achievement if verified.

3. **Collaborative Foundation Model Development**: The project demonstrates a new paradigm where foundation model labs (BFL) collaborate directly with application-focused companies (Krea) to create specialized base models optimized for downstream fine-tuning, rather than purely general-purpose models.

4. **Strategic Market Positioning**: The model is positioned as both a standalone text-to-image generator and a flexible base for customization, suggesting a dual-purpose design that could influence how future foundation models are architected.

The emphasis on overcoming the "AI look" through intentional aesthetic bias is particularly notable, as it suggests a move toward models that embed specific stylistic preferences rather than attempting aesthetic neutrality.
   </research>
   <research source="https://www.krea.ai/blog/flux-krea-open-source-release">
## Summary for AI Model Researchers

Several surprising and valuable insights emerge from this blog post that extend beyond the model card:

**1. Access to "flux-dev-raw" - A Rare Research Asset:**
The most significant revelation is that Krea had access to **flux-dev-raw**, a pre-trained, guidance-distilled base model from Black Forest Labs that hasn't been post-trained. This is extremely valuable for research as most open-source models are already "baked" with extensive post-training. Having access to such a "raw" foundation model allows for controlled experimentation with post-training techniques without inherited biases.

**2. Quantified Data Requirements:**
The paper provides a concrete data point: **less than 1M samples** are sufficient for effective post-training when quality is prioritized over quantity. This challenges the assumption that massive datasets are always necessary for fine-tuning.

**3. Novel "TPO" Technique:**
They mention using a variant of preference optimization called **TPO** (not elaborated, but appears to be their proprietary technique), applied in multiple rounds for calibration.

**4. Custom CFG Loss for Guidance-Distilled Models:**
They developed a **custom loss function to fine-tune directly on classifier-free guided distributions** for guidance-distilled models - a technical detail that could be valuable for researchers working with similar architectures.

**5. Critique of Existing Aesthetic Evaluation:**
The systematic critique of LAION-Aesthetics revealing biases toward "women, blurry backgrounds, overly soft textures, and bright images" provides concrete evidence of evaluation model limitations that many researchers may unknowingly inherit.

**6. "Global Preference" Problem:**
Their insight that merging diverse aesthetic preferences creates a "marginal preference distribution" that satisfies no one is a fundamental challenge to current RLHF approaches in creative domains. This suggests the field may need to move toward more personalized or specialized models rather than universal ones.

**7. Strategic Use of "Bad" Data:**
Their argument for intentionally including "bad" data during pre-training (as long as properly conditioned) to enable effective negative prompting is counterintuitive but theoretically sound.

This represents a significant case study in aesthetic-focused model development with practical insights about data curation, evaluation metrics, and the fundamental trade-offs between generalization and specialization in creative AI systems.
   </research>
   <research source="https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev">
Based on my analysis of the FLUX.1 [dev] Non-Commercial License, here are the key points that would be particularly interesting or surprising to an AI model researcher:

## Particularly Notable License Provisions:

**1. Output Commercial Rights vs Model Restrictions**
The license creates an unusual split: while you cannot use the model itself commercially, you retain full commercial rights to any outputs generated. However, there's a critical restriction - you cannot use these outputs to train competing models against FLUX.1 [dev] or FLUX.1 Kontext [dev].

**2. Mandatory Content Filtering Requirements**
Section 2(e) mandates that users must either implement content filters OR conduct manual review of all outputs before distribution. This is enforced through potential random audits by Black Forest Labs - they reserve the right to approach known deployers to verify compliance.

**3. Automatic License Termination for Patent Challenges**
Section 8(c) includes a "patent peace" clause - if you sue Black Forest Labs for IP infringement related to FLUX models, your license automatically terminates. This is a defensive mechanism becoming more common in AI model licensing.

**4. Broad Export Control Compliance**
The license extensively addresses export controls and trade sanctions, including prohibitions on IP proxying to disguise location - reflecting the strategic importance of AI models in current geopolitical contexts.

**5. High-Risk Use Liability Shift**
Section 6 explicitly defines "High-Risk Use" (applications where failure could cause serious injury, discrimination, or privacy violations) and completely shifts liability to the user while requiring them to implement appropriate safeguards.

**6. Indemnification Scope**
Users must indemnify Black Forest Labs not just for their direct use, but also for failures in their content filtering implementations - creating potential liability for inadequate safety measures.

These provisions suggest Black Forest Labs is taking an unusually proactive approach to both commercialization control and risk mitigation compared to typical open-source AI model licenses.
   </research>
   <research source="https://github.com/black-forest-labs/flux">
# Summary for AI Model Researchers

## Key Technical Insights Beyond the Model Card

**Architecture Diversity**: FLUX.1 represents a comprehensive **model ecosystem** rather than a single model, with 10+ specialized variants spanning:
- **Structural conditioning** models (Canny, Depth) with both full and LoRA variants
- **Task-specific architectures**: Fill (inpainting), Redux (image variation), Kontext (image editing)
- **Performance tiers**: Schnell (fast), Dev (quality-focused), and implied Pro (commercial API-only)

**Novel Flow Matching Implementation**: The repository contains a **custom flow matching implementation** for latent space generation, suggesting proprietary advances beyond standard diffusion approaches. The recent Kontext model specifically uses "Flow Matching for In-Context Image Generation and Editing in Latent Space" (arXiv:2506.15742).

**Commercial Usage Tracking Infrastructure**: Most notably, BFL has implemented **automated usage tracking via API calls** embedded directly in the inference code - a unique approach where open-weight models phone home to enable commercial licensing. This represents a novel business model bridging open-source and commercial deployment.

**TensorRT Integration**: The repository provides official **TensorRT optimization support** via NVIDIA's PyTorch containers, indicating serious production deployment considerations and performance optimization beyond typical research releases.

**Modular Licensing Strategy**: Each model variant has **distinct licensing** (Apache 2.0 for Schnell vs Non-Commercial for Dev variants), suggesting a sophisticated approach to balancing research access with commercial viability.

The technical infrastructure and business model innovations here may be as significant as the models themselves for understanding the future of open-weight AI deployment.
   </research>
   <research source="https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux">
Based on the Diffusers documentation for Flux, here's a comprehensive summary of the most interesting and surprising technical details for an AI model researcher:

## Technical Architecture Surprises

**Canny/Depth Control as Non-ControlNet Architecture**: Surprisingly, Flux's Canny and Depth control models are *not* traditional ControlNet architectures. Instead, they use **channel-wise concatenation** with input control conditions, making the transformer learn structure control directly rather than using separate residual connections. This is architecturally distinct from the standard ControlNet approach.

**True CFG Support**: Unlike the base Flux models that use embedded guidance, Flux supports **true classifier-free guidance** when `true_cfg_scale > 1` and negative prompts are provided. This allows for more precise control over generation quality vs prompt adherence.

## Memory and Optimization Innovations

**Group Offloading Architecture**: The documentation reveals sophisticated **leaf-level offloading** capabilities that can offload the lowest leaf-level parameters to CPU rather than module-level offloading. This can be mixed across different pipeline components for fine-grained memory management.

**FP8 Quantization Challenges**: Interestingly, FP8 inference is described as **"brittle"** depending on GPU type, CUDA version, and PyTorch version. The recommended approach is using `optimum-quanto` library with specific quantization patterns for the transformer and T5 text encoder.

**FP16 Inference Quirks**: Running Flux in FP16 produces **different outputs** compared to FP32/BF16 due to activation clipping in text encoders. The solution requires forcing text encoders to run in FP32 while keeping other components in FP16.

## Advanced Pipeline Combinations

**Turbo LoRA Stacking**: The documentation shows how to combine Flux Control LoRAs with Turbo LoRAs (like ByteDance/Hyper-SD) using **adapter weight mixing** (`set_adapters(["depth", "hyper-sd"], adapter_weights=[0.85, 0.125])`). This enables few-step inference with control guidance.

**Redux Two-Stage Architecture**: Flux Redux uses a unique two-pipeline approach where `FluxPriorReduxPipeline` generates embeddings that are then fed into the base `FluxPipeline` with disabled text encoders for memory efficiency.

## Specialized Features

**Fill Pipeline Without Strength Parameter**: Unlike traditional inpainting pipelines, FluxFillPipeline doesn't require a `strength` parameter, suggesting a different mathematical formulation for inpainting/outpainting tasks.

**Kontext In-Context Learning**: Flux Kontext enables in-context control for editing, refinement, relighting, and style transfer - essentially allowing the model to understand editing intent from examples rather than explicit control inputs.

**Reset Parameter Warning**: There's a specific warning about using `unload_lora_weights(reset_to_overwritten_params=True)` when working with Control LoRAs to completely reset the transformer, which is crucial for pipeline reuse.

## Performance Insights

**Guidance Scale Extremes**: The documentation shows extremely high guidance scales (30.0) for control tasks, much higher than typical diffusion models, suggesting Flux requires stronger guidance for precise control.

**Single File Loading**: Support for loading community-finetuned checkpoints in the original Black Forest Labs format, including FP8 quantized versions, indicates strong community ecosystem support.

The most surprising aspect is how Flux diverges from standard diffusion architectures in multiple ways - from control mechanisms to guidance systems - while maintaining compatibility with advanced optimization techniques.
   </research>
   <research source="https://github.com/black-forest-labs/flux/blob/main/src/flux/content_filters.py">
Based on the content filters code, here are the key findings that would be interesting to an AI model researcher:

## Technical Implementation Surprises

**1. Dual-Stage Filtering Architecture**: The content filter uses a sophisticated two-stage approach - first running a specialized NSFW classifier (`Falconsai/nsfw_image_detection`) with a high threshold (0.85), then falling back to a general-purpose vision-language model (Pixtral-12b) for copyright/public figure detection. This hybrid approach is notable as most implementations rely on single-stage filtering.

**2. Constrained Generation via Logit Manipulation**: The filter uses an elegant `yes_no_logit_processor` that forces the model to only output "yes" or "no" tokens by setting all other vocabulary tokens to minimum values. This is a clean implementation of constrained generation that ensures reliable binary classification without complex parsing.

**3. Image Downsampling Strategy**: Images are automatically downsampled to 512² pixels for the vision-language model check, which is a practical optimization balancing computational efficiency with detection accuracy.

## Content Policy Insights

**4. Strict Copyright/Public Figure Detection**: The prompts reveal an extremely strict interpretation of copyright concerns - requiring the ability to "name the exact copyrighted work or specific person depicted." This suggests they're targeting very specific, identifiable content rather than style mimicry or general resemblance.

**5. Living vs. Historical Figure Distinction**: Interestingly, the text filter specifically targets "living public figures" while the image filter includes "historical figures," suggesting different risk assessments between modalities.

**6. Brand/Trademark Focus**: The inclusion of trademarked logos and brands in the filtering criteria indicates they're taking a broad approach to intellectual property protection beyond just character depictions.

This implementation reveals a production-ready, multi-layered safety system that goes well beyond simple keyword filtering, representing a sophisticated approach to AI safety in generative models.
   </research>

</model_research>


```

Your task is to produce a concise 300-500 word summary of the specified model highlighting 
key points of interest for AI researchers. Make sure that major information and traits are
specified, and prefer bullet points for notable points of interest. Use markdown links to 
refer to sources if warranted.

---ASSISTANT
# FLUX.1 Krea [dev]: A 12B Parameter Text-to-Image Model Summary

FLUX.1 Krea [dev] is a 12 billion parameter rectified flow transformer developed through collaboration between Black Forest Labs and Krea, specifically designed to combat the "oversaturated AI look" problem in generated images.

## Key Technical Features

• **Flow Matching Architecture**: Uses custom flow matching implementation in latent space rather than standard diffusion, with guidance distillation for improved efficiency
• **Aesthetic-Focused Training**: Deliberately trained with aesthetic bias to produce more realistic, diverse outputs that avoid typical artificial AI appearance
• **Drop-in Compatibility**: Can replace original FLUX.1 [dev] in existing systems and workflows
• **Multi-Platform Support**: Available in both ComfyUI and Diffusers with optimized inference pipelines

## Notable Research Insights

• **Raw Foundation Access**: Trained from "flux-dev-raw" - a rare pre-trained, guidance-distilled base model without post-training, enabling controlled experimentation
• **Efficient Data Requirements**: Achieved competitive performance with **less than 1M high-quality samples**, challenging assumptions about massive dataset necessity
• **Custom TPO Technique**: Uses proprietary preference optimization variant applied in multiple calibration rounds
• **Novel CFG Loss**: Developed custom loss function for fine-tuning guidance-distilled models directly on classifier-free guided distributions

## Unique Licensing Model

The model introduces an innovative licensing approach:
• **Commercial Output Rights**: Generated content can be used commercially despite non-commercial model license
• **Mandatory Content Filtering**: Legally requires either automated filters or human review of all outputs
• **Competitive Training Prohibition**: Outputs cannot be used to train competing models
• **Patent Peace Clause**: License terminates automatically if users initiate IP litigation

## Safety and Content Control

Implements sophisticated multi-stage content filtering:
• **Dual-stage architecture** with NSFW classifier (0.85 threshold) plus vision-language model fallback
• **Constrained generation** via logit manipulation for reliable binary classification  
• **Strict copyright detection** targeting identifiable works and public figures

## Research Significance

FLUX.1 Krea [dev] represents a paradigm shift toward "opinionated" models that embed specific aesthetic preferences rather than pursuing neutrality. The collaboration model between foundation labs and application companies, combined with the novel licensing approach balancing open research with commercial protection, may influence future AI model development and deployment strategies.

The model's focus on overcoming the "global preference problem" - where merged aesthetic preferences satisfy no one - suggests the field may need specialized models rather than universal ones for creative applications.