---
marp: true
theme: freud
paginate: true
---


<style>
     .cite-author {  
      text-align        : right;
   }
   .cite-author:after {
      color             : orangered;
      font-size         : 125%;
      /* font-style        : italic; */
      font-weight       : bold;
      font-family       : Cambria, Cochin, Georgia, Times, 'Times New Roman', serif; 
      padding-right     : 130px;
   }
   .cite-author[data-text]:after {
      content           : " - "attr(data-text) " - ";      
   }

   .cite-author p {
      padding-bottom : 40px
   }
</style>

<!-- _class: titlepage -->

<div class="title"         > Securing the Model Context Protocol</div>
<div class="subtitle"      > securese.ai, Stockholm   </div>
<div class="author"        > Shaun Smith                       </div>
<div class="date"          > Sep 2025                                    </div>
<div class="organization"  > huggingface.co/evalstate</div>
<div class="organization"  > huggingface.co/evalstate</div>
<div class="organization"  > github.com/evalstate</div>
<div class="organization"  > x.com/evalstate</div>

<!-- -->


---

## Part 0 - Introduction

<!-- who knows about mcp, who i am, what the presentation entails -->

MCP Steering Group Member
Community Moderator, Working Groups.

Work @ Hugging Face on MCP and Open Source initiatives.

<!-- if you are using MCP you are an LLM Systems integrator -->

---

<!-- _class: transition -->

# <span class="mcp-model">Model Context</span>  <span class="mcp-context">Protocol</span>

---

# The Model

![w:700](./images/model_parameters.png)

---

<!-- _class: dataset-makeup -->

<style scoped>
  section.dataset-makeup table td:first-child {
    font-weight: 700;
    white-space: nowrap;
  }
  section.dataset-makeup table tr:hover {
    background-color: var(--table-hover-background-color) !important;
    color: var(--table-hover-color) !important;
    font-weight: 700;
  }
</style>

# Training Data Makeup __Meta Llama 2023__


| Source | Content | Weighting | Size (GB) |
| --- | --- | --- | ---: |
| **🌐 English CommonCrawl** | English language web content | Very High (73.7%) | 3,379 |
| **🌐 C4** | Cleaned web pages | High (15.9%) | 783 |
| **💻 GitHub** | Open-source code | Medium (2.9%) | 328 |
| **📚 Wikipedia** | Encyclopedia articles in 20 languages | Medium (11.0%) | 83 |
| **📘 Books** | Project Gutenberg and Books3 collection | Medium (10.0%) | 85 |
| **🧪 ArXiv** | Scientific papers | Low (2.7%) | 92 |
| **💬 Stack Exchange** | Q&amp;A from various domains | Low (2.1%) | 78 |

<p class="small">Sizes normalized to gigabytes for straightforward comparisons.</p>

---

<!-- _class: mcp-features -->

<style scoped>
  section.mcp-features table {
    width: 100%;
    margin-top: 1.1rem;
    table-layout: fixed;
    border-collapse: collapse;
    font-size: 1.02rem;
  }
  section.mcp-features thead th {
    padding: 0 0.75rem 0.55rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.74rem;
    font-weight: 600;
    border-bottom: 1.5px solid rgba(0, 0, 0, 0.3);
    color: rgba(0, 0, 0, 0.68);
  }
  section.mcp-features tbody td {
    padding: 0.65rem 0.75rem;
    border-bottom: 1px solid rgba(0, 0, 0, 0.12);
    vertical-align: top;
  }
  section.mcp-features tbody tr:last-child td {
    border-bottom: none;
  }
  section.mcp-features table td:first-child {
    font-weight: 700;
  }
  section.mcp-features table tr:nth-child(even),
  section.mcp-features table tr:nth-child(odd) {
    background-color: transparent !important;
  }
  section.mcp-features table tr:hover {
    background-color: var(--table-hover-background-color) !important;
    color: var(--table-hover-color) !important;
    font-weight: 700;
  }
  section.mcp-features .cell-title {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
  }
  section.mcp-features .cell-title::before {
    content: "";
    display: inline-block;
    width: 28px;
    height: 28px;
    flex-shrink: 0;
    background-repeat: no-repeat;
    background-position: center;
    background-size: 24px 24px;
    opacity: 0.92;
  }
  section.mcp-features .cell-title.is-tools::before {
    background-image: url("./images/lucide-wrench.svg");
  }
  section.mcp-features .cell-title.is-resources::before {
    background-image: url("./images/lucide-database-zap.svg");
  }
  section.mcp-features .cell-title.is-prompts::before {
    background-image: url("./images/lucide-sparkles.svg");
  }
  section.mcp-features .examples {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
</style>

# MCP Server Feature Roles

| Feature | Explanation | Examples | Control  |
| --- | --- | --- | ---: |
| <span class="cell-title is-tools">Tools</span> | Functions the model may call to act on the world: write to databases, invoke APIs, modify files, or trigger workflows. | <div class="examples"><span>Search flights</span><span>Send messages</span><span>Create calendar events</span></div> | Model |
| <span class="cell-title is-resources">Resources</span> | Read-only context surfaces like file contents, schemas, and docs that enrich prompts without side effects. | <div class="examples"><span>Retrieve documents</span><span>Access knowledge bases</span><span>Read calendars</span></div> | Application |
| <span class="cell-title is-prompts">Prompts</span> | Instruction templates that steer the model to combine tools and resources for specific workflows. | <div class="examples"><span>Plan a vacation</span><span>Summarize my meetings</span><span>Draft an email</span></div> | User |


---

# privacy

![w:600](./images/chatgpt-privacy.png)

---

# completions[0..1]

<div class="columns">
<div>

![w:700](./images/completion-1-lightbox.png)

</div>
<div>

![w:700](./images/completion-2-lightbox.png)

</div>

</div>

---

# foobar

<div align="center">

## poifsdfsfsf nt one
## point two

| foo | bar | baz |
|-----|-----|-----|
| test | test | test |



</div>

---


<!-- points to make here -->
Models are trained using lots of text. 
Models were then trained to be conversational
Models were then trained to follow instructions
Models generate text using probabilities. [SHOW DEMO]


<!-- 

This isn't a long "history lesson" style talk; but i wanted to reground us 

Conversational Training. Hand Noted. RLHF. 
Instruction Training.

How do we make a model?
Ingredients. Lots of CPU, lots of compute.

Text Completions 

 given . The text we ask it to complete is known as the "Context".
Computational Complexity and Model Size.

-->

---

<!-- points to make here -->
The context is _tiny_ compared to the model
The context is precious
Instruction following has a precedence problem
Generations are intentionally different each time (completions[0])

---

## Transition

<!-- points to make here -->
There's quite a lot of hyperbole around MCP Security; 

<!-- _class: columns -->

## Model Footprint vs Context Window

<div class="columns">
  <div>
    <h3>Weights</h3>
    <p><strong>gpt-oss-120b</strong></p>
    <p class="small">~60 GB to keep resident (8-bit sharded weights)</p>
    <p>Multiple GPUs just to host the parameters.</p>
  </div>
  <div>
    <h3>Active Context</h3>
    <p><strong>131,072 tokens</strong></p>
    <p class="small">~95k words (~400 printed pages)</p>
    <p>Only a few MB of live conversation at a time.</p>
  </div>
</div>

<p class="small">We pour gigabytes into the model weights, yet the working memory per turn stays comparatively tiny.</p>


---

## Model Memory Budget

<div class="footprint-bars">
  <div class="row">
    <div class="label">
      <strong>Model weights</strong>
      <div class="meta">gpt-oss-120b • ~60 GB resident across shards</div>
    </div>
    <div class="bar" data-note="full scale (1.0)">
      <div class="fill" style="--scale: 1;">
        <span>~60 GB</span>
      </div>
    </div>
  </div>
  <div class="row">
    <div class="label">
      <strong>Live context</strong>
      <div class="meta">131,072 tokens • ~0.5 MB transient buffer</div>
    </div>
    <div class="bar" data-note="visual x50">
      <div class="fill is-tiny" style="--scale: 0.02;">
        <span>~0.5 MB</span>
      </div>
    </div>
  </div>
</div>

<div class="footprint-note">Actual memory budget ratio is roughly 120000:1 in favor of the weights.</div>


---

## Frozen weights, fleeting context

<div class="footprint-nesting">
  <div class="box">
    <strong>Parameters stay resident</strong>
    <p class="small">~60 GB loaded for inference at all times.</p>
    <div class="chip">weights</div>
    <p class="small">Sharded across multiple accelerators to keep latency down.</p>
    <div class="inner">~0.5 MB live context</div>
  </div>
  <div class="box emphasis">
    <strong>Context stream resets</strong>
    <p class="small">131,072 tokens per turn (~95k words).</p>
    <div class="chip">conversation</div>
    <ul>
      <li>Rebuilt for every request.</li>
      <li>Tool outputs and prompts fight for headroom.</li>
      <li>Older turns vanish first when the window fills.</li>
    </ul>
  </div>
</div>

<p class="small">Inference runs on a huge static memory map, while the conversational working set remains a tiny, constantly refreshed slice.</p>


---

## Scale at a glance

<div class="footprint-grid">
  <div class="panel weights" data-caption="resident on accelerators">
    <small>Model weights</small>
    <strong>~60 GB</strong>
    <p>Each inference node keeps the complete parameter shard map warm, trading cost for latency and throughput.</p>
  </div>
  <div class="panel context" data-caption="rebuilt per turn">
    <small>Context window</small>
    <strong>131k tokens</strong>
    <p>Roughly 0.5 MB of conversational state—orders of magnitude smaller than the model it steers.</p>
  </div>
</div>

<p class="small">Area ratio is illustrative: even when we exaggerate the context box, it barely registers next to the weight footprint.</p>


---

## Context windows keep stretching

<div class="window-grid">
  <div class="window-card" data-year="2020">
    <strong>GPT-3 (davinci)</strong>
    <div class="tokens">2,048 tokens <span>~1.5k words • ~6 pages</span></div>
    <p>The original production window: enough for a prompt, a short brief, and a small completion.</p>
    <div class="badge">baseline</div>
  </div>
  <div class="window-card" data-year="2023">
    <strong>GPT-3.5 Turbo 16k</strong>
    <div class="tokens">16,385 tokens <span>~12k words • ~49 pages</span></div>
    <p>Extended chat history and lightweight tool transcripts became practical at this scale.</p>
    <div class="badge">multi-turn</div>
  </div>
  <div class="window-card" data-year="2023">
    <strong>GPT-4 Turbo / GPT-4o</strong>
    <div class="tokens">128,000 tokens <span>~96k words • ~384 pages</span></div>
    <p>Full product manuals and large retrieval blocks can live in a single request.</p>
    <div class="badge">128k</div>
  </div>
  <div class="window-card" data-year="2024">
    <strong>Claude 3 Opus (Projects)</strong>
    <div class="tokens">256,000 tokens <span>~192k words • ~768 pages</span></div>
    <p>Anthropic’s Workspace mode lets the assistant reason across multi-volume dossiers.</p>
    <div class="badge">256k</div>
  </div>
  <div class="window-card" data-year="2024">
    <strong>Gemini 1.5 Pro</strong>
    <div class="tokens">1,000,000 tokens <span>~750k words • ~3,000 pages</span></div>
    <p>Streaming multimodal context; entire product repositories or video transcripts at once.</p>
    <div class="badge">1m window</div>
  </div>
  <div class="window-card" data-year="2024">
    <strong>Claude 3.5 Sonnet (Projects)</strong>
    <div class="tokens">1,000,000 tokens <span>~750k words • ~3,000 pages</span></div>
    <p>Beta “Projects” mode mirrors Gemini-scale context, aimed at long-running workflows.</p>
    <div class="badge">1m window</div>
  </div>
</div>


---

## What fits inside?

<div class="analog-table">
  <table>
    <thead>
      <tr>
        <th>Window / Model</th>
        <th>≈ Words</th>
        <th>≈ Pages*</th>
        <th>Comparable text</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>GPT-3 · 2,048 tokens</td>
        <td>~1,500</td>
        <td>~6</td>
        <td>Executive brief or short story draft</td>
      </tr>
      <tr>
        <td>GPT-3.5 Turbo · 16,385 tokens</td>
        <td>~12,300</td>
        <td>~49</td>
        <td>Conference paper plus reviewer discussion</td>
      </tr>
      <tr>
        <td>GPT-4o · 128,000 tokens</td>
        <td>~96,000</td>
        <td>~384</td>
        <td>Full-length novel or product spec binder</td>
      </tr>
      <tr>
        <td>Claude 3 Opus · 256,000 tokens</td>
        <td>~192,000</td>
        <td>~768</td>
        <td>Two packed technical manuals back-to-back</td>
      </tr>
      <tr>
        <td>Gemini / Claude Projects · 1,000,000 tokens</td>
        <td>~750,000</td>
        <td>~3,000</td>
        <td>The entire Bible plus appendices &amp; footnotes</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="analog-notes">*Pages estimated at 250 words each; words ≈ tokens × 0.75.</div>


---

## Reference texts by tokens

<div class="window-grid">
  <div class="window-card" data-year="">
    <strong>Average contemporary novel</strong>
    <div class="tokens">~120,000 tokens <span>~90k words • ~360 pages</span></div>
    <p>Fits comfortably inside a 128k window—GPT-4o can absorb an entire book-length draft.</p>
    <div class="badge">book-length</div>
  </div>
  <div class="window-card" data-year="1851">
    <strong>Moby-Dick (Melville)</strong>
    <div class="tokens">~280,000 tokens <span>~210k words • ~840 pages</span></div>
    <p>Requires a 256k-class context; too large for 128k without chunking.</p>
    <div class="badge">epic</div>
  </div>
  <div class="window-card" data-year="1st c.">
    <strong>The Bible (complete)</strong>
    <div class="tokens">~1,040,000 tokens <span>~783k words • ~3,100 pages</span></div>
    <p>Only the emerging 1M-token windows can take the entire canon in one pass.</p>
    <div class="badge">1m scale</div>
  </div>
</div>


Set the context of Large Language Models, training data, context windows.
Age of Prompt Engineering.
What goes in, model ownership is important.

What if we add training that's conversational? We'll have lots of human/assistant pairs and show it how to act conversationally?

- Appears knowledgeable
- Semantic understanding of text
- Better.

Donald Trump clause in the System Prompt.

- How do we make it better. We have bigger and bigger models; 

- Training data
- OpenAI training data opt-in (future models may know).
- The era of Prompt Engineering
  - Inference. 

HuggingFace Smol 3b training set.

If you don't own the model, you don't own the weights.
What facts the model has been trained on. What biases the model has are out of your control.

- Copy and Paste context management. Custom RAG systems.

Where that data comes from. Anthropic $1.5bn books, shredding.
OpenAI Privacy dialgoue.
The model may end up knowing a suspicous 

Security Risks?
 Cognitive Risks.
 Bias Risks.
 Training Data.

Need to trust the model, trust the provider.


## Part 2 - The Context

Trained on chat
Trained on instruction



OpenAI Codex



## Part 2 - Tools and Agents

It's all the same trick!
Live demonstration : Teach our model to do Tool Calling.

The age of Agents.
What is an Agent

- Harness that self-adjusts its context. 
- 
Prompt Injection.


---

IGNORE ALL PREVIOUS INSTRUCTONS

similar problem to "solving hallucinations".



---


## Part 3 - MCP

### Introduction

Can't deflect responsibility in to the Protocol
Can't transfer the risk

### Distribution

As Community Moderator get to see a lot of MCP Servers. One-shot prompted in to existence. 



Introducing the Model Context Protocol.


We see automation not augmentation.

So now that we know what a bit more about Models, and a bit more about Context let's see where MCP fits.
Show MCP-Webcam.

Less than 12 months old. Distribution Statistics. Weekend in Apr

- what mcp is -- do a deep dive explanation on the components and the parts.

- json-rpc; transports, hosts, client, servers.
- show all of the different datasources that can work.
- transport, data, layer??  (d)
- two specifications
OAUTH2.1
- Package and distribution of MCP-B/DXT. GitHub, Webiste.
- Registry

Bi-Directional Communication

co-minglign


## 

---

## 

Assume that the data in your context window is privileged. 

The reason for the preamble is so that we can have a balanced discussion about MCP Security

---


# Model Context Protocol is an open-source standard for connecting AI applications to external systems.


> Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect electronic devices, MCP provides a standardized way to connect AI applications to external systems.

---

# Architecture

<!-- launched in november last year, and immediately proved popular -->
<!-- why? for the first time rather than handling complex RAG or custom tool calling you had ready-made applications to integrate with -->
- Parts (MCP Servers, Host Application, Model)

- MCP Servers: Primitives
- MCP Servers: Connectivity
- MCP Servers: Priniciple of simple development
- Distribution Problem
- Remote Servers had no Authentication.

<!-- we'll do a high level walkthrough, then look at some of these in more detail -->
<!-- we talk a lot about MCP Servers, and that's not quite the right name -->
- Host
- Client
- Server
- LLM (Model/Context)
- User!

---

# Primitives (Server)

Primitives, Transports

When using an MCP Server

- Prompts
- Resources
- Tools
- _Instructions_

<writing style suite>
<use these things> 


<!-- maybe i'm always tired of typing the same thing -->
<!-- maybe there's a website link the host application should follow -->

---


# Primitives (Host/Client)

- Sampling
- Elicitation
- Logging

---




# Transports (and Distribution)

STDIO
SSE (Deprecated)
Streamable HTTP
The rise of Hosting Services and Proxies.

---
# Instruction and Tool Challenges

- MCP Servers can add Instructions - most hosts inject these in to the System Prompt
- Tool Descriptions may not match 
- Tool can look safe (but run later)
- Entire Tool list can change 
- Tool Results may be unwelcome

---

# Early days of MCP. Server List. 

Review the Server, make sure there are no obvious.

# What happens


MCP Server Instructions injected in to Context.
Auto-injection in to the Context.

Context co-mingling.
Data sent to the LLM 
Tools that know about each other

Tool Descriptions narrow. Do they mutate? ToolListChangedNotifications.


Data is accessible at the privilege level of the User.
https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/

---

# Distribution

- StreamableHTTP gives deployment options, and the latest OAuth is intended to make integration easier.
- This makes deployment - and auditing easier. far better to have telemetry from your MCP Infrastructure than having people copy-paste from unknown sources.


---

# MCP "Server"

MCP Server is a bad name. We are used to Servers . MCP Servers can actively contact the Host Application, or User.
With Sampling and Elicitations

---

# Problems

We've integrated data and actions from different systems via a Host Application.

Co-mingling of data.
Co-mingling of instructions.


---

# Registry
# MCP-B
# Registries
# Demo Claude Registry

### Distribution

 - Demo Claude registry
 - Community Registry

### 

Architecture. 

Host, Client and Server

Primitives, Transport. 


Vulnerabilities.
- Session 
- Non-text content.

---
