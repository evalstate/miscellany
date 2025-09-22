---
marp: true
html: true
sanitize: false
theme: freud
paginate: false
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
<div class="subtitle"      > secureai.se, Stockholm   </div>
<div class="author"        > Shaun Smith                       </div>
<div class="date"          > September 2025                                    </div>
<table class="social-table">
  <tbody>
    <tr>
      <td><img src="./images/huggingface-mark-logo.svg" alt="Hugging Face" /></td>
      <td><a class="organization" href="https://huggingface.co/evalstate">huggingface.co/evalstate</a></td>
    </tr>
    <tr>
      <td><img src="./images/github-mark.svg" alt="GitHub" /></td>
      <td><a class="organization" href="https://github.com/evalstate">github.com/evalstate</a></td>
    </tr>
    <tr>
      <td><img src="./images/xcom-logo-black.png" alt="X" /></td>
      <td><a class="organization" href="https://x.com/evalstate">x.com/evalstate</a></td>
    </tr>
  </tbody>
</table>

<!-- -->

---

# Hello!

<div class="columns">

<div class="center">

![w:350](./images/greeting_huggy_left.png)

### Hugging Face MCP/Open Source Projects

</div>

<div class="center">

![w:350](./images/mcp-icon.svg)


### MCP Steering Committee Member

</div>

</div>

<center> 

### `fast-agent.ai`

</center>
<!-- who knows about mcp, who i am, what the presentation entails -->

<!-- Community Moderator, Working Groups. -->

<!-- Work @ Hugging Face on MCP and Open Source initiatives. -->

<!-- if you are using MCP you are an LLM Systems integrator -->

---

<!-- _class: transition -->

# <span class="mcp-model">Model Context</span>  <span class="mcp-context">Protocol</span>

---

# Text Generation Models

<div class="columns">

<div class="center"> 

![w:600](./images/model_parameters.png)

</div>

<div>

Continues generating text from a prompt

</div> 
</div>

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

# Training Data Composition __Meta Llama 2023__

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

<style scoped>
  section iframe.web-embed {
    width: 100%;
    min-height: 520px;
    border: 0;
    border-radius: 18px;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
  }
</style>

# Token Prediction

<iframe class="web-embed" src="file:///X:/source/mikupad/mikupad.html" allowfullscreen loading="lazy" referrerpolicy="no-referrer"></iframe>

---

<!-- _class: completions-zoom -->

# Completions[0..1]

## Identical Prompt, Identical Model... 

<div class="columns">
<div>

<div class="zoom-target first" tabindex="0">

![w:600](./images/completion-1-lightbox.png)

<span class="zoom-lens" aria-hidden="true"></span>

</div>

</div>
<div>

<div class="zoom-target second" tabindex="0">

![w:600](./images/completion-2-lightbox.png)

<span class="zoom-lens" aria-hidden="true"></span>

</div>

</div>

</div>

---


# Privacy and Content

<div class="center">

![w:600](./images/chatgpt-privacy.png)

</div>

<!-- we trust providers to x,y,z -->

---

# Guardrails (and Prompt Engineering)

(donald trump clause)

https://docs.claude.com/en/release-notes/system-prompts#august-5-2025

## These are all things that LLM Systems Integrators need to consider.

---

Charles Dickens, A Tale of Two Cities : 206,022 Tokens (139,000 Words)

---

# Frozen weights, fleeting context

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

<!-- _class: transition -->

# <span class="mcp-model">Model Context Protocol</span> is an open-source standard for connecting AI applications to external systems.

#### Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect electronic devices, MCP provides a standardized way to connect AI applications to external systems.

---

# Solving a problem

<div class="columns">

<div>

## Interactive

- ### Copy/Paste Context Management
- ### Custom RAG Pipelines and Integrations

</div>

<div>

## Agents

- ### Plug and Play

</div>

</div>

---

# Architecture

![650px](./images/architecture_1.png)

---

<!-- _class: mcp-features -->

# MCP Server Capabilities

| Icon | Feature | Explanation | Example | 
| --- | --- | --- | --- | 
| <span class="feature-icon"><img src="./images/lucide-wrench.svg" alt="Tools" width="100%" /></span> | <span class="cell-title">Tools</span> | Functions the model may call to act on the world: write to databases, invoke APIs, modify files,  workflows. | <div class="examples"><span>Send messages</span></div> | Model |
| <span class="feature-icon"><img src="./images/lucide-database-zap.svg" alt="Resources" width="100%" /></span> | <span class="cell-title">Resources</span> | Read-only data sources like file contents, schemas, and documents that enrich prompts without. | <div class="examples"><span>Attach  documents</span></div> | Application |
| <span class="feature-icon"><img src="./images/lucide-messages-square.svg" alt="Prompts" width="100%"/></span> | <span class="cell-title">Prompts</span> | Instruction templates that steer the model to for specific workflows. | <div class="examples"><span>Draft an email</span></div> | User |


---

<!-- _class: mcp-features -->

# MCP Client Primitives

| Icon | Feature | Explanation | Example | 
| --- | --- | --- | --- | 
| <span class="feature-icon"><img src="./images/folders.svg" alt="Roots" width="100%" /></span> | <span class="cell-title">Roots</span> | Specify which files and directories the Server can access | <div class="examples"><span>Share Local Files</span></div> | Model |
| <span class="feature-icon"><img src="./images/cpu.svg" width="100%" alt="Sampling" /></span> | <span class="cell-title">Sampling</span> | Allow the MCP Server to request an LLM Completion. | <div class="examples"><span>Process unstructured data</span></div> | Application |
| <span class="feature-icon"><img src="./images/message-circle-question-mark.svg" width="100%" alt="Elicitations" /></span> | <span class="cell-title">Elicitations</span> | Request specific information from the User, bypassing the LLM | <div class="examples"><span>Collect specific booking information</span></div> | User |


<!-- the name mcp server is a bit misleading -->

---

# Transports/Distribution __Dev Preview Nov 2024__

Protocol defines a minimum __how__ Client and Server connect and communicate, which official SDKs must support.

<div class="columns">

<div>

## STDIO (Local)

Run Locally (within the Host Process)
Running at User Privilege Level 
_Ad-hoc distribution_

</div>

<div class="sse-deprecated">

## <strike>SSE (Remote)</strike>

<strike>

Remote Hosting (ex. Process)
Limited Host Application Support
_No standard authentication_
</strike>

</div>

</div>

---

# Locally Deployed Servers

<div class="columns">

<div align="center">

![w:560px](./images/local_context.png)

</div>

<div>

- ### Usually started as a sub-process from the Host Application
- ### Access to local resources and files.
- ### Can execute commands on the Users computer
- ### Especially useful for Developer Tools
- ### Authentication through Config Files
- ### Updates, Usage and Telemetry Data can be difficult to capture +/-

</div>

</div>

---

# OAuth 2.1 and Streamable HTTP __2025-06-18__

- ## First Protocol update (__2025-03-26__) introduced a new Streamable HTTP Transport for Remote Servers and OAuth authentication.
- ## OAuth spec was revised to simplify implementation for MCP Server authors: 
  - ## No need to implement Authorization Server (easily use 3rd Party) 
  - ## Straightforward redirect from MCP Server so Client can handle authorization flow.
- ## First-Party Remote Servers will often have Privacy, Access policies in place. 
<!-- works really nicely demo HF MCP Integration -->

---

# Registries and MCP Bundle Format


<div class="columns">

<div>

### [registry.modelcontextprotocol.io](registry.modelcontextprotocol.io) 

Standardises MCP Server description format and provides a basic level of assurance.
<!-- Speak well of the community efforts here -->

### [MCP Bundle Format](https://github.com/anthropics/mcpb) (formerly DXT)

__MCPB__ archive distribution format for local MCP Servers to make __curation__, installation and update easier for End Users. 

</div>

<div>

<br />
<img src="./images/ecosystem-diagram.excalidraw.svg">


</div>

---

# Registries and Curation

<div class="registry-layout">
  <div class="text-column">
    <h3>MCP  Registries</h3>
    <ul>
      <li><p>Trusted Sources (e.g. Claude MCP Partners)</p></li>
      <li><p>Managed MCP Enterprise Registry (e.g. Azure)</p></li>
      <li><p>Tool Integration (e.g. VSCode + GitHub)
   </ul>
  </div>
  <div class="collage-column">
    <div class="registry-collage">
      <img class="shot-azure" src="./images/azure_registry.png" alt="Azure registry screenshot" />
      <img class="shot-claude" src="./images/claude_mcp_partners.png" alt="Claude MCP partners screenshot" />
      <img class="shot-github" src="./images/github_mcp_registry.png" alt="GitHub MCP registry screenshot" />
    </div>
  </div>

</div>

---


# LLM Integration Risks - Lethal Trifecta


<div class="columns"> 

<div>

- ### Access to your private data—one of the most common purposes of tools in the first place!
- ### Exposure to untrusted content—any mechanism by which text (or images) controlled by a malicious attacker could become available to your LLM
- ### The ability to externally communicate in a way that could be used to steal your data

> Source: https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/

</div>

<div>

<br />

![](./images/lethaltrifecta.jpg)

</div>

---

# LLM Integration Risks - Context Management

- ### Function Calling includes Tool and Parameter descriptions to your Context.
- ### LLM not able to distinguish between intended, unintended and malicious instructions.
- ### Unused Tools / Servers degrade LLM Performance and increase inference costs.
- ### Exfiltration may not always appear obvious: _Host Application rendering_ of Images/Markdown/Mermaid links
- ### Tool can look safe on first run (pre-approval) but modify behaviour on second run.
- ### Tool Results may include unvetted data (e.g. Instructions embedded in a GitHub Issue or JIRA Ticket or Word Document).

<!-- similar risks exist for copy/paste context management -->

---

# MCP Specific Guidance

- ### Review Tool, Parameter and Instructions inclusion
- ### MCP Server `instructions` may be added to the Context.
- ### Multimodal Content (e.g. Images) returned via tools expose the same risks
- ### MCP Server Tool List Change Notifications - revalidation 
- ### Tools should not reference other Tools (especially NOT other MCP Servers) 
- ### Prioritise which things need Human-in-the-Loop 
- ### Risk assess specific Server/Tool mixes.

> (Read Ola's Post).

---

<!-- _class: transition -->

# <span class="mcp-model">Community and Contributing</span>

---

# Getting Involved

- ### Open Source Specification and SDKs
- ### Recently updated governance model - in the open via SEP Process
- ### Active community discussions on Discord
- ### https://modelcontextprotocol.io/community
- ### https://github.com/modelcontextprotocol/
- ### Huge ecosystem of Open Source MCP Clients and Servers
- ### Explore Open Source Models, Datasets and Training Courses on MCP, LLMs and Transformers on https://huggingface.co
- ### Also use Open Source Models/Datasets

---

<!-- 

points to make here 
Models are trained using lots of text. 
Models were then trained to be conversational
Models were then trained to follow instructions
Models generate text using probabilities. [SHOW DEMO]

This isn't a long "history lesson" style talk; but i wanted to reground us 

Conversational Training. Hand Noted. RLHF. 
Instruction Training.

How do we make a model?
Ingredients. Lots of CPU, lots of compute.

Text Completions 

 given . The text we ask it to complete is known as the "Context".
Computational Complexity and Model Size.

The context is _tiny_ compared to the model
The context is precious
Instruction following has a precedence problem
Generations are intentionally different each time (completions[0])

-->

---

---

Assume that the data in your context window is privileged. 

The reason for the preamble is so that we can have a balanced discussion about MCP Security

<!-- launched in november last year, and immediately proved popular -->
<!-- why? for the first time rather than handling complex RAG or custom tool calling you had ready-made applications to integrate with 
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





-->
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


<!-- maybe i'm always tired of typing the same thing -->
<!-- maybe there's a website link the host application should follow -->

# Transports (and Distribution)

STDIO
SSE (Deprecated)
Streamable HTTP
The rise of Hosting Services and Proxies.

---

# Early days of MCP. Server List. 

Review the Server, make sure there are no obvious.

# What happens

MCP Server Instructions injected in to Context.
Auto-injection in to the Context.

Context co-mingling.
Data sent to the LLM 
Tools that know about each other


# Distribution

- StreamableHTTP gives deployment options, and the latest OAuth is intended to make integration easier.
- This makes deployment - and auditing easier. far better to have telemetry from your MCP Infrastructure than having people copy-paste from unknown sources.

---
