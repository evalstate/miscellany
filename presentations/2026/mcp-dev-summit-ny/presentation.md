---
marp: true
html: true
size: 16:9
theme: ny-noir
paginate: false
header: '<span class="header-logos"><img src="./images/hf_logo.svg" alt="Hugging Face" /><img src="./images/github-mark.svg" alt="GitHub" />github.com/evalstate</span>'
style: |
  .header-logos {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(5, 10, 18, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #edf5ff;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: none;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.24);
  }

  .header-logos img {
    width: auto;
    height: 22px;
    padding: 0.18rem;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.92);
    box-shadow: none !important;
    border: none !important;
    object-fit: contain;
    filter: none !important;
  }

  .header-logos img[src*="hf_logo"] {
    height: 24px;
    padding: 0.22rem 0.28rem;
  }

  video.full-slide {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 24px;
  }

  .hero-note {
    max-width: 24em;
    margin-top: 1rem;
    color: #dbe6f7;
    font-size: 0.68em;
    line-height: 1.45;
  }

  .mini-list {
    margin-top: 0.8rem;
    font-size: 0.64em;
    color: #d8e4f5;
  }

  .intro-logos {
    display: grid;
    gap: 1rem;
    justify-items: center;
    align-content: center;
    margin-top: 0.8rem;
  }

  .intro-logos img {
    box-shadow: none !important;
    border-radius: 0 !important;
    border: none !important;
  }

  .intro-logos .hf-mark {
    width: 240px;
  }

  .intro-logos .mcp-mark-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 220px;
    height: 220px;
    border-radius: 32px;
    background:
      radial-gradient(circle at 50% 40%, rgba(255, 255, 255, 0.18), transparent 58%),
      rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
  }

  .intro-logos .mcp-mark {
    width: 160px;
    filter: invert(1) brightness(1.2);
  }

  .image-strip {
    display: grid;
    grid-template-columns: 1.45fr 1fr;
    gap: 1rem;
    align-items: start;
    margin-top: 1rem;
  }

  .image-strip img {
    width: 100%;
    max-height: 280px;
    object-fit: contain;
  }

  .rl-layout {
    display: grid;
    grid-template-columns: 1.08fr 1.12fr;
    gap: 0.85rem;
    align-items: start;
  }

  .rl-layout > div:first-child {
    min-width: 0;
  }

  .rl-layout > div:first-child ul {
    margin-top: 0;
  }

  .rl-stack {
    display: grid;
    gap: 0.8rem;
    align-content: start;
  }

  .rl-stack img {
    width: 100%;
    max-height: 285px;
    object-fit: contain;
  }

  .flow-compare {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.8rem;
    margin-top: 0.7rem;
  }

  .flow-card {
    padding: 0.75rem 0.8rem 0.7rem;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.045);
  }

  .flow-card h3 {
    margin: 0 0 0.35rem;
    color: #ffc76b;
    font-size: 0.76em;
  }

  .flow-subtitle {
    margin-bottom: 0.5rem;
    color: #c9d7ea;
    font-size: 0.46em;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .swimlane {
    display: grid;
    gap: 0.24rem;
  }

  .step {
    position: relative;
    display: grid;
    grid-template-columns: 5.1em 1fr;
    gap: 0.2rem;
    align-items: center;
    padding: 0.38rem 0.5rem;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.06);
  }

  .step::after {
    content: "↓";
    position: absolute;
    left: 50%;
    bottom: -0.48rem;
    transform: translateX(-50%);
    color: rgba(255, 255, 255, 0.45);
    font-size: 0.58em;
    line-height: 1;
    pointer-events: none;
  }

  .step:last-child::after {
    content: "";
    display: none;
  }

  .lane {
    color: #9bdcff;
    font-size: 0.4em;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    text-align: left;
  }

  .action {
    color: #eef6ff;
    font-size: 0.5em;
    line-height: 1.18;
  }

  .step.owner-model {
    border-color: rgba(255, 199, 107, 0.34);
    background: rgba(255, 199, 107, 0.1);
  }

  .step.owner-tool {
    border-color: rgba(115, 210, 255, 0.34);
    background: rgba(115, 210, 255, 0.1);
  }

  .flow-note {
    margin-top: 0.45rem;
    color: #b8c8dc;
    font-size: 0.44em;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .dynamic-tool-layout {
    display: grid;
    grid-template-columns: 0.66fr 1.34fr;
    gap: 0.9rem;
    align-items: stretch;
    margin-top: 0;
  }

  .dynamic-tool-layout > div:first-child {
    min-width: 0;
  }

  .dynamic-tool-layout h1,
  .dynamic-tool-layout h2 {
    margin: 0 0 0.6rem;
    color: #fff7eb;
    font-family: 'Instrument Serif', serif;
    font-size: 1.28em;
    line-height: 0.96;
    border-bottom: none;
  }

  .dynamic-tool-layout > div:first-child p:first-child {
    margin-top: 0;
  }

  .dynamic-tool-layout ul {
    margin-top: 0.45rem;
  }

  .dynamic-tool-video {
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .dynamic-tool-video video {
    width: 100%;
    max-height: 620px;
    object-fit: contain;
    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 18px 36px rgba(0, 0, 0, 0.28);
    background: rgba(255, 255, 255, 0.03);
  }

  .launch-grid {
    margin-top: 1rem;
    gap: 1.15rem;
  }

  .launch-grid .card {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 150px;
    padding: 1rem 1rem 0.95rem;
    border-radius: 28px;
    text-align: center;
  }

  .launch-grid .card h3 {
    margin: 0;
    color: #fff9f1;
    font-family: 'Instrument Serif', serif;
    font-size: 1.12em;
    line-height: 0.98;
    letter-spacing: 0.01em;
    text-wrap: balance;
  }

  .launch-grid .card p,
  .launch-grid .card ul {
    display: none;
    margin-top: 0.9rem;
    font-size: 0.58em;
    line-height: 1.32;
    color: #c7d5e8;
  }

  .acp-combo {
    display: grid;
    grid-template-columns: 1.12fr 0.88fr;
    gap: 0.9rem;
    align-items: start;
    margin-top: 0.5rem;
  }

  .acp-combo-main {
    min-width: 0;
  }

  .acp-combo-visual {
    display: grid;
    gap: 0.85rem;
    align-content: start;
  }

  .acp-logo-panel {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 180px;
    padding: 1rem;
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.045);
  }

  .acp-logo-panel img {
    width: 100%;
    max-width: 300px;
    max-height: 120px;
    object-fit: contain;
    box-shadow: none;
    border: none;
    border-radius: 0;
  }

  .acp-summary {
    padding: 0.85rem 0.95rem;
    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.045);
  }

  .acp-summary h3 {
    margin: 0 0 0.5rem;
    color: #9bdcff;
    font-size: 0.74em;
  }

  .acp-summary p {
    margin: 0;
    font-size: 0.6em;
    line-height: 1.34;
    color: #d7e4f6;
  }

  .acp-kicker {
    max-width: 22em;
    margin: 0.35rem 0 0.8rem;
    color: #d7e4f6;
    font-size: 0.68em;
    line-height: 1.32;
  }

  .acp-video {
    display: flex;
    justify-content: center;
    align-items: flex-start;
  }

  .acp-video video {
    width: 100%;
    max-height: 500px;
    object-fit: contain;
    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 18px 36px rgba(0, 0, 0, 0.28);
    background: rgba(255, 255, 255, 0.03);
  }

  .acp-points {
    display: grid;
    gap: 0.65rem;
    margin-top: 0.8rem;
  }

  .acp-points .panel {
    padding: 0.85rem 0.95rem;
    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.045);
  }

  .acp-points .panel h3 {
    margin-top: 0;
    margin-bottom: 0.45rem;
  }

  .acp-points .panel p {
    margin: 0;
    font-size: 0.62em;
    line-height: 1.3;
  }

  .open-responses-layout {
    display: grid;
    grid-template-columns: 0.9fr 1.1fr;
    gap: 1rem;
    align-items: start;
    margin-top: 0.6rem;
  }

  .open-responses-intro {
    padding: 1rem 1rem 0.95rem;
    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.045);
  }

  .open-responses-intro p {
    margin: 0;
    font-size: 0.68em;
    line-height: 1.34;
    color: #dbe6f7;
  }

  .open-responses-intro p + p {
    margin-top: 0.7rem;
    color: #c6d6ea;
    font-size: 0.6em;
  }

  .open-responses-panels {
    display: grid;
    gap: 0.8rem;
  }

  .open-responses-panels .panel {
    padding: 1rem 1rem 0.95rem;
  }

  .open-responses-panels .panel h3 {
    margin-top: 0;
    margin-bottom: 0.55rem;
  }

  .open-responses-panels .panel ul {
    margin: 0;
    padding-left: 1rem;
  }
---

<!-- _class: titlepage -->

<div class="city-stamp">MCP Dev Summit · New York · 2026</div>
<div class="title">MCP at 18 Months</div>
<div class="subtitle">Protocols, patterns, and the things we did not see coming.</div>
<div class="author">Shaun Smith · <code>@evalstate</code></div>
<div class="date">April 2026</div>

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

---


<div class="columns">

<div>

# Shaun Smith `@evalstate`

- Open Source @ Hugging Face
- MCP maintainer / transports working group
- `huggingface/hf-mcp-server`
- `huggingface/upskill`
- `huggingface/skills`
- Maintainer of `fast-agent`

<div class="hero-note">
</div>

</div>

<div class="center">

<div class="intro-logos">
  <img class="hf-mark" src="./images/hf_logo.svg" alt="Hugging Face" />
  <div class="mcp-mark-wrap">
    <img class="mcp-mark" src="./images/mcp-icon.svg" alt="MCP" />
  </div>
</div>

</div>

</div>

---

<!-- _class: transition -->

# The debate about MCP is more interesting than MCP itself

<div class="lede">
...and that is a good thing!
</div>

<div class="signal-strip">
</span></div>
</div>


---

# Things we didn't have at launch

<div class="card-grid launch-grid">
  <div class="card">
    <img src="./images/claude-code.png" />
  </div>

  <div class="card">
    <h3>Streamable HTTP Transport and OAuth</h3>
  </div>
  <div class="card">
    <h3>AGENTS.MD and Agent Skills</h3>
  </div>
  <div class="card">
    <h3>Internal Tools in Inference APIs</h3>
  </div>
  <div class="card">
    <h3>Agent Client Protocol<br />
    Responses API</h3>
  </div>
  <div class="card">
    <h3>Long Running Tool Loops (and reasoning models)</h3>
  </div>

</div>

---


# Reinforcement Learning 

<div class="rl-layout">
  <div>

Models are placed in an environment, given a task and scored with a reward function:

  - <strong>discover</strong>
  - <strong>self-correct</strong>
  - <strong>problem solve</strong>
  - keep <strong>driving the loop</strong> without constant human steering

> mini-SWE-Agent: A single 100 line python and single freeform (non JSON) tool can score 76.0% on SWE-Bench!

It's hard to compete against that efficiency.

![w:400 alt text](images/openenv.png)

  </div>
  <div class="rl-stack">
    <img src="./images/image-1.png" alt="Reinforcement learning environment diagram" />
    <img src="./images/swe-bash-tool.png" alt="SWE-Bench bash tool benchmark result" />
  </div>
</div>


---

# Smarter Tool Loops

<div class="comparison">
  <div class="panel">
      Harness Changes

General Purpose Agent Harnesses are given direct Shell access

Fewer pre/post Tool/LLM Stop checks/hacks to keep model on-track.

Snapshot/Checkpointing techniques (AgentFS, Execution Monitoring)

Remote runtime environments (e.g. Codex Web, Claude Code)
    
  </div>
  <div class="panel">
    Why this enabled Skills

Simple navigable, native hierarchy of content

Reusable procedures become strong scaffolding for capable models

Bash is token dense and unsurprising compared to custom JSON Tools / mid-context tool enablement

Between deterministic program and documentation. 

  </div>
</div>

<blockquote>
<p>Once models can discover, recover, and keep going, a “skill” becomes a practical acceleration layer rather than a brittle scripted hack.</p>
</blockquote>

---

<div class="dynamic-tool-layout">
  <div>

<h1>Dynamic Tool Calling</h1>

Dynamic Space Tool: **45 tokens**

MCP provides an **inference gateway** to thousands of specialized and custom models covering Audio, Video, Text, 3D Models, Environments and more.

**MCP** provides Authentication and Multimodal support. 


<code>Qwen 3.5-35B-A3B</code>
<code>Flux.1-Krea-Dev </code>
<code>Qwen-Edit-2509-Multiple-angles-LoRA</code>
<code>Wan2.2 First/Last Frame</code>

  </div>
  <div class="dynamic-tool-video">
    <video autoplay muted loop playsinline>
      <source src="./images/dynamic_space_final.mp4.mp4" type="video/mp4" />
    </video>
  </div>
</div>

---

# Code Execution Tools 

<div class="columns">

<div>

A model with access to general purposes tools has crossed into a very real form of <strong>code mode</strong>.

Bash provides a general purpose, token dense-execution language. 

Task-specific tools generated on demand. Example: **HF Tool Builder** navigates OpenAPI spec to build composable CLI tools.

Some models are trained to use **code tools natively**, and are bundled with interpreters.

</div>

<div class="center">
<div class="rl-stack">

<img src="./images/smolagents.png" />
<img src="./images/image-3.png" />
</div>

</div>

</div>

---

# Generation and Execution Environments

<div class="flow-compare">
  <div class="flow-card">
    <h3>Style 1 - Main Model owns Code Generation</h3>
    <div class="swimlane">
      <div class="step">
        <div class="lane">Main model</div>
        <div class="action">Generates Search Function</div>
      </div>
      <div class="step">
        <div class="lane">Execution Tool</div>
        <div class="action">Uses Search Function to return API definitions</div>
      </div>
      <div class="step ">
        <div class="lane">Main model</div>
        <div class="action">Generates code from that API surface</div>
      </div>
      <div class="step">
        <div class="lane">Execution tool</div>
        <div class="action">Runs the code and returns output</div>
      </div>
      <div class="step">
        <div class="lane">Main model</div>
        <div class="action">Reads result and writes final answer</div>
      </div>
    </div>
    <div class="flow-note">Code Generation: Main Model</div>
    <div class="flow-note">Code Execution: Tool Environment</div>
    
  </div>
  <div class="flow-card">
    <h3>Style 2 - Delegated Code Generation</h3>
    <div class="swimlane">
      <div class="step">
        <div class="lane">Main model</div>
        <div class="action">Sends a natural-language task to the tool</div>
      </div>
      <div class="step owner-tool">
        <div class="lane">Execution tool</div>
        <div class="action">System Prompt contains API definitions</div>
      </div>
      <div class="step">
        <div class="lane">Execution tool</div>
        <div class="action">Returns the result</div>
      </div>
      <div class="step">
        <div class="lane">Main model</div>
        <div class="action">Packages it as the final answer</div>
      </div>
    </div>
    <div class="flow-note">Code Generation: Tool Model</div>
    <div class="flow-note">Code Generation: Tool Environment</div>
    <div class="flow-note">API Definitions Cacheable</div>
  </div>
</div>

<br />

<center>

## **MCP** makes it easy to transfer **generation** and **execution** between models and environments! <br> (and who pays for inference)


</center>

---

<div class="dynamic-tool-layout">
  <div>

# LLMs for Navigating: GenUI, Apps SDK **(Prefect Prefab)**

A common pattern:
1. user asks for navigation or retrieval
1. tools fetch the answer
1. the model then spends expensive output tokens reprocessing a result that was already good enough
2. The **MCP Apps** pattern fixes this by letting the result become <strong>final for the user</strong>.

  </div>
  <div class="dynamic-tool-video">
    <video autoplay muted loop playsinline>
      <source src="./images/gen_ui_one.mp4" type="video/mp4" />
    </video>
  </div>
</div>


---

# Inference and Environment Boundaries are Blurring

## The new abstraction

<div class="card-grid">
  <div class="card">
Runtime Environments

Options from YOLO, Local/Remote containers, exe.dev-style or lightweight sandboxes (Monty, Just-Bash). Simple persistent storage (e.g. HF Buckets)</p>

  </div>
  <div class="card">
Model Selection

Mixed Model workloads handle different modalitites, specializations and price points. Token efficient task agent delegation.
  </div>
  <div class="card">
Inference APIs
    
Increasingly absorb search, tools, code, and state into one bundled execution surface.
  </div>
</div>


---

# Agent Client Protocol

<div class="columns">
  <div>
    <p></p>
    <div class="acp-points">
      <div class="panel">
        File and Shell Tools
        <p>Client provided tools, enables "follow along" in editors </p>
      </div>
      <div class="panel">
        Session Based
        <p>Listing, Resumption and Rehydration of Agent sessions</p>
      </div>
      <div class="panel">
        Streaming Results and Observability
        <p>Agent Results and Tool Status stream, are cancellable</p>
      </div>
      <div class="panel">
        MCP Native Support
        <p>Uses MCP Data Model. Client sends MCP Sever Configurations</p>
      </div>
    </div>
  </div>
  <div class="acp-video">
    <br/ >
    <video autoplay muted loop playsinline>
      <source src="./images/toad-subagent.mp4" type="video/mp4" />
    </video>
  </div>
</div>

---

# Open Responses


<div class="open-responses-layout">
  <div class="open-responses-intro">


<h2>Open standard extending OpenAI's Responses API. Provides a consistent, provider neutral way to interact with modern LLMs. Repairs Chat Completion API drift.</h2>

> It defines a shared schema, and tooling layer that enable a unified experience for calling language models, streaming results, and composing agentic workflows—independent of provider.

<br />

<h2>Usage as a Provider / Router allows creation of rich Agent Environments</h2>

  </div>
  <div class="open-responses-panels">
    <div class="panel">
      Internal Tools - (Model or Provider)
      <ul>
        <li><code>shell</code> and <code>local_shell</code></li>
        <li><code>code_interpreter</code></li>
        <li><code>apply_patch</code></li>
        <li><code>web_search</code></li>
        <li><code>etc..</code></li>
      </ul>
      External Tools (Client Supplied)
      <ul>
        <li>MCP Servers</li>
        <li>Standard JSON function calls</li>
        <li>Free-Form Tools</li>
        <li>Grammar constrained Tools</li>
      </ul>
    </div>
  </div>
</div>

---

# It was close....! PMF for MCP

<div class="comparison">
  <div class="panel">
  MCP is a Commodity Standard    

Supports Consumer, Enterprise and Developer use-cases. 

Single URL to install authenticated JSON tools across thousands of clients

MCP's "fit" features *weren't present* at launch!

URI/Resources based extensions deliver innovation and extensibility...

...Which  enabled rapid MCP Apps distribution on a solid support base.


  </div>
  <div class="panel">
   Model/Host Changes and STDIO
   
   Host applications with Shell tool reduce the need for STDIO Servers.
   
   In many cases for local running tools such as Apify **mcp-cli** or Pete Steinberger's **MCPorter** offer a _better_ experience for MCP usage.

   Distribution via MCPB is one potential advantage

   Simple one-shot server design meant that distribution of ideas was more important than code.
  </div>
</div>


---

<!-- _class: transition -->

# Thank You!
