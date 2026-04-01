---
marp: true
html: true
size: 16:9
theme: ny-noir
paginate: false
header: '<span class="header-logos"><img src="./images/hf_logo.svg" alt="Hugging Face" /><img src="./images/github-mark.svg" alt="GitHub" />github.com/evalstate</span>'
style: |
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
    grid-template-columns: 1.18fr 0.82fr;
    gap: 1rem;
    align-items: start;
    margin-top: 0.7rem;
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
    padding: 0.95rem 1rem;
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

<div class="eyebrow">1 · Introduction</div>

# Shaun Smith `@evalstate`

- Open Source @ Hugging Face
- MCP maintainer / transports working group
- `huggingface/hf-mcp-server`
- `huggingface/upskill`
- `huggingface/skills`
- Maintainer of `fast-agent`

<div class="hero-note">
My angle here is practical: what actually changed in agent systems after MCP launched, what parts of the protocol clearly found product-market fit, and what the new execution patterns are doing to the boundary of the stack.
</div>

</div>

<div class="center">

![w:240](./images/hf_logo.svg)
![w:220](./images/mcp-icon.svg)

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
    Open Responses</h3>
  </div>
  <div class="card">
    <h3>Long Running Tool Loops (and reasoning models)</h3>
  </div>

</div>

---


# Reinforcement Learning 

<div class="rl-layout">
  <div>

Models are placed in an environment, given a task and scored with a reward function :

  - <strong>discover</strong>
  - <strong>self-correct</strong>
  - <strong>problem solve</strong>
  - keep <strong>driving the loop</strong> without constant human steering

> mini-SWE-Agent: A single 100 line python and single freeform (non JSON) tool can scoe 76.0% on SWE-Bench!

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
      <h3>Harness Changes</h3>
      <li>General Purpose Agent Harnesses are given direct Shell access</li>
      <li>Models are able to reason and act on discovered content
      <li>Progressive Disclosure has follow automatic follow-through</li>
    </ul>
  </div>
  <div class="panel">
    <h3>Why this enabled Skills</h3>
      <li>Simple, shell Native hierarchy of content</li>
      <li>Reusable procedures become strong scaffolding for capable models</li>
      <li>Shell surface is perfect for token dense discovery and navigation</li>
  </div>
</div>

<blockquote>
<p>Once models can discover, recover, and keep going, a “skill” becomes a practical acceleration layer rather than a brittle scripted hack.</p>
</blockquote>

---

<div class="dynamic-tool-layout">
  <div>

<h1>Dynamic Tool Calling</h1>


Models can now discover and dynamically call Tools. 

Dynamic Space Tool: **45 tokens**


**MCP** provides Authentication and Multimodal support. 

MCP provides an **inference gateway** to thousands of specialized and custom models covering Audio, Video, Text, 3D Models, Environments and more.



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

Agent Skills are powerful:
- Between deterministic program and documentation. 
- Model discoverable context loading
- task-specific tools generated on demand. Example: **HF Tool Builder** navigates OpenAPI spec to build composable CLI tools.

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

# Model Generation and Execution Environments

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
        <div class="action">Return API definitions</div>
      </div>
      <div class="step ">
        <div class="lane">Main model</div>
        <div class="action">Generates code from that tool surface</div>
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
  </div>
</div>

<br />

<center>

## **MCP** makes it easy to transfer **generation** and **execution** between models and environments! 


</center>

---

<div class="dynamic-tool-layout">
  <div>

<h1>LLMs for Navigating, GenUI and the Apps SDK</h1>

A common pattern:
1. user asks for navigation or retrieval
1. tools fetch the answer
1. the model then spends expensive output tokens reprocessing a result that was already good enough
1. The **MCP Apps SDK** pattern fixes this by letting the result become <strong>final for the user</strong>.

  </div>
  <div class="dynamic-tool-video">
    <video autoplay muted loop playsinline>
      <source src="./images/gen_ui_one.mp4" type="video/mp4" />
    </video>
  </div>
</div>

---

# Inference and Execution Boundaries are Blurring

<div class="card-grid">
  <div class="card">
    <h3>Execution Environments</h3>
    <p>Useful when local context matters and the user already trusts the environment.</p>
  </div>
  <div class="card">
    <h3>Model Selection</h3>
    <p>Lets us transport workload to a better environment while preserving a usable interface boundary.</p>
  </div>
  <div class="card">
    <h3>Inference APIs</h3>
    <p>Increasingly absorb search, tools, code, and state into one bundled execution surface.</p>
  </div>
</div>

<div class="signal-strip">
  <div class="signal"><strong>MCP + OAuth</strong><span>Moves execution somewhere safer or more capable.</span></div>
  <div class="signal"><strong>Remote tools</strong><span>Make workload transport part of normal architecture.</span></div>
  <div class="signal"><strong>Inference bundling</strong><span>Pulls some of that same logic into model APIs.</span></div>
  <div class="signal"><strong>Result</strong><span>The old “where does the tool live?” question is no longer stable.</span></div>
</div>

---

# Agent Client Protocol

<div class="acp-combo">
  <div class="acp-combo-main">
    <div class="comparison">
      <div class="panel">
        <h3>ACP bundles from the client side</h3>
        <ul>
          <li>Model + tools + skills + interaction shape</li>
          <li>Streaming, sessions, and rehydration</li>
          <li>Observability and control points</li>
          <li>Useful when we care about continuity and UX quality</li>
        </ul>
      </div>
      <div class="panel">
        <h3>Why it matters here</h3>
        <ul>
          <li>MCP handles tool interaction well</li>
          <li>ACP distributes a fuller agent experience</li>
          <li>Good fit when model and API quirks need one wrapper</li>
          <li>Lets us bundle interaction, state, and execution shape together</li>
        </ul>
      </div>
    </div>
  </div>
  <div class="acp-combo-visual">
    <div class="acp-logo-panel">
      <img src="./images/acp-docs-logo.avif" alt="Agent Client Protocol logo" />
    </div>
    <div class="acp-summary">
      <h3>The important point</h3>
      <p>MCP still matters because both patterns need a coherent way to talk to tools and remote capability surfaces.</p>
    </div>
  </div>
</div>




---

# Open Responses



> Open Responses is an open-source specification and ecosystem for building multi-provider, interoperable LLM interfaces based on the OpenAI Responses API. It defines a shared schema, and tooling layer that enable a unified experience for calling language models, streaming results, and composing agentic workflows—independent of provider.


<div class="open-responses-layout">
  <div class="open-responses-intro">
    <p>Open Responses is the inference-side version of bundling: the model API can expose tool use, state, and execution surfaces as part of the response loop itself.</p>



  </div>
  <div class="open-responses-panels">
    <div class="panel">
      <h3>Internal tools - Model/Provider</h3>
      <ul>
        <li><code>code_interpreter</code></li>
        <li><code>apply_patch</code></li>
        <li><code>web_search</code></li>
      </ul>
    </div>
    <div class="panel">
      <h3>External tools</h3>
      <ul>
        <li>MCP servers</li>
        <li>Standard JSON function calls</li>
        <li>Grammar Constrained non-JSON tools</li>
      </ul>
    </div>
  </div>
</div>

<div class="signal-strip">
  <div class="signal"><strong>MCP + OAuth</strong><span>Moves execution somewhere safer or more capable.</span></div>
  <div class="signal"><strong>Remote tools</strong><span>Make workload transport part of normal architecture.</span></div>
  <div class="signal"><strong>Inference bundling</strong><span>Pulls some of that same logic into model APIs.</span></div>
  <div class="signal"><strong>Result</strong><span>The old “where does the tool live?” question is no longer stable.</span></div>
</div>


---

<div class="eyebrow">5 · Why STDIO servers mattered</div>

# Before shell execution in clients, STDIO filled a real gap

<div class="comparison">
  <div class="panel">
    <h3>What the gap was</h3>
    <ul>
      <li>Clients often could not execute arbitrary shell commands themselves</li>
      <li>If you wanted access to the local machine, you had to expose it somehow</li>
      <li>STDIO servers became the path to “do something here”</li>
      <li>That was not accidental — it was a real capability gap</li>
    </ul>
  </div>
  <div class="panel">
    <h3>Why distribution was about ideas</h3>
    <ul>
      <li>Many servers could be one-shotted into existence</li>
      <li>Sometimes the value was the interface idea, not the code artifact</li>
      <li>The durable value-add was not wrapper code itself</li>
      <li>It was access to <strong>data</strong>, <strong>resources</strong>, and <strong>compute</strong></li>
    </ul>
  </div>
</div>


---

<div class="eyebrow">Closing</div>

# MCP - Product Market Fit

<div class="closing-grid">
  <div class="panel">
    <h3>The through-line</h3>
    <ul>
      <li>MCP </li>
      <li>The ecosystem around MCP changed faster than almost anyone expected.</li>
      <li>RL made self-propelling tool loops more viable.</li>
      <li>Skills and shell execution turned into practical code mode.</li>
      <li>Apps SDK patterns taught us when not to spend more tokens.</li>
      <li>Remote execution, ACP, and Open Responses are all redrawing the boundary.</li>
    </ul>
  </div>
  <div class="panel">
    <h3>The good news</h3>
    <ul>
      <li>The debate is more interesting because the protocol is now real infrastructure.</li>
      <li>The next phase is about placement: what belongs in MCP, what belongs in the client, and what belongs in inference.</li>
      <li>That is a much better problem to have than irrelevance.</li>
    </ul>
  </div>
</div>

---

<!-- _class: transition -->

# Thank You!
