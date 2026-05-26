---
marp: true
html: true
size: 16:9
theme: nvidia-meetup
paginate: false
header: '<span class="header-logos"><img src="./images/hf_logo.svg" alt="Hugging Face" /><img src="./images/github-mark.svg" alt="GitHub" />github.com/evalstate</span>'

---

<!-- _class: titlepage -->

<div class="city-stamp">Agentic AI on CPU · London · 2026</div>
<div class="title">Upskilling Local Models and Agents</div>
<div class="author">Shaun Smith · Adrian Lepers</div>
<div class="date">May 2026</div>

<table class="social-table">
  <tbody>
    <tr>
      <td><img src="./images/huggingface-mark-logo.svg" alt="Hugging Face" /></td>
      <td><a class="organization" href="https://huggingface.co/evalstate">huggingface.co/evalstate</a></td>
    </tr>
    <tr>
      <td><img src="./images/huggingface-mark-logo.svg" alt="Hugging Face" /></td>
      <td><a class="organization" href="https://huggingface.co/AdrianLepers">huggingface.co/AdrianLepers</a></td>
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
- MCP Maintainer [Transports] / Community Moderator
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


<video class="full-slide" src="./images/intro-spaces.webm" autoplay loop muted playsinline></video>


---

<!-- _class: transition -->

# The evolution of Tool Calling and Model Inference....

<div class="lede">
</div>

<div class="signal-strip">
</span></div>
</div>

---

# Things we didn't have 18 Months ago...

<div class="card-grid launch-grid">
  <div class="card">
    <img src="./images/claude-code.png" />
  </div>

  <div class="card">
    <h3>MCP Streamable HTTP Transport and OAuth</h3>
  </div>
  <div class="card">
    <h3>AGENTS.MD and Agent Skills</h3>
  </div>
  <div class="card">
    <h3>MoE Models and Efficient Quantizations</h3>
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

# Smaller and Simpler Harnesses

<div class="big-points">

- General-purpose agent harnesses are given direct(*) shell access
- Fewer pre/post tool and LLM stop checks to keep models on track
- API surface and Custom Workflows replaced by Model capabilities
- Snapshot and checkpointing techniques
- Movable runtime environments
- Scripting (code generation) allows immediate specialization

</div>

---

# Self Directing Models

<img class="nia-flow-img no-shadow" src="./images/skill-flow.svg" alt="Task flows to Navigate, Ingest, Act, then loops back to Task" />

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
  <div class="dynamic-tool-video dynamic-tool-video-unframed dynamic-tool-video-tight">
    <video autoplay muted loop playsinline>
      <source src="./images/dynamic_space_final.mp4.mp4" type="video/mp4" />
    </video>
  </div>
</div>


---

# Why This Enabled Skills

<div class="big-points">

- Simple to navigate native content hierarchy
- Unsurprising Token Dense format (`bash`!)
- Reusable procedures become scaffolding for capable models
- Script access requires fewer mid-context tool  tricks
- Between deterministic program and documentation

</div>

---

# Training Models

<div class="columns">

<div>
LLM Trainer Skill

`Fine-tune Qwen3-0.6B on the dataset open-r1/codeforces-cots`

Handles:
- Dataset Construction
- Dataset Selection and Validation
- Hardware Selection
- Training Scripts
- Job Submission and Monitoring
- Trackio Supervision
- GGUF Conversion 

_Recently added Vision Training!_

</div>


<div class="center">
<div class="rl-stack">

<img src="./images/2026-04-30-claude-blog.png" />
<img height="300" src="./images/2026-04-30-trackio-training.png" />
</div>

</div>

</div>

---



# `https://github.com/huggingface/upskill`



<div class="center">

<img width="900" src="./images/2026-02-11-upskill.png" />

---

# Upskill

<div class="big-points">

- Run in Sandboxes, View Traces, Optimise and Benchmark

</div>


<div class="upskill-image-pair">

<div>

<img src="./images/2026-04-30-skill-chart.png" alt="Upskill trace chart" />

</div>

<div>

<img src="./images/2026-04-30-skill-output2.png" alt="Upskill benchmark output" />

</div>

</div>

<div class="big-points">

- Tutor and Select best Price/Performance Models

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

<style scoped>
header, section::after {
  display: none !important;
}
.macbook-quote-slide {
  position: relative;
  box-sizing: border-box;
  display: grid;
  grid-template-rows: auto 1fr auto auto;
  height: 100%;
  margin: -72px -78px;
  padding: 68px 78px 62px;
  background:
    radial-gradient(circle at 82% 14%, rgba(255, 255, 255, 0.34), transparent 23%),
    radial-gradient(circle at 12% 88%, rgba(8, 16, 25, 0.18), transparent 30%),
    linear-gradient(135deg, #ffb44f 0%, #ffd37d 52%, #7dd3fc 118%);
  color: #081019;
}
.macbook-kicker {
  color: #17324a;
  font-size: 0.56em;
  font-weight: 900;
  letter-spacing: 0.17em;
  text-transform: uppercase;
}
.macbook-quote {
  max-width: 24.5em;
  margin-top: 0.85rem;
  padding: 1.1rem 1.25rem;
  border-left: 7px solid #081019;
  border-radius: 0 24px 24px 0;
  background: rgba(255, 248, 235, 0.48);
  box-shadow: 0 20px 42px rgba(8, 16, 25, 0.14);
  color: #122033;
  font-size: 0.54em;
  font-weight: 700;
  line-height: 1.32;
}
.macbook-quote strong {
  color: #081019;
  font-weight: 900;
}
.macbook-hero {
  align-self: end;
  max-width: 10.5em;
  color: #081019;
  font-family: 'Instrument Serif', serif;
  font-size: 1.42em;
  line-height: 0.96;
  letter-spacing: 0.01em;
}
.macbook-hero .metric {
  color: #7c2d12;
  font-weight: 700;
}
.macbook-hero .sub {
  display: block;
  margin-top: 0.38rem;
  color: #17324a;
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.34em;
  font-weight: 900;
  letter-spacing: 0.09em;
  line-height: 1.22;
  text-transform: uppercase;
}
.macbook-chart {
  position: absolute;
  right: 62px;
  top: 142px;
  width: 520px;
  padding: 0.35rem;
  border-radius: 26px;
  background: rgba(255, 248, 235, 0.42);
  box-shadow: 0 22px 46px rgba(8, 16, 25, 0.16);
}
.macbook-chart img {
  display: block;
  width: 100%;
  border: none !important;
  border-radius: 20px;
  box-shadow: none !important;
}
.macbook-source {
  margin-top: 0.55rem;
  color: rgba(8, 16, 25, 0.72);
  font-size: 0.42em;
  font-weight: 800;
  letter-spacing: 0.04em;
}
.macbook-source strong {
  color: #081019;
}
.macbook-source a {
  color: #17324a;
  text-decoration: none;
  border-bottom: 1px solid rgba(23, 50, 74, 0.42);
}
</style>

<div class="macbook-quote-slide">
  <div>
    <div class="macbook-kicker">Local model capability is compounding</div>
    <div class="macbook-quote">
      Between May 2024 and May 2026, the most expensive MacBook Pro you could buy stayed at
      <strong>128 GB of unified memory</strong>. The hardware ceiling barely moved. But the smartest
      open-weight model you could actually run on it went from a score of <strong>10</strong>
      — Llama 3 70B — to <strong>47</strong> — DeepSeek V4 Flash on antirez's mixed-Q2 GGUF —
      on the Artificial Analysis Intelligence Index.
    </div>
  </div>

  <div class="macbook-chart">
    <img src="./images/chart.png" alt="Artificial Analysis local model intelligence chart" />
  </div>

  <div></div>

  <div class="macbook-hero">
    That is <span class="metric">4.7×</span> in 24 months
    <span class="sub">or a doubling of intelligence every 10.7 months</span>
  </div>

  <div class="macbook-source">
    Source: <strong>“Local Moore’s Law”</strong> by Mishig Davaadorj ·
    <a href="https://huggingface.co/blog/mishig/local-moores-law">huggingface.co/blog/mishig/local-moores-law</a>
  </div>
</div>

---

![](./images/gepa-optimize.png)

---

<style scoped>
.privacy-slide {
  position: relative;
  height: 100%;
}
.privacy-title {
  margin: 0 0 0.75rem !important;
  border-bottom: none !important;
  color: #fff7eb;
  font-family: 'Instrument Serif', serif;
  font-size: 2.05em !important;
  line-height: 0.95;
}
.privacy-filter-visual {
  position: absolute;
  top: 3.4rem;
  right: 0rem;
  width: 520px;
}
.privacy-filter-visual img {
  display: block;
  width: 100%;
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 16px 34px rgba(0, 0, 0, 0.32);
}
.privacy-demo-card {
  position: relative;
  width: 48%;
  margin-top: 0.8rem;
  padding: 0.82rem 0.9rem 0.78rem;
  border: 1px solid rgba(255, 255, 255, 0.11);
  border-radius: 22px;
  background: rgba(8, 14, 24, 0.78);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 18px 36px rgba(0, 0, 0, 0.22);
}
.privacy-copy-button {
  position: absolute;
  top: calc(0.7rem + 20px);
  right: calc(0.8rem + 20px);
  z-index: 2;
  border: 1px solid rgba(155, 220, 255, 0.48);
  border-radius: 999px;
  padding: 0.28rem 0.68rem;
  background: rgba(155, 220, 255, 0.14);
  color: #e8f8ff;
  font-size: 0.38em;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  cursor: pointer;
}
.privacy-copy-button:hover {
  background: rgba(155, 220, 255, 0.26);
}
.privacy-demo-card pre {
  margin: 0 !important;
  padding: 0.82rem 0.92rem !important;
  border: 1px solid rgba(226, 232, 240, 0.55);
  border-radius: 18px;
  background: rgba(1, 6, 13, 0.78) !important;
  color: #ffe7a8;
  white-space: pre-wrap;
  word-break: normal;
  overflow-wrap: anywhere;
  overflow: hidden;
}
.privacy-demo-card code {
  display: block;
  background: transparent !important;
  color: #ffe7a8 !important;
  padding: 0 !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 15.5px !important;
  line-height: 1.26;
  white-space: pre-wrap !important;
  overflow-wrap: anywhere;
  word-break: break-word;
}
.privacy-demo-note {
  margin-top: 0.48rem;
  color: #9bdcff;
  font-size: 0.46em;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
</style>

<div class="privacy-slide">
  <h1 class="privacy-title">OpenAI Privacy Filter</h1>

  <div class="privacy-filter-visual">
    <img src="./images/privacy_filter.png" alt="Privacy filter screenshot" />
  </div>

  <div class="privacy-demo-card">
    <button class="privacy-copy-button" type="button" onclick="const code=document.getElementById('privacy-pii').innerText; navigator.clipboard.writeText(code); const old=this.textContent; this.textContent='Copied'; setTimeout(()=>this.textContent=old,1200);">Copy</button>
<pre><code id="privacy-pii">can you help me.

my name is "shaun smith"
my credit card is "4929 1003 4422 4042"
the API key I have been using is
"sk-proj-rr3399393922220202".

You can reach me at shaun.smith@private-email.com
or +44 7700 900123.

My home address is 221B Baker Street,
London NW1 6XE.

My AWS access key is AKIAIOSFODNN7EXAMPLE
and the secret is
wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY.

My national insurance number is QQ 12 34 56 C.</code></pre>
  </div>
</div>

---

# Some thoughts

<div class="big-points">

- Owning and usefully customising and improving your own models is accessible
- Frontier Models are overused: Price/Performance 
- Inference and Execution environments are blending
- Self Improvement is here if you want it

</div>

---

<style scoped>
header, section::after {
  display: none !important;
}
.takeaway-slide {
  position: relative;
  box-sizing: border-box;
  height: 100%;
  margin: -72px -78px;
  padding: 72px 78px 58px;
  overflow: hidden;
  background:
    radial-gradient(circle at 102% -8%, rgba(255, 215, 75, 0.18) 0 19%, transparent 19.3%),
    radial-gradient(circle at 102% -8%, transparent 0 24%, rgba(255, 215, 75, 0.95) 24.1% 24.45%, transparent 24.6%),
    radial-gradient(circle at 12% 18%, rgba(255, 199, 107, 0.06), transparent 28%),
    linear-gradient(135deg, #050707 0%, #081019 54%, #07111a 100%);
  color: #f8fafc;
}
.takeaway-kicker {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  color: #f7d94c;
  font-size: 0.58em;
  font-weight: 900;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.takeaway-kicker::before {
  content: "";
  width: 2.2rem;
  height: 0.16rem;
  background: #f7d94c;
}
.takeaway-headline {
  margin-top: 2.9rem;
  max-width: 11.8em;
  color: #f8fafc;
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 2.2em;
  font-weight: 900;
  line-height: 0.98;
  letter-spacing: -0.045em;
}
.takeaway-headline .accent {
  display: block;
  margin-top: 0.24em;
  color: #f7d94c;
}
.takeaway-divider {
  width: 4.3rem;
  height: 0.18rem;
  margin-top: 2.6rem;
  background: #f7d94c;
}
.takeaway-points {
  margin-top: 0.85rem;
  display: grid;
  gap: 0.18rem;
  font-family: 'IBM Plex Sans', sans-serif;
}
.takeaway-points .primary {
  color: #ffffff;
  font-size: 0.82em;
  font-weight: 900;
  line-height: 1.12;
}
.takeaway-points .muted {
  margin-top: 0.25rem;
  color: rgba(248, 250, 252, 0.66);
  font-size: 0.72em;
  font-weight: 500;
}
.takeaway-brand {
  position: absolute;
  right: 78px;
  bottom: 46px;
  display: flex;
  align-items: center;
  gap: 0.45rem;
  color: #fff;
  font-size: 0.52em;
  font-weight: 900;
}
.takeaway-brand img {
  width: 28px;
  height: 28px;
  border: 1px solid #f7d94c !important;
  border-radius: 0 !important;
  background: #f7d94c;
  box-shadow: none !important;
}
</style>

<div class="takeaway-slide">
  <div class="takeaway-kicker">The takeaway</div>

  <div class="takeaway-headline">
    Open source doesn't mean
    <span class="accent">ungoverned.</span>
  </div>

  <div class="takeaway-divider"></div>

  <div class="takeaway-points">
    <div class="primary">Own the weights.</div>
    <div class="primary">Set the policy.</div>
    <div class="muted">Don’t only trust someone else’s black box.</div>
  </div>

  <div class="takeaway-brand">
    <img src="./images/huggingface-mark-logo.svg" alt="Hugging Face" />
    <span>Hugging Face</span>
  </div>
</div>

---


<!-- _class: transition -->


# Thank You!

<div class="thanks-links">
  <div>
    <img src="./images/github-mark.svg" alt="GitHub" />
    <span>github.com/evalstate</span>
  </div>
  <div>
    <img src="./images/xcom-logo-black.png" alt="X" />
    <span>x.com/evalstate</span>
  </div>
</div>

<div class="bottom-image-flush no-shadow">

![](./images/hugs.svg)

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
