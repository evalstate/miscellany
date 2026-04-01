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
---

<!-- _class: titlepage -->

<div class="city-stamp">MCP Dev Summit · New York · 2026</div>
<div class="title">MCP at 18 Months</div>
<div class="subtitle">Protocols, patterns, and the things we did not see coming.</div>
<div class="author">Shaun Smith · <code>@evalstate</code></div>
<div class="date">March 2026 draft</div>

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

<div class="eyebrow">2 · Opening thesis</div>

# The debate about MCP is more interesting than MCP itself

<div class="lede">
...and that is a good thing.
</div>

<div class="signal-strip">
</span></div>
</div>

<blockquote>
The world has changed around MCP 
</blockquote>


---

<!-- _class: transition -->

# 3 · What we did not have when MCP launched

---

<div class="card-grid">
  <div class="card">
    <h3>Coding Agents</h3>
    <p>Goose (Jan 2025)</p>
    <p>Claude Code (Feb 2025)</p>

  </div>
  <div class="card">
    <h3>MCP Streamable HTTP Transport</h3>
    <p>No Auth mechanism</p>
    <p></p>
  </div>
  <div class="card">
    <h3>AGENTS.MD or AgentSkills</h3>
  </div>
  <div class="card">
    <h3>Tools in Inference API's</h3>
    <p>Remote tool execution, e.g. Code Interpreter, Shell Sandbox, Web Search, Bash</p>
  </div>
</div>


![w:420](./images/claude-code.png)
![w:420](./images/deepseek.png)


---


<div class="eyebrow">4 · RL changed the loop</div>

# Reinforcement learning made agentic behavior feel real

- Models are increasingly trained to:
  - <strong>discover</strong>
  - <strong>self-correct</strong>
  - <strong>problem solve</strong>
  - keep driving the loop without constant human steering
- That is the missing context for why <strong>self-propelling tool loops</strong> suddenly started working better.
- And it explains why even very simple agent harnesses can now do surprisingly serious work.

![w:980](./images/image-1.png)
![alt text](swe-bash-tool.png)

---

<div class="eyebrow">mini-SWE-agent</div>

# Simplicity matters more than ceremony

<div class="comparison">
  <div class="panel">
    <h3>Why mini-SWE-agent matters</h3>
    <ul>
      <li>Single tool</li>
      <li>Minimal loop</li>
      <li>Non-persistent bash between fenced commands</li>
      <li>Still good enough to show the new model behavior is real</li>
    </ul>
  </div>
  <div class="panel">
    <h3>Why this enabled skills</h3>
    <ul>
      <li>Skills no longer need mystical orchestration to be useful</li>
      <li>Reusable procedures become strong scaffolding for capable models</li>
      <li>Simple environments can produce compound behavior</li>
      <li>The loop quality moved closer to the model</li>
    </ul>
  </div>
</div>

<blockquote>
<p>Once models can discover, recover, and keep going, a “skill” becomes a practical acceleration layer rather than a brittle scripted hack.</p>
</blockquote>

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

<div class="eyebrow">Value-add</div>

# The real value-add of MCP was never “just wrap an API”

<div class="signal-strip">
  <div class="signal"><strong>Data</strong><span>Access to information you do not already have locally.</span></div>
  <div class="signal"><strong>Resources</strong><span>Structured access to remote state, files, and protected surfaces.</span></div>
  <div class="signal"><strong>Compute</strong><span>Running work where it makes sense, not where the user happens to be.</span></div>
  <div class="signal"><strong>Distribution</strong><span>A shareable interface idea that agents and clients can both understand.</span></div>
</div>

<div class="quote-wall">
  <p>STDIO solved a local capability problem. MCP’s longer-term value was that it gave those capabilities a standard shape.</p>
  <small>and let the useful parts survive transport changes</small>
</div>

---

<div class="eyebrow">6 · Shell is code too</div>

# Arbitrary shell execution is basically code mode

<div class="columns">

<div>

- Once a model can drive shell intelligently, it has crossed into a very real form of <strong>code mode</strong>.
- Bash is not “lesser” than code here — it is a compact, high-leverage execution surface.
- This is where skills get powerful:
  - reusable shell loops
  - environment setup
  - remote execution wrappers
  - task-specific tools generated on demand
- This is also where the <strong>HF jobs skill</strong> pattern becomes interesting: a skill can launch work into a more suitable environment instead of doing everything on the main thread.

</div>

<div class="center">

![w:720](./images/image-2.png)

</div>

</div>

---

<div class="eyebrow">7 · Apps SDK pattern</div>

# Sometimes the model should stop and the UI should take over

- A common failure mode:
  1. user asks for navigation or retrieval
  2. tools fetch the answer
  3. the model then spends expensive output tokens reprocessing a result that was already good enough
- The Apps SDK pattern fixes this by letting the result become <strong>terminal for the user</strong>.
- That is not just UX polish — it is an efficiency pattern.

![w:1100](./images/image.png)

---

<div class="eyebrow">8 · Boundaries are blurring</div>

# Now the execution environment starts to move

<div class="card-grid">
  <div class="card">
    <h3>Client-side</h3>
    <p>Useful when local context matters and the user already trusts the environment.</p>
  </div>
  <div class="card">
    <h3>Remote MCP with auth</h3>
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

<div class="eyebrow">9 · ACP</div>

# Agent Client Protocol lets us bundle the experience

<div class="comparison">
  <div class="panel">
    <h3>What ACP gives us</h3>
    <ul>
      <li>Bundle model + tools + skills + interaction shape</li>
      <li>Streaming</li>
      <li>Sessions and rehydration</li>
      <li>Observability and control points</li>
    </ul>
  </div>
  <div class="panel">
    <h3>Why it matters here</h3>
    <ul>
      <li>MCP handles tool interaction well</li>
      <li>ACP helps distribute a fuller agent experience</li>
      <li>Useful when we care about continuity, controls, and UX quality</li>
      <li>Feels like the right wrapper for richer agent bundles</li>
    </ul>
  </div>
</div>

---

<div class="eyebrow">10 · Open Responses</div>

# Open Responses wants to bundle again — from the other side

<div class="comparison">
  <div class="panel">
    <h3>What gets bundled</h3>
    <ul>
      <li>Inference</li>
      <li>State</li>
      <li>Tool calling</li>
      <li>Code and search surfaces</li>
      <li>Potentially remote MCP itself</li>
    </ul>
  </div>
  <div class="panel">
    <h3>Why that matters</h3>
    <ul>
      <li>It changes where orchestration can live</li>
      <li>It pressures protocol boundaries from inside the model API</li>
      <li>It competes with some agent harness responsibilities</li>
      <li>It also makes some workflows dramatically easier to ship</li>
    </ul>
  </div>
</div>

<blockquote>
<p>ACP bundles from the client side. Open Responses bundles from the inference side. MCP still matters because both need a coherent way to talk to tools and remote capability surfaces.</p>
</blockquote>

---

<div class="eyebrow">Closing</div>

# So the story is not “MCP won” or “MCP lost”

<div class="closing-grid">
  <div class="panel">
    <h3>The through-line</h3>
    <ul>
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

# Questions

### and which layer should own the loop?
