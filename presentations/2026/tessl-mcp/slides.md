---
theme: default
title: Tessl + MCP
author: Shaun Smith
info: |
  Starter Slidev deck for Tessl + MCP.
class: deck-root
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
# transition: fade-out
mdc: true
fonts:
  sans: Inter
  mono: JetBrains Mono
---

<div class="title-intro">
<div class="title-intro-mark">MCP</div>
<main>
<h1>Connecting<br />Context</h1>
<h2>The Future of MCP Transports</h2>
<p>Shaun Smith · Hugging Face · 2026</p>
</main>
</div>

---

<div class="about-intro">
<section>
<div class="kicker">about me</div>

<h1>Shaun Smith <code>@evalstate</code></h1>

<ul class="about-points">
<li>Open Source @ Hugging Face</li>
<li>MCP Maintainer and Moderator</li>
<li>huggingface/mcp</li>
<li>huggingface/skills</li>
<li>huggingface/upskill</li>
<li>Maintainer of <code>fast-agent</code></li>
</ul>

<table class="about-social-table">
<tbody>
<tr>
<td class="social-icon-cell"><img class="social-icon social-icon-hf" :src="'intro/huggingface-mark-logo.svg'" alt="Hugging Face" /></td>
<td>huggingface.co/evalstate</td>
</tr>
<tr>
<td class="social-icon-cell"><img class="social-icon social-icon-github" :src="'intro/github-mark.svg'" alt="GitHub" /></td>
<td>github.com/evalstate</td>
</tr>
<tr>
<td class="social-icon-cell"><img class="social-icon social-icon-x" :src="'intro/xcom-logo-black.png'" alt="X" /></td>
<td>x.com/evalstate</td>
</tr>
</tbody>
</table>
</section>

<aside class="about-logo-panel deck-panel">
<img class="about-hf-logo" :src="'intro/hf_logo.svg'" alt="Hugging Face" />
<div class="about-logo-divider"></div>
<img class="about-mcp-logo" :src="'intro/mcp-icon.svg'" alt="Model Context Protocol" />
</aside>
</div>

---

<IntroVideo />

---

<div class="agenda-slide">
<p class="kicker">today</p>

# Topics

<div class="agenda-list">
<div>MCP at Hugging Face</div>
<div>Client Behaviour and Analytics</div>
<div>Issues related to MCP implementation</div>
<div>New! MCP specification changes</div>
</div>
</div>



---

# MCP At Hugging Face

<div class="hf-mcp-slide">
<section class="hf-mcp-copy">
<h2>The Hub for Agents and Assistants</h2>

<div class="hf-mcp-capabilities">
<div><strong>Inference gateway</strong><span>Route agents to multimodal models and hosted endpoints.</span></div>
<div><strong>Research workspace</strong><span>Inspect datasets, find models, and run experiments.</span></div>
<div><strong>Sandboxes</strong><span>Create and manage isolated execution environments.</span></div>
<div><strong>Access modes</strong><span>Support authenticated and unauthenticated workflows.</span></div>
</div>

<div class="hf-mcp-models" aria-label="Example models">
<span>Qwen 3.5-35B-A3B</span>
<span>Flux.1-Krea-Dev</span>
<span>Qwen-Edit LoRA</span>
<span>Wan2.2 First/Last Frame</span>
</div>
</section>

<aside class="hf-mcp-video-frame deck-panel">
<HfMcpServerVideo />
</aside>
</div>

---

# Today,  MCP design is bi-directional 

<div class="protocol-diagram">
  <ProtocolStack />
</div>


---

# Current Transports

<div class="spec-timeline-diagram">
  <McpSpecTransportTimeline variant="before" />
</div>

---

<div class="weekly-activity-slide chart-slide">
  <header class="chart-slide__header">
    <div>
      <h1>Streamable HTTP adoption</h1>
      <h2>Proportion of <code>mcp-remote</code> usage</h2>
    </div>
  </header>
  <McpRemoteNoFallbackChart />
</div>

---

<div class="traffic-chart-slide chart-slide">
  <header class="chart-slide__header">
    <div>
      <h1>Claude Code</h1>
      <h2>Weekly <code>mcp-remote</code> share · usage index </h2>
    </div>
  </header>
  <McpRemoteTrafficChart client="Claude Code" :showHeader="false" />
</div>

---

<div class="weekly-activity-slide chart-slide">
  <header class="chart-slide__header">
    <div>
      <h1>Weekly MCP activity</h1>
    </div>
  </header>
  <McpWeeklyActivityChart />
</div>

---

# Understanding Activity

<div class="understanding-activity-slide">
<section class="understanding-activity-copy">

<div class="compact-point-list">
<div>
<strong>Initializations are a bad proxy</strong>
<span>Can’t measure ambient installation or caching; doesn’t correlate to tool calls.</span>
</div>

<div>
<strong>Tool calls: more ≠ better</strong>
<span>High volume may indicate poor tool design, discovery problems, or erratic clients.</span>
</div>

<div>
<strong>Session conversion is preferred</strong>
<span>Clients that connect and make at least one tool call reduce skew from testing and idle installs.</span>
</div>

<div>
<strong>Workload shape still matters</strong>
<span>Session length and burstiness distinguish interactive use from agentic loops.</span>
</div>
</div>
</section>

<aside class="understanding-activity-table">
  <ClientConversionEfficiencyTable />
</aside>
</div>

---

<div class="conversion-chart-slide chart-slide">
  <header class="chart-slide__header">
    <div>
      <h1>Daily session conversion</h1>
      <h2>Session → query conversion rate</h2>
    </div>
  </header>
  <SessionConversionChart />
</div>

---

# MCP Clients Dataset

<div class="clients-dataset-slide text-image-slide">
  <section class="text-image-slide__copy">
    <p class="kicker">Open dataset</p>

  <a class="dataset-link-card" href="https://hf.co/datasets/evalstate/mcp-clients">
    <span>hf.co/datasets/evalstate</span>
    <strong>mcp-clients</strong>
  </a>

  <div class="compact-point-list">
    <div>
      <strong>Clients</strong>
      <span>names, versions, last-seen activity</span>
    </div>
    <div>
      <strong>Capabilities</strong>
      <span>tools, prompts, roots, sampling, elicitation</span>
    </div>
    <div>
      <strong>Extensions</strong>
      <span>track emerging feature support over time</span>
    </div>
  </div>
  </section>

  <figure class="dataset-screenshot deck-panel">
    <img :src="'/images/clients-data.png'" alt="Hugging Face Data Studio table for the mcp-clients dataset" />
  </figure>
</div>


---

# Scaling MCP in Production...

<div class="remote-mcp-diagram">
  <RemoteMcpLoadBalancerStoryboard />
</div>



---

<div class="protocol-efficiency-slide chart-slide">
  <header class="chart-slide__header">
    <div>
      <h1>Examining 10M Protocol Messages</h1>
    </div>
  </header>
  <McpProtocolEfficiency />
</div>





---

# Main Issues with Statefulness

<div class="statefulness-slide">
<section class="statefulness-list statefulness-list--accent">
<h2>Operational coupling</h2>
<ul>
<li><strong>“Sticky” sessions in the load balancer</strong>
  <ul>
  <li>scalability</li>
  <li>fault tolerance</li>
  <li>in-place changes</li>
  </ul>
</li>
<li><strong>Speculative open connections are expensive</strong></li>
<li><strong>SSE cut-off times</strong> on popular hosting platforms</li>
</ul>
</section>

<section class="statefulness-list">
<h2>Protocol ambiguity</h2>
<ul>
<li><strong>Elicitation and Sampling</strong> require a Server → Client channel to stay open</li>
<li><strong>Session state is not well defined</strong>
  <ul>
  <li>STDIO lifecycle</li>
  <li>tool list and capability changes</li>
  </ul>
</li>
<li><strong>Basic analytics</strong> requires handling sessions</li>
</ul>
</section>
</div>

---
layout: section
---

# 2026-07-28 Specification: `The Stateless Core`

---

# 2026-07-28 Release Candidate

<div class="spec-timeline-diagram">
  <McpSpecTransportTimeline variant="after" />
</div>


---

# SEP-2260, SEP-2257: "Simplifications"

<div class="simplifications-slide">
<section class="simplifications-copy">

  ## Simplify and Deprecate

  <div class="compact-point-list">
    <div>
      <strong>No unsolicited Server → Client calls</strong>
      <span>Server requests must be contained inside a client-initiated request.</span>
    </div>
    <div>
      <strong>Drop the fragile GET/SSE handler</strong>
      <span>No speculative open channel just in case the server wants to call back.</span>
    </div>
    <div>
      <strong>Deprecate Sampling + Roots</strong>
      <span>Retire underused protocol surface instead of standardizing around it.</span>
    </div>
  </div>
</section>

<section class="simplifications-visual">
  <ClickableImagePopover
    class="simplifications-webcam deck-panel"
    src="/images/mcp-webcam.png"
    alt="mcp-webcam demo screenshot"
  />

  <aside class="simplifications-protocol deck-panel">
    <SimplificationsProtocolRails />
  </aside>
</section>
</div>


---



# SEP-2575: Make MCP Stateless

<div class="stateless-discovery-slide">
<section class="stateless-discovery-copy">


## Remove Initialization Handshake

<dl class="compact-point-list">
<div>
<dt>Handshake Info to Data Layer</dt>
<dd>Version, Capability and Client identity move into the JSON-RPC <code>_meta</code> envelope on each request/response.</dd>
</div>
<div>
<dt>New <code>server/discover</code> endpoint</dt>
<dd>Optional Client Probe to share Capability information for compatibility/User Experience reasons.</dd>
</div>
<div>
<dt>New <code>subscriptions/listen</code> endpoint</dt>
<dd>Endpoint to allow Client to initiate a notification stream for Resource Subscriptions or List Changed events</dd>
</div>
</dl>

</section>

<section class="stateless-discovery-json deck-panel">
<div class="http-json http-json--packet">
<div class="http-json-line stateless-discovery-json__gap">→ request</div>
<div class="http-json-line">{</div>
<div class="http-json-line http-json-line--indent"><em>"jsonrpc"</em>: <strong>"2.0"</strong>,</div>
<div class="http-json-line http-json-line--indent"><em>"method"</em>: <mark>"server/discover"</mark>,</div>
<div class="http-json-line http-json-line--indent"><em>"params"</em>: {</div>
<div class="http-json-line http-json-line--indent-2"><mark>"_meta"</mark>: {</div>
<div class="http-json-line http-json-line--indent-3"><em>"protocolVersion"</em>: <strong>"2026-07-28"</strong>,</div>
<div class="http-json-line http-json-line--indent-3"><em>"clientInfo"</em>: { <em>"name"</em>: <strong>"ExampleClient"</strong> }</div>
<div class="http-json-line http-json-line--indent-2">}</div>
<div class="http-json-line http-json-line--indent">}</div>
<div class="http-json-line">}</div>
<div class="http-json-line stateless-discovery-json__gap">← response</div>
<div class="http-json-line">{</div>
<div class="http-json-line http-json-line--indent"><em>"result"</em>: {</div>
<div class="http-json-line http-json-line--indent-2"><em>"supportedVersions"</em>: [<strong>"2026-07-28"</strong>],</div>
<div class="http-json-line http-json-line--indent-2"><mark>"capabilities"</mark>: {</div>
<div class="http-json-line http-json-line--indent-3"><em>"tools"</em>: {},</div>
<div class="http-json-line http-json-line--indent-3"><em>"resources"</em>: {},</div>
<div class="http-json-line http-json-line--indent-3"><em>"prompts"</em>: {}</div>
<div class="http-json-line http-json-line--indent-2">},</div>
<div class="http-json-line http-json-line--indent-2"><em>"serverInfo"</em>: { <em>"name"</em>: <strong>"ExampleServer"</strong> }</div>
<div class="http-json-line http-json-line--indent">}</div>
<div class="http-json-line">}</div>
</div>
</section>
</div>

---

# SEP-2459: Cache Control

<div class="cache-control-slide">
<section class="cache-control-copy">

## Cacheable Results

<div class="compact-point-list">
<div>
<strong>Applies to discovery and reads</strong>
<span><code>tools/list</code>, <code>prompts/list</code>, <code>resources/list</code>, <code>resources/templates/list</code>, and <code>resources/read</code>.</span>
</div>
<div>
<strong><code>ttlMs</code> is freshness</strong>
<span>Clients may consider the result fresh until <code>received + ttlMs</code>; <code>0</code> means immediately stale.</span>
</div>
<div>
<strong>Notifications invalidate</strong>
<span>TTL avoids unnecessary refetches between changes; list-changed notifications make cached results stale immediately.</span>
</div>
</div>

</section>

<aside class="cache-scope-table deck-panel">
<div class="kicker">cacheScope</div>

| Value | Meaning |
| --- | --- |
| `"public"` | Does not contain user-specific data. Any client, gateway, or caching proxy may store and serve it to any user. |
| `"private"` | May contain caller-specific data. Reuse only within the same authorization context; never share across access tokens. |

</aside>
</div>

---

# SEP-2322: Stateful Elicitations

<div class="mrtr-contrast-slide">
<section class="mrtr-contrast-copy">

## Before: wait for the answer

<div class="compact-point-list">
<div>
<strong>SSE POST response stream stays open</strong>
<span>The server asks for more input on the original stream.</span>
</div>
<div>
<strong>Client POSTs the answer</strong>
<span>The elicitation response is a new JSON-RPC HTTP request.</span>
</div>
<div>
<strong>Load balancer parses JSON</strong>
<span>It must route by JSON-RPC request id, or use shared storage.</span>
</div>
</div>

</section>

<aside class="mrtr-stateful-flow deck-panel">
<div class="kicker">stateful turn-taking</div>
<div class="mrtr-flow-row">
  <div class="mrtr-node mrtr-node--client">Client</div>
  <div class="mrtr-arrow">POST tools/call</div>
  <div class="mrtr-node mrtr-node--lb">LB</div>
  <div class="mrtr-arrow">route</div>
  <div class="mrtr-node">A</div>
</div>
<div class="mrtr-sse">elicitation over SSE · Server A waits…</div>
<div class="mrtr-flow-row">
  <div class="mrtr-node mrtr-node--client">Client</div>
  <div class="mrtr-arrow">POST answer</div>
  <div class="mrtr-node mrtr-node--warn">LB</div>
  <div class="mrtr-arrow">inspect id</div>
  <div class="mrtr-node">A</div>
</div>
<div class="mrtr-problem">Routing depends on the JSON-RPC request id</div>
</aside>
</div>

---

# SEP-2322: Stateless Elicitations

<div class="mrtr-cumulative-slide">
<section class="mrtr-cumulative-copy">

## After: retry with context

<p class="mrtr-lede">The server returns <code>input_required</code>. The client retries with everything learned so far.</p>

<div class="mrtr-field-strip">
<span>original arguments</span>
<span>inputResponses</span>
<span>requestState?</span>
</div>

</section>

<aside class="mrtr-cumulative-flow deck-panel">
<div class="mrtr-step">
  <strong>1</strong>
  <span>Client sends <code>tools/call</code></span>
</div>
<div class="mrtr-step mrtr-step--accent">
  <strong>2</strong>
  <span>Server returns <code>resultType: "input_required"</code></span>
</div>
<div class="mrtr-step">
  <strong>3</strong>
  <span>Client collects elicitation / sampling / roots responses</span>
</div>
<div class="mrtr-step mrtr-step--final">
  <strong>4</strong>
  <span>Client replays <code>tools/call</code> with cumulative input</span>
</div>
</aside>
</div>

---

# SEP-2243: HTTP Standardization

<div class="http-standardization-problem">
  <HttpRouteMap mode="problem" />
  <HttpHeaderExample variant="problem" />
</div>

---

# SEP-2243: Tool Data in HTTP Headers 

<div class="http-standardization-schema">
  <HttpHeaderExample variant="tool" />
</div>

---

# SEP-2243: Routable MCP Traffic 

<div class="http-standardization-problem">
  <HttpRouteMap mode="solution" />
  <HttpHeaderExample variant="solution" />
</div>

---

# Scaling MCP in Production

<div class="remote-mcp-diagram remote-mcp-diagram--wide">
  <RoutableMcpTrafficBurst />
</div>

---

<div class="migration-path-slide">
<p class="kicker">migration path</p>

# Migration Path

<div class="migration-path-list">
<div><span>Now</span><strong>Release Candidate Specification</strong></div>
<div><span>30 Jun 2026</span><strong>Beta SDKs</strong></div>
<div><span>28 Jul 2026</span><strong>Planned release date</strong></div>
</div>
</div>

---

<div class="hackmonty-slide">

# Hugging Face sponsors Hack Monty

<img :src="'/images/hackmonty.png'" alt="Hack Monty" />

</div>
