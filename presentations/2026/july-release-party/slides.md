---
theme: default
layout: intro
title: July Release Party
author: Shaun Smith
info: |
  July Release Party presentation.
class: deck-root
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
# transition: fade-out
mdc: true
fonts:
  sans: Source Sans 3
  mono: IBM Plex Mono
---


<div class="brand-machine-wrap">
  <BrandSlotMachine />
</div>

<!--
AAIF and the AAIF logo design are registered trademarks of the Linux Foundation.
Click the imagery to replay the animation.
-->

---


# July Release Party

## What shipped, what changed, and what comes next

::meta::

Shaun Smith · July 2026

---

<div class="about-intro">
<section>
<div class="kicker">about me</div>

<h1><span>Shaun Smith</span><span class="about-handle">@evalstate</span></h1>

<ul class="about-points">
<li>Open Source @ Hugging Face</li>
<li>MCP Maintainer and Moderator</li>
<li>huggingface/mcp</li>
<li>huggingface/skills</li>
<li>huggingface/upskill</li>
<li>Maintainer of <span class="about-repo">fast-agent</span></li>
</ul>

<table class="about-social-table">
<tbody>
<tr>
<td class="social-icon-cell"><img class="social-icon social-icon-hf" src="/brand/hugging-face.svg" alt="Hugging Face" /></td>
<td>huggingface.co/evalstate</td>
</tr>
<tr>
<td class="social-icon-cell"><img class="social-icon social-icon-github" src="/intro/github-mark.svg" alt="GitHub" /></td>
<td>github.com/evalstate</td>
</tr>
<tr>
<td class="social-icon-cell"><img class="social-icon social-icon-x" src="/intro/x-mark.svg" alt="X" /></td>
<td>x.com/evalstate</td>
</tr>
</tbody>
</table>
</section>

<aside class="about-logo-panel deck-panel">
<img class="about-hf-logo" src="/brand/hugging-face.svg" alt="Hugging Face" />
<div class="about-logo-divider"></div>
<img class="about-mcp-logo" src="/brand/mcp-symbol-black.svg" alt="Model Context Protocol" />
</aside>
</div>

---

<IntroVideo />

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

# Legacy MCP is fully bi-directional

<div class="protocol-diagram">
  <ProtocolStack />
</div>


---

# Current Transports

<div class="spec-timeline-diagram">
  <McpSpecTransportTimeline variant="before" />
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

# Scaling MCP in Production

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
layout: default
class: ported-section-slide
---

# What's new

## 2026-07-28 Specification

---

# 2026-07-28 Specification

<div class="spec-timeline-diagram">
  <McpSpecTransportTimeline variant="after" />
</div>


---

# Simplifications (SEP-2260,SEP-2577)

<div class="simplifications-slide">
<section class="simplifications-copy">

  <div class="compact-point-list">
    <div class="simplifications-point simplifications-point--evidence">
      <div>
        <strong>No unsolicited Server → Client calls</strong>
        <span>Server requests must be contained inside a client-initiated request.</span>
      </div>
      <ClickableImagePopover
        class="simplifications-evidence deck-panel"
        src="/images/mcp-webcam.png"
        alt="mcp-webcam demo screenshot"
        prompt="enlarge"
      />
    </div>
    <div>
      <strong>Drop the fragile GET/SSE handler</strong>
      <span>Modern clients no longer need it; previous-version Streamable HTTP remains available by fallback.</span>
    </div>
    <div>
      <strong>Deprecate Roots, Sampling + Logging</strong>
      <span>Roots and Sampling remain available through MRTR during deprecation.</span>
    </div>
  </div>
</section>

<section class="simplifications-visual">
  <aside class="simplifications-protocol deck-panel">
    <SimplificationsProtocolRails />
  </aside>
</section>
</div>

---

# Remove Initialization Handshake (SEP-2575)

<div class="stateless-discovery-slide">
<section class="stateless-discovery-copy">



<dl class="compact-point-list">
<div>
<dt>Handshake Info to Data Layer</dt>
<dd>Protocol version, Client identity and Client capabilities move into namespaced JSON-RPC <code>_meta</code> fields on every request.</dd>
</div>
<div>
<dt>New <code>server/discover</code> endpoint</dt>
<dd>Optional Client call to learn the Server’s supported versions, capabilities and implementation metadata.</dd>
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
<div class="http-json-line">{ <em>"jsonrpc"</em>: <strong>"2.0"</strong>, <em>"id"</em>: <strong>"discover-1"</strong>,</div>
<div class="http-json-line http-json-line--indent"><em>"method"</em>: <mark>"server/discover"</mark>,</div>
<div class="http-json-line http-json-line--indent"><em>"params"</em>: { <mark>"_meta"</mark>: {</div>
<div class="http-json-line http-json-line--indent-2 http-json-line--meta"><em>"io.modelcontextprotocol/protocolVersion"</em>: <strong>"2026-07-28"</strong>,</div>
<div class="http-json-line http-json-line--indent-2 http-json-line--meta"><em>"io.modelcontextprotocol/clientInfo"</em>: { <em>"name"</em>: <strong>"Client"</strong>, <em>"version"</em>: <strong>"1.0"</strong> },</div>
<div class="http-json-line http-json-line--indent-2 http-json-line--meta"><em>"io.modelcontextprotocol/clientCapabilities"</em>: {}</div>
<div class="http-json-line">} } }</div>
<div class="http-json-line stateless-discovery-json__gap">← response</div>
<div class="http-json-line">{ <em>"jsonrpc"</em>: <strong>"2.0"</strong>, <em>"id"</em>: <strong>"discover-1"</strong>,</div>
<div class="http-json-line http-json-line--indent"><em>"result"</em>: {</div>
<div class="http-json-line http-json-line--indent-2"><em>"resultType"</em>: <strong>"complete"</strong>,</div>
<div class="http-json-line http-json-line--indent-2"><em>"supportedVersions"</em>: [<strong>"2026-07-28"</strong>],</div>
<div class="http-json-line http-json-line--indent-2 http-json-line--meta"><mark>"capabilities"</mark>: { <em>"tools"</em>: {}, <em>"resources"</em>: {}, <em>"prompts"</em>: {} },</div>
<div class="http-json-line http-json-line--indent-2 http-json-line--meta"><em>"serverInfo"</em>: { <em>"name"</em>: <strong>"Server"</strong>, <em>"version"</em>: <strong>"1.0"</strong> }</div>
<div class="http-json-line">} }</div>
</div>
</section>
</div>

---

# List Caching (SEP-2549)

<div class="cache-control-slide">
<section class="cache-control-copy">

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
<span>Relevant list-changed or resource-updated notifications make cached results stale immediately.</span>
</div>
</div>

</section>

<aside class="cache-scope-table deck-panel">
<div class="kicker">cacheScope</div>

| Value | Meaning |
| --- | --- |
| `"public"` | Does not contain user-specific data. Any client, gateway, or caching proxy may store and serve it to any user. |
| `"private"` | Contains user-specific data. Only the requesting user’s Client may cache it; shared caches must not serve it to another user. |

</aside>
</div>

---

# SEP-2322: Modern Elicitations

<div class="modern-elicitation-story deck-panel">
  <ModernElicitationFlow />
</div>

---

# SEP-2322: Request 1 Ends

<div class="elicitation-packet-slide elicitation-packet-slide--focused">
<section class="elicitation-packet-copy">
  <div class="compact-point-list">
    <div>
      <strong>A final response</strong>
      <span><code>input_required</code> closes request 1. Nothing remains open on the Server.</span>
    </div>
    <div>
      <strong>Ask the person</strong>
      <span>The keyed <code>elicitation/create</code> asks the Client to confirm the hourly cost.</span>
    </div>
    <div>
      <strong>Keep state opaque</strong>
      <span>The Client stores <code>requestState</code> without inspecting or changing it.</span>
    </div>
  </div>
</section>

<aside class="elicitation-packet-json deck-panel">
<div class="http-json http-json--packet">
  <div class="http-json-line elicitation-packet-json__gap">→ request 1</div>
  <div class="http-json-line">{ <em>"id"</em>: <strong>1</strong>, <em>"method"</em>: <mark>"tools/call"</mark>,</div>
  <div class="http-json-line http-json-line--indent"><em>"name"</em>: <strong>"hf.create_sandbox"</strong>,</div>
  <div class="http-json-line http-json-line--indent"><em>"arguments"</em>: { <em>"hardware"</em>: <strong>"t4-small"</strong> } }</div>

  <div class="http-json-line elicitation-packet-json__gap">← response 1 · <mark>request closed</mark></div>
  <div class="http-json-line">{ <em>"id"</em>: <strong>1</strong>, <em>"result"</em>: {</div>
  <div class="http-json-line http-json-line--indent"><mark><em>"resultType"</em>: <strong>"input_required"</strong></mark>,</div>
  <div class="http-json-line http-json-line--indent"><em>"inputRequests"</em>: {</div>
  <div class="http-json-line http-json-line--indent-2"><mark><strong>"confirm_cost"</strong></mark>: {</div>
  <div class="http-json-line http-json-line--indent-3"><em>"method"</em>: <strong>"elicitation/create"</strong>,</div>
  <div class="http-json-line http-json-line--indent-3"><em>"mode"</em>: <strong>"form"</strong>,</div>
  <div class="http-json-line http-json-line--indent-3"><em>"message"</em>:</div>
  <div class="http-json-line http-json-line--indent-4"><strong>"Create sandbox for <mark>$0.40/hour</mark>?"</strong>,</div>
  <div class="http-json-line http-json-line--indent-3"><em>"schema"</em>: { <em>"confirmed"</em>: <strong>"boolean"</strong> }</div>
  <div class="http-json-line http-json-line--indent-2">} },</div>
  <div class="http-json-line http-json-line--indent"><em>"requestState"</em>:</div>
  <div class="http-json-line http-json-line--indent-2"><mark><strong>"opaque-sbx-state"</strong></mark> } }</div>
</div>
</aside>
</div>

---

# SEP-2322: Request 2 Starts Fresh

<div class="elicitation-packet-slide elicitation-packet-slide--focused">
<section class="elicitation-packet-copy">
  <div class="compact-point-list">
    <div>
      <strong>A new request</strong>
      <span>Request 2 has a different id and repeats the original tool arguments.</span>
    </div>
    <div>
      <strong>The key matches</strong>
      <span><code>confirm_cost</code> identifies both the input request and its accepted response.</span>
    </div>
    <div>
      <strong>State returns exactly</strong>
      <span>The accepted confirmation and unchanged opaque state complete the retry.</span>
    </div>
  </div>

  <div class="elicitation-result-callout">
    <span>Result to carry forward</span>
    <strong>sbx-7f3c</strong>
    <small>created · ready</small>
  </div>
</section>

<aside class="elicitation-packet-json deck-panel">
<div class="http-json http-json--packet">
  <div class="http-json-line elicitation-packet-json__gap">→ request 2 · independent retry</div>
  <div class="http-json-line">{ <em>"id"</em>: <mark><strong>2 · new id</strong></mark>,</div>
  <div class="http-json-line http-json-line--indent"><em>"method"</em>: <strong>"tools/call"</strong>,</div>
  <div class="http-json-line http-json-line--indent"><em>"name"</em>: <strong>"hf.create_sandbox"</strong>,</div>
  <div class="http-json-line http-json-line--indent"><em>"arguments"</em>: { <em>"hardware"</em>: <strong>"t4-small"</strong> },</div>
  <div class="http-json-line http-json-line--indent"><em>"inputResponses"</em>: {</div>
  <div class="http-json-line http-json-line--indent-2"><mark><strong>"confirm_cost"</strong></mark>: {</div>
  <div class="http-json-line http-json-line--indent-3"><mark><em>"action"</em>: <strong>"accept"</strong></mark>,</div>
  <div class="http-json-line http-json-line--indent-3"><em>"content"</em>: { <mark><em>"confirmed"</em>: <strong>true</strong></mark> }</div>
  <div class="http-json-line http-json-line--indent-2">} },</div>
  <div class="http-json-line http-json-line--indent"><em>"requestState"</em>:</div>
  <div class="http-json-line http-json-line--indent-2"><mark><strong>"opaque-sbx-state"</strong></mark> }</div>

  <div class="http-json-line elicitation-packet-json__gap">← response 2</div>
  <div class="http-json-line">{ <em>"id"</em>: <strong>2</strong>, <em>"result"</em>: {</div>
  <div class="http-json-line http-json-line--indent"><mark><em>"resultType"</em>: <strong>"complete"</strong></mark>,</div>
  <div class="http-json-line http-json-line--indent"><em>"text"</em>:</div>
  <div class="http-json-line http-json-line--indent-2"><strong>"Sandbox <mark>sbx-7f3c</mark> created"</strong> } }</div>
</div>
</aside>
</div>

---

# Replace Sessions with State Handles

<div class="state-handle-slide">
<div class="kicker">SEP-2567 · explicit application state</div>
<div class="state-handle-flow">
<div class="state-handle-card deck-panel">
<span>Previous tool result</span>
<strong>sbx-7f3c</strong>
<small>server-minted sandbox ID</small>
</div>
<div class="state-handle-arrow" aria-hidden="true">&rarr;</div>
<div class="state-handle-call deck-panel">
<span>Next tool call</span>
<code>hf.run_in_sandbox({</code>
<code>&nbsp;&nbsp;sandbox_id: <mark>"sbx-7f3c"</mark>,</code>
<code>&nbsp;&nbsp;command: "python train.py"</code>
<code>})</code>
</div>
</div>
<div class="state-handle-summary">
<span><s>Mcp-Session-Id</s></span>
<strong>State becomes explicit tool data.</strong>
<small>No hidden protocol session to create, route, or resume.</small>
</div>
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
<p class="kicker">What’s Next</p>

# Migration Path

<div class="migration-path-list">
<div><span>Now</span><strong>Release Candidate Specification</strong></div>
<div><span>30 Jun 2026</span><strong>Beta SDKs</strong></div>
<div><span>28 Jul 2026</span><strong>Planned release date</strong></div>
</div>
</div>

---
layout: intro
---

# Thank you

## Questions?

::meta::

huggingface.co/evalstate · github.com/evalstate

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
