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
<li>Open Responses Maintainer</li>
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

# What we are talking about today

## MCP at Hugging Face and Current Transports

## Client Behaviour and Analytics

## Issues Related to MCP Implementation

## New MCP Specification Changes



---

# MCP At Hugging Face

<div class="hf-mcp-slide">
<section class="hf-mcp-copy">
<p class="kicker">MCP server</p>
<h2>One protocol surface for the Hugging Face ecosystem</h2>

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

# Current MCP is bi-directional 

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

<div class="traffic-chart-slide">
  <McpRemoteTrafficChart client="Claude Code" title="Claude Code" />
</div>

---

# Understanding Activity

## Interactive vs Agentic Workloads

- Session Length
- Burstiness

## Initializations are a bad proxy for use

- Can't measure ambient installation (caching, tool search)
- Doesn't correlate to Tool Calls

## Tool Calls: More != Better

- High Tool Call volume may indicate poor tool design or discovery
- Client behaviour can be unpredictable

## Session Conversion is preferred

- Clients that connect and make at least one tool call.
- Reduces skew of erratic clients or excessive testing


---

<div class="weekly-activity-slide chart-slide">
  <header class="chart-slide__header">
    <div>
      <h1>Weekly MCP activity</h1>
      <p>Initialization requests as bars · tool calls as line</p>
    </div>
  </header>
  <McpWeeklyActivityChart />
</div>


---

<div class="protocol-efficiency-slide chart-slide">
  <header class="chart-slide__header">
    <div>
      <p class="chart-slide__kicker">MCP Protocol Efficiency</p>
      <h1>Overhead in 10M Protocol Messages</h1>
    </div>
  </header>
  <McpProtocolEfficiency />
</div>

---

<div class="conversion-chart-slide chart-slide">
  <header class="chart-slide__header">
    <div>
      <h1>Daily session conversion</h1>
      <p>Session → query conversion rate · 3-day converted-session average</p>
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

# Remote MCP through a load balancer

<div class="remote-mcp-diagram">
  <RemoteMcpLoadBalancerStoryboard />
</div>

---

# Main Issues with Statefulness and Server-to-Client Comms

- "Sticky" sessions in the load balancer
  - Scalability
  - Fault Tolerance
  - In-Place Changes
  - 
- Maintaining open connections speculatively is expensive
- SSE "cut-off" times in popular hosting platforms
- Fault Tolerance and Scalability Concerns
- Elicitation and Sampling require Server->Client channel open
- Session State not well defined (e.g STDIO, Tool List)
- Basic analytics requires handling Sessions

---
layout: section
---

# 2026-07-28 Specification: `The Stateless Core`

---

# New Specification

<div class="spec-timeline-diagram">
  <McpSpecTransportTimeline variant="after" />
</div>


---

# Simplifications

## 80% of the complexity is in 20% of the Protocol...

## No longer allow Server to Client initiated requests (mcp-webcam) - SEP2260

- Server cannot make an unsolicited call to the Client.
  Server Requests must be contained within a Client initiated request.
- Removes need for unreliable "GET" SSE Handler on the Server

## Deprecate Sampling, Roots (and Logging)

- Simplify protocol surface for underutilized features.


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
<div class="http-json-line http-json-line--indent-3"><em>"protocolVersion"</em>: <strong>"2026-07-30"</strong>,</div>
<div class="http-json-line http-json-line--indent-3"><em>"clientInfo"</em>: { <em>"name"</em>: <strong>"ExampleClient"</strong> }</div>
<div class="http-json-line http-json-line--indent-2">}</div>
<div class="http-json-line http-json-line--indent">}</div>
<div class="http-json-line">}</div>
<div class="http-json-line stateless-discovery-json__gap">← response</div>
<div class="http-json-line">{</div>
<div class="http-json-line http-json-line--indent"><em>"result"</em>: {</div>
<div class="http-json-line http-json-line--indent-2"><em>"supportedVersions"</em>: [<strong>"2026-07-30"</strong>],</div>
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

# Cache Control

## Tool List Changed

- Tool Metadata can be acquired infrequently and shared
- Supports "per user" MCP Server configurations too
- Also allows configuration through `mcp-server.json` files 

---

# Multi Round-Trip Request

## Problem / Solution

Moving to a stateless protocol means that state-based turn taking doesn't apply

Return `inputRequired` rather than SSE Stream.



---

# SEP-2243: The problem

<div class="http-standardization-problem">
  <HttpRouteMap mode="problem" />
  <HttpHeaderExample variant="problem" />
</div>

---

# SEP-2243: Tool metadata

<div class="http-standardization-schema">
  <HttpHeaderExample variant="tool" />
</div>

---

# SEP-2243: HTTP headers make it routable

<div class="http-standardization-problem">
  <HttpRouteMap mode="solution" />
  <HttpHeaderExample variant="solution" />
</div>

---

# Migration Path

Draft Specification Release : Now
Beta SDKs : Soon
Planned Release Date  : 

---

# Related SEPs



---

<div class="traffic-chart-slide">
  <McpRemoteTrafficChart client="Codex" title="Codex" />
</div>

---

<div class="traffic-chart-slide">
  <McpRemoteTrafficChart client="Codex" range-mode="claude" title="Codex" subtitle="same date range as Claude Code" />
</div>
