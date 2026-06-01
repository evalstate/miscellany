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

<ul>
<li>Gateway to multi-modal inference</li>
<li>Conduct Research, Inspect Datasets, Find Models</li>
<li>Run and manage sandboxes</li>
<li>Allow Authenticated and Unauthenticated access</li>
</ul>

<code>Qwen 3.5-35B-A3B</code> <br/>
<code>Flux.1-Krea-Dev </code> <br/>
<code>Qwen-Edit-2509-Multiple-angles-LoRA</code> <br/>
<code>Wan2.2 First/Last Frame</code>

</section>

<aside class="hf-mcp-video-frame deck-panel">
<HfMcpServerVideo />
</aside>
</div>

---

# MCP Protocol is bidirectional

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

## Open Source

- Clients, Versions and Capabilities
- Track availability of Features and Extensions
- 


---

# Remote MCP through a load balancer

<div class="remote-mcp-diagram">
  <RemoteMcpLoadBalancerStoryboard />
</div>

---

# Main Issues with Statefulness and Server-to-Client Comms

- "Sticky" sessions in the load balancer
- Maintaining open connections speculatively is expensive
- SSE "cut-off" times in popular hosting platforms
- Fault Tolerance and Scalability Concerns
- Elicitation and Sampling require 

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


- No longer allow Server to Client initiated requests (MCP WebCam) - SEP2260
- Deprecate Sampling, Roots (and Logging)
- Removes the need for a "GET" handler on the MCP Server

---


# Stateless Protocol

## Remove Initialize and Mcp-Session-Id

## Add `/discover` endpoint

- Can be helpful for UX

## Use model-driven state handles

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

# HTTP Standardization: Problem

<div class="http-standardization-problem">
  <HttpRouteMap />

  <section class="http-request-panel deck-panel">
    <h2>MCP over HTTP</h2>
    <div class="http-request-line"><strong>POST /mcp/ HTTP/1.1</strong><span>Host: mcp-server.example</span></div>
    <pre class="http-json"><code>{
  "jsonrpc": "2.0",
  "method": <mark>"tools/call"</mark>,
  "params": {
    "name": <mark>"spanner.execute_sql"</mark>,
    "arguments": {
      "project": <mark>"senseai-prod"</mark>,
      "region": <mark>"us-west1"</mark>,
      "instance": <mark>"finance-db-01"</mark>,
      "query": "SELECT ..."
    }
  }
}</code></pre>
  </section>
</div>

---

# Migration Path



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
