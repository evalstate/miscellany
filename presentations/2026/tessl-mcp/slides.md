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

# MCP Protocol - Transports

<div class="protocol-diagram">
  <ProtocolStack />
</div>

MCP is bidirectional.

---

# Remote MCP through a load balancer

<div class="remote-mcp-diagram">
  <RemoteMcpLoadBalancer />
</div>

---


# Transport Evolution

<div class="spec-timeline-diagram">
  <McpSpecTransportTimeline />
</div>

---


<div class="traffic-chart-slide">
  <McpRemoteTrafficChart client="Claude Code" title="Claude Code" />
</div>

---


<div class="traffic-chart-slide">
  <McpRemoteTrafficChart client="Codex" title="Codex" />
</div>

---


<div class="traffic-chart-slide">
  <McpRemoteTrafficChart client="Codex" range-mode="claude" title="Codex" subtitle="same date range as Claude Code" />
</div>

---


<div class="weekly-activity-slide">
  <McpWeeklyActivityChart />
</div>

---


<div class="conversion-chart-slide">
  <SessionConversionChart />
</div>

---

# Understanding Activity

## Initializations are a bad proxy for use

- Can't measure ambient installation (caching, tool search)
- Doesn't correlate to Tool Calls

## Tool Calls: More != Better

- High Tool Call volume may indicate poor tool design or discovery
- Client behaviour can be unpredictable

## Session Conversion is preferred

- Tells us Clients that have connected at least once.

---

# Some Issues

MCP is noisy.
MCP is complicated.

---

# 2026-07-28 Specification: `The Stateless Core`

---

# Simplifications

- No longer allow Server to Client initiated requests (MCP WebCam)
- Remove Sampling, Roots (and Logging)

---

# Stateless Protocol

## Remove Initialize and Mcp-Session-Id

## Add /discover endpoint

---

# Cache Control

## Tool List Changed 

---

# HTTP Headers

