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
layout: default
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
layout: default
---

<IntroVideo />


---
layout: default
---

# MCP At Hugging Face

<div class="hf-mcp-slide">
<section class="hf-mcp-copy">
<div class="kicker">hugging face mcp server</div>
<h2>One protocol surface for the Hub</h2>

<ul>
<li>Search models, datasets, papers, Spaces, and docs</li>
<li>Inspect repos and metadata from the agent loop</li>
<li>Call hosted tools and demos through MCP</li>
</ul>

<p>Discovery · context · execution</p>
</section>

<aside class="hf-mcp-video-frame deck-panel">
<HfMcpServerVideo />
</aside>
</div>


---


# Protocol Features

<div class="protocol-diagram">
  <ProtocolStack />
</div>

---
# What's new

## Stateless


---
layout: default
kicker: streamable http
---

# Remote MCP through a load balancer

<div class="remote-mcp-diagram">
  <RemoteMcpLoadBalancer />
</div>

---
layout: default
---

# Transport Evolution

<div class="spec-timeline-diagram">
  <McpSpecTransportTimeline />
</div>

---
layout: default
---

<div class="traffic-chart-slide">
  <McpRemoteTrafficChart client="Claude Code" title="Claude Code" />
</div>

---
layout: default
---

<div class="traffic-chart-slide">
  <McpRemoteTrafficChart client="Codex" title="Codex" />
</div>

---
layout: default
---

<div class="traffic-chart-slide">
  <McpRemoteTrafficChart client="Codex" range-mode="claude" title="Codex" subtitle="same date range as Claude Code" />
</div>

---
layout: default
---

<div class="weekly-activity-slide">
  <McpWeeklyActivityChart />
</div>

---



# What's gone?


---
