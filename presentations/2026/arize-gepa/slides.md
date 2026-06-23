---
theme: default
title: Optimizing Skills and Tools, with Vision and GEPA
author: Shaun Smith
info: |
  Starter Slidev deck for Optimizing Skills and Tools, with Vision and GEPA.
class: deck-root
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
transition: slide-left
mdc: true
fonts:
  sans: Inter
  mono: JetBrains Mono

---

<div class="title-intro">
<div class="title-intro-mark">GEPA</div>
<main>
<h1>Optimizing<br />Skills and Tools</h1>
<h2>with Vision and GEPA</h2>
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
layout: default
kicker: next
---

# Presentation shape

- Audience and desired outcome
- What skill/tool optimization changes about agent workflows
- Where vision belongs in the evaluation loop
- What GEPA adds to prompt and policy refinement
- Demo path and visual story
- Key visuals worth building as reusable components
- Closing takeaway

---
layout: default
kicker: core loop
class: gepa-loop-slide
---

# GEPA loop

<GepaLoop />

---
layout: default
class: story-score-slide
---

# Score breakdown

<StoryScoreBreakdown />

---
layout: default
class: gepa-run-slide
---

# GEPA improves what it sees

<GepaRunExplorer />

---
layout: center
class: text-center
---

---

# Score

Prompt: "Write a story".

Score: 

NOTE --> Higher is Better


---

# HTML Skill

# Labelling

---

# Generating Labels

Tempting to use an LLM _but_


---

# Benchmark

Does it do the same thing twice?

Does the score transfer to my held out set?


---

# Other Uses

<div class="deck-grid deck-grid--three mt-8">
  <div class="deck-card">
    <h2>1. Tool Optimisation</h2>
    <p>Keep narrative in <code>slides.md</code>; use layouts and shared CSS before custom components.</p>
  </div>
  <div class="deck-card">
    <h2>2. Code Generation</h2>
    <p>Run deterministic layout checks before asking a vision model to judge aesthetics.</p>
  </div>
  <div class="deck-card">
    <h2>3. Labelling</h2>
    <p>Run deterministic layout checks before asking a vision model to judge aesthetics.</p>
  </div>

</div>
