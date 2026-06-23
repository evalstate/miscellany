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
class: story-trace-slide
---

# Story Writing GEPA trace

<StoryGepaTrace />

---
layout: default
class: html-effectiveness-video-slide
---

<div class="html-effectiveness-video">
  <div class="html-effectiveness-video__frame">
    <video
      :src="'videos/birch-html-html-intro-sequence.mp4'"
      autoplay
      loop
      muted
      playsinline
      preload="metadata"
    ></video>
  </div>

  <nav class="html-effectiveness-video__links" aria-label="HTML and Birch resources">
    <a href="https://thariqs.github.io/html-effectiveness/" target="_blank" rel="noreferrer">
      The Unreasonable Effectiveness of HTML
    </a>
    <a href="https://evalstate-birch-html.hf.space/analysis/report.html" target="_blank" rel="noreferrer">
      The Birch Benchmark
    </a>
  </nav>
</div>

---
layout: default
class: birch-scoring-slide
---

# Birch scoring → ASI

<div class="birch-scoring">
  <section class="birch-scoring__inputs" aria-label="Birch skill inputs">
    <div class="birch-scoring__file birch-scoring__file--skill">
      <span>skill</span>
      <strong>SKILL.md</strong>
      <p>How to produce standalone Birch HTML.</p>
    </div>
    <div class="birch-scoring__plus">+</div>
    <div class="birch-scoring__file birch-scoring__file--recipe">
      <span>recipe</span>
      <strong>numeric-data.md</strong>
      <p>What this benchmark asks the skill to render.</p>
    </div>
  </section>

  <section class="birch-scoring__flow" aria-label="Scoring inputs become actionable system intelligence">
    <article>
      <span class="birch-scoring__tag">deterministic</span>
      <strong>render checks</strong>
      <p>generation success · Birch CSS · semantic primitives · chart/table contracts · mobile geometry</p>
    </article>
    <article>
      <span class="birch-scoring__tag birch-scoring__tag--vision">VLM</span>
      <strong>screenshot feedback</strong>
      <p>“table is clipped on mobile” · readability · visual defects the DOM checks miss</p>
    </article>
    <div class="birch-scoring__arrow">→</div>
    <article class="birch-scoring__asi">
      <span>ASI packet</span>
      <strong>score + failures + fixes</strong>
    </article>
  </section>

  <figure class="birch-scoring__evidence">
    <div class="birch-scoring__phone">
      <img :src="'birch-scoring/numeric-data-mobile-clipped.png'" alt="Mobile Birch artifact with clipped table evidence" />
      <div class="birch-scoring__bbox"></div>
    </div>
    <figcaption>
      <span>example feedback</span>
      <strong>mobile table clipped</strong>
      <p>one screenshot becomes actionable mutation pressure</p>
    </figcaption>
  </figure>
</div>

---
layout: default
class: gepa-run-slide
---

# Optimising the Skill

<GepaRunExplorer />



---
layout: center
class: text-center
---

# Batches, Labels and Tools


---
layout: default
class: f1-labels-slide
---

<figure class="f1-labels-figure">
  <img :src="'images/f1-labels.png'" alt="F1 label examples" />
</figure>

---

# OpenClaw Github Issue Labelling

- Over 700 GitHub PRs and Issues over complicated product surface

- Label Creation: Tempting to use LLM but you need reproducable and correct labels to train and evaluate against. Try a "laddering" process to save time.
- Label Application: 
- Optimisation: 

---
layout: default
class: fast-agent-docs-slide
---

# fast-agent docs

<div class="fast-agent-docs-grid">
  <a class="fast-agent-doc-card" href="https://fast-agent.ai/guides/batch-processing/" target="_blank" rel="noreferrer">
    <img :src="'fast-agent-batch-processing-social.png'" alt="fast-agent social card — Batch Processing" />
    <span>Batch Processing</span>
  </a>
  <a class="fast-agent-doc-card" href="https://fast-agent.ai/guides/gepa/" target="_blank" rel="noreferrer">
    <img :src="'fast-agent-gepa-social.png'" alt="fast-agent social card — GEPA Optimization" />
    <span>GEPA Optimization</span>
  </a>
</div>

---


---

# Generating Labels

Tempting to use an LLM _but_


---

# Benchmark

- Does it do the same thing twice?
- Does the score transfer to my held out set?


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
