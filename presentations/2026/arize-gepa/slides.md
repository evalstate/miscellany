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
    <ReliableVideo src="videos/birch-html-html-intro-sequence.mp4" />
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

# HTML Skill → Actionable Side Information

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

# Optimising the Skill: gpt-oss-120b

<GepaRunExplorer demo-auto-start :demo-step-delay-ms="3000" />



---
layout: default
class: gepa-run-slide
---

# With a Small Model: Qwen-3.6 35-A3B

<SmallModelRunExplorer />



---
layout: center
class: text-center
---

# Batches, Labels and Tools and Other Uses


---
layout: default
class: batches-labels-tools-image-slide
---

<figure class="batches-labels-tools-image">
  <img :src="'images/batches-labels-tools-twitter.png'" alt="Batches, Labels and Tools example" />
</figure>

---
layout: default
class: f1-labels-slide
---

<figure class="f1-labels-figure">
  <img :src="'images/f1-labels.png'" alt="F1 label examples" />
</figure>

---
layout: default
class: good-benchmarks-slide
---

# Good benchmarks

<GoodBenchmarksSlide />


---
layout: default
class: openclaw-labeling-slide
---

# OpenClaw GitHub issue labelling

<div class="openclaw-labeling">
  <section class="openclaw-labeling__hero">
    <div class="kicker">batch labelling case study</div>
    <strong>700+</strong>
    <span>PRs and issues</span>
    <p>Complicated product surface; labels need to be reproducible enough to train and benchmark against.</p>
  </section>

  <section class="openclaw-labeling__pipeline" aria-label="Labelling workflow">
    <article>
      <span>01</span>
      <strong>Curate</strong>
      <p>high quality seed set</p>
    </article>
    <i>→</i>
    <article>
      <span>02</span>
      <strong>Ladder</strong>
      <p>consensus over runs</p>
    </article>
    <article>
      <span>03</span>
      <strong>Partition</strong>
      <p>held-out benchmark</p>
      <p>training / pareto set</p>
    </article>
  </section>
</div>

---
layout: default
class: openclaw-lessons-slide
---

# Common Issues

<div class="openclaw-lessons">
  <article class="openclaw-lesson openclaw-lesson--accent">
    <span>data</span>
    <strong>Don’t blindly trust LLM labels</strong>
    <p>Use LLMs to save time, but keep human-curated examples and consensus checks in the loop.</p>
  </article>
  <article class="openclaw-lesson">
    <span>hygiene</span>
    <strong>Keep row IDs out of prompts</strong>
    <p>Identifiers leaking into reflection or task text create brittle, non-transferable optimizations.</p>
  </article>
  <article class="openclaw-lesson">
    <span>tools</span>
    <strong>Make routing instructions explicit</strong>
    <p>Don’t hide crucial tool-use behavior inside descriptions where the optimizer can miss it.</p>
  </article>
  <article class="openclaw-lesson">
    <span>pressure</span>
    <strong>Constrain prompt length</strong>
    <p>Use length pressure to avoid runaway instructions that overfit a labelling run.</p>
  </article>
  <article class="openclaw-lesson openclaw-lesson--wide">
    <span>benchmark</span>
    <strong>Check transfer and variance</strong>
    <p>Separate the held-out benchmark from training / pareto examples. Do post-loop scores transfer, and how much do they vary across models and runs?</p>
  </article>
</div>


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
layout: center
class: thank-you-slide
---

<div class="thank-you">
  <h1>Thank You!</h1>
  <p>Questions?</p>
</div>
