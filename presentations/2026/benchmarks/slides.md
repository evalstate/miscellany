---
theme: default
title: Benchmarking agents
titleTemplate: "%s · fast-agent"
colorSchema: light
transition: none
layout: cover
fonts:
  sans: Figtree
  serif: Fraunces
  mono: DM Mono
  provider: none
drawings:
  persist: false
---

# Benchmarking agents

Why we run them, what we run, how to run them, and how to read them.

::meta::

**Shaun Smith** · Hugging Face

Event name · date TBC

---
layout: section
number: 1
---

# Why benchmark?

Who's asking, and what they want to know.

---

# Four audiences, four questions

<div class="fa-grid-4" style="margin-top: 12px">
<Card>

### Model developers

Is the new checkpoint better, and where?

</Card>
<Card>

### Harness optimisers

Did my change help, and what did it cost?

</Card>
<Card>

### Consumers

Which model and harness suit my task and budget?

</Card>
<Card>

### Investors

Who is actually ahead, and by how much?

</Card>
</div>

<!--
DRAFT questions under each audience. Replace with your own framing.
Cards are deliberately equal (paper): no one audience is "us", so no amber.
-->

---
layout: section
number: 2
---

# What are we running?

Verifier designs, and what each costs to run.

---
clicks: 3
---

# Where does the verifier run?

<div style="height: 318px">
<VerifierDesigns :phase="$clicks" />
</div>

<!--
Click 1: Setup. Click 2: Agent time. Click 3: Verifier.
Harbor's terms: "shared" = verifier runs in the agent's environment (the default);
"separate" = agent env is stopped, then a verifier runs in its own container.
- TB 2.1: no environment_mode set, so shared. test.sh even installs curl/uv at verify time,
  inside the container the agent just had full control of.
- DeepSWE 1.1: separate since v1.1. A [[verifier.collect]] hook runs
  `git diff --binary <base> HEAD > /logs/artifacts/model.patch`; the patch is applied in a
  pristine container and graded with held-out tests. Agent and verifier both "no-network".
  The reference solution patch is never used at grading time.
- TB 4: separate is required. The agent container is torn down; the verifier is built from
  tests/Dockerfile (ground truth baked in, never visible to the agent) and reads only: declared
  `artifacts`, its own image, and persistent sidecars from environment/docker-compose.yaml
  (11 of 66 tasks have one). Open internet for the agent (64 of 66 tasks).
Point: same harness mechanism (Harbor separate verifier), different contract:
a diff for DeepSWE, arbitrary declared files and live services for TB 4.
Sources: harbor@0dc28dd src/harbor/trial/single_step.py; terminal-bench-2-1@7131e43;
deep-swe@0b9fabb README.md + tasks/*/task.toml; terminal-bench v4.0.0 CONTRIBUTING.md "Tests".
-->

---

# Orders of magnitude

<div class="fa-compact">

| | TB 2.1 | DeepSWE 1.1 | TB 4 |
| --- | --- | --- | --- |
| Tasks | 89 | 113 | 66 |
| Agent timeout (median) | 15 min | 3 h | 8 h |
| Resources (median) | 1 CPU · 2 GB | 2 CPU · 8 GB | 2 CPU · 4 GB |
| Agent internet | Yes | No | Yes (64 of 66) |
| Wall-clock per full run | TBC | TBC | TBC |
| Estimated cost per full run | TBC | TBC | TBC |

</div>

<Receipt mark="*" source="task.toml configs: terminal-bench-2-1@7131e43 · deep-swe@0b9fabb · terminal-bench v4.0.0">
Configured limits, not measured runs. Wall-clock and cost TBC from our runs.
</Receipt>

<!--
Top rows computed from every task.toml on 29 Sep 2026 (script: medians of agent.timeout_sec,
environment.cpus/memory_mb; network from allow_internet / network_mode).
Totals if useful: summed agent-timeout budget per single pass is ~42 h (TB 2.1), ~339 h (DeepSWE), ~528 h (TB 4).
TB 4 max: 16 CPU / 32 GB; 3 GPU tasks. TB 2.1 max agent timeout 12000 s.
Wall-clock and cost need our run data; state concurrency next to any time-to-run number.
-->

---
layout: section
number: 3
---

# Planning and running

What you need to know before you press go.

---

# Before you press go

<div class="fa-grid-2">
<div>

- Benchmark and **version** pinned
- Model, reasoning setting and provider
- Harness and **version**
- Trials per task, and timeouts

</div>
<div>

- Sandbox provider and concurrency
- Rate limits and token budget
- Traces and logs kept for every trial
- A retry policy, decided **up front**

</div>
</div>

<!-- DRAFT list. -->

---

# Choosing a sandbox provider

<div class="fa-grid-3" style="margin-top: 8px">
<Card tone="paper">

### Capacity

Concurrent sandboxes, cold-start time, limits per account.

</Card>
<Card tone="paper">

### Compatibility

Image size, architecture, privileged or nested containers, networking.

</Card>
<Card tone="paper">

### Cost and reliability

Price per sandbox-hour, regions, outage history, support.

</Card>
</div>

<!-- DRAFT criteria. Add named providers and your experience with each. -->

---

# Things that will go wrong

<div class="fa-grid-2">
<Card tone="deep">

### The agent

- Harness or agent crashes mid-trial
- Context overflows and runaway loops
- Timeouts that end a trial that was succeeding

</Card>
<Card tone="deep">

### The infrastructure

- Sandbox provider outages
- Model API rate limits and 5xx errors
- Image pulls and package mirrors failing

</Card>
</div>

<p style="margin-top: 20px"><strong>Decide in advance:</strong> is an infrastructure failure a retry or a fail?</p>

<!-- DRAFT. Add real incidents from our runs. -->

---
layout: presenter
pose: C
height: 420
---

# Models are clever

Some of the tricks models use to pass:

- <span class="fa-draft">Example TBC</span>
- <span class="fa-draft">Example TBC</span>
- <span class="fa-draft">Example TBC</span>

<!--
Collect concrete examples from our traces (with task IDs).
Candidate categories to check against our traces: reading or editing the tests, finding answers in git
history or caches, special-casing the verifier's inputs, network lookups where not intended.
-->

---
layout: section
number: 4
---

# Reading benchmarks

How to read a result, and what to watch for.

---

# Every number needs a receipt

<div class="fa-grid-2">
<div>

- **n**: tasks × trials
- **Date** of the run
- **Model** and reasoning setting
- **Harness** and version

</div>
<div>

- **Cost basis**: estimated or billed?
- **Timeouts**: standard or custom?
- **Scoring**: pass@1, average or best-of-k?
- **Status**: verified or self-reported?

</div>
</div>

---

# Common distortions

<div class="fa-grid-3 fa-compact">
<Card>

### Selection

Cherry-picked comparisons, and stale competitor results.

</Card>
<Card>

### Scoring

Best-of-k presented as a single try. Retries left unreported.

</Card>
<Card>

### Conditions

Custom timeouts, extra tools, or a different benchmark version.

</Card>
<Card>

### Cost

Estimated cost presented as spend. Missing cache pricing.

</Card>
<Card>

### Charts

Truncated axes and log scales that aren't labelled.

</Card>
<Card>

### Contamination

Tasks or solutions that were in the training data.

</Card>
</div>

<!-- DRAFT list; trim to the ones you'll talk to, and add real examples. -->

---
layout: cover
tagline: Thank you.
---

# Questions?

::meta::

huggingface.co/evalstate · github.com/evalstate · x.com/evalstate
