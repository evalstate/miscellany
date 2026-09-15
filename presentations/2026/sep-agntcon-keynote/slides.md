---
theme: default
title: September AGNTCON Keynote
titleTemplate: "%s"
transition: none
class: keynote-brand-slide
fonts:
  sans: Source Sans 3
  mono: IBM Plex Mono
drawings:
  persist: false
defaults:
  layout: default
  class: keynote-section
---
<div class="speaker-intro">
<section class="speaker-copy">

# Shaun Smith

<div class="speaker-subtitle">Hugging Face</div>

<nav class="speaker-socials" aria-label="Social profiles">
<a href="https://huggingface.co/evalstate"><img src="/brand/hugging-face.svg" alt="Hugging Face" /><span>huggingface.co/evalstate</span></a>
<a href="https://github.com/evalstate"><img src="/intro/github-mark.svg" alt="GitHub" /><span>github.com/evalstate</span></a>
<a href="https://x.com/evalstate"><img src="/intro/x-mark.svg" alt="X" /><span>x.com/evalstate</span></a>
</nav>
</section>

<aside class="speaker-logos" aria-label="Hugging Face and Model Context Protocol">
<img class="speaker-hf" src="/brand/hugging-face.svg" alt="Hugging Face" />
<div class="speaker-logo-divider"></div>
<img class="speaker-mcp" src="/brand/mcp-symbol-black.svg" alt="Model Context Protocol" />
</aside>
</div>


---

# In the beginning

<!--
Section 1 of 7 · 0:00–2:00 · 2 minutes
Working section placeholder, not final slide copy.
Hugging Face MCP overview; early MCP as a bidirectional protocol.
-->

- Locally running STDIO Servers
- MCP Client and Server were closely coupled
- Came with a simple remote transport (`SSE`).

---
class: native-data-slide
---

<ChartRecordingStage chart="legacy-protocol-video" title="MCP Communications" subtitle="" />

<!--
New presentation/video study; existing Legacy/Modern slides remain unchanged.
Same established legacy 2025-11-25 conceptual communication, not a lifecycle trace
or paired round trips. Each pulse is an independent one-way activation; the actor
and capability highlight together. Client stays left and Server stays right.
Sequence: Tools → Sampling → Resources → Elicitation → Prompts → Roots, alternating
message direction. These are illustrative cues, not method-specific exchanges.
Travel is 1275ms (two-thirds the former speed); arrival highlight holds for 5s.
Original layout proportions and original CapabilityIcon glyphs are preserved;
the arrows are continuous SVG paths, without direction labels.
Play manually, once by default; optional looping only after activation. Capture
seeks exact semantic frames; the final Roots/Client highlight remains for the poster.
Protocol behaviour is unchanged. The documented local protocol checkout remains
unavailable; no new specification claim or deprecation label is introduced here.
-->

---

# Streamable HTTP

<!--
Section 2 of 7 · 2:00–3:30 · 90 seconds
Working section placeholder, not final slide copy.
Introduction of Streamable HTTP and protocol auth; client migration to it.
Services that were hosting over SSE/SHTTP and 
-->

- Remote connectivity became practical
- Single URL to connect and authenticate
- Simpler distribution: Connectors and Plugins

---

# Practical Challenges

- Shared state between Client and Server 
- Connection management
- Compatibility with HTTP infrastructure

<!--
Section 3 of 7 · 3:30–5:00 · 90 seconds
There were a number of features we were interested in using - for
example allowing people to dynamically change tools from our site.
Working section placeholder, not final slide copy.
Benefits of Streamable HTTP, including observability and optimisation. Traffic-management challenges: client/server behaviour in practice.
-->

---
class: message-ratio-slide
---

<div class="message-ratio-frame">
<ChartRecordingStage chart="legacy-message-ratio" title="1 tool call. 73 other messages." subtitle="">
<template #heading="{ otherCount }">
<h1><span class="message-ratio-tool-head">1 TOOL CALL.</span><span>{{ otherCount }} OTHER MESSAGES.</span></h1>
</template>
<template #footnote>Hugging Face MCP · June 2026 · Not a literal message sequence</template>
</ChartRecordingStage>
</div>

<!--
Native Slidev migration of /home/evalstate/source/data-analysis/charts/legacy-message-animation/index.html.
Click Play to begin; Pause/Resume, Replay, Show all, speed, and optional looping are
available. A controls-free ?capture=1 view now supports deterministic frame capture
through the same recording bridge as the other charts. Source timing and counts are
unchanged; original source typography, colours and geometry are restored.
Source HTML autoplays, but the talk version deliberately starts manually.
Slidev owns fullscreen and navigation; there are no chart-global keyboard
shortcuts. Reduced-motion and print/overview show the completed static chart.

74 equal-sized squares: 1 tool call, 27 initialization, 39 listing, 7 other.
The exact June 2026 aggregate ratio was 72.91097292652765 non-tool messages per
tool-call attempt. Largest-remainder rounding preserves 73 non-tool squares.
This counts inbound MCP method invocations and notifications, not responses,
bytes or tokens. Tool-call attempts include failures. All observed clients,
including test traffic; no fixed-five exclusion. These are historical transport
counters, not the canonical five-excluded protocol query-log chart.
Initialization includes initialize and notifications/initialized. Listing includes
tools/list, prompts/list, resources/list and resources/templates/list. Other includes
pings, resource operations, other notifications and prompt retrievals; some are
legitimate. Non-tool does not mean useless or avoidable. Cascade timing/grouping is
illustrative, not observed session ordering or latency. This is historical data,
not a new claim about protocol lifecycle behaviour.

Source: evalstate/hf-mcp-stats at 3554a77eba80ec06d06af689bf26ee80b795fe14.
Window: 2026-06-01 through 2026-06-30; coverage 713.82468 of 720 wall hours.
Full source provenance, coverage caveats and hashes retained in
 data/legacy-message-ratio.provenance.json. No data re-query or cohort changes.
-->

---

# Stateless MCP (2026-07-28)

- Changes both STDIO and HTTP Transports
- Removes lifecycle handshakes
- Adds caching and HTTP native features
- Improved support for Elicitations
<!--
Section 4 of 7 · 5:00–6:00 · 1 minute
Working section placeholder, not final slide copy.
The new stateless version: key changes and deprecations. Verify exact changes against protocol sources when developing this section.
-->

---
class: native-data-slide
---

<div class="native-chart-stage" data-flow="modern" data-phase="deprecated-overview">
<LegacyProtocolVideo title="2026-07-28 MCP Communications" deprecated />
</div>

<!--
Static companion to MCP Communications. Original icons, actor positions and card
proportions are unchanged. Roots and Sampling are crossed out at the presenter's
request. The documented protocol checkout /home/ssmith/source/modelcontextprotocol
is still unavailable and no equivalent local checkout was located; the deprecation
claim remains pending independent source verification, as on the earlier Modern slide.
The one double-headed arrow depicts client/server request and response traffic,
not permission for the Server to initiate independent modern requests. Elicitation
remains shown conceptually; modern retries carry original parameters, keyed
inputResponses and unchanged opaque requestState when supplied, not a literal
question echo. This variant has no pulses, highlights, timers started, or controls.
-->

---
class: native-data-slide
---

<ChartRecordingStage chart="protocol-adoption" title="2026-07-28 Protocol Adoption (Tool Calls)" subtitle="" />

<!--
Native port of charts/protocol-adoption-dashboard from the local data-analysis workspace.
Private aggregates: review before external sharing. This is a pinned snapshot, not live.
Source evalstate/hf-mcp-logs @ 1b5b83fbc3b4be2870345edf498b2644167caf18.
Window July 28–September 15, 2026 UTC; September 15 is partial. Source publication
cutoff September 15 16:06:46 UTC does not guarantee ingestion completeness.

Logged tool calls, not protocol messages or users; failures included, no deduplication.
Fixed five hashes excluded across every client/version; missing hashes retained.
Only recognized protocol versions enter the rate denominator; unknown/unreviewed
versions remain in count tables. Null is unavailable, not zero. No sample threshold.
Rolling7 is a strict count-weighted ratio over seven completed UTC dates, not a mean
of daily percentages. A partial day uses the preceding seven completed calendar dates.
Viewport selection changes visible dates, not the rolling denominator. KPIs use the
last reached daily anchor (floor position). No lines bridge nulls/calendar gaps.
Purple connections are illustrative reveals, not intraday estimates. Amber marks
actual partial observations. Overall includes all retained client identities.
Chat UI = chat-ui-mcp, not chat-ui-intern. Codex CLI = codex-mcp-client, not openai-mcp.
No version/event evidence or causal interpretation is inferred.

Data, raw selected-client count table and downloads are available via Data & provenance.
Unmodified source JSON, CSV and provenance live in data/protocol-adoption/.
Default playback is 4× (7.5 seconds of motion); source ratios/timestamps are unchanged.
Reveal layout leads with the purple count-weighted trailing-seven share; daily is
thin slate, with amber for the partial observation. The 0–100% axis is retained
with a 50% reference. Completion-only stars label actual final values, not forecasts.
Final overall rolling share is 49.3% (nearly half), not a claim of crossing 50%.
Stage decoration is omitted for keynote use; caveats remain in these notes and data.
Append ?capture=1 for a controls-free stage; automated recording uses exact timestamps.
-->

---
class: native-data-slide
---

<div class="native-chart-stage">
<ClientAdoptionComparison title="2026-07-28 Protocol Adoption (Tool Calls)" :capture="$route.query.capture === '1'" />
</div>

<!--
Six-client contact sheet restyled from client-migration-chat-ui-20260914/client_migration_focus.png.
Native SVG; source data are accessible in data/client-adoption-comparison/ and via
Data & provenance. Private review aggregates: review before public sharing.

IMPORTANT: this reference uses the completed July 28–September 14 snapshot at
12c284c96e556c16aa7cb09ff1370dbbd6342941, not the preceding dashboard's partial
September 15 snapshot. It also requires ≥100 valid calls for rates, unlike the
unsuppressed dashboard. Do not treat missing rates as zero or compare those
snapshots as if their dates/support policy were identical. No rate recomputation.
Last7 September 8–14: 95.2%, 99.3%, 100.0%, 28.4%, 15.1%, 0.0% in panel order.

Purple daily / slate count-weighted strict trailing-seven shares, common 0–100% axes.
Activity bars retain actual all-protocol call counts with independent client scales:
compare activity patterns within each panel, not relative popularity across panels.
Unknown/unreviewed protocols enter volumes but not rate denominators. Fixed five
excluded across client/version; missing hashes retained. Blank rates are unavailable.
Identity labels preserve chat-ui-mcp (not chat-ui-intern) and codex-mcp-client
(not openai-mcp (Codex)). Self-reported clients are not users or proven migrations.
Coverage of padded partitions does not prove complete real-world traffic.
-->

---

# Deployment Tips

- Migrate to latest SDK Versions
- Enable List Caching
- Stateless:- do you need `discover` or `subscribe`? 

---
class: keynote-brand-slide
---

# Hugging Face MCP

<div class="hf-stack-topology" data-phase="overview">
<div class="hf-stack-clients" data-node="clients">
<svg viewBox="0 0 64 48" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="7" y="7" width="38" height="27" rx="3"/><path d="M26 34v7m-9 0h18"/><rect x="43" y="19" width="14" height="23" rx="3" fill="white"/><path d="M49 37h2"/></svg>
<strong>Clients</strong>
</div>
<div class="hf-topology-link" aria-hidden="true"></div>
<div class="hf-topology-router" data-node="router">
<svg viewBox="0 0 64 48" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 24h16m-5-5 5 5-5 5M40 12h15m-5-5 5 5-5 5M40 36h15m-5-5 5 5-5 5M32 24v-9a3 3 0 0 1 3-3h5M32 24v9a3 3 0 0 0 3 3h5"/><circle cx="32" cy="24" r="4" fill="white"/></svg>
<strong>HTTP Router</strong>
</div>
<div class="hf-topology-link" aria-hidden="true"></div>
<section class="hf-stack-server" aria-label="Hugging Face MCP Server and its capabilities">
<div class="hf-stack-node" data-node="hf-mcp-server">
<img src="/brand/hugging-face.svg" alt="Hugging Face" />
<strong>MCP Server</strong>
</div>
<div class="hf-stack-branches" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i></div>
<div class="hf-stack-capabilities">
<div class="hf-topology-capability" data-capability="models"><img src="/hf-icons/models.svg" alt="" /><span>Models</span></div>
<div class="hf-topology-capability" data-capability="datasets"><img src="/hf-icons/datasets.svg" alt="" /><span>Datasets</span></div>
<div class="hf-topology-capability" data-capability="buckets"><img src="/hf-icons/buckets.svg" alt="" /><span>Buckets</span></div>
<div class="hf-topology-capability" data-capability="papers"><img src="/hf-icons/papers.svg" alt="" /><span>Papers</span></div>
<div class="hf-topology-capability" data-capability="gpu-apps"><img class="hf-topology-fan" src="/hf-icons/gpu-fan.svg" alt="" /><span>GPU Apps</span></div>
<div class="hf-topology-capability" data-capability="compute"><img src="/hf-icons/hardware.svg" alt="" /><span>Compute</span></div>
</div>
</section>
</div>

<!--
Alternative topology layout: Clients → HTTP Router → Hugging Face MCP Server node,
with six vertically stacked capability cards. The warm grouping contains both the
MCP Server node and its capabilities; branches show offerings, not separate physical
servers. Tight card padding preserves large icons and labels. Original topology
slide is retained. No routing animation yet, apart from the decorative GPU fan.
Website icon provenance: public/hf-icons/SOURCES.md.
-->

---

# Observability

- As models get smarter, we can make more token dense and efficient tools
- Get better idea of interaction surfaces. 
- Reduce Tool error rates, track Client capabilities and versions



---
class: native-data-slide
---

<ChartRecordingStage chart="tool-quality-version" title="Tool Error Rate" subtitle="hf_fs · daily quality-classified errors" />

<!--
Native port of charts/tool-quality-version-focus from the local data-analysis workspace.
Private aggregate outputs: review before external sharing. No source refresh or remote scan.
August 25–September 13, 2026 published tool-quality cells; newer local protocol snapshots
lack the classifications required to extend this chart. Published cell coverage ~98–99%.
Metric: exclusive tool_quality failed calls / selected calls, not all failures/operations.
Disjoint Claude/non-Claude client cells. Fixed five excluded; missing hashes retained;
suppressed cells are not zero. Numeric rates use the last revealed daily observation.
Daily noon UTC anchors; straight connections and guide effects are illustrative.
Design experiment: fixed 0–6% vertical viewport. Higher values continue above the
plot and are clipped, not capped at 6%; numerical readouts retain the actual rates.

Client ribbon = daily usage leader, NOT verified release date. MCP Server ribbon =
first-observed build hour, NOT a claimed global rollout. Initial .12 is not an event.
Exact .14/.15 six-hour boundary is retained without widening or merging.
Verified tool-description/argument-schema notes at .13/.15/.18/.19; .14 is unchanged.
.17 guidance appears at observed .18, not an invented .17 deployment. Notes persist
until superseded. Version changes and error rates are not causally attributed.

Unmodified source data.json, daily.csv, changes.json, provenance.json and README are
available through methodology/download controls and in data/tool-quality-version/.
Append ?capture=1 for a controls-free stage; automated recording uses exact timestamps.
-->
