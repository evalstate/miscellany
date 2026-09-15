---
theme: default
title: Modern Elicitations — storyboard
titleTemplate: "%s"
transition: none
fonts:
  sans: Source Sans 3
  mono: IBM Plex Mono
drawings:
  persist: false
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

# Legacy MCP is fully bi-directional

<div class="legacy-horizontal" aria-label="Client and Server exchange messages in both directions">
<div class="legacy-types legacy-types-client">
<div class="legacy-type"><CapabilityIcon name="roots" /><span>Roots</span></div>
<div class="legacy-type"><CapabilityIcon name="sampling" /><span>Sampling</span></div>
<div class="legacy-type"><CapabilityIcon name="elicitation" /><span>Elicitation</span></div>
</div>
<div class="legacy-actor legacy-actor-client"><span>Client</span></div>
<div class="legacy-channels">
<div class="legacy-lane legacy-lane-right"><span>Client → Server</span><i></i></div>
<div class="legacy-lane legacy-lane-left"><i></i><span>Server → Client</span></div>
</div>
<div class="legacy-actor legacy-actor-server"><span>Server</span></div>
<div class="legacy-types legacy-types-server">
<div class="legacy-type"><CapabilityIcon name="tools" /><span>Tools</span></div>
<div class="legacy-type"><CapabilityIcon name="resources" /><span>Resources</span></div>
<div class="legacy-type"><CapabilityIcon name="prompts" /><span>Prompts</span></div>
</div>
</div>

<!--
Static horizontal layout study of July's ProtocolStack: client capabilities on the
far left, server capabilities on the far right. Arrows represent message direction,
not separate physical transports. Established connection with negotiated capabilities
is assumed; not every implementation offers every capability. Legacy 2025-11-25.
No lifecycle sequence or protocol method names have been ported from the animation.
-->

---

# Modern MCP

<div class="legacy-horizontal legacy-horizontal-modern" aria-label="Modern MCP: Client initiates requests to Server">
<div class="legacy-types legacy-types-client">
<div class="legacy-type legacy-type-deprecated"><CapabilityIcon name="roots" /><span>Roots</span><small>DEPRECATED</small></div>
<div class="legacy-type legacy-type-deprecated"><CapabilityIcon name="sampling" /><span>Sampling</span><small>DEPRECATED</small></div>
<div class="legacy-type"><CapabilityIcon name="elicitation" /><span>Elicitation</span></div>
</div>
<div class="legacy-actor legacy-actor-client"><span>Client</span></div>
<div class="legacy-channels">
<div class="legacy-lane legacy-lane-right"><span>Client → Server</span><i></i></div>
</div>
<div class="legacy-actor legacy-actor-server"><span>Server</span></div>
<div class="legacy-types legacy-types-server">
<div class="legacy-type"><CapabilityIcon name="tools" /><span>Tools</span></div>
<div class="legacy-type"><CapabilityIcon name="resources" /><span>Resources</span></div>
<div class="legacy-type"><CapabilityIcon name="prompts" /><span>Prompts</span></div>
</div>
</div>

<!--
Modern 2026-07-28 conceptual comparison. Roots and Sampling are marked deprecated
as requested by the presenter; the documented local protocol checkout was unavailable
on this host, so that status still needs independent source verification.
The single arrow represents request initiation, not all message traffic: the Server
still returns responses. Elicitation is retained through MRTR rather than a separate
Server → Client request. Modern retry carries original parameters, keyed inputResponses,
and unchanged opaque requestState when supplied, not necessarily a literal question echo.
-->

---

# Legacy MCP is fully bi-directional

<div class="legacy-horizontal" aria-label="Client and Server exchange messages in both directions">
<div class="legacy-types legacy-types-client">
<div class="legacy-type" data-capability="roots"><CapabilityIcon name="roots" /><span>Roots</span></div>
<div class="legacy-type" data-capability="sampling"><CapabilityIcon name="sampling" /><span>Sampling</span></div>
<div class="legacy-type" data-capability="elicitation"><CapabilityIcon name="elicitation" /><span>Elicitation</span></div>
</div>
<div class="legacy-actor legacy-actor-client"><span>Client</span></div>
<CommunicationTraffic variant="legacy" />
<div class="legacy-actor legacy-actor-server"><span>Server</span></div>
<div class="legacy-types legacy-types-server">
<div class="legacy-type" data-capability="tools"><CapabilityIcon name="tools" /><span>Tools</span></div>
<div class="legacy-type" data-capability="resources"><CapabilityIcon name="resources" /><span>Resources</span></div>
<div class="legacy-type" data-capability="prompts"><CapabilityIcon name="prompts" /><span>Prompts</span></div>
</div>
</div>

<!--
Animated layout experiment; original static slides are preserved immediately before
these copies. Start manually; loops until Stop or slide departure.
Each pulse is an independent visual activation, not a request/response round trip.
Legacy alternates activation direction; Modern only shows Client → Server activation.
The receiving actor and a capability glow on arrival. These are illustrative cues,
not exact method mappings or a literal wire trace. Server responses still exist but
are deliberately omitted. Modern Elicitation remains visible without a reverse pulse;
deprecated capabilities never activate. No lifecycle sequence is implied.
Illustrated versions remain legacy 2025-11-25 and modern 2026-07-28. The documented
local protocol checkout is unavailable on this host; source verification remains
outstanding, including the presenter-requested Roots/Sampling deprecation labels.
-->

---

# Modern MCP

<div class="legacy-horizontal legacy-horizontal-modern" aria-label="Modern MCP: Client initiates requests to Server">
<div class="legacy-types legacy-types-client">
<div class="legacy-type legacy-type-deprecated"><CapabilityIcon name="roots" /><span>Roots</span><small>DEPRECATED</small></div>
<div class="legacy-type legacy-type-deprecated"><CapabilityIcon name="sampling" /><span>Sampling</span><small>DEPRECATED</small></div>
<div class="legacy-type" data-capability="elicitation"><CapabilityIcon name="elicitation" /><span>Elicitation</span></div>
</div>
<div class="legacy-actor legacy-actor-client"><span>Client</span></div>
<CommunicationTraffic variant="modern" />
<div class="legacy-actor legacy-actor-server"><span>Server</span></div>
<div class="legacy-types legacy-types-server">
<div class="legacy-type" data-capability="tools"><CapabilityIcon name="tools" /><span>Tools</span></div>
<div class="legacy-type" data-capability="resources"><CapabilityIcon name="resources" /><span>Resources</span></div>
<div class="legacy-type" data-capability="prompts"><CapabilityIcon name="prompts" /><span>Prompts</span></div>
</div>
</div>

<!--
Animated layout experiment; original static slides are preserved immediately before
these copies. Start manually; loops until Stop or slide departure.
Each pulse is an independent visual activation, not a request/response round trip.
Legacy alternates activation direction; Modern only shows Client → Server activation.
The receiving actor and a capability glow on arrival. These are illustrative cues,
not exact method mappings or a literal wire trace. Server responses still exist but
are deliberately omitted. Modern Elicitation remains visible without a reverse pulse;
deprecated capabilities never activate. No lifecycle sequence is implied.
Illustrated versions remain legacy 2025-11-25 and modern 2026-07-28. The documented
local protocol checkout is unavailable on this host; source verification remains
outstanding, including the presenter-requested Roots/Sampling deprecation labels.
-->

---

# Hugging Face MCP Server

<div class="hf-topology" data-phase="overview">
<div class="hf-topology-router" data-node="router">
<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true"><rect x="8" y="23" width="32" height="16" rx="4"/><path d="M15 23V10m18 13V10M11 13l4-4 4 4m10 0 4-4 4 4"/><circle cx="16" cy="31" r="1"/><path d="M23 31h10"/></svg>
<strong>Router</strong>
</div>
<div class="hf-topology-link" aria-hidden="true"></div>
<section class="hf-topology-server" data-node="hf-mcp-server" aria-label="Hugging Face MCP Server and its capabilities">
<div class="hf-topology-identity">
<img src="/brand/hugging-face.svg" alt="Hugging Face" />
<strong>MCP Server</strong>
</div>
<div class="hf-topology-internal" aria-hidden="true"></div>
<div class="hf-topology-capabilities">
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
Conceptual Hugging Face MCP Server topology, based on the presenter's requested
capability grouping, not a claim about individual physical services or deployments.
The router is outside and in front of the server. All six capabilities belong within
the server boundary. No traffic animation or routing policy is illustrated yet.
Semantic node/capability attributes provide stable targets for later routing work.
Icons are extracted from huggingface.co navigation and its Spaces GPU status fan;
see public/hf-icons/SOURCES.md. Compute uses Hardware; GPU Apps uses the GPU fan.
Only the decorative fan rotates; routing and topology remain static.
-->

---

# Hugging Face MCP Server

<div class="hf-stack-topology" data-phase="overview">
<div class="hf-stack-clients" data-node="clients">
<svg viewBox="0 0 64 48" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true"><path d="M18 38h31a11 11 0 0 0 1-22 17 17 0 0 0-32-4 13 13 0 0 0 0 26Z"/><rect x="22" y="20" width="19" height="12" rx="2"/><path d="M28 35h7m-4-3v3"/></svg>
<strong>Clients</strong>
</div>
<div class="hf-topology-link" aria-hidden="true"></div>
<div class="hf-topology-router" data-node="router">
<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true"><rect x="8" y="23" width="32" height="16" rx="4"/><path d="M15 23V10m18 13V10M11 13l4-4 4 4m10 0 4-4 4 4"/><circle cx="16" cy="31" r="1"/><path d="M23 31h10"/></svg>
<strong>Router</strong>
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
Alternative topology layout: Clients → Router → Hugging Face MCP Server node,
with six vertically stacked capability cards. The warm grouping contains both the
MCP Server node and its capabilities; branches show offerings, not separate physical
servers. Tight card padding preserves large icons and labels. Original topology
slide is retained. No routing animation yet, apart from the decorative GPU fan.
Website icon provenance: public/hf-icons/SOURCES.md.
-->

---
<div class="es-heading"><span class="es-mode ">OLD WAY</span><span>Modern Elicitations</span><span>01 / 05</span></div>

# Create a sandbox.

<div class="es-stage" data-flow="old" data-phase="request">
<div class="es-client" aria-label="Client computer"><div class="es-screen"><i></i><i></i><i></i><div></div><div></div></div><div class="es-stand"></div></div>
<div class="es-server" aria-label="Server"><div><i></i><span></span></div><div><i></i><span></span></div><div><i></i><span></span></div></div>
<div class="es-name es-name-client">Client</div><div class="es-name es-name-server">Server</div>
<div class="es-transfer "><strong>Create sandbox →</strong><div class="es-arrow"></div><small>Client starts a tool call</small></div>
</div>

<div class="es-caption">The same starting point.</div>

<!--
Client initiates tools/call. No sandbox is created before approval.
Static storyboard: advance manually. Fixed actor positions, no timers or animation.
Sources checked against local modelcontextprotocol/dcd ocs/specification: 2025-11-25/client/elicitation.mdx and 2026-07-28/basic/patterns/mrtr.mdx.
-->

---

<div class="es-heading"><span class="es-mode ">OLD WAY</span><span>Modern Elicitations</span><span>02 / 05</span></div>

# The Server needs permission.

<div class="es-stage" data-flow="old" data-phase="question">
<div class="es-client" aria-label="Client computer"><div class="es-screen"><i></i><i></i><i></i><div></div><div></div></div><div class="es-stand"></div></div>
<div class="es-server" aria-label="Server"><div><i></i><span></span></div><div><i></i><span></span></div><div><i></i><span></span></div></div>
<div class="es-name es-name-client">Client</div><div class="es-name es-name-server">Server</div>
<div class="es-bubble"><small>QUESTION</small><strong>Do you want to create a<br>paid GPU Sandbox?</strong></div><div class="es-transfer es-left"><strong>Ask the Client</strong><div class="es-arrow"></div><small>Server → Client request</small></div>
</div>

<div class="es-caption">A paid resource needs a decision.</div>

<!--
Legacy: Server sends elicitation/create to the Client while the original tools/call remains pending.
Static storyboard: advance manually. Fixed actor positions, no timers or animation.
Sources checked against local modelcontextprotocol/docs/specification: 2025-11-25/client/elicitation.mdx and 2026-07-28/basic/patterns/mrtr.mdx.
-->

---

<div class="es-heading"><span class="es-mode ">OLD WAY</span><span>Modern Elicitations</span><span>03 / 05</span></div>

# The Client shows the question.

<div class="es-stage" data-flow="old" data-phase="dialog">
<div class="es-client" aria-label="Client computer"><div class="es-screen"><i></i><i></i><i></i><div></div><div></div></div><div class="es-stand"></div></div>
<div class="es-server" aria-label="Server"><div><i></i><span></span></div><div><i></i><span></span></div><div><i></i><span></span></div></div>
<div class="es-name es-name-client">Client</div><div class="es-name es-name-server">Server</div>
<div class="es-bubble"><small>QUESTION</small><strong>Do you want to create a<br>paid GPU Sandbox?</strong></div>
<div class="es-dialog"><div class="es-dialog-bar"><span>Sandbox confirmation</span><span>×</span></div><div class="es-dialog-body"><small>SERVER ASKS</small><strong>Do you want to create a<br>paid GPU Sandbox?</strong><div class="es-buttons"><span>No</span><span class="es-yes">Yes</span></div></div></div><div class="es-transfer es-left"><strong>Ask the Client</strong><div class="es-arrow"></div><small>Server → Client request</small></div><div class="es-status es-waiting es-rack-waiting"><strong>OPEN</strong><span><i class="es-spinner" aria-hidden="true"></i> WAITING</span></div>
</div>

<div class="es-caption">The dialog is on the Client. The question is still open on the Server.</div>

<!--
Modern: input_required terminates the first request. Client displays the question and retains returned context. Legacy: elicitation/create remains pending. These are conceptual states, not literal wire payloads.
Advance manually. Only the decorative waiting indicator spins; no timed slide transitions.
Sources checked against local modelcontextprotocol/docs/specification: 2025-11-25/client/elicitation.mdx and 2026-07-28/basic/patterns/mrtr.mdx.
-->

---

<div class="es-heading"><span class="es-mode ">OLD WAY</span><span>Modern Elicitations</span><span>04 / 05</span></div>

# Send just the answer.

<div class="es-stage" data-flow="old" data-phase="answer">
<div class="es-client" aria-label="Client computer"><div class="es-screen"><i></i><i></i><i></i><div></div><div></div></div><div class="es-stand"></div></div>
<div class="es-server" aria-label="Server"><div><i></i><span></span></div><div><i></i><span></span></div><div><i></i><span></span></div></div>
<div class="es-name es-name-client">Client</div><div class="es-name es-name-server">Server</div>
<div class="es-bubble"><small>OPEN QUESTION</small><strong>Do you want to create a<br>paid GPU Sandbox?</strong></div>
<div class="es-dialog"><div class="es-dialog-bar"><span>Sandbox confirmation</span><span>×</span></div><div class="es-dialog-body"><small>SERVER ASKS</small><strong>Do you want to create a<br>paid GPU Sandbox?</strong><div class="es-buttons"><span>No</span><span class="es-selected">✓ Yes</span></div></div></div><div class="es-transfer "><strong>Yes</strong><div class="es-arrow"></div><small>Response to the pending question</small></div><div class="es-status es-waiting es-rack-waiting"><strong>OPEN</strong><span><i class="es-spinner" aria-hidden="true"></i> WAITING</span></div>
</div>

<div class="es-caption">“Yes” relies on the question the Server kept pending.</div>

<!--
Yes is an illustrative user decision, not a literal JSON-RPC response. Modern retry: new id, original params, keyed inputResponses and unchanged requestState if supplied. Question context is not necessarily the literal question text. Server validates context and authorization.
Advance manually. Only the decorative waiting indicator spins; no timed slide transitions.
Sources checked against local modelcontextprotocol/docs/specification: 2025-11-25/client/elicitation.mdx and 2026-07-28/basic/patterns/mrtr.mdx.
-->

---

<div class="es-heading"><span class="es-mode ">OLD WAY</span><span>Modern Elicitations</span><span>05 / 05</span></div>

# Now create the sandbox.

<div class="es-stage" data-flow="old" data-phase="complete">
<div class="es-client" aria-label="Client computer"><div class="es-screen"><i></i><i></i><i></i><div></div><div></div></div><div class="es-stand"></div></div>
<div class="es-server" aria-label="Server"><div><i></i><span></span></div><div><i></i><span></span></div><div><i></i><span></span></div></div>
<div class="es-name es-name-client">Client</div><div class="es-name es-name-server">Server</div>
<div class="es-result"><span>✓</span><strong>GPU Sandbox created</strong><small>Approved · ready to use</small></div><div class="es-transfer es-left"><strong>Sandbox ready</strong><div class="es-arrow"></div><small>Original call completes</small></div><div class="es-status es-free"><span>✓</span> Complete</div>
</div>

<div class="es-caption">The waiting question is resolved. The original call can finish.</div>

<!--
Approved path shown. No/decline does not create a paid sandbox. Modern has two independent calls; legacy has a pending elicitation within the original operation.
Static storyboard: advance manually. Fixed actor positions, no timers or animation.
Sources checked against local modelcontextprotocol/docs/specification: 2025-11-25/client/elicitation.mdx and 2026-07-28/basic/patterns/mrtr.mdx.
-->

---

<div class="es-heading"><span class="es-mode es-modern">MODERN WAY</span><span>Modern Elicitations</span><span>01 / 05</span></div>

# Create a sandbox.

<div class="es-stage" data-flow="modern" data-phase="request">
<div class="es-client" aria-label="Client computer"><div class="es-screen"><i></i><i></i><i></i><div></div><div></div></div><div class="es-stand"></div></div>
<div class="es-server" aria-label="Server"><div><i></i><span></span></div><div><i></i><span></span></div><div><i></i><span></span></div></div>
<div class="es-name es-name-client">Client</div><div class="es-name es-name-server">Server</div>
<div class="es-transfer "><strong>Create sandbox →</strong><div class="es-arrow"></div><small>Client starts a tool call</small></div>
</div>

<div class="es-caption">The same starting point.</div>

<!--
Client initiates tools/call. No sandbox is created before approval.
Static storyboard: advance manually. Fixed actor positions, no timers or animation.
Sources checked against local modelcontextprotocol/docs/specification: 2025-11-25/client/elicitation.mdx and 2026-07-28/basic/patterns/mrtr.mdx.
-->

---

<div class="es-heading"><span class="es-mode es-modern">MODERN WAY</span><span>Modern Elicitations</span><span>02 / 05</span></div>

# The Server needs permission.

<div class="es-stage" data-flow="modern" data-phase="question">
<div class="es-client" aria-label="Client computer"><div class="es-screen"><i></i><i></i><i></i><div></div><div></div></div><div class="es-stand"></div></div>
<div class="es-server" aria-label="Server"><div><i></i><span></span></div><div><i></i><span></span></div><div><i></i><span></span></div></div>
<div class="es-name es-name-client">Client</div><div class="es-name es-name-server">Server</div>
<div class="es-bubble"><small>TOOL REQUIRES ELICITATION</small><strong>Do you want to create a<br>paid GPU Sandbox?</strong></div><div class="es-status ">Tool requires Elicitation</div>
</div>

<div class="es-caption">A paid resource needs a decision.</div>

<!--
This frame is before delivery. Modern server prepares input_required, including an elicitation and optional opaque requestState.
Static storyboard: advance manually. Fixed actor positions, no timers or animation.
Sources checked against local modelcontextprotocol/docs/specification: 2025-11-25/client/elicitation.mdx and 2026-07-28/basic/patterns/mrtr.mdx.
-->

---

<div class="es-heading"><span class="es-mode es-modern">MODERN WAY</span><span>Modern Elicitations</span><span>03 / 05</span></div>

# The Client shows the question.

<div class="es-stage" data-flow="modern" data-phase="dialog">
<div class="es-client" aria-label="Client computer"><div class="es-screen"><i></i><i></i><i></i><div></div><div></div></div><div class="es-stand"></div></div>
<div class="es-server" aria-label="Server"><div><i></i><span></span></div><div><i></i><span></span></div><div><i></i><span></span></div></div>
<div class="es-name es-name-client">Client</div><div class="es-name es-name-server">Server</div>
<div class="es-bubble es-bubble-delivered"><small>DELIVERED QUESTION</small><strong>Do you want to create a<br>paid GPU Sandbox?</strong></div>
<div class="es-dialog es-dialog-delivered"><div class="es-dialog-bar"><span>Sandbox confirmation</span><span>×</span></div><div class="es-dialog-body"><strong>Do you want to create a<br>paid GPU Sandbox?</strong><div class="es-buttons"><span>No</span><span class="es-yes">Yes</span></div><div class="es-context">▣ &nbsp; Question context attached</div></div></div><div class="es-transfer es-left"><strong>Question delivered</strong><div class="es-arrow"></div><small>First request ends</small></div><div class="es-status es-free"><span>✓</span> No open question</div>
</div>

<div class="es-caption">The user can take their time. Nothing is held open on the Server.</div>

<!--
Modern: input_required terminates the first request. Client displays the question and retains returned context. Legacy: elicitation/create remains pending. These are conceptual states, not literal wire payloads.
Static storyboard: advance manually. Fixed actor positions, no timers or animation.
Sources checked against local modelcontextprotocol/docs/specification: 2025-11-25/client/elicitation.mdx and 2026-07-28/basic/patterns/mrtr.mdx.
-->

---

<div class="es-heading"><span class="es-mode es-modern">MODERN WAY</span><span>Modern Elicitations</span><span>04 / 05</span></div>

# Send the question context with the answer.

<div class="es-stage" data-flow="modern" data-phase="answer">
<div class="es-client" aria-label="Client computer"><div class="es-screen"><i></i><i></i><i></i><div></div><div></div></div><div class="es-stand"></div></div>
<div class="es-server" aria-label="Server"><div><i></i><span></span></div><div><i></i><span></span></div><div><i></i><span></span></div></div>
<div class="es-name es-name-client">Client</div><div class="es-name es-name-server">Server</div>
<div class="es-bubble es-bubble-delivered"><small>DELIVERED QUESTION</small><strong>Do you want to create a<br>paid GPU Sandbox?</strong></div>
<div class="es-dialog es-dialog-delivered"><div class="es-dialog-bar"><span>Sandbox confirmation</span><span>×</span></div><div class="es-dialog-body"><strong>Do you want to create a<br>paid GPU Sandbox?</strong><div class="es-buttons"><span>No</span><span class="es-selected">✓ Yes</span></div><div class="es-context">▣ &nbsp; Question context attached</div></div></div><div class="es-transfer "><strong>Question context + Yes</strong><div class="es-arrow"></div><small>New call · original arguments included</small></div><div class="es-status es-free"><span>✓</span> Ready for a new call</div>
</div>

<div class="es-caption">The Server receives what it needs to continue.</div>

<!--
Yes is an illustrative user decision, not a literal JSON-RPC response. Modern retry: new id, original params, keyed inputResponses and unchanged requestState if supplied. Question context is not necessarily the literal question text. Server validates context and authorization.
Static storyboard: advance manually. Fixed actor positions, no timers or animation.
Sources checked against local modelcontextprotocol/docs/specification: 2025-11-25/client/elicitation.mdx and 2026-07-28/basic/patterns/mrtr.mdx.
-->

---

<div class="es-heading"><span class="es-mode es-modern">MODERN WAY</span><span>Modern Elicitations</span><span>05 / 05</span></div>

# Now create the sandbox.

<div class="es-stage" data-flow="modern" data-phase="complete">
<div class="es-client" aria-label="Client computer"><div class="es-screen"><i></i><i></i><i></i><div></div><div></div></div><div class="es-stand"></div></div>
<div class="es-server" aria-label="Server"><div><i></i><span></span></div><div><i></i><span></span></div><div><i></i><span></span></div></div>
<div class="es-name es-name-client">Client</div><div class="es-name es-name-server">Server</div>
<div class="es-delivered-pair" aria-label="Question and confirmation received together"><div class="es-bubble"><small>QUESTION + CONFIRMATION</small><strong>Do you want to create a<br>paid GPU Sandbox?</strong></div><div class="es-confirmation"><small>CONFIRMATION</small><strong>✓ Yes</strong></div></div>
<div class="es-result"><span>✓</span><strong>GPU Sandbox created</strong><small>Approved · ready to use</small></div><div class="es-transfer es-left"><strong>Sandbox ready</strong><div class="es-arrow"></div><small>New call completes</small></div><div class="es-status es-free"><span>✓</span> Complete</div>
</div>

<div class="es-caption">Question context + Answer. Not an open conversation.</div>

<!--
Question + Confirmation is conceptual shorthand, not a literal question echo. The retry carries original parameters, keyed inputResponses, and unchanged opaque requestState when supplied. Both context and answer arrive together on the Server.
Approved path shown. No/decline does not create a paid sandbox. Modern has two independent calls; legacy has a pending elicitation within the original operation.
Static storyboard: advance manually. Fixed actor positions, no timers or animation.
Sources checked against local modelcontextprotocol/docs/specification: 2025-11-25/client/elicitation.mdx and 2026-07-28/basic/patterns/mrtr.mdx.
-->


---

<div class="es-heading"><span class="es-mode ">OLD WAY</span><span>HTTP Standardization</span><span>01 / 04</span></div>

# The routing clues are inside the body.

<div class="hs-stage" data-flow="http" data-phase="opaque" role="img" aria-label="Client sends an HTTP request through an LB/router to inference, API, or sandbox servers. ">
<div class="es-client"><div class="es-screen"><i></i><i></i><i></i><div></div><div></div></div><div class="es-stand"></div></div>
<div class="hs-client-name">Client</div>
<div class="hs-packet"><small>POST /mcp</small><strong>JSON body</strong><span>Tool + arguments inside</span></div>
<div class="hs-incoming"></div>
<div class="hs-router"><span>?</span><strong>LB / Router</strong><small>HTTP infrastructure</small></div>
<div class="hs-branches"><div class="hs-branch hs-inference"></div><div class="hs-branch hs-api"></div><div class="hs-branch hs-sandbox"></div></div>
<div class="hs-backends">
<div class="hs-backend hs-inference"><span class="hs-icon">✦</span><div><strong>Inference</strong><small>Generate</small></div></div>
<div class="hs-backend hs-api"><span class="hs-icon">↔</span><div><strong>API</strong><small>Fetch data</small></div></div>
<div class="hs-backend hs-sandbox"><span class="hs-icon">&gt;_</span><div><strong>Sandbox</strong><small>Run code</small></div></div>
</div>
</div>

<div class="es-caption">HTTP routing needs MCP-aware body parsing.</div>

<!--
The body is not encrypted or inherently unreadable. Existing HTTP infrastructure can parse JSON with custom logic, but ordinary header-based routing does not see the MCP method, tool name, or tool arguments. The LB/router is shown at the HTTP inspection point, after TLS termination.
Static storyboard: advance manually. Client, LB/router, and backend positions are fixed across all four frames. Yellow highlights indicate inspectable HTTP metadata, not an elicitation question.
-->


---

<div class="es-heading"><span class="es-mode es-modern">MODERN WAY</span><span>HTTP Standardization</span><span>02 / 04</span></div>

# Make the message visible to HTTP.

<div class="hs-stage" data-flow="http" data-phase="inspect" role="img" aria-label="Client sends an HTTP request through an LB/router to inference, API, or sandbox servers. ">
<div class="es-client"><div class="es-screen"><i></i><i></i><i></i><div></div><div></div></div><div class="es-stand"></div></div>
<div class="hs-client-name">Client</div>
<div class="hs-packet"><small>HTTP metadata</small><strong>Method · Tool name</strong><span>JSON body unchanged</span></div>
<div class="hs-incoming"></div>
<div class="hs-router"><span>Inspect</span><strong>LB / Router</strong><small>HTTP infrastructure</small></div>
<div class="hs-branches"><div class="hs-branch hs-inference"></div><div class="hs-branch hs-api"></div><div class="hs-branch hs-sandbox"></div></div>
<div class="hs-backends">
<div class="hs-backend hs-inference"><span class="hs-icon">✦</span><div><strong>Inference</strong><small>Generate</small></div></div>
<div class="hs-backend hs-api"><span class="hs-icon">↔</span><div><strong>API</strong><small>Fetch data</small></div></div>
<div class="hs-backend hs-sandbox"><span class="hs-icon">&gt;_</span><div><strong>Sandbox</strong><small>Run code</small></div></div>
</div>
</div>

<div class="es-caption">Existing HTTP infrastructure can read the routing clues.</div>

<!--
Conceptual HTTP metadata, not a complete wire request. Standardization exposes selected message information in HTTP headers; it does not move the entire body into headers. This lets configured L7 load balancers, gateways, and routers inspect and route using their normal HTTP facilities. Not all LBs inspect HTTP; TLS termination or equivalent access is required.
Static storyboard: advance manually. Client, LB/router, and backend positions are fixed across all four frames. Yellow highlights indicate inspectable HTTP metadata, not an elicitation question.
-->


---

<div class="es-heading"><span class="es-mode es-modern">MODERN WAY</span><span>HTTP Standardization</span><span>03 / 04</span></div>

# The right tool. The right server.

<div class="hs-stage" data-flow="http" data-phase="route" role="img" aria-label="Client sends an HTTP request through an LB/router to inference, API, or sandbox servers. Sandbox route highlighted.">
<div class="es-client"><div class="es-screen"><i></i><i></i><i></i><div></div><div></div></div><div class="es-stand"></div></div>
<div class="hs-client-name">Client</div>
<div class="hs-packet"><small>HTTP metadata</small><strong>Tool: run_in_sandbox</strong><span>JSON body unchanged</span></div>
<div class="hs-incoming"></div>
<div class="hs-router"><span>Route</span><strong>LB / Router</strong><small>HTTP infrastructure</small></div>
<div class="hs-branches"><div class="hs-branch hs-inference"></div><div class="hs-branch hs-api"></div><div class="hs-branch hs-sandbox"></div></div>
<div class="hs-backends">
<div class="hs-backend hs-inference"><span class="hs-icon">✦</span><div><strong>Inference</strong><small>Generate</small></div></div>
<div class="hs-backend hs-api"><span class="hs-icon">↔</span><div><strong>API</strong><small>Fetch data</small></div></div>
<div class="hs-backend hs-sandbox"><span class="hs-icon">&gt;_</span><div><strong>Sandbox</strong><small>Run code</small></div></div>
</div>
</div>

<div class="es-caption">One front door. Different kinds of servers.</div>

<!--
Illustrative tool names and deployment topology. Header-visible tool names allow configured rules to send inference tools to inference servers, API tools to API servers, and sandbox tools to sandbox servers. Three possible routes are shown; this call follows only the sandbox route. Standardization enables these rules; it does not automatically discover backends or configure the router.
Static storyboard: advance manually. Client, LB/router, and backend positions are fixed across all four frames. Yellow highlights indicate inspectable HTTP metadata, not an elicitation question.
-->


---

<div class="es-heading"><span class="es-mode es-modern">OPTIONAL</span><span>HTTP Standardization</span><span>04 / 04</span></div>

# Copy a routing key into HTTP metadata.

<div class="hs-stage" data-flow="http" data-phase="copy" role="img" aria-label="Client sends an HTTP request through an LB/router to inference, API, or sandbox servers. Sandbox route highlighted.">
<div class="es-client"><div class="es-screen"><i></i><i></i><i></i><div></div><div></div></div><div class="es-stand"></div></div>
<div class="hs-client-name">Client</div>
<div class="hs-packet"><small>Tool argument → HTTP metadata</small><strong>sandbox_id: sbx-7f3c</strong><span>Still in the JSON body</span></div>
<div class="hs-incoming"></div>
<div class="hs-router"><span>Route</span><strong>LB / Router</strong><small>HTTP infrastructure</small></div>
<div class="hs-branches"><div class="hs-branch hs-inference"></div><div class="hs-branch hs-api"></div><div class="hs-branch hs-sandbox"></div></div>
<div class="hs-backends">
<div class="hs-backend hs-inference"><span class="hs-icon">✦</span><div><strong>Inference</strong><small>Generate</small></div></div>
<div class="hs-backend hs-api"><span class="hs-icon">↔</span><div><strong>API</strong><small>Fetch data</small></div></div>
<div class="hs-backend hs-sandbox"><span class="hs-icon">&gt;_</span><div><strong>Sandbox</strong><small>sbx-7f3c</small></div></div>
</div>
</div>

<div class="es-caption">Tool-defined keys. Ordinary HTTP routing.</div>

<!--
Conceptual illustration of optional tool-declared argument-to-header copying, not literal header syntax. A declared sandbox_id argument can be mirrored into HTTP metadata so infrastructure can route to the appropriate sandbox backend. The body remains the source of truth; copied metadata must match it. Do not imply that arbitrary tool arguments, secrets, or the entire message are copied. Exact declaration and header syntax must be rechecked against the 2026-07-28 protocol checkout, which was unavailable in this workspace; the historical July deck was consulted read-only for context.
Static storyboard: advance manually. Client, LB/router, and backend positions are fixed across all four frames. Yellow highlights indicate inspectable HTTP metadata, not an elicitation question.
-->

---
class: message-ratio-slide
---

<div class="message-ratio-frame">
<LegacyMessageRatio>
<template #heading="{ otherCount }">
<h1><span class="message-ratio-tool-head">1 TOOL CALL.</span><span>{{ otherCount }} OTHER MESSAGES.</span></h1>
</template>
<template #footnote>Hugging Face MCP · June 2026 · Rounded average ratio, not a literal message sequence</template>
</LegacyMessageRatio>
</div>

<!--
Native Slidev migration of /home/evalstate/source/data-analysis/charts/legacy-message-animation/index.html.
Click Play to begin; Pause/Resume, Replay, Show all, speed, and optional looping are
available. Slidev owns fullscreen and navigation; there are no chart-global keyboard
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
