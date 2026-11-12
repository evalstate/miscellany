---
theme: default
title: September AGNTCON Transport
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
class: elicitation-overview
---

<div class="eo-kicker">STATELESS ELICITATION · MULTI ROUND-TRIP REQUESTS</div>

# Multiple turns. Stateless round trips.

<div class="eo-story" data-flow="modern" data-phase="independent-round-trips" role="img" aria-label="Two independent, complete round trips. First: Client requests a sandbox; Server responds with a question and context, ending that call. Between calls, Client asks User for approval and User answers Yes. No call remains pending on Server; Client holds the context. Second: Client sends a new request with original parameters, context and Yes; Server returns the sandbox and ends the second call.">
<div class="eo-actor eo-user"><svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="10" r="5"/><path d="M6 28v-3a10 10 0 0 1 20 0v3"/></svg><strong>User</strong></div>
<div class="eo-actor eo-client"><svg viewBox="0 0 32 32" aria-hidden="true"><rect x="4" y="4" width="24" height="18" rx="2"/><path d="M16 22v6M10 28h12M4 10h24"/></svg><strong>Client</strong></div>
<div class="eo-actor eo-server"><svg viewBox="0 0 32 32" aria-hidden="true"><rect x="4" y="4" width="24" height="10" rx="2"/><rect x="4" y="18" width="24" height="10" rx="2"/><path d="M9 9h1M9 23h1M16 9h7M16 23h7"/></svg><strong>Server</strong></div>
<div class="eo-lifeline eo-user-line"></div><div class="eo-lifeline eo-client-line"></div><div class="eo-lifeline eo-server-line"></div>
<div class="eo-round eo-round-one"><strong>ROUND TRIP 1</strong><span>CALL ENDS</span></div>
<div class="eo-round eo-round-two"><strong>ROUND TRIP 2</strong><span>NEW CALL</span></div>
<div class="eo-between"><strong>No call pending</strong><span>Client holds the context</span></div>
<div class="eo-message eo-server-lane eo-start"><span><b>1</b>Create sandbox</span></div>
<div class="eo-message eo-server-lane eo-left eo-need"><span><b>2</b>Question + context</span></div>
<div class="eo-message eo-user-lane eo-left eo-ask"><div class="eo-question"><b>3</b><span>Create a paid<br>GPU sandbox?</span></div></div>
<div class="eo-message eo-user-lane eo-answer"><span><b>4</b>Yes</span></div>
<div class="eo-message eo-server-lane eo-retry"><span><b>5</b>Original request + context + Yes</span></div>
<div class="eo-message eo-server-lane eo-left eo-done"><span><b>6</b>Sandbox ready</span></div>
</div>

<!--
Fresh one-slide explanation, not an old/new comparison. Two separate boxes are
two independent request/response pairs; both terminate. The gap is the user turn,
not a suspended server call. The server lifeline deliberately breaks across it.
“Stateless” describes continuation of this conversation: the next request carries
everything needed to resume, so a different server instance may process it without
access to the prior pending call. It does not prohibit application state, remove
authorization checks, or remove server-side enforcement of single-use invariants.
Read the numbered arrows top to bottom. Client mediates both the question and the answer; there is no direct
Server ↔ User channel in this form-mode example. User may also decline or cancel;
we show the accepted path. No sandbox is created before approval. Dialog text is
illustrative; nothing on this slide is an interactive control.

Illustrated wire behavior is modern 2026-07-28 MRTR:
1. Client initiates tools/call with the original sandbox parameters.
2. Server responds with InputRequiredResult (resultType input_required), including
   a keyed inputRequests entry for elicitation/create and optional requestState.
   This is a RESPONSE that ends the initial call, not a reverse JSON-RPC request.
3–4. Client presents the server's question to User and gathers an ElicitResult.
   There is no pending call on Server during this user interaction. The Client UI
   must identify the requesting server and allow decline/cancel. This example
   asks for a decision, never payment credentials.
5. Client retries tools/call with a NEW JSON-RPC id, original parameters, keyed
   inputResponses, and EXACT unchanged opaque requestState if supplied. If absent,
   requestState must not be added. Required per-request _meta is omitted visually.
   “Request + context + Yes” / “Question context + Answer” is conceptual shorthand,
   not a literal echo of the question or a growing transcript. Client must not
   inspect or modify opaque state. Further input_required rounds are possible.
6. Server validates the inputs, authorization and state integrity where required,
   creates the approved sandbox, and returns the final result to Client.

Sources rechecked: ~/reference/modelcontextprotocol/docs/specification/2026-07-28/
basic/patterns/mrtr.mdx and client/elicitation.mdx. Static: no timers or animation.
-->

---
class: mrtr-slide
---

<div class="mrtr-kicker">MODERN ELICITATIONS <span>OLD WAY</span></div>

# Ask back. Keep the call open.

<div class="mrtr-story" data-flow="old" data-phase="shared-state" role="img" aria-label="Legacy: Client calls Server to create a sandbox. Server asks for permission in a reverse request. While the user decides, the original call and question remain open. Client answers Yes, then Server returns the sandbox.">
<div class="mrtr-actor mrtr-client">Client</div><div class="mrtr-actor mrtr-server">Server</div>
<div class="mrtr-rail mrtr-client-rail"></div><div class="mrtr-rail mrtr-server-rail"></div>
<div class="mrtr-message mrtr-request"><span>Request: <b>Create sandbox</b></span></div>
<div class="mrtr-message mrtr-left mrtr-question"><div class="mrtr-question-card"><div><strong>REQUEST</strong> back to the client</div><span>“Create a paid GPU sandbox?”</span></div></div>
<div class="mrtr-pause"><div class="mrtr-user">User decides…<span>Conversation state on both sides</span></div><div class="mrtr-wait"><strong>OPEN / WAITING</strong><span>Original call still pending</span></div></div>
<div class="mrtr-message mrtr-retry"><strong>Reply to the pending question</strong><span class="mrtr-example">Yes</span></div>
<div class="mrtr-message mrtr-left mrtr-result"><span>Final response: <b>Sandbox ready</b></span></div>
</div>

<!--
Legacy 2025-11-25. Read this diagram top to bottom. Time flows down; Client stays
left and Server stays right on both slides. Arrows are logical messages, not
separate physical connections. This is the nested, form-mode elicitation path.

Client sends tools/call. Server sends a separate elicitation/create request while
tools/call remains pending. Client displays the question; Server retains the
pending continuation. “Yes” stands for an ElicitResult (action + structured
content), correlated to the server's elicitation request; it does not repeat the
original tool parameters. Server then completes the original tools/call.

The problem is a live conversation spanning both sides, not that a server thread
must block or that all server work stops. Scaling/restarts require preserving or
coordinating that pending state (e.g. affinity or shared storage). No sandbox is
created before approval. This example requests a decision, not payment credentials.

Verified against ~/reference/modelcontextprotocol/docs/specification/2025-11-25/client/elicitation.mdx.
Static illustration, no animation or interactive dialog.
-->

---
class: mrtr-slide
---

<div class="mrtr-kicker">MODERN ELICITATIONS <span>NEW WAY · MULTI ROUND-TRIP REQUESTS</span></div>

# Return the question. End the call.

<div class="mrtr-story" data-flow="modern" data-phase="cumulative-retry" role="img" aria-label="MRTR: Client requests a sandbox. Server returns the question and context, ending call one. User decides with no call pending on the Server. Client makes a new request carrying the original sandbox request plus context plus Yes. Server returns the sandbox, ending call two.">
<div class="mrtr-actor mrtr-client">Client</div><div class="mrtr-actor mrtr-server">Server</div>
<div class="mrtr-rail mrtr-client-rail"></div><div class="mrtr-rail mrtr-server-rail"></div>
<div class="mrtr-message mrtr-request"><span>Request: <b>Create sandbox</b></span></div>
<div class="mrtr-message mrtr-left mrtr-question"><div class="mrtr-question-card"><div><strong>RESPONSE</strong> with question + context</div><span>“Create a paid GPU sandbox?”</span></div></div>
<div class="mrtr-pause"><div class="mrtr-user">User decides…<span>Context waits with the client</span></div><div class="mrtr-free"><strong>CALL ENDED</strong><span>No pending call to remember</span></div></div>
<div class="mrtr-message mrtr-retry"><strong>New request = original + context + answer</strong><span class="mrtr-bundle"><b>Create sandbox</b><i>+</i><b class="mrtr-context">context</b><i>+</i><b class="mrtr-answer">Yes</b></span></div>
<div class="mrtr-message mrtr-left mrtr-result"><span>Final response: <b>Sandbox ready</b></span></div>
</div>

<!--
Modern 2026-07-28 MRTR basic workflow. Read top to bottom, comparing the same four
message directions and same yellow question card with the legacy slide.

Round trip 1: tools/call receives an InputRequiredResult (resultType:
input_required), with a keyed inputRequests map containing elicitation/create,
and optionally requestState. This is a response, not a separate reverse JSON-RPC
request. It terminates call 1; the user interaction happens between independent
calls. “CALL ENDED” describes this operation, not the whole server.

Round trip 2: Client retries tools/call with a NEW JSON-RPC id, the original
parameters, keyed inputResponses (e.g. approval: {action: accept, content: ...}),
and the exact, unchanged opaque requestState if supplied. If requestState was
absent, Client MUST NOT add it. Required per-request _meta is omitted here.

“Context + Yes” / “Question context + Answer” is conceptual shorthand: not a
literal echo of the question and not a growing chat transcript. Each retry carries
the information needed to continue. More input_required rounds may follow; the
latest returned opaque state may encapsulate prior progress. Do not suggest the
client can inspect or merge requestState, or that all prior payloads must accumulate.

The retry is self-contained for this continuation: another server instance can
handle it without the original pending call. MRTR enables this stateless workflow;
it does not ban all application state or remove business/security obligations.
Server must validate inputs and authorization and protect requestState integrity
where it influences business logic; single-use invariants still need enforcement.

Verified against ~/reference/modelcontextprotocol/docs/specification/2026-07-28/basic/patterns/mrtr.mdx
and 2026-07-28/client/elicitation.mdx. No sandbox is created before approval.
Static illustration, no animation or interactive dialog.
-->

---
class: elicit-editorial
---

<div class="ee-masthead"><span>MODERN ELICITATIONS</span><b>01 — OLD WAY</b></div>

<div class="ee-composition" data-flow="old" data-phase="shared-state">
<section class="ee-copy">

# Hold the<br>conversation.

<div class="ee-explanation">The question is a<br><strong>request back</strong><br>to the client.</div>
<div class="ee-principle"><i></i><span>Both sides keep<br>conversation state.</span></div>
</section>
<section class="ee-diagram" aria-label="One open call: Client requests a sandbox, Server asks back, both sides retain the conversation while the user decides, Client answers, then Server completes the original call.">
<div class="ee-actors"><strong>Client</strong><strong>Server</strong></div>
<div class="ee-open-call">
<div class="ee-round-label"><span>ONE OPEN CALL</span><i></i></div>
<div class="ee-message ee-outbound"><span>Create sandbox</span><i>→</i></div>
<div class="ee-question"><i>←</i><div><strong>REQUEST BACK</strong><span>Create a paid GPU sandbox?</span></div></div>
<div class="ee-wait"><span>User decides…</span><strong>OPEN / WAITING</strong></div>
<div class="ee-message ee-outbound"><span>Yes <small>· answer only</small></span><i>→</i></div>
<div class="ee-message ee-inbound"><i>←</i><span>Sandbox ready <small>· call ends</small></span></div>
</div>
</section>
</div>

<!--
Design variant of the legacy 2025-11-25 slide. Static. Client stays left and Server
right. Read the exchange card top to bottom. The single enclosing boundary is the
lifetime of the original tools/call, not a physical connection or a shared database.

Server sends a separate elicitation/create request while tools/call remains
pending. The yellow question is a new server-to-client request. Client displays it
and returns an ElicitResult; “Yes” is conceptual shorthand, not a wire payload.
Both sides retain parts of the live conversation. OPEN / WAITING applies to this
call, not the entire server or a necessarily blocked thread. After approval,
Server creates the sandbox and completes the original call. No payment credentials
are requested. Scaling or restarts require preserving/coordinating pending state.

Source: ~/reference/modelcontextprotocol/docs/specification/2025-11-25/client/elicitation.mdx.
-->

---
class: elicit-editorial
---

<div class="ee-masthead"><span>MODERN ELICITATIONS</span><b>02 — MULTI ROUND-TRIP REQUESTS</b></div>

<div class="ee-composition" data-flow="modern" data-phase="cumulative-retry">
<section class="ee-copy">

# Carry the<br>context.

<div class="ee-explanation">The question is<br><strong>the response.</strong><br>That call is done.</div>
<div class="ee-principle"><i></i><span>The next request<br>carries what it needs.</span></div>
</section>
<section class="ee-diagram" aria-label="Two independent round trips. First: request a sandbox, receive the question and context, end call. Between calls the client gathers the answer; no call is pending on the Server. Second: send the original request plus context plus answer, receive the sandbox.">
<div class="ee-actors"><strong>Client</strong><strong>Server</strong></div>
<div class="ee-round ee-first">
<div class="ee-round-label"><span>ROUND TRIP 1</span><b>CALL ENDS</b></div>
<div class="ee-message ee-outbound"><span>Create sandbox</span><i>→</i></div>
<div class="ee-question"><i>←</i><div><strong>RESPONSE <em>+ context</em></strong><span>Create a paid GPU sandbox?</span></div></div>
</div>
<div class="ee-between"><span>User decides…</span><strong>No call pending</strong></div>
<div class="ee-round ee-second">
<div class="ee-round-label"><span>ROUND TRIP 2</span><b>NEW CALL</b></div>
<div class="ee-message ee-outbound ee-retry"><div><strong>Original request + context + answer</strong><span>Create sandbox <i>+</i> <b>context</b> <i>+</i> <b>Yes</b></span></div><i>→</i></div>
<div class="ee-message ee-inbound"><i>←</i><span>Sandbox ready <small>· call ends</small></span></div>
</div>
</section>
</div>

<!--
Design variant of the modern 2026-07-28 MRTR basic workflow. Static. Two separate
exchange cards show independent request/response pairs, with the user interaction
in the gap. The same yellow question is now inside a response, not a reverse call.

Round 1: tools/call receives InputRequiredResult (resultType input_required), a
keyed inputRequests map containing elicitation/create, and optional requestState.
The response terminates call 1. No call pending refers to this operation only.
Round 2: Client retries tools/call with a NEW JSON-RPC id, original parameters,
keyed inputResponses, and the EXACT unchanged opaque requestState if supplied.
If requestState was absent, it MUST NOT be added. Required per-request _meta is
omitted in this conceptual diagram. “Yes” stands for an ElicitResult.

“Question context + Answer” / “Original request + context + answer” is conceptual
shorthand, not a literal echo of the question or a growing transcript. The latest
opaque state may encapsulate prior progress; clients must not inspect or merge
it. Further input_required rounds are possible. The retry is self-contained for
this continuation and can be handled by another server instance. MRTR does not
remove application state, authorization checks, or single-use enforcement.
Server validates inputs and protects requestState integrity where required.

Source: ~/reference/modelcontextprotocol/docs/specification/2026-07-28/basic/patterns/mrtr.mdx.
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
