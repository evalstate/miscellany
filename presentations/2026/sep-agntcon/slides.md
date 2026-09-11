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
Sources checked against local modelcontextprotocol/docs/specification: 2025-11-25/client/elicitation.mdx and 2026-07-28/basic/patterns/mrtr.mdx.
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
