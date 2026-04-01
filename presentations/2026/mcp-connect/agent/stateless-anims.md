# Stateless Animation Summary

This document summarises the animation design thinking for explaining MCP's statelessness problem and the solutions.

## Two Animation Sets

The animations split into two complementary sets:

### 1. MRTR Animations (Message-Level Statelessness)
Shows how individual request/response patterns can be made stateless using cumulative context.

| File | Purpose |
|------|---------|
| `mcp-stateful-request.html` | **Before**: Single message at a time, server must remember |
| `mcp-mrtr-request.html` | **After**: Cumulative payload (chat API pattern) |
| `mcp-mrtr-flow.html` | Message flow diagram (Client ↔ Server) |
| `chat-demo.html` | Familiar chat bubbles for inspiration |
| `chat-api-view.html` | How chat APIs actually transmit (cumulative JSON) |

**Key insight**: Chat completion APIs solved this—each request contains full conversation history. MRTR applies the same pattern to MCP.

### 2. Session Affinity Animations (Deployment-Level Statelessness)  
Shows why stateful sessions break in clustered/scaled deployments.

| File | Purpose |
|------|---------|
| `session-affinity-servers-v6.html` | Full scenario: LB routes to wrong node, request fails |
| `session-scenario-1.html` | Same scenario with client state panel visible |

**The Problem Illustrated**:
1. Agent → Node 1: `initialize` → session `sess_42` created
2. Agent → Node 1: `tools/call` → Node 1 sends `ElicitationRequest`
3. Agent → **Node 3**: `ElicitationResult` → **404 Not Found!**

The load balancer has no session affinity, so the response lands on a node that doesn't know about `sess_42`.

## Visual Design Language

### Consistent Colour Coding
- **Blue** (#1976d2): Client/Agent, requests from client
- **Orange** (#ef6c00): Server cluster, server responses
- **Green**: Success states, session IDs
- **Red** (#c62828): Errors, failures

### Animation Techniques
- Messages animate along paths (Agent → LB → Node)
- Nodes highlight when "active" (processing)
- Error states use red + shake animation
- Session state panels update in real-time
- Narration bar at bottom explains each step

### Timing
- ~500ms per path segment
- 1.5-2s pause between steps (time to read narration)
- 2.5-3s pause at error state
- Automatic loop with reset

## Narrative Structure

The animations tell a three-part story:

### Part 1: The Problem (Stateful)
Current MCP requires servers to maintain session state. This causes:
- **Timeouts**: Long interactions may exceed connection limits  
- **Deployments**: Server restarts lose in-flight state
- **Scaling**: Sticky sessions required, can't load-balance freely

### Part 2: The Inspiration (Chat APIs)
Chat APIs are stateless—each request includes full conversation history. The server needs no memory.

### Part 3: The Solution (MRTR / Stateless Sessions)
Apply the chat API pattern to MCP. Two approaches:
1. **MRTR**: Each request includes cumulative context
2. **Shared session storage**: Sessions stored externally (Redis, etc.)

## Files by Modification Date

Most recent work (Feb 2, 2026):
```
15:49  session-scenario-1.html         ← latest iteration
15:42  session-affinity-servers-v6.html
14:02  mcp-stateful-request.html
13:59  mcp-mrtr-request.html
13:59  mcp-mrtr-flow.html
13:49  chat-api-view.html
13:18  chat-demo.html
```

## Usage Recommendations

**For "Why Stateless?" slide:**
Use `session-affinity-servers-v6.html` full-width—it shows the clustered server problem clearly.

**For "MRTR Solution" slides:**  
Side-by-side iframes:
```html
<div style="display:flex; gap:20px;">
  <iframe src="./animations/mcp-mrtr-flow.html" width="45%" height="500"></iframe>
  <iframe src="./animations/mcp-mrtr-request.html" width="50%" height="500"></iframe>
</div>
```

**For "Chat API Inspiration" slide:**
```html
<div style="display:flex; gap:20px;">
  <iframe src="./animations/chat-demo.html" width="48%" height="480"></iframe>
  <iframe src="./animations/chat-api-view.html" width="48%" height="480"></iframe>
</div>
```

## Design Evolution Notes

The session-affinity animations went through multiple iterations (v1→v6) refining:
- Node layout (horizontal → vertical stack)
- Session state display (under nodes → beside nodes)
- LB positioning (centered → offset for clearer return paths)
- Message content (simple → showing session IDs + request IDs)
- Narration (added step numbers, error highlighting)

The key improvement in later versions was showing the **session state on each node**, making it visually obvious why the request fails when routed to the wrong node.
