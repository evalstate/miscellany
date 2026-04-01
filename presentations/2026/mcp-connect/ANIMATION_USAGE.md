# Animations

## Story Flow

The test page (`test-chat-animation.html`) tells the story in three parts:

### 1. Before — The Problem
**Current MCP is stateful**: Each message sent individually, server must maintain state.

- Left: Message flow (same arrows as solution)
- Right: Single message at a time, replacing previous, with "server must maintain state" warning

### 2. Inspiration — Chat APIs  
**Chat APIs are stateless**: Each request includes full conversation history.

- Left: Chat bubbles appearing
- Right: Cumulative JSON payload growing

### 3. After — MRTR Solution
**MCP adopts the same pattern**: Client sends cumulative context.

- Left: Message flow (same as before)
- Right: Request #1, #2, #3 showing accumulating payload

## Files

| File | Purpose |
|------|---------|
| `animations/mcp-mrtr-flow.html` | MCP message flow (Client↔Server) |
| `animations/mcp-stateful-request.html` | **Before**: Single message at a time |
| `animations/mcp-mrtr-request.html` | **After**: Cumulative requests |
| `animations/chat-demo.html` | Chat UI bubbles |
| `animations/chat-api-view.html` | Chat cumulative payload |
| `test-chat-animation.html` | Full story: Before → Inspiration → After |

## For Presentation Slides

**Slide: Current Problem**
```html
<div style="display:flex; gap:20px; justify-content:center;">
  <iframe src="./animations/mcp-mrtr-flow.html" width="45%" height="500"></iframe>
  <iframe src="./animations/mcp-stateful-request.html" width="50%" height="500"></iframe>
</div>
```

**Slide: How Chat APIs Work**
```html
<div style="display:flex; gap:20px; justify-content:center;">
  <iframe src="./animations/chat-demo.html" width="48%" height="480"></iframe>
  <iframe src="./animations/chat-api-view.html" width="48%" height="480"></iframe>
</div>
```

**Slide: MRTR Solution**
```html
<div style="display:flex; gap:20px; justify-content:center;">
  <iframe src="./animations/mcp-mrtr-flow.html" width="45%" height="500"></iframe>
  <iframe src="./animations/mcp-mrtr-request.html" width="50%" height="500"></iframe>
</div>
```

## Timing

- Stateful demo: ~2.5s per message (replaces previous)
- MRTR demo: ~2.5s between flow messages, requests appear at 0s, 5s, 10s
- Chat demo: 2s between messages
