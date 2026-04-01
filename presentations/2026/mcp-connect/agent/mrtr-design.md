# MRTR Animation Design Summary

## Purpose

These animations were created to explain the **Multi-Round-Trip Request (MRTR)** proposal for MCP transports. The goal is to help audiences intuitively understand:

1. The problem with the current stateful approach
2. How chat APIs solve a similar problem
3. How MRTR applies the same solution to MCP

## The Problem: Stateful MCP

Currently, MCP server-to-client requests (Elicitation, Sampling) are handled stateully:

```
Client → Server:  CallToolRequest
Server → Client:  ElicitationRequest
Client → Server:  ElicitationResponse      ← Just the response, no context
Server → Client:  SamplingRequest
Client → Server:  SamplingResponse         ← Just the response, no context
Server → Client:  CallToolResult
```

Each message is sent **individually**. The server must maintain state to remember:
- The original tool call
- The elicitation it sent and the response it received
- The sampling request it sent

This creates problems:
- **Timeouts**: Long-running interactions may exceed connection limits
- **Deployments**: Server restarts lose all in-flight state
- **Scaling**: Sticky sessions required, can't load-balance freely
- **Complexity**: Server implementations must handle state management

## The Inspiration: Chat APIs

Chat completion APIs (OpenAI, Anthropic, etc.) solved this long ago. Each request includes the **full conversation history**:

```
Request 1:  [user message]
Request 2:  [user message, assistant message, user message]
Request 3:  [user message, assistant message, user message, assistant message, user message]
```

The server is stateless — all context needed to generate a response is in the request itself.

## The Solution: MRTR

MRTR applies the same pattern to MCP. Each client request includes **cumulative context**:

```
Request 1:  [CallToolRequest]

Request 2:  [CallToolRequest, ElicitationRequest, ElicitationResponse]

Request 3:  [CallToolRequest, ElicitationRequest, ElicitationResponse, 
             SamplingRequest, SamplingResponse]
```

The server can now be stateless. It receives everything it needs to continue processing.

## Animation Files

| File | Description |
|------|-------------|
| `animations/mcp-mrtr-flow.html` | Message flow diagram (Client ↔ Server arrows) |
| `animations/mcp-stateful-request.html` | **Before**: Shows one message at a time replacing previous |
| `animations/mcp-mrtr-request.html` | **After**: Shows cumulative request payloads |
| `animations/chat-demo.html` | Chat UI with message bubbles |
| `animations/chat-api-view.html` | Chat API cumulative payload |
| `test-chat-animation.html` | Full story: Before → Inspiration → After |

## Visual Design Decisions

### Color Coding
- **Blue**: Client requests (CallToolRequest, ElicitationResponse, SamplingResponse)
- **Orange**: Server requests (ElicitationRequest, SamplingRequest)
- **Green**: Client responses / success states

### "Before" Animation
- Single message displayed at a time, **replacing** previous
- State indicator at bottom with dots lighting up
- Emphasizes server must "remember" each step
- Warning styling to indicate this is problematic

### "After" Animation  
- Three request groups shown sequentially
- Prior context shown **dimmed** to indicate "already sent"
- New items highlighted
- Clear visual of payload growing with each round trip

### Chat Reference
- Familiar chat bubble UI on left
- JSON payload on right growing with each message
- Helps audience connect MRTR to something they already understand

## Timing

All animations loop automatically:
- ~2-2.5 seconds between messages
- 4 second pause after last message
- Fade out and restart

Designed for presentation use — can leave on screen while speaking.

## Usage in Presentation

Three slides recommended:

1. **"Current MCP: The Problem"** — stateful animation
2. **"How Chat APIs Work"** — chat reference  
3. **"MRTR: The Solution"** — cumulative animation

Each slide uses side-by-side iframes:
- Left: The conversation/flow view
- Right: What's actually being transmitted

See `ANIMATION_USAGE.md` for embed code snippets.
