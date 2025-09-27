---
marp: true
theme: freud
paginate: true

---

stdio has blocking behaviour.
one-shot.



---

## What happened here? 

GRAPH OF INBOUND TRAFFIC, THEN DROPPING... WHAT HAPPENED?

<!-- educational, bit of data, what next -->

---

## Experience is it Just works

<!-- when i was here last time, i said the great thing about MCP was it just worked -->
<!-- we have lots of options for Streamable HTTP and not much time, so we'll build up -->
Decision Point - How do I want to connect? We have a lot of options for Streamable HTTP and not much time. MCP is bi-directional, so with ordinary HTTP it's just Request/Response - how can this work for MCP?


---

## Methods Supported.  

[Methods Supported Chip]

Stateless JSON-RPC. All we can do is respond. Fine! 

If you don't need state. MCP SDK can still do some of the lifting for you.

---

## Sending Tool Progress Notifications

<!-- we need to decide early whether to SSE or not -->
If we have a Tool that takes a while to execute, then we might want to send Progress Updates. 

Respond with an SSE Stream, send associated events, then send the Result and close.

We need to decide on the. The SDK _should_ reset timeout counters on received events....

---

# WE ARE CROSSING A HUGE BRIDGE

---

## Request associated Sampling and Elicitation

If the Server wants to use Sampling or Elicitation we can send those on the associated SSE stream. The Client handles them, and sends the response back with a POST message.

**MAKE SURE YOU HAVE ALLOCATED A SESSION ID** so the Response can be associated with the request. The server responds with a 202 only (no new streams etc).

---

# Allocating an Mcp-Session-Id

### Server Controlled
### Only need it request/response association
### Three way Message Dance to do this.
### If you are using SDK this will all happen anyway.

---

## Server initiated Sampling and Elicitation, 

In this case we need to maintain an open GET channel. 

Client Listens, handles requests it receives, and POSTs back as previously. 

-SDK behaviour:
- Python ALWAYS opens this up on the Server.

---

## Change Notifications Subscriptions.

Per-Server state - now I need to think about saving state somehow - what things are switched on/off? So persistence becomes the issue. 

Although these are separate methods, the request is the same: re-read.

(Note this isn't an actual requirement - the notification just says read the list again - but practically....)

---

# Session Deletion

At the end. Reconstruct that state. 

---

# Identity and State

Hugging Face MCP Server in Production allows configuration of Tools per User. We use the OAuth identity to enable that. 

OAUTH : Persistent Settings for User
SessionID : Host/Conversational State

---

# Host Application and User Behaviour. 

- Users install MCP Servers and Host Applications create, but don't delete Sessions.
- If you are using MCP Sessions this creates resource demands.

---

## PAIN POINTS (SO FAR)

- Silent Failures
- Using the wrong "Channel"
- Maintaining the GET channel Open
- Lack of PING

---

## In Practice

Ran the Hugging Face server. 
Enormous amount of traffic

- Tools List refresh could be handled via a Header. 
- Client implementations call things that aren't supported.
- 

A simple HTTP Request 
 - Has to handle the JSON-RPC body to identify the requested operation. (Writing low-value code to fulfil the transport requirements).

 - 

---

# In Practice

## Data c/o Jeff Richter , Microsoft Azure.

---

# In Practice

Load Balancing and Fault Tolerance

1. Sampling/Elicitation - we expect a response on our own Thread of execution.
1. 

---

STDIO; serializing requests
blocking behaviour, "taking turns". Use of STDOUT breaks connection.

Single Process, Single "User".
No "Sessions", isolation through instancing.

I want to connect by LLM . "just works"

How does it work?

Trade-offs

Developer Experience - good.


How to use the correct channel.
    The requestId is a hint to the channel.

Differences between SDKs.

---

### section


What do we mean by Statelessness of the Protocol vs. Sessions

- Interesting Stats
- Clients that Delete Sessions
- "Chattiness"
- User Behaviour

---

Options. 

Supporting Multiple Threads/Agents.

Optimized for what Clients/Users do?


---

Data 

- Without roadmap from claude.ai it's hard to know what mcp server developers can rely on .
- surety from 
- Coding Tools are reasonably well supported.
- IDEs. Utility of MCP constrained due to Tool confusion; developer experience building custom solutions. Commoditized servers.

---

