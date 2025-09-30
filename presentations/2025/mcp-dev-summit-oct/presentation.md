---
marp: true
theme: freud
paginate: false
---



<style>
     .cite-author {  
      text-align        : right;
   }
   .cite-author:after {
      color             : orangered;
      font-size         : 125%;
      /* font-style        : italic; */
      font-weight       : bold;
      font-family       : Cambria, Cochin, Georgia, Times, 'Times New Roman', serif; 
      padding-right     : 130px;
   }
   .cite-author[data-text]:after {
      content           : " - "attr(data-text) " - ";      
   }

   .cite-author p {
      padding-bottom : 40px
   }

   /* Bottom-positioned wide image */
   .bottom-image {
     position: absolute;
     bottom: 20px;
     left: 50%;
     transform: translateX(-50%);
     width: calc(100% - 40px);
     max-width: 95%;
   }

   .bottom-image img {
     width: 100%;
     height: auto;
     object-fit: contain;
   }

   /* Alternative: Fixed to bottom with no padding */
   .bottom-image-flush {
     position: absolute;
     bottom: 0;
     left: 0;
     right: 0;
     width: 100%;
   }

   .bottom-image-flush img {
     width: 100%;
     height: auto;
     object-fit: contain;
   }

</style>

<!-- _class: titlepage -->

<div class="title"         > Streams, Sessions, Stats: Transport and Client Behaviour</div>
<div class="subtitle"      > MCP Dev Summit, London   </div>
<div class="author"        > Shaun Smith                       </div>
<div class="date"          > October 2025                                    </div>
<table class="social-table">
  <tbody>
    <tr>
      <td><img src="./images/huggingface-mark-logo.svg" alt="Hugging Face" /></td>
      <td><a class="organization" href="https://huggingface.co/evalstate">huggingface.co/evalstate</a></td>
    </tr>
    <tr>
      <td><img src="./images/github-mark.svg" alt="GitHub" /></td>
      <td><a class="organization" href="https://github.com/evalstate">github.com/evalstate</a></td>
    </tr>
    <tr>
      <td><img src="./images/xcom-logo-black.png" alt="X" /></td>
      <td><a class="organization" href="https://x.com/evalstate">x.com/evalstate</a></td>
    </tr>
  </tbody>
</table>

<!-- -->

---

## Huge Traffic Drop!

![w:1200](./images/traffic-aug.png)

<!-- educational, bit of data, what next -->

---

# Simplest Streamable HTTP MCP Server


<!-- Stateless JSON-RPC. All we can do is respond. Fine!

If you don't need state. MCP SDK can still do some of the lifting for you. -->
<!-- when i was here last time, i said the great thing about MCP was it just worked -->
<!-- we have lots of options for Streamable HTTP and not much time, so we'll build up -->

### 

<div class="columns">

<div>


### Set up an ordinary HTTP `POST` Handler that directly returns the JSON-RPC in the response.

### If all you need is Tools, Prompts, Resources (and Completions) this is enough!

###  `enableJSONResponse` can give 10-20% improvement in throughput and latency

</div>

<div align="center">

![w:500](./images/diag_rpc_post.png)


```typescript
new StreamableHTTPServerTransport(){
    // Note: Not Default!
    enableJSONResponse: true 
}
```


</div>

</div>

<!-- <div class="bottom-image">
  <img src="./images/capabilities_stateless_1.png" alt="MCP Capabilities - Stateless" />
</div>
-->

---

# Adding Tool Progress Notifications


<!-- Stateless JSON-RPC. All we can do is respond. Fine!

If you don't need state. MCP SDK can still do some of the lifting for you. -->


<div class="columns">

<div>


### Long running tools (like Image Generation) can send Progress Updates.

### Rather than responding directly, we respond with an SSE Event Stream.

### MCP Server should make sure Notifications are sent on the correct channel.

```
requestHandlerExtra etc.

```


</div>

<div>

DIAGRAM OF NOTIFICATIONS HERE


</div>

</div>

<!-- >
<div class="bottom-image">
  <img src="./images/capabilities_stateless_1.png" alt="MCP Capabilities - Stateless" />
</div>
-->

---

# Server Request to Client - (Client Initiated)


<div class="columns">

<div>

### MCP Servers can make Sampling and Elicitation requests __to__ the Client associated with a Tool Call.

### Server sends the Elicitation Request via the Post SSE response. 

### Result is POSTed back to the MCP Server using the Request ID for association - and returns a 202.

### Server then returns the Result via the Post SSE Channel.

</div>

<div>

PICTURE OF AN ELICITATION TAKING PLACE

</div>

</div>


---


# Server Request to Client (Server Initiated)


![w:800](./images/diag_full_setup.png)
<div class="columns">

<div>

### To use Resource Subscriptions, List Change Notifications or Server-Initiated Sampling/Elicitation, we send the request over an HTTP GET SSE Stream the client opens after initialize.




</div>

<div>



```typescript

// big SDK Gotcha!


```


</div>

</div>


---




# The Full MCP Experience


<div class="columns">

<div>

### Now we add a `GET` handler to maintain an open stream to send notifications to the Host.

### Server needs to make sure Notifications and Requests are sent via the correct channel for correlation.

</div>

<div>



```typescript

// big SDK Gotcha!


```


</div>

</div>



<div class="bottom-image">
  <img src="./images/capabilities_everything_4.png" alt="MCP Capabilities - Stateless" />
</div>

---


# Ping!

The Host can `POST` to the Server to tell it's alive
The Server can tell whether or not the Host is Responsive.
Host Ping tells you if the POST endpoint works.
Server Ping tells you if the GET channel is open.
<div class="columns">

<div>

#### Examples: `webcam.fast-agent.ai/mcp`
#### Hugging Face MCP Server in StreamableHTTP Mode.

![w:500](./images/fa_ping.png)

</div>

<div>

![w:400](./images/ping.png)


</div>

</div>


---

## PAIN POINTS (SO FAR)

- Using the wrong "Channel"
- Maintaining the GET channel Open
- Handling PING failures.
- Python ALWAYS opens GET when used as a Server.
- Silent failures in SDK on transmission failures

- Jeff R. [Elicitation for agreeing to return PII]
- Jsff R. [MS Teams] conversation to file defect, make sure template is filled out (select template to fill out defect template).

---

![w:700](./images/traffic-aug.png)
![w:500](./images/claude_enhancements.png)


---



### MCP Method Call Ratios / Hugging Face MCP Server)
<!-- _class: mcp-features -->

<div class="columns">

<div>

<table class="show-headers">
<thead>
<tr><th>MCP Method</th><th>Aug</th><th>Sep</th></tr>
</thead>
<tbody>
<tr><td><code>initialize</code></td><td>1.000</td><td>1.000</td></tr>
<tr><td><code>tools/list</code></td><td>0.400</td><td>1.175</td></tr>
<tr><td><code>notifications/initialized</code></td><td>0.995</td><td>0.982</td></tr>
<tr><td><code>prompts/list</code></td><td>1.081</td><td>0.685</td></tr>
<tr class="warning-row"><td><code>resources/list</code></td><td>1.039</td><td>0.606</td></tr>
<tr><td><code>notifications/cancelled</code></td><td>0.063</td><td>0.150</td></tr>
<tr class="highlight-row"><td><code>Actual Tool/Prompt Calls</code></td><td>0.011</td><td>0.032</td></tr>
<tr><td><code>ping</code></td><td>0.001</td><td>0.027</td></tr>
<tr class="warning-row"><td><code>resources/templates/list</code></td><td>0.000</td><td>0.022</td></tr>
</tbody>
</table>

</div>

<div>

### What This Shows

Initialization sequence is usually at least 3 calls. 

MCP has significant overhead. For September:
 - We see __~3__ Tool/Prompt Calls per 100 Initialize events.
 - and __~165__ MCP Method Calls for every Tool/Prompt Call. 
 - compared to __547__ in August!


The `resource` methods aren't supported, yet clients still request them.

</div>

</div>


---

# Initialize != Usage

<div class="columns">

<div>

## Interactive Hosts (IDE/UI)

### Sessions may remain open for minutes/hours. 

### Tool/Prompt Usage is User driven so idle sessions are normal.

### VSCode has a high "efficiency ratio"

### Examples are __claude-ai__, __windsurf-client__ and __mcp-inspector__.

</div>

<div>

## Gateways / Embedded Hosts

### MCP Server is used as part of an automation, from a "Remote" Host or a gateway. 

### Burst of activity to Initialize and Call Tool that typically lasts under 5 seconds. 

### Examples are __openai-mcp__, __docker-mcp-gateway__, and __javelin-mcp-client__.

</div>


</div>


---

### Client Top 20 (by Usage) with Capabilities / Approx ~1.5m Sessions Sep '25

<!-- _class: top-clients -->
<!--
Usage Guide:
- Session deletion icon: <span class="icon-delete"><img src="./images/trash-2.svg" alt="Delete" /></span>
- Alert/warning icon: <span class="icon-alert"><img src="./images/circle-alert.svg" alt="Alert" /></span>
- Enabled capability: <span class="capability-icon"><img src="./images/[icon].svg" /></span>
- Disabled capability: <span class="capability-icon disabled"><img src="./images/[icon].svg" /></span>
  - folders.svg = Roots
  - cpu.svg = Sampling
  - message-circle-question-mark.svg = Elicitation
- Icons are placed in the <div class="client-icons"> container
- Enabled capabilities: bold blue color with light blue background
- Disabled capabilities: subtle gray with minimal background
- Delete/Alert icons: red color with light red background
-->

<table>
<thead>
<tr><th>#</th><th>Client</th><th>Icons</th><th>#</th><th>Client</th><th>Icons</th></tr>
</thead>
<tbody>
<tr>
  <td>1</td><td>claude-ai</td><td><div class="client-icons"><span class="capability-icon disabled"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
  <td>11</td><td>docker-mcp-gateway</td><td><div class="client-icons"><span class="icon-delete"><img src="./images/trash-2.svg" alt="Delete" /></span><span class="capability-icon"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
</tr>
<tr>
  <td>2</td><td>lmstudio-mcp-bridge</td><td><div class="client-icons"><span class="capability-icon disabled"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
  <td>12</td><td>groq-mcp-client</td><td><div class="client-icons"><span class="capability-icon disabled"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
</tr>
<tr>
  <td>3</td><td>Visual Studio Code</td><td><div class="client-icons"><span class="capability-icon"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
  <td>13</td><td>openai-mcp</td><td><div class="client-icons"><span class="icon-delete"><img src="./images/trash-2.svg" alt="Delete" /></span><span class="capability-icon disabled"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
</tr>
<tr>
  <td>4</td><td>test-client</td><td><div class="client-icons"><span class="icon-alert"><img src="./images/circle-alert.svg" alt="Alert" /></span><span class="capability-icon"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
  <td>14</td><td>Cherry Studio</td><td><div class="client-icons"><span class="capability-icon disabled"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
</tr>
<tr>
  <td>5</td><td>cursor-vscode</td><td><div class="client-icons"><span class="icon-alert"><img src="./images/circle-alert.svg" alt="Alert" /></span><span class="capability-icon"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
  <td>15</td><td>fast-agent-mcp</td><td><div class="client-icons"><span class="icon-delete"><img src="./images/trash-2.svg" alt="Delete" /></span><span class="capability-icon disabled"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
</tr>
<tr>
  <td>6</td><td>claude-code</td><td><div class="client-icons"><span class="capability-icon"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
  <td>16</td><td>codex (via mcp-remote)</td><td><div class="client-icons"><span class="capability-icon disabled"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
</tr>
<tr>
  <td>7</td><td>mcp (via mcp-remote)</td><td><div class="client-icons"><span class="capability-icon"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
  <td>17</td><td>claude-ai (via mcp-remote)</td><td><div class="client-icons"><span class="capability-icon disabled"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
</tr>
<tr>
  <td>8</td><td>Anthropic/ClaudeAI</td><td><div class="client-icons"><span class="capability-icon disabled"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
  <td>18</td><td>@n8n/langchain.mcpClientT</td><td><div class="client-icons"><span class="icon-alert"><img src="./images/circle-alert.svg" alt="Alert" /></span><span class="capability-icon disabled"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
</tr>
<tr>
  <td>9</td><td>mcp</td><td><div class="client-icons"><span class="icon-delete"><img src="./images/trash-2.svg" alt="Delete" /></span><span class="capability-icon"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
  <td>19</td><td>lobehub-mcp-client</td><td><div class="client-icons"><span class="capability-icon disabled"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
</tr>
<tr>
  <td>10</td><td>Manus</td><td><div class="client-icons"><span class="icon-delete"><img src="./images/trash-2.svg" alt="Delete" /></span><span class="capability-icon disabled"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
  <td>20</td><td>dev.warp.Warp-Stable</td><td><div class="client-icons"><span class="capability-icon disabled"><img src="./images/folders.svg" alt="R" /></span><span class="capability-icon disabled"><img src="./images/cpu.svg" alt="S" /></span><span class="capability-icon disabled"><img src="./images/message-circle-question-mark.svg" alt="E" /></span></div></td>
</tr>
</tbody>
</table>

<div class="legend">
  <div class="legend-item">
    <span class="capability-icon"><img src="./images/folders.svg" alt="R" /></span>
    <span class="legend-label">Roots</span>
  </div>
  <div class="legend-item">
    <span class="capability-icon"><img src="./images/cpu.svg" alt="S" /></span>
    <span class="legend-label">Sampling</span>
  </div>
  <div class="legend-item">
    <span class="capability-icon"><img src="./images/message-circle-question-mark.svg" alt="E" /></span>
    <span class="legend-label">Elicitation</span>
  </div>
  <div class="legend-item">
    <span class="icon-delete"><img src="./images/trash-2.svg" alt="Delete" /></span>
    <span class="legend-label">Session Deletion</span>
  </div>
  <div class="legend-item">
    <span class="icon-alert"><img src="./images/circle-alert.svg" alt="Alert" /></span>
    <span class="legend-label">Invalid Capabilities</span>
  </div>
</div>


---

<!-- _class: mcp-features -->

# MCP Client Primitives

| Icon | Feature | Usage |
| --- | --- | --- |
| <span class="feature-icon"><img src="./images/folders.svg" alt="Roots" width="100%" /></span> | <span class="cell-title">Roots</span> | _7.1%_ of all sessions, _33.6%_ of sessions that use tools <br /> Not currently useful for Remote Servers |
| <span class="feature-icon"><img src="./images/cpu.svg" width="100%" alt="Sampling" /></span> | <span class="cell-title">Sampling</span> | _0.9%_ of all sessions, _22.2%_ of sessions that use tools |
| <span class="feature-icon"><img src="./images/message-circle-question-mark.svg" width="100%" alt="Elicitations" /></span> | <span class="cell-title">Elicitations</span> | _3.2%_ of all sessions, _21.6%_ of sessions that use tools |
| <span class="feature-icon"><img src="./images/trash-2.svg" alt="Session Deletion" width="100%" /></span> | <span class="cell-title">Session Deletion</span> | 4 of the top 20 clients delete sessions, only _6.64%_ of sessions get deleted overall |


---

# Try it out!

## Hugging Face MCP Server supports STDIO, and Stateful + Stateless Deployment Modes. One Click Deployment to a FreeCPU Space via Docker.



---



# Allocating a "GET" to each user

## Data c/o Jeff Richter , Microsoft Azure.

Jeff's Graph Here showing top-out

## GET Disconnect behaviour (silent failures).

---

# Thread of Execution? Handling POST Responses.

## Hosts responding to a Request from _either_ the `GET` or `POST SSE` stream send their Response with a new `POST` to which the MCP Server responds with a 202. 

## In a multi-server environment (e.g. load balanced), it requires either a sticky session or.

## Scale 


---

# Sessions

## Typically OAuth used for Identity, Mcp-Session-Id used for correlation .

## Causes the "Conversational Context" problem.

## VSCode uses a new Connection per Conversation Thread (find discord image)

---

# Relevant SEPs

## Initialize Language Wording - Allow us to 

## Elevate Sessions #1364 - Host controlled Sessions, SessionId to the Data, not Transport layer.

## SEP #1442 - 

## HTTP/gRPC Transport Proposals - Reduce overhead

---



## Request associated Sampling and Elicitation

If the Server wants to use Sampling or Elicitation we can send those on the associated SSE stream. The Client handles them, and sends the response back with a POST message.

**MAKE SURE YOU HAVE ALLOCATED A SESSION ID** so the Response can be associated with the request. The server responds with a 202 only (no new streams etc).

---

# Allocating an Mcp-Session-Id

### Server Controlled
### Only need it request/response association
### If you are using SDK this will all happen anyway.
### We need to decide on the. The SDK _should_ reset timeout counters on received events....

---

## Server initiated Sampling and Elicitation, 

In this case we need to maintain an open GET channel. 

Client Listens, handles requests it receives, and POSTs back as previously. 

- SDK behaviour:


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

| OAuth / API Key | SessionID |
|----------------|------------|
| User Identity  | Conversational State |
| (MCP Server Configuration) | ???? |

---

# Host Application and User Behaviour. 

- Users install MCP Servers in Host Applications create, but don't delete Sessions.
- If you are using MCP Sessions this creates resource demands.

---

Back to the Chart of "what happened".

---

# 

# For simple JSON-RPC, MCP has overhead

## Lots of calls (e.g. 2 step initialization, followed by prompts listings etc.)

## Having to inspect JSON-RPC packets for routing rather than typical HTTP handler patterns (makes caching even of static things like Tool Lists hard)
## Has to handle the JSON-RPC body to identify the requested operation. (Writing low-value code to fulfil the transport requirements).


---

## Client / User Behaviour

Show ratio of initialization to Tool Calls. Note that lots of initialize != lots of usage necessarily.

---

# Allocating a "GET" to each user

## Data c/o Jeff Richter , Microsoft Azure.


---



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

