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

<div class="columns">

<div>

# Shaun Smith `@evalstate`

- ### @ Hugging Face MCP
- ### MCP Maintainer / Transports WG
- ### Transports WG
- ### Maintainer of `fast-agent` 

</div>



<div align="center">

![w:250](./images/hf_logo.svg)
![w:250](./images/mcp-icon.svg)

</div>


</div>


---

# Hugging Face MCP Server: Huge Traffic Drop!

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

### MCP Server developer should make sure Notifications are sent on the correct channel.


</div>

<div>

![](./images/diag_notifications.png)


```typescript
async (request, extra) => {
  await server.notification({
    method: "notifications/statusUpdate",
    params: { /* your params */ }
  }, { relatedRequestId: extra.requestId });
};

```

```python
await session.send_progress_notification(
    progress_token="token-789",
    progress=50,
    related_request_id="tool-call-456"  
)
```


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

### Eliciation Request from Server delivered via the Post SSE response stream. 

### Elicitation Result is POSTed returned with a __new__ POST using the Request ID for association - and returns a 202.

### Server then returns the Tool Result via the original Post SSE stream.

</div>

<div>

![](./images/diag_request.png)


### Note that the Elicitation Request __must complete__ before the SSE Connection times out!


</div>

</div>


---


# Server Request to Client (Server Initiated)

### To use Resource Subscriptions, List Change Notifications or Server-Initiated Sampling/Elicitation, we send the request over an HTTP GET SSE Stream the client opens after initialize.


<div align="center"> 

![w:800](./images/diag_full_setup.png)

</div>

---



# Ping!

The Host can `POST` a Ping to the Server to tell it's alive.

<div class="columns">

<div>

![w:400](./images/ping.png)

</div>

<div>

![w:500](./images/fa_ping.png)


</div>

</div>

The Server can __Ping__ the Host via the `GET` Channel if open. 

---


# `Mcp-Session-Id` for State? 

<div class="emphasis-box">

### An MCP "session" consists of logically related interactions between a client and a server, beginning with the initialization phase.

</div>

## Sessions are controlled by the MCP Server, not the Host - and are tightly coupled to the Streamable HTTP Transport.

## 
## Typically OAuth used for Identity, Mcp-Session-Id used for correlation .

## Causes the "Conversational Context" problem.

## VSCode uses a new Connection per Conversation Thread (find discord image)

## Spec just says "Related Messages"

---

# `Mcp-Session-Id` for Routing

<div class="columns">

<div>

![alt text](./images/deploy_pain_1.png)

</div>

<div>

<div class="emphasis-box">

### With Multiple MCP Server instances, the Response needs to go the correct Server.

</div>

### `Mcp-Session-Id` HTTP Header can be used for Routing to the initiating MCP Server (sticky sessions).

### Sharing `Mcp-Session-Id` state amongst the cluster is not enough - Both `Mcp-Session-Id` and JSON-RPC `RequestId` are needed for correlation.

</div>

</div>


---

# Hugging Face MCP Server: Huge Traffic Drop!

![w:1200](./images/traffic-aug.png)

<div class="zoom-effect zoom-effect--claude">

![w:600](./images/claude_enhancements.png)

</div>

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
| <span class="feature-icon"><img src="./images/folders.svg" alt="Roots" width="100%" /></span> | <span class="cell-title">__Roots__</span> | _7.1%_ of all sessions, _33.6%_ of sessions that use tools <br /> __Not currently useful for Remote Servers__ |
| <span class="feature-icon"><img src="./images/cpu.svg" width="100%" alt="Sampling" /></span> | <span class="cell-title">__Sampling__</span> | _0.9%_ of all sessions, _22.2%_ of sessions that use tools |
| <span class="feature-icon"><img src="./images/message-circle-question-mark.svg" width="100%" alt="Elicitations" /></span> | <span class="cell-title">__Elicitations__</span> | _3.2%_ of all sessions, _21.6%_ of sessions that use tools |
| <span class="feature-icon"><img src="./images/trash-2.svg" alt="Session Deletion" width="100%" /></span> | <span class="cell-title">__Session Deletion__</span> | 4 of the top 20 clients delete sessions, only _6.64%_ of sessions get deleted overall |


---

# Thoughts and Guidance

- #### Hugging Face MCP Server supports STDIO, and Stateful + Stateless Deployment Modes. One Click Deployment to a FreeCPU Space via Docker. Link from `huggingface.co/mcp`

- #### `mcp-remote` used to be the most popular Client... Streamable HTTP Support in Hosts has picked up. Also means people are actively managing config?

- #### SDK DevEx differs between Transports and Capability usage - cosider deployment options carefully. Don't forget to Use `extra`/`related-request-id`  and configure `JSON-RPC` mode. `fast-agent` can help with diagnosis and debugging.

- #### Consider whether Server -> Client features are necessary for your use-case - especially in an uncontrolled environment. If they are, ![h:50](./images/ultrathink.gif) harder!

- #### Don't rely on Clients managing sessions - for now consider what you want to use `Mcp-Session-Id` for.


---


# Transport WG / Relevant SEPs

- ## Make Initialize Step Optional (Spec Change)

- ## Elevate Sessions #1364 - Host controlled Sessions, SessionId to the Data, not Transport layer.

- ## Delaminate JSON-RPC layer from Protocol.

- ## SEP #1442 - Make MCP Stateless by Default.

- ## Pure HTTP Transport - `https://github.com/mikekistler/pure-http-transport`


---

<!-- _class: transition -->

The end

---
