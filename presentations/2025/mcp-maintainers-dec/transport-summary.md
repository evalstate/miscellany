# Transport `stateless` flag Summary

Some MCP Server SDKs has a flag for `stateless` mode - which means allowing calls beyond `Initialize` and `InitializeNotification` without an `Mcp-Session-Id` header. 


## Typescript

### Server

Application Developer has to set up HTTP Routes [examples use Express]. 

Examples folder contains separate "Stateless" StreamableHTTP Example. The Stateless example is simpler, and returns 405 on the GET/DELETE methods.

### Client

Client always attempts the GET call. If no session id it is `undefined`

## Python

### Server

FastMCP is configured with `stateless_http` True or False.

For Server: `stateless_http` = True: 
 - Returns "None" for Session
 - **Does** open the GET stream on the Server.
 - Allows multiple connections.

### Client

Does not attempt to connect if no `Mcp-Session-Id` is present.

 
_NB_ TS Client and Python Server behave differently.

## Golang

### Server

In "Stateless" mode the SDK generates a Session ID per-request - but responds with a 405 on the GET stream. Session-Id validation is switched off.

(to-test). Requires a SessionID 
Concept Name

Stateless (Sessionless)
StreamableHTTPOptions  has Stateless and JSONRPC
GET is rejected if Stateless==true or empty sessionid

Contains Session timeouts

Does open the GET handler

### Client

Client always connects to the `GET` handler (regardless of SessionId), and attempts to `DELETE` on closedown. 

```
INFO:     127.0.0.1:46770 - "DELETE /mcp HTTP/1.1" 405 Method Not Allowed

ERROR:mcp.server.streamable_http:Error in message router
Traceback (most recent call last):
  File "/home/shaun/source/fast-agent/.venv/lib/python3.13/site-packages/mcp/server/streamable_http.py", line 961, in message_router
    async for session_message in write_stream_reader:
    ...<45 lines>...
            )
  File "/home/shaun/source/fast-agent/.venv/lib/python3.13/site-packages/anyio/abc/_streams.py", line 41, in __anext__
    return await self.receive()
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/shaun/source/fast-agent/.venv/lib/python3.13/site-packages/anyio/streams/memory.py", line 111, in receive
    return self.receive_nowait()
           ~~~~~~~~~~~~~~~~~~~^^
  File "/home/shaun/source/fast-agent/.venv/lib/python3.13/site-packages/anyio/streams/memory.py", line 93, in receive_nowait
    raise ClosedResourceError
anyio.ClosedResourceError
```


## C#

### Server

### Client

Client always connects to 

---

---

## Hugging Face

### MCP Usage

Gradio Applications / SDK [Python]
 - Used for ML Application Development
 - Hosting on HF Spaces - Both GPU and ZeroGPU > 2,500 MCP Servers hosted
 - Typically . _Alternative is Inference providers/Gradio API_ 
 - Replica affinity (sticky) required (IP Based) to manage load and generation results
 - We run replicas 

HF MCP Server [Typescript]
 - Server and Configurable Gateway/Proxy
 - Was run in session-free mode for quite a while - now uses Sessions for Analytics. THIS REQUIRED.
 - Relays Progress notifications, Supports anonymous, OAuth and Token access, use both JSON-RPC and SSE responses depending on need.
 - Helped diagnose issues with VSCode 
 - ~1m MCP Calls per-day

Hugging Chat UI [Typescript]
 - Chat MCP Client.

Inference Providers
 - Remote MCP Client 

fast-agent
 - 

 
 


---





## Other Miscellany

Servers can advertise their `associationId` handling through their Server Card or Capabilities with the following metadata:

```
associationMode: [None | Ephemeral | Persistent]
associationTtl: [timeout] 
```

associationMode Enum:
- `None` - Behaviour not defined.
- `Ephemeral` - associationId recognised and respected for this Connection
- `Persistent` - associationId recognised and respected for this [Logical] Server

associationTtl:
- `associationTtl` - Amount of time that `associationId` is maintained. For `Ephemeral` this would usually be the idle session timeout, for `Persistent` it would relate to persistent storage reclamation policy and is not a guarantee. 

In this design the Server decides when and how to compartmentalise associationIds. This would usually be with the Users Identity.

When a Requestor sends an `associationId`, the Responder can:
 - Echo the sent `associationId` in the Response if the request has been logically associated with other Requests
 - Omit an `associationId` in the Response if the request has not been associated.
 - Send an Error with descriptive text if the `associationId` is not considered valid.
 - Return a different `associationId` in the Response. In this case the Client MAY ignore the associationId, treat it as either as an Error, or use it for future Requests to the Server.

When a Requestor does not send an `associationId`, the Responder can:
 - Return an `associationId` in the Response to indicate that the Server supports future related Requests being associated with this Request. The Requestor MAY ignore this.
