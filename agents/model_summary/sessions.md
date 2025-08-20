Abstract

This proposal makes MCP Session Management operations explicit within the Protocol Schema, rather than part of the Streamable HTTP transport supplement.

Motivation

Sessions are an important logical concept required by System Integrators to correctly reason about context window and resource management. 
 - Whether the MCP Server is stateful
 - Behaviour and suitability of Servers to Resume Conversations, Participate in Connection Pools or Service Multiple Context Windows is unknowable to the Host application.
 - Implementors of Custom Transports must design their own Session Management semantics, breaking the abstraction between the Host, Client,  Transport and Server layers. [4]
 - There is a lack of equivalence between the existing STDIO and Streamable HTTP Transports and their Session Management semantics.
 - 


Proposal

- Specify transport specific handshake (retain HTTP Specific working and optimisations) [3] within 
- Introduce a "Session" capability. 
- Introduce metadata that provide information at run time and <offline/registry>

Proposal 

The 

Whilst the "Initialize" handshake protocol negotiates the version 



continuations and. 




[1] The whole MCP Specification comprises the "Protocol Schema" and related supplements (for example Authorization and Transport). This proposal distinguishes the two, and chooses the "Protocol Schema" as the main point of reference for MCP Developers. 
[1] The current Schema `initialize` and `notifications/initialize` protocol do not describe Session Identification, Resumption or Deletion.


[4] The issue is with the Transport leaking inwards to the Schema-defined Protocol 

Abstract

Proposal is to decouple Session Management from the Streamable HTTP Transport, and make it an explicit part of the Model Context Protocol.

Motivation

Sessions within MCP are [currently defined](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#session-management) as part of the Streamable HTTP Transport.


The enhancement is to decouple Session management from the Transport, 

At the moment, Session management is coupled. 

The allocation of SessionIDs is coupled to the 


Sessions
 - None. The Server does not support Sessions.
 - One. The Server 
 - Many. 



https://github.com/modelcontextprotocol/modelcontextprotocol/issues/823
https://github.com/modelcontextprotocol/modelcontextprotocol/issues/984
https://github.com/modelcontextprotocol/modelcontextprotocol/issues/958
https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087




 export interface ServerCapabilities {
   // ... existing capabilities
   sessions?: {
     // Whether server supports multiple concurrent sessions
     multiSession?: boolean;
     // Whether sessions persist across transport disconnections
     persistent?: boolean;
     // Whether server maintains state between requests
     stateful?: boolean;
     // Session lifecycle management
     lifecycle?: {
       clientInitiated?: boolean;
       serverInitiated?: boolean;
       explicitTermination?: boolean;
     };
   };
 }

