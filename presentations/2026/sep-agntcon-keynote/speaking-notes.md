# Act One


- When MCP Launched everyone used STDIO, and there was a simple SSE transport. Assumptions were that you would extend desktop applications like an IDE.
- The March '25 release added Streamable HTTP, which and 




- A long, long time ago before we had Claude Code, Codex, Skills or AGENTS.md we had MCP
- In those early days, we would usually connect from a desktop application; giving downloaded software full permissions.
- 


Need to mention about Session Lifecycles


In the beginning...

Before Claude Code, before Codex there was MCP. So, there are a couple of different ways to [add functionality] connect an MCP Server to whatever you're using. You can either download the MCP Server and run it as a local program, or connect to a remote Server.

Right at the beginnning the early hosts only really did that download and embed thing. So you'd trust some software you downloaded and connect it. Also at the time it was relatively unusual for these tools to have shell access -- Goose was the first major tool to do this in January 2025.

That's kind of OK if you are extending a desktop application but a lot harder if you want to use for example Web based systems like ChatGPT.

There was a way to connect Clients and Servers over HTTP, but there was no auth, and it became end of life quickly. That ends the first chapter.

# Act Two

The first spec revision introduced Streamable HTTP as a way to provide good remote connectivity. At the time a lot of Clients were set up to still use only STDIO, so a lot of people connected through an adapter. Between June and August, the use of the adapter went close to zero. Tool call volume also increased considerably during that period, with Anthropic launching "connectors" and switching on modern OAuth and streamable connections.


Some practical challenges arise though. Assumptions made for a local bi-directional connection don't hold true when you are hosting a public internet-facing service.

Features that we wanted to make use of - like being able to click and configure the available tools and update the client dynamically required too much fragile resource overhead. The protocol was also remarkably chatty; once more this can make sense if you have a local embedded server with low overhead, but doesn't work quite so well when you have a many to one relationship between clients and servers. You go from a one-to-one relationship to a many to one relationship. Another that was tough to get right was elicitations. So eliciations are when the Server sends the User question. The Server kind of "waits" for the answer - which makes it quite difficult to connect later on. The other thing we were finding that is that MCP was quite chatty. Some clients can attach quite aggressively and create traffic floods.


The Transports W-G led by Kurtis 
So a group of us, Microsoft, Google, Anthropic, Independents, thrashed out what needed to be done. 

The main changes to simplify the protocol:
 - Deprecate Sampling, Roots [and Logging] 
 - Disallow Server initiated communications
 - Remove handshaking and need for Clients and Servers to remember each other.

# Act 3

These type of remote connections are helpful for observability. We are able to better understand what to optimise for. We can see client capabilities, as models have become more capable, we can increase the Token Density of tools and get rapid feedback. We want descriptions, responses to be efficient, but also tool call arguments.
SDKs
Token Density of tools.
Observability; we can optimise our tools for new models, see what client versions and capabilities are available.


This was launched in the July 2026 revision of the spec. 


It's always been a good time to connect things with MCP -- and now is an even better time. 

# Act Three 


# Close

MCP lets us do immense amounts of data processing close to where the data is. Train model with large datasets. Conduct research/analysis 
Power of Open Source -- MS, Google, HF, Anth, OpenAI as well as dozens of individual contributors. 
Extensions, Tasks that can save us money.

----




# Beyond the transport

<div class="keynote-topics">Routing by type<br>Tasks for keeping models warm</div>

- HTTP Level Routing
- Apps for Interactivity
- Triggers and Events
- Tasks for 

Move 


