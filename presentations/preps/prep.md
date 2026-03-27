-> Economics and Ergonomics of MCP Server Hosting
-> MCP Servers are a gateway, like an API because you either want the compute, or the data there.
-> Remote Execution
-> Agents are a context management hack
-> RL enabled skills
-> Models aren't trained on JSON alone
-> Chat Templates break tool calling
-> Code Mode deployment options
-> Local Models, Spaces
-> The rise of more sophisticated inference APIs
-> The need for Agent Environment's
-> Open Responses

----

# MCP at 18 Months: Protocols, Patterns, and what we didn't see coming.

## Introduction

The debate about MCP is more interesting than MCP (and that's a good thing). Nassim Taleb called the Lindy Effect. (NY based). Roughy speaking the longer something has existed, the .

The surface area that has become standardized through _use_ is critically important.
 - Clients that provide Authentication, 
 - CLI conversion tools 

A CLI conversion tool is not a failure of MCP, rather it's a success. The Human Factors that led us to design LLM friendly API's. 

Elicitations. 

A common patterns (you can see these online, many started by me). 
 Interlocutor: MCP is just tool calling
 X: No, you've misunderstood.

 
MCP Protocol Surface Area

Application, Model, User

Elicitations, Sampling, Root

-> Travel Planner.
-> Often 
Mesh of Browser to navigate to the site, show an API. Branding.
Argument for MCP is that it's a Human Forcing Function. 
Traffic Interactive 
----> Economics 

MCP Achieving Product Market Fit. 
----------------------------------
Tool Calling, certainly not new. ReACT, not new. Code Generation not new. Resurgent
Token Dense.
Pointlessness of mutli-processing for generative sub-agents (and the importance of Apps SDK)
MCP achieved PMF by overextending at the start (feature rich) and the market deciding on fit. MCP's problem if you will is that it hit perfect PMF for a chunk of the protocol. Churn in Transports and OAuth --> cost of change people are paying 

MCPs surface area is quite large/ambitious. 


Bitter Lesson
The idea was that we could one-shot simple MCP Server to fulfil a function. Distribution became a problem. 
One-shot it, the sharing the idea is more important than the code.
Human Factors -> API; forcing function to make API design friendlier. But search/execute Code Mode is a step along the path. 
Authenticated, Remote Tool Calling. 
v8/monty sandboxes.
Usage:
Sometimes an API wrap is perfect, I've got a focussed sub-agent why not?
Difference between what people should do and what they actually do.
Avoid post-processing.
Chaining together tool loops. 

Understanding the tensions between Consumer, Enterprise and Developer. Efficiency trade-offs and performance.


Some things that happened post MCP.
 - Reasoning Models.??
 - ACP
 - Responses API, OpenResponses
 - Skill Standard (and Claude Code). Goose. Early versions of fast-agent.
 - Popularization of RL/OpenEnv.

Inference
---------
One of the main things we use 


Benefit of Hindsight; 


Why does software cost so much? Compared to what?


## Remote Tool Calling and Batching
---------------
Talk about how MCP provides a mechanism for "hands-free" remote loops that can be with batches.


## Code Generation v2. 
----------------------
v1. CodeACT.

Search->Execute Pattern.

What changed? 
Lightweight Sandboxes, Monty, Piodide + Deno, V8. 
Where does the code generation take place?
Where does the code 

## Generative UI
----------------

We have a common problem that people use LLMs for Search/Navigation.

Local Models
------------
Rise of Local Models.
Qwen 3.5 Series. 
TTS/transformers.js

TODO: talk to nico?
Token Density in our primary agents.


MCP->ACP->OpenResponses

ACP has a difficult job in that almost by definition it is following rather than leading and trying to find the sweet spot between the tools. 

Some comparisons:

- Streaming
- Sessions + Rehydration


What is ACP doing that MCP isn't doing?
 - Streams content
 - Notifications
 - Sessions with Resumption/Rehydration

MCP over ACP proposal.

Helps with design constraints for "normal" style usage.

Bringing 

Multimodal - superpower.

It's earned it's place in the stack. Worse is better; but that's OK. UTCP and SLOP might have technical advantages. 


The debate (or even debate about the debate) is more interesting than MCP itself. (and that's a good thing). 

What is OpenResponses doing that ACP Isn't.
 - Stateful conversation
 - 

Code Mode -> Shift in to inference APIs
Models are trained to do it.
(Show Chat Templates)


Irony is that the Model Contextpart. 

Agent operates on our behalf; where is our avatar?

Tool Results are post-processed. 



## Transportable



- Contrast how ACP and Open Responses work within our MCP infrastructure and our experience deploying them.

MCP is often used for inference; Progress Notifications (not tasks) are ideal for this. Jobs are ideal for this. We also have Dynamic Spaces, 


- Explore how Skills, and Code Generation fit, the impact future models will have on the landscape - and why things that may seem obvious in hindsight weren't.
- Discuss the value of multi-model environments and how subagent patterns are critical for certain tasks.
- Report on the trends we see in shifting Agentic and Inference workloads, and MCPs continued role in supporting them
- Examine our unique multimodal demands - and what needs we and our community have to support mixed content.







---


MCP at 18 Months: Protocols, Patterns, and what we didn't see coming.
Description
MCP launched a revolution in peoples expectations of what Generative AI could achieve. Since then, MCP has been supplemented by other protocols, techniques and extensions.

At Hugging Face we have 1000s of AI Applications deployed, using MCP for interconnectivity, as well as supporting Remote MCP via Inference Providers.

This experience based session explores how inference, compute and storage workloads are shifting in an agentic world, and how MCP supports us in a rapidly changing environment.

We will:
- Contrast how ACP and Open Responses work within our MCP infrastructure and our experience deploying them.
- Explore how Skills, and Code Generation fit, the impact future models will have on the landscape - and why things that may seem obvious in hindsight weren't.
- Discuss the value of multi-model environments and how subagent patterns are critical for certain tasks.
- Report on the trends we see in shifting Agentic and Inference workloads, and MCPs continued role in supporting them
- Examine our unique multimodal demands - and what needs we and our community have to support mixed content.


----


This session takes a balanced look at the changes within MCP over the last year - how it continues to fit and support the adjacent ecosystem.

Grounded in experience this will give attendees insights to the options and tradeoffs now available without presenting false dichotomies or hype.

HF are in a unique position with our focus on open source, open weights and leading edge research and practical implementation experience of the topics.