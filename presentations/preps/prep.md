-> Economics and Ergonomics of MCP Server Hosting
-> MCP Servers are a gateway, like an API because you either want the compute, or the data there.
-> Remote Execution
-> Agents are a context management hack
-> RL enabled skills
-> Models aren't trained on JSON alone
-> Chat Templates break tool calling
-> Code Mode deployment options
-> Local Models.
-> The rise of more sophisticated inference APIs
-> The need for Agent Environment's
-> Open Responses

----

# MCP at 18 Months: Protocols, Patterns, and what we didn't see coming.

## Introduction

For me, this conference being in New York is rather prescient. 

The debate about MCP is more interesting than MCP (and that's a good thing). 

Nassim Taleb called the Lindy Effect. (NY based). Roughy speaking the longer something has existed, the.. So something that has been around 75 years. In c. At current industry pace (like Dog Years) that's forever.

It's answered the question 

## Product Market Fit

The surface area that has become standardized through _use_ is critically important.

Clients that provide Authentication services and tool discovery  mechanisms. 

Perspective of the Product.
Perspective of the Market.

The normal story of PMF is to start small, pivot, rewrite, adapt. MCP started with a big Product and some speculative ideas.

Some parts of MCP have died several times over.

Servers that "just work" in the ecosystem. Solves .

Consumer, Enterprise, Developer. 

 - CLI conversion tools 

A CLI conversion tool is not a failure of MCP, rather it's a success. The Human Factors that led us to design LLM friendly API's. 

A common pattern is (you can see these online). 
 Interlocutor: MCP is just tool calling
 X: No, you've misunderstood. There are prompt, and resources and sampling.

MCP Protocol Surface Area
MCP achieved PMF by overextending at the start (feature rich) and the market deciding on fit. 

MCP's problem if you will is that it hit perfect PMF for a chunk of the protocol. Churn in Transports and OAuth --> cost of change people are paying 

MCPs surface area is/was quite large/ambitious. 

Application, Model, User

Elicitations, Sampling, Root
Multimodal - superpower.

Token Dense.
Pointlessness of multi-processing for generative sub-agents (and the importance of Apps SDK)

## Economics of Hosting and Distribution was about ideas.

API vendors offering value-add.
MCP Servers / APIs need to be offering something - a resource you don't have:
 - They are a gateway to compute you need. Example
 - Access Control
 - Value add through computation or aggregation

## Using MCP for Inference

Sub-agent through inference.

## These things did not exist

Tool Calling, certainly not new. ReACT, not new. Code Generation not new. Resurgent

Some things that happened post MCP.
 - Reasoning Models.??
 - Claude Code, Goose.
 - Responses API, OpenResponses
 - Skill Standard (and Claude Code). Goose. Early versions of fast-agent.
 - Popularization of RL/OpenEnv.

Security of running MCP Servers. 

## Agents are a hack

Bitter Lesson
The idea was that we could one-shot simple MCP Server to fulfil a function. Distribution became a problem. 
One-shot it, the sharing the idea is more important than the code.
Human Factors -> API; forcing function to make API design friendlier. But search/execute Code Mode is a step along the path. 
Remote Tool Calling (with authentication too). 
v8/monty sandboxes.
Usage:
Sometimes an API wrap is perfect, I've got a focussed sub-agent why not?
Difference between what people should do and what they actually do.
Avoid post-processing.
Chaining together tool loops. 
Understanding the tensions between Consumer, Enterprise and Developer. Efficiency trade-offs and performance.

## Reinforcement Learning, Tool Loops and Skills.

LLM wrangling. Place the agent in to an environment and reward it for completing tasks. 

Shell loop rewards have enabled skills; bash tooling is particularly efficient (especially in non-JSON mode).

Model controlled ingest. There's no reason not to offer a bash environment via MCP

Versioned delivery easier wih MCP. 

## Inference

Are these capabilities that my Agent Harness has, or capabilities that my Inference back-end has?

Remote execution sandboxes (Skills). Remote MCP (tool loops). 

Benefit of Hindsight; 

Why does software cost so much? Compared to what?
-> Travel Planner.
-> Often 
Mesh of Browser to navigate to the site, show an API. Branding.
Most of our traffic comes from chat-style clients. Traffic Interactive 

## Remote Tool Calling and Batching

Talk about how MCP provides a mechanism for "hands-free" remote loops that can be with batches.



## JSON Fixation

Stripped by templates.
Models not trained for it.
New tools tending to not 
TOON doesn't work.

## Apps SDK

Apps SDK gives us something just as important. That is the ability to present data to the User and not the LLM. 

Problem:
 -> User asks LLM to find data
 -> Tool calls are executed to complete the task
 -> The task completes with a dump of data (Tool Result)
 -> That tool result gets added to the Input Tokens, and often simply regenerated with no value add.

Simply put, there often comes a point at which a Tool Call is *terminal* for the User and so inference is not just unwelcome, but actively harmful.

Is that how people should use LLMs? Probably not. Is that how they _do_ use LLMs? Yes!

## Dynamic Spaces


## Code Generation v2. 

MCP Acted as a human forcing function to design LLM friendly APIs. 

v1. CodeACT.
Search->Execute Pattern.

What changed? 
Lightweight Sandboxes, Monty, Piodide + Deno, V8. 
Where does the code generation take place?
Where does the code 

API wrapper MCP Servers are bad. Poor take, subagents that produce code or can chain tool calls love this.

Embedding Tool Calling within reasoning. Inference provided interpreters.

Local Models

For a lot of tasks, searching/generating with . Common actions, 
GraphQL; 

Expensive Model bouncing around the API?
Expe

## Generative UI

We have a common problem that people use LLMs for Search/Navigation.

Simply selecting suitable components is probably good enough.

## LLM Distribution

The rise of small models, fast models, accessible RL. Filters down through work like unsloth/openenv. We're now at the point where agents can self-optimize for tasks.

Rise of Local Models.
Qwen 3.5 Series. 
TTS/transformers.js

TODO: talk to nico?
Token Density in our primary agents.

## Agent Client Protocol

ACP let us distribute a high quality experience. 

Chat Completion compatible != Good Experience. `gpt-oss` requires stripping keeping reasoning between tool calls, and stripping .

The rules aren't necessarily complex, but they are important - and 

MCP lets LLMs interact with Tools
ACP lets users interact with LLMs+Tools

MCP and ACP share large parts of their data model.

ACP has a difficult job in that almost by definition it is following rather than leading and trying to find the sweet spot between the tools. 

Some comparisons:

- Streaming
- Sessions + Rehydration

What is ACP doing that MCP isn't doing?
 - Streams content
 - Notifications
 - Sessions with Resumption/Rehydration

MCP over - proposal.

Provides a normalized view of agent interactions. 

Another way to distribute our Inference Provider offerings.



## Open Responses, Observability

Bringing 

It's earned it's place in the stack. Worse is better; but that's OK. UTCP and SLOP might have technical advantages. 

The debate (or even debate about the debate) is more interesting than MCP itself. (and that's a good thing). 


What is OpenResponses doing that ACP Isn't.
 - Stateful conversation
 - 
OpenResponses normalizes inference, and opens up the possibility of routing.

Speed 

Code Mode -> Shift in to inference APIs
Models are trained to do it.
(Show Chat Templates)

Irony is that the Model Contextpart. 

Agent operates on our behalf; where is our avatar?

Tool Results are post-processed. 



## Concluding



---


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


-----****

This session takes a balanced look at the changes within MCP over the last year - how it continues to fit and support the adjacent ecosystem.

Grounded in experience this will give attendees insights to the options and tradeoffs now available without presenting false dichotomies or hype.

HF are in a unique position with our focus on open source, open weights and leading edge research and practical implementation experience of the topics.

-------

Enhance 34 to 46
Pull back
Wait a minute, go right
Stop
Enhance 57,19
Track 45 left. 
Stop
Enhance 15 to 23

----


