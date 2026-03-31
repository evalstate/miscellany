---
marp: true
html: true
size: 16:9
theme: freud
paginate: false
header: '<span class="header-logos"><img src="./images/hf_logo.svg"   /><img src="./images/github-mark.svg" />github.com/evalstate</span>'
style: |
  iframe.demo {
    width: 100%;
    height: 70vh;
    border: none;
    border-radius: 16px;
    background: transparent;
  }

  iframe.demo.demo--column {
    height: 65vh;
  }

  iframe.demo.demo--short {
    height: 52vh;
  }

  video.full-slide {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

---


---

# MCP and Agent Automation @ Hugging Face

- 1000's of Machine Learning Applications
- Hub MCP Server provides MCP gateway for Sandboxes, Inference and Research
- Dynamic Inference
- Clients in ChatUI, llama.cpp and more 

Supplemented by huggingface/skills and `hf` CLI tool


---

# The *debate* about MCP is more interesting than MCP  (and that's a good thing).

## MCP achieved PMF

<!--
- at least parts of it did.
- large surface area
- The way things 

-->

## Market Perspectives

- Consumer
- Enterprise
- Developer

---

# Things that weren't a thing when MCP Launched

Things that weren't a thing when MCP Launched:

 - Reasoning Models (o1 was 2 months old) Deepseek R1.
 - Goose (Jan 2025), *Claude Code (Feb 2025)*
 - *Responses API (march 2025)*
 - MCP Bundles
 - *Remote MCP*
 - *Streamable HTTP Transport*
 - *Self Propelling Tool Loops*
 - *Vibe Coding*
 - *AGENTS.md* (Aug '25), *Agent Skills* (Dec '25).

---

# Reinforcement Learning and Model Behaviour

- Reinforcement Learning
  - Place model in Environment (OpenEnv, NemoEnv) 
  - Code Reward Functions 
  - Reinforce "good" trajectories
  - 
- Make Models good at goal seeking, discovery and trial/error
- Enabled Skills
- Generalized Tool Calls 

![w:600 alt text](image-1.png)

---

# mini-SWE-agent (SWE-Bench)

- Single Tool
- Non-persistent Bash with commands between code fences.
- 76% on SWE Bench
- *Agents are trained to learn and solve problems*
- "Agent Skills" supercharge this

---

# Bash > STDIO 

MCPorter: https://github.com/steipete/mcporter
mcp-cli: https://github.com/apify/mcp-cli

- Distribution / MCP Bundlers (MCPB)

But fortunately we have Streamable HTTP + OAuth

- Hobbyist building
- MCP Specific Hosting Platforms
- Value Add????
- $ Access to Data
- $ Processed Data
- $ Access to Compute
- Issue was never actually API Wrappers....

- Claude Code Channels?

---

# Shell is Code Too

Skills 
SmolAgents
![w:250 alt text](image-2.png)

`hf-tool-builder` skill. Queries an OAuth spec, and then builds reusable, composable CLI tools (or Python programs) on demand for any purpose.

## Code Tools

Lightweight Runtime

- When models can build and run their own tools, where does MCP fit in?
- In that case, 


MCP win is in the human factors that improved API design.

---

# Where do I want to run this?

- MCP Gives us deployment options
- Agent-as-Tool
- 

- CHALLENGE: Search/Execute happens on the main thread
- LLM Overprocessing
- Cost/Efficiency/Specialization questions.
- Sub Agents (Dynamic Tooling)

Remote MCP 
e2b
YOLO
Container (Local)
Container (Remote)
Inference Provider.

---

# Agent Client Protocol

- Execution Bundles
- How do you distribute?
- Content Streaming 
- Sessions with Resumption/Rehydration
- Observability
- Shares Data Structures with MCP

---


# Open Responses 

Tool Call Shapes - Custom / Grammar Constrained responses.
Remote MCP Embedded.

---

# Code Mode and Generative UI

Search/Execute Pattern
Sandboxes
![w:700 alt text](image.png)

---

# MCP for Navigation

- User enteres a query: "What are the .
- Model generates tool call to fulfil the query
- CallToolResult contains the content (often in )
- LLM then spends expensive **output tokens** re-rendering the information.
- Navigating
- Providing access to 

---

# Delegating to Task Specific Agents


---

# Open Responses

- Generic Services are taken care of.
- Web Fetch / Citations
- Code Interpreter
- Shell Environments
- Schemaless Tools

---

# What didn't we talk about?

## Prompts and Sampling

## Resources

## 

--- 

# MCP / Product Market Fit

Include venn diagrams, 

---

# Things we didn't have

- Claude Code
- Reasoning Models
- AGENTS.md , Agent Skills
- Remote MCP
- A "proper" remote transport
- Long Running Tool Loops
![w:200 alt text](goose.png)
![w:400alt text](claude-code.png)
![alt text](deepseek.png)


<!-- Generate Code, Fault diagnostics etc. -->

---

# Reinforcement Learning

- Models placed in environment and rewarded to goal seek
- Examples include OpenEnv, NVEnv
- Often harnesses are very simple (mini-SWE-agent)
- Skills are an effect of model training
- Skills provide focussed education for the model

<!-- 
needed to have "hacks" in tool loops to keep them running 


-->

---

# STDIO Servers

- MCP had a "human factors" in optimising API design for LLM usage.
- Discovery is token efficient (potentially no JSON envelopes)
- mcp-cli, MCPporter offer MCP connectivity.

---

# The Value-Add

- Distribution and Value-Add change
- Client Landscape at the time didn't. No proper remote connectivity.
- Remote Connectivity [Connectors] in Claude in July 
- Claude Code Connector Distribution
- APIs that offer access to Resources or Data
- Remote Access via a URL and Auth.

---

# Code Tools and Generative UI

- Shell scripts are code (e.g. Perl file manipulation)
- Python is code / TypeScript 
- Generating Code > JSON

---

# Apps SDK for efficiency (The Navigation Problem)

Diagram / Demo here of GenUI bouncing, stats, post-processing.

- Overprocessing Results.
- Can you find out my train times.
- Model may be processing/looping which is unreliable and inefficient
- It's processing which would be better done by a deterministic.
- Unnecessary context windows space, or expensive output token processing

---

# Code Execution Location

- "Main" execution thread
- Agents as Tools
- Delegate 
* Find / Execute vs. Sub Agent
* Picking the right model for the job
* Sandboxing (Monty, v8)
* Just Bash
* 

- MCP Gives me Portability
- Straightforward . 

---

 # Blending

---

# Open Responses

- Replaces and normalizes Chat Completions API
- Models trained on 
- Open Responses is inference
- Internal / External Tools
- Open
- Observability. Assumption that models have been built with tool calls


---

# Agent Client Protocol

- Normalization and Bundling
- 
- Client supplies File and Shell tools
- Streaming, Reasoning and Tool Calls
- MCP Servers are first-class

---

---

# Finale



---