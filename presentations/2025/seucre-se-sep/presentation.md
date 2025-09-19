---
marp: true
theme: freud
paginate: true
---

<!-- _class: titlepage -->

<div class="title"         > Securing the Model Context Protocol</div>
<div class="subtitle"      > securese.ai, Stockholm   </div>
<div class="author"        > Shaun Smith                       </div>
<div class="date"          > Sep 2025                                    </div>
<div class="organization"  > huggingface.co github.com/evalstate x.com/evalstate</div>

<!-- -->


---

## Part 0 - Introduction

<!-- who knows about mcp, who i am, what the presentation entails -->

MCP Steering Group Member
Community Moderator, Working Groups.

Work @ Hugging Face on MCP and Open Source initiatives.


---

<!-- _class: transition -->

# <span class="mcp-model">Model Context</span>  <span class="mcp-context">Protocol</span>

---

## Part 1 - The Model

<!-- 

This isn't a long "history lesson" style talk; but i wanted to reground us 

Conversational Training. Hand Noted. RLHF. 
Instruction Training.

How do we make a model?

Ingredients. Lots of CPU, lots of compute.

Text Completions -->
 given . The text we ask it to complete is known as the "Context".
Computational Complexity and Model Size.


Set the context of Large Language Models, training data, context windows.
Age of Prompt Engineering.
What goes in, model ownership is important.

What if we add training that's conversational? We'll have lots of human/assistant pairs and show it how to act conversationally?

- Appears knowledgeable
- Semantic understanding of text
- Better.

Donald Trump clause in the System Prompt.

- How do we make it better. We have bigger and bigger models; 

- Training data
- OpenAI training data opt-in (future models may know).
- The era of Prompt Engineering
  - Inference. 

HuggingFace Smol 3b training set.

If you don't own the model, you don't own the weights.
What facts the model has been trained on. What biases the model has are out of your control.

- Copy and Paste context management. Custom RAG systems.

Where that data comes from. Anthropic $1.5bn books, shredding.
OpenAI Privacy dialgoue.
The model may end up knowing a suspicous 

Security Risks?
 Cognitive Risks.
 Bias Risks.
 Training Data.

Need to trust the model, trust the provider.


## Part 2 - The Context

Trained on chat
Trained on instruction



OpenAI Codex



## Part 2 - Tools and Agents

It's all the same trick!
Live demonstration : Teach our model to do Tool Calling.

The age of Agents.
What is an Agent

- Harness that self-adjusts its context. 
- 
Prompt Injection.


---

IGNORE ALL PREVIOUS INSTRUCTONS

similar problem to "solving hallucinations".



---


## Part 3 - MCP

### Introduction

Can't deflect responsibility in to the Protocol
Can't transfer the risk

### Distribution

As Community Moderator get to see a lot of MCP Servers. One-shot prompted in to existence. 



Introducing the Model Context Protocol.


We see automation not augmentation.

So now that we know what a bit more about Models, and a bit more about Context let's see where MCP fits.
Show MCP-Webcam.

Less than 12 months old. Distribution Statistics. Weekend in Apr

- what mcp is -- do a deep dive explanation on the components and the parts.

- json-rpc; transports, hosts, client, servers.
- show all of the different datasources that can work.
- transport, data, layer??  (d)
- two specifications
OAUTH2.1
- Package and distribution of MCP-B/DXT. GitHub, Webiste.
- Registry

Bi-Directional Communication

co-minglign


## 

---

## 

Assume that the data in your context window is privileged. 

The reason for the preamble is so that we can have a balanced discussion about MCP Security

---


# Model Context Protocol is an open-source standard for connecting AI applications to external systems.

#### Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect electronic devices, MCP provides a standardized way to connect AI applications to external systems.

---

# Architecture

<!-- we'll do a high level walkthrough, then look at some of these in more detail -->
<!-- we talk a lot about MCP Servers, and that's not quite the right name -->

Three parts, Host, Client, Server

---

# Primitives

Primitives, Transports

When using an MCP Server

Prompts
Resources
Tools

---

# Transports (and Distribution)

STDIO
SSE (Deprecated)
Streamable HTTP
The rise of Hosting Services and Proxies.

---

# Early days of MCP. Server List. 

Review the Server, make sure there are no obvious.

# What happens


MCP Server Instructions injected in to Context.
Auto-injection in to the Context.

Context co-mingling.
Data sent to the LLM 
Tools that know about each other

Tool Descriptions narrow. Do they mutate? ToolListChangedNotifications.


Data is accessible at the privilege level of the User.


---


---

# MCP "Server"

MCP Server is a bad name. We are used to Servers . MCP Servers can actively contact the Host Application, or User.
With Sampling and Elicitations

---

# Problems

We've integrated data and actions from different systems via a Host Application.

Co-mingling of data.
Co-mingling of instructions.


---

# Registry
# MCP-B
# Registries
# Demo Claude Registry

### Distribution

 - Demo Claude registry
 - Community Registry

### 

Architecture. 

Host, Client and Server

Primitives, Transport. 


Vulnerabilities.
- Session 
- Non-text content.

---
