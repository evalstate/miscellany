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
<div class="organization"  > huggingface.co github.com/evalstate x.com/llmindsetuk</div>

<!-- -->


---

## Part 0 - Introduction

<!-- who knows about mcp, who i am, what the presentation entails -->

Community Moderator, Working Groups.



---

## Part 1 - The Model


Conversational Training. Hand Noted. RLHF. 
Instruction Training.

How do we make a model?

Ingredients. Lots of CPU, lots of compute.

Text Completions --> given . The text we ask it to complete is known as the "Context".
Computational Complexity and Model Size.


Set the context of Large Language Models, training data, context windows.
Age of Prompt Engineering.
What goes in, model ownership is important.


- Training data
- OpenAI training data opt-in (future models may know).
- The era of Prompt Engineering
  - Inference. 

HuggingFace Smol 3b training set.

If you don't own the model, you don't own the weights.
What facts the model has been trained on. What biases the model has are out of your control.

Copy and Paste context management. Custom RAG systems.

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

## Making a Model

## 

---

## 


---

## Part 3 : MCP


### Distribution

 - Demo Claude registry
 - Community Registry


### 

Architecture. 

Host, Client and Server

Primitives, Transport. 

All built on JSON-RPC.


Vulnerabilities.
- Session 
- Non-text content.
- 



---