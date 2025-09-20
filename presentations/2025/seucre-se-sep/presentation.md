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

---

## Model Size vs Context Window (True Scale)

<style scoped>
.comparison-container {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 500px;
  padding: 20px;
}

.box-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
}

.model-box {
  width: 500px;
  height: 500px;
  background: linear-gradient(90deg, #2563eb 1px, transparent 1px),
              linear-gradient(#2563eb 1px, transparent 1px);
  background-size: 20px 20px;
  background-color: rgba(37, 99, 235, 0.1);
  border: 3px solid #2563eb;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}

.context-box {
  width: 1px;
  height: 1px;
  background-color: #ef4444;
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

.label {
  font-size: 22px;
  font-weight: bold;
  color: #2b3446;
  margin-top: 20px;
  text-align: center;
}

.size-label {
  font-size: 16px;
  color: #6c6c6c;
  margin-top: 8px;
}

.ratio-text {
  font-size: 32px;
  font-weight: bold;
  color: #1e40af;
  text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>

<div class="comparison-container">
  <div class="box-wrapper">
    <div class="model-box">
      <div class="ratio-text">60 GB</div>
      <div class="context-box"></div>
    </div>
    <div class="label">Model Weights</div>
    <div class="size-label">GPT-OSS-120B (Can you see the context window?)</div>
  </div>
</div>

<div style="text-align: center; margin-top: 40px; font-style: italic; color: #6c6c6c;">
  Context window (140KB) is a single pixel at true scale
</div>

---

## Model Size vs Context Window (True Scale)

<style scoped>
.comparison-container {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 500px;
  gap: 60px;
  padding: 20px;
}

.box-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
}

.model-box {
  width: 450px;
  height: 450px;
  background: linear-gradient(90deg, #2563eb 1px, transparent 1px),
              linear-gradient(#2563eb 1px, transparent 1px);
  background-size: 15px 15px;
  background-color: rgba(37, 99, 235, 0.1);
  border: 3px solid #2563eb;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}

.context-box {
  width: 1px;
  height: 1px;
  background-color: #ef4444;
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  box-shadow: 0 0 8px 4px rgba(239, 68, 68, 0.9),
              0 0 16px 8px rgba(239, 68, 68, 0.7),
              0 0 24px 12px rgba(239, 68, 68, 0.5);
}

.context-indicator {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 100px;
  height: 100px;
  border: 2px dashed #ef4444;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 0.8; }
}

.arrow {
  position: absolute;
  top: 50%;
  right: -40px;
  transform: translateY(-50%);
  color: #ef4444;
  font-size: 24px;
}

.context-callout {
  position: absolute;
  right: -200px;
  top: 50%;
  transform: translateY(-50%);
  background: white;
  border: 2px solid #ef4444;
  border-radius: 8px;
  padding: 10px 15px;
  width: 150px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.label {
  font-size: 20px;
  font-weight: bold;
  color: #2b3446;
  margin-top: 20px;
  text-align: center;
}

.size-label {
  font-size: 16px;
  color: #6c6c6c;
  margin-top: 8px;
}

.ratio-text {
  font-size: 28px;
  font-weight: bold;
  color: #1e40af;
  text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.context-text {
  font-size: 14px;
  font-weight: bold;
  color: #ef4444;
  text-align: center;
}

.scale-note {
  position: absolute;
  top: 20px;
  right: 40px;
  background: rgba(255, 255, 255, 0.95);
  padding: 12px 20px;
  border-radius: 8px;
  border: 1px solid #ccc;
  font-size: 14px;
  color: #6c6c6c;
}
</style>

<div class="comparison-container">
  <div class="box-wrapper">
    <div class="model-box">
      <div class="ratio-text">60 GB</div>
      <div class="context-box"></div>
      <div class="context-indicator">
        <div class="arrow">→</div>
      </div>
      <div class="context-callout">
        <div class="context-text">Context Window</div>
        <div style="font-size: 12px; color: #6c6c6c; margin-top: 5px;">140KB (≈0.00023%)</div>
      </div>
    </div>
    <div class="label">Model Weights</div>
    <div class="size-label">GPT-OSS-120B</div>
  </div>
</div>

<div class="scale-note">
  <strong>True Scale:</strong><br>
  1 pixel = 140KB context<br>
  450px × 450px = 60GB model
</div>

<div style="text-align: center; margin-top: 40px; font-style: italic; color: #6c6c6c;">
  The context window is 428,571× smaller than the model weights
</div>


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