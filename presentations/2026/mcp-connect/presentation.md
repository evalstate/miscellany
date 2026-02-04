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

<script>
  (() => {
    const slideSelector = 'svg[data-marpit-svg]';
    const activeClass = 'bespoke-marp-active';

    const reloadIframes = (slide) => {
      const iframes = slide.querySelectorAll('iframe.demo');
      iframes.forEach((iframe) => {
        const baseSrc = iframe.dataset.baseSrc || iframe.getAttribute('src');
        if (!baseSrc) {
          return;
        }

        iframe.dataset.baseSrc = baseSrc;
        const separator = baseSrc.includes('?') ? '&' : '?';
        iframe.src = `${baseSrc}${separator}reload=${Date.now()}`;
      });
    };

    let lastActiveSlide = null;
    const handleSlideChange = () => {
      const activeSlide = document.querySelector(`${slideSelector}.${activeClass}`) ||
                          document.querySelector(`.${activeClass}${slideSelector}`);
      if (!activeSlide || activeSlide === lastActiveSlide) {
        return;
      }

      lastActiveSlide = activeSlide;
      reloadIframes(activeSlide);
    };

    const observeSlides = () => {
      const observer = new MutationObserver(handleSlideChange);
      document.querySelectorAll(slideSelector).forEach((slide) => {
        observer.observe(slide, { attributes: true, attributeFilter: ['class'] });
      });

      handleSlideChange();
      window.addEventListener('pageshow', handleSlideChange);
    };

    // Wait for bespoke to initialize
    const waitAndInit = () => {
      const slides = document.querySelectorAll(slideSelector);
      if (slides.length > 0) {
        observeSlides();
      } else {
        requestAnimationFrame(waitAndInit);
      }
    };

    if (document.readyState === 'complete') {
      waitAndInit();
    } else {
      window.addEventListener('load', waitAndInit, { once: true });
    }
  })();
</script>

<!-- _class: titlepage -->

<div class="title"         > Connecting Context: Future Transports</div>
<div class="subtitle"      > MCP Connect, Paris   </div>
<div class="author"        > Shaun Smith                       </div>
<div class="date"          > February 2026                                    </div>
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

---

<div class="columns">

<div>

# Shaun Smith `@evalstate`

- ### Open Source @ Hugging Face 
- ### MCP Maintainer / Transports WG
- ### huggingface/hf-mcp-server
- ### huggingface/upskill
- ### huggingface/skills
- ### huggingface.co/datasets/mcp-clients
- ### Maintainer of `fast-agent` 

</div>



<div class="center">

![w:250](./images/hf_logo.svg)
![w:250](./images/mcp-icon.svg)

</div>


</div>


---


<video class="full-slide" src="./images/intro-spaces.webm" autoplay loop muted playsinline></video>

---

# HF Hub MCP Server API Activity (from Jun 2025)

<center>

![w:1500](./images/2026-02-03-mcpremote1.png)

</center>




---

# HF Hub MCP Server API Activity (from Jun 2025)

<center>

![w:1500](./images/2026-02-03-mcpremote2.png)

</center>

---

# What's going on??

<div class="columns">


<div>

## Initialize Requests?
- ### 1% MCP Traffic -> Tool Call
- ### Unreliable proxy for MCP install

## Tool Calls: More != Better
- ### Human vs. Agent usage. High Call rate may indicate failure.

## *Sessions with at least 1 Tool Call*
- ### 2% Conversion rate from initialize

</div>

<div>

<center>

![w:550](./images/2026-02-04-efficient.png)

</center>

</div>

</div>


<!-- Initialize Events might tell us whether someone has the MCP Server installed -->
<!-- Raw Tool Calls is a potentially misleading vanity metric -->
<!-- Sessions that convert to at least one tool call --> 
<!-- typical ratio is 1.77% of initialize events are "interesting" -->

---

# Last 6 weeks of activity

<center>

![w:1500](./images/2026-02-03-mcp-server-stats.png)

</center>


---

# Open Source Client Data!!

## https://huggingface.co/datasets/evalstate/mcp-clients


<center>

![w:820](./images/2026-02-04-client-dataset.png)

</center>


---

# Preparing MCP for the future

<!-- _class: transition -->

<!--

## Robust Agent Loops of hundreds of turns

## Adoption of MCP Apps at Internet Scale 

## Create, control and manage Resources


-->


---

# Preparing MCP for the future

<iframe class="demo" loading="lazy" src="./animations/stdio-simple.html"></iframe>


---

<!-- _class: transition -->

# Change 1:  Stateless Protocol

---

<div style="text-align: center; font-size: 22px; font-weight: 600; margin-bottom: 6px;">Future Protocol - No Initialize, Capabilities with Request/Response Pair</div>

<iframe class="demo" loading="lazy" src="./animations/http-multinode-stateless.html"></iframe>

<!-- discoverable with a "discovery" endpoint -->

---

<div style="text-align: center; font-size: 22px; font-weight: 600; margin-bottom: 6px;">Current Protocol - Stateful Connection</div>

<iframe class="demo" loading="lazy" src="./animations/http-multinode.html"></iframe>

---

<div style="text-align: center; font-size: 22px; font-weight: 600; margin-bottom: 6px;">Sampling and Elicitation</div>

<iframe class="demo" loading="lazy" src="./animations/http-multinode-shared-storage.html"></iframe>

---

<!-- _class: transition -->

# Change 2: - Multi Round Trip Requests

---

<div style="text-align: center; font-size: 22px; font-weight: 600; margin-bottom: 6px;">Current Protocol - Needs to manage Request Id</div>

<iframe class="demo" loading="lazy" src="./animations/http-multinode-shared-storage.html"></iframe>

---


<div style="text-align: center; font-size: 22px; font-weight: 600; margin-bottom: 6px;">Multi-round trip flow + stateful request</div>

<div class="columns">

<div>

<iframe class="demo demo--column" loading="lazy" src="./animations/mcp-mrtr-flow.html"></iframe>

</div>

<div>

<iframe class="demo demo--column" loading="lazy" src="./animations/mcp-stateful-request.html"></iframe>

</div>

</div>


---

<div style="text-align: center; font-size: 22px; font-weight: 600; margin-bottom: 6px;">Chat interaction vs API payload</div>

<div class="columns">

<div>

<iframe class="demo demo--column" loading="lazy" src="./animations/chat-demo.html"></iframe>

</div>

<div>

<iframe class="demo demo--column" loading="lazy" src="./animations/chat-api-view.html"></iframe>

</div>

</div>


---

<div style="text-align: center; font-size: 22px; font-weight: 600; margin-bottom: 6px;">MRTR flow + accumulated request</div>

<div class="columns">

<div>

<iframe class="demo demo--column" loading="lazy" src="./animations/mcp-mrtr-flow.html"></iframe>

</div>

<div>

<iframe class="demo demo--column" loading="lazy" src="./animations/mcp-mrtr-request.html"></iframe>

</div>

</div>


---

<!-- _class: transition -->

# Change 3: Real Sessions (Cookies)

---

<div style="text-align: center; font-size: 22px; font-weight: 600; margin-bottom: 6px;">MCP cookies for session semantics</div>

<iframe class="demo" loading="lazy" src="./animations/http-multinode-cookie.html"></iframe>

---

# Supporting Changes

## Duplication of JSON-RPC content within HTTP Headers.

## Clarification of Sampling / Elicitation usage:

<center>

![w450](images/2026-02-03-mcp-webcam.png)

</center>

---

# Transport WG / Relevant SEPs

- ## Handle inconsistencies between transports
- ## Separate JSON-RPC layer from Protocol Data Layer.
- ## SEP #1442 - Make MCP Stateless by Default: Move State captured in Initialize to Request/Response cycle.

- ## Pure HTTP Transport - `https://github.com/mikekistler/pure-http-transport`


---


---

<!-- _class: transition -->

### _Thanks to the Transport Working Group_

<!-- _class: biblio -->

![bg left:33% opacity:20% blur:8px](https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80)

1. Everything Server PR 1: https://github.com/modelcontextprotocol/servers/pull/2789
1. Everything Server PR 2: https://github.com/modelcontextprotocol/servers/pull/2672
1. Hugging Face MCP Server: https://huggingface.co/mcp
1. MCP community Working Groups https://modelcontextprotocol-community.github.io/working-groups/

---


# Streamable HTTP — Dual Cluster

<iframe class="demo" loading="lazy" src="./animations/http-dual-cluster.html"></iframe>

---
