---
theme: default
title: Kit · layouts and components
titleTemplate: "%s · fast-agent"
colorSchema: light
transition: none
layout: cover
fonts:
  sans: Figtree
  serif: Fraunces
  mono: DM Mono
  provider: none
drawings:
  persist: false
---

# Same model, better results.

How the harness changes what your model can do.

::meta::

**Shaun Smith** · Hugging Face

Event name · date TBC

<!--
Cover layout: headline left, homepage splash right (burst swings in, tagline types out).
Change the tagline with `tagline:` in this slide's frontmatter.
-->

---

# One claim, with its receipt

<div class="fa-grid-2" style="margin-top: 8px">
<div>

- Short declaratives. The receipt sits next to the claim.
- Round down, never up.
- Give n, date, model and harness version for every number.
- Keep the disclaimer on the same slide.

</div>
<Card tone="paper">

### Placeholder result

## Accuracy: <span class="fa-mark">TBC</span>

Replace with a measured figure.

<Receipt mark="*" source="benchmark run id TBC" />

</Card>
</div>

<!--
Default layout: the footer shows the tile mark and page number.
Set `footer: false` in frontmatter to hide it.
-->

---
layout: section
number: 1
---

# Why benchmark the harness?

Section divider on petrol, with a single score burst.

---
layout: presenter
pose: B
---

# She presents. The data stays level.

- One mascot per slide, always outside the chart area.
- Pose **B** (open palm) sits to the right of what she presents.
- Poses A–F come from the approved sheet. See `README.md`.

<Sticker tone="amber">New!</Sticker>

---

# Placeholder results table

| Harness | Model | Accuracy | Tokens / task |
| --- | --- | --- | --- |
| <span class="fa-us">fast-agent</span> | model TBC | — | — |
| Baseline A | model TBC | — | — |
| Baseline B | model TBC | — | — |

<Receipt mark="*" n="TBC" date="TBC" model="TBC" harness="fast-agent vTBC">
Illustrative layout only. No measured data yet.
</Receipt>

<!--
Mark the fast-agent row by wrapping its first cell in <span class="fa-us">. Amber = fast-agent only.
Competitor chart colours are still open in the design system.
-->

---
layout: presenter
pose: C
height: 400
---

# Run it yourself

<Terminal title="fast-agent" :lines="[
  '$ uvx fast-agent-mcp@latest -x',
  'Requires uv.',
]" />

<p style="margin-top: 18px">Setup guide: <a href="https://fast-agent.ai">fast-agent.ai</a></p>

---

# Code blocks use DM Mono on paper

```python
import asyncio
from fast_agent import FastAgent

fast = FastAgent("benchmarks")

@fast.agent(instruction="You are a helpful agent")
async def main():
    async with fast.run() as agent:
        await agent.interactive()

asyncio.run(main())
```

---
layout: cover
tagline: Thank you.
---

# Questions?

::meta::

huggingface.co/evalstate · github.com/evalstate · x.com/evalstate
