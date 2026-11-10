---
class: answer-lookup-slide
---

# The basic hack: look the answer up

<div class="al-grid">
<div>
<p class="al-lead">Flags in 55,569 Terminal-Bench 2.1 trials:</p>
<table class="al-funnel">
<tbody>
<tr><td class="al-n">414</td><td>requested the benchmark's own task material</td></tr>
<tr><td class="al-n">184</td><td>asked for a solution, a test or a finished answer by URL</td></tr>
<tr><td class="al-n">334</td><td>flagged with benchmark content arriving in context. In context means known.</td></tr>
</tbody>
</table>
<p class="al-sub">Four tasks hold 93 of the 184 answer-request flags (50%), from 4% of trials.</p>
<table class="al-tasks">
<tbody>
<tr><td>extract-elf</td><td class="al-bar"><span style="width:100%"></span></td><td class="al-n2">37</td></tr>
<tr><td>build-pov-ray</td><td class="al-bar"><span style="width:70%"></span></td><td class="al-n2">26</td></tr>
<tr><td>torch-pipeline-parallelism</td><td class="al-bar"><span style="width:43%"></span></td><td class="al-n2">16</td></tr>
<tr><td>db-wal-recovery</td><td class="al-bar"><span style="width:37%"></span></td><td class="al-n2">14</td></tr>
</tbody>
</table>
</div>
<div>
<h3>Who asks for the answer</h3>
<table class="al-models">
<thead><tr><th>Model</th><th>Request flags</th><th>Rate</th></tr></thead>
<tbody>
<tr><td>SWE-1.7 Lightning</td><td>36 / 441</td><td>8.1%</td></tr>
<tr><td>Grok 4.7</td><td>6 / 445</td><td>1.3%</td></tr>
<tr><td>Flint</td><td>34 / 2,729</td><td>1.2%</td></tr>
<tr><td>DeepSeek V4.1 Flash</td><td>48 / 4,770</td><td>1.0%</td></tr>
<tr><td>GPT-5.x and GPT-6.x</td><td>24 / 14,234</td><td>0.1%</td></tr>
<tr><td>Claude (Fable, Haiku, Sonnet)</td><td>0 / 1,665</td><td>0%</td></tr>
</tbody>
</table>
<p class="al-sub">Terminal-Bench 4 says <q>Do not cheat by using online solutions</q>: 0 of 12,896 trials flagged.</p>
</div>
</div>

<Receipt n="55,569 deduplicated TB 2.1 trials; 12,896 TB4" date="8 Nov scan, aggregated 10 Nov 2026" model="configured model labels" harness="mixed" source="atif-scan lookup checks">
Detector flags, not verified downloads or adjudicated hacks. Cohorts differ.
</Receipt>

<style>
.answer-lookup-slide h1 { font-size: 36px; margin-bottom: 10px; }
.answer-lookup-slide .al-grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 30px; align-items: start; margin-bottom: 6px; }
.answer-lookup-slide .al-lead { font-size: 19px; font-weight: 700; margin: 0 0 6px; }
.answer-lookup-slide table { width: 100%; border-collapse: collapse; }
.answer-lookup-slide td, .answer-lookup-slide th { padding: 3px 8px; border-bottom: 2px solid var(--line); text-align: left; font-size: 15px; line-height: 1.25; vertical-align: middle; }
.answer-lookup-slide .al-funnel td { font-size: 16px; }
.answer-lookup-slide .al-n { font: 700 26px/1 var(--font-voice); color: var(--petrol); width: 82px; font-variant-numeric: tabular-nums; }
.answer-lookup-slide .al-sub { font-size: 14px; color: var(--text-muted); margin: 6px 0 4px; }
.answer-lookup-slide .al-tasks td { border-bottom: 0; padding: 2px 6px; font-size: 14px; }
.answer-lookup-slide .al-bar { width: 45%; }
.answer-lookup-slide .al-bar span { display: block; height: 12px; background: var(--petrol); }
.answer-lookup-slide .al-n2 { font-variant-numeric: tabular-nums; width: 36px; text-align: right; }
.answer-lookup-slide h3 { font-size: 22px; margin: 0 0 8px; }
.answer-lookup-slide .al-models th { font-size: 13px; color: var(--text-muted); }
.answer-lookup-slide .al-models td { font-variant-numeric: tabular-nums; }
</style>

<!--
Source: ~/.cache/atif-scan/work/reference-lookup-stats-20261110 (first-pass aggregates over the 94,119-trial deduplicated selection, corpus_scan_v2).
414 = broad request union (benchmark_source, solution_url, task_files, own_task_files, task_named_skill).
184 = lookup.benchmark_solution_url (solution/test/cheat paths and other agents' finished trials).
334 = lookup.benchmark_content_obtained, an older heuristic; it can over-match unrelated canary-bearing output. Not a subset of 184.
Per the "in context means known" rule, arrival counts; reuse isn't required. Still a detector count, not adjudicated.
GPT row = gpt-5.6 terra/luna/sol, gpt-5.5, gpt-6-luna, gpt-6.1-terra. Claude row = fable-5, haiku-4.5, sonnet-5.
TB4 zero is not proof of no access: fresher tasks, fewer public solutions. See the TB4 anti-cheat-line analysis.
Verified examples: Grok extracts solution.sh JS (reward 1); GLM applies a reference patch from a task page (DeepSWE, reward 1).
-->
