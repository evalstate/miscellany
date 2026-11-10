---
class: zork-slide
---

# Transcribe a video. What could that involve?

<div class="zork-task">
<figure>
<video poster="/zork/opening-screen.png" autoplay muted loop playsinline aria-label="The supplied Zork task video, played at four times speed"><source src="/zork/zork-run-4x.webm" type="video/webm" /><source src="/zork/zork-run-4x.mp4" type="video/mp4" /></video>
<figcaption>The full 190-second task video at 4× speed, looped.</figcaption>
</figure>
<div>
<p class="zork-lead">A scrolling DOS game.<br>190 seconds to transcribe.</p>
<p>Recover every typed command, not the game’s replies or repeated screens.</p>
<p><strong>The video is supplied locally.</strong></p>
</div>
</div>

<div class="zork-path">Video → frames → OCR or image reading → reconcile → output</div>

<Receipt n="1 task video" date="inspected 10 Nov 2026" model="not applicable" harness="local media inspection" source="local TB 2.1 task video">
Duration 190.10 s, shown at 4× speed. No reference move list reproduced.
</Receipt>

<style>
.slidev-layout.zork-slide h1 { font-size: 36px; margin-bottom: 18px; }
.zork-slide .zork-task { display: grid; grid-template-columns: 1fr 1.25fr; gap: 28px; align-items: start; }
.zork-slide figure { margin: 0; }
.zork-slide figure img { width: 100%; border: var(--border); border-radius: var(--radius-md); }
.zork-slide figure video { display: block; width: 100%; border: var(--border); border-radius: var(--radius-md); }
.zork-slide figcaption { font: 500 13px/1.4 var(--font-read); color: var(--text-muted); margin-top: 6px; }
.zork-slide .zork-lead { font-size: 22px; line-height: 1.3; font-weight: 700; margin: 0 0 12px; }
.zork-slide p { font-size: 17px; line-height: 1.4; margin: 0 0 12px; }
.zork-slide .zork-path { border: var(--border); border-radius: var(--radius-md); padding: 10px 16px; margin: 12px 0; font-weight: 700; font-size: 18px; background: var(--ivory-deep); }
.zork-slide .zork-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; margin-bottom: 18px; }
.zork-slide .zork-columns .fa-card { padding: 18px; }
.zork-slide .zork-columns h3 { font-size: 23px; margin-bottom: 12px; }
.zork-slide .zork-columns ul { font-size: 18px; margin: 0 0 14px; }
.zork-slide .zork-quote { font: 700 23px/1.22 var(--font-voice); letter-spacing: var(--track-voice); margin: 0 0 16px; }
.zork-slide .zork-small { font-size: 15px; line-height: 1.4; color: var(--text-muted); }
.zork-slide .zork-stats { font-size: 15px; line-height: 1.2; margin-bottom: 14px; }
.zork-slide .zork-stats th, .zork-slide .zork-stats td { padding: 5px 10px; }
.zork-slide .zork-route { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 8px; }
.zork-slide .zork-route .fa-card { padding: 18px; }
.zork-slide .zork-route h3 { font-size: 22px; margin-bottom: 10px; }
.zork-slide .zork-route p { font-size: 17px; }
.zork-slide .fa-receipt { font-size: 12px; line-height: 1.3; }
</style>

<!--
Task supplied a local video. The older task README described downloading it.
The opening screen is benchmark media, not a copied benchmark solution.
The diagram is a narrative map, not an exhaustive or mutually exclusive census.
-->

---
class: zork-slide
---

# The winning detail was reconciliation

<div class="zork-columns">
<div>
<p class="zork-lead">GPT-6.1 Sol uses ordinary tools,<br>then keeps careful accounts.</p>
<ul>
<li>Extract frames and run Tesseract.</li>
<li>Align scrolling screens by fuzzy overlap.</li>
<li>Repair OCR with explicit skip/fix tables.</li>
<li>Revisit ambiguous regions with denser crops.</li>
</ul>
<p><strong>Rewarded.</strong> Image reading is also recorded; this is not an OCR-only example.</p>
</div>
<Card>
<p class="zork-quote">“My plan is to do line fuzzy matching against the previous screen by selecting the longest contiguous approximate run.”</p>
<p class="zork-small">Masked reasoning, reviewed against the recorded tool calls.</p>
</Card>
</div>

<Receipt n="1 inspected run" date="3 Oct 2026" model="gpt-6.1-sol" harness="fast-agent 0.10.42" source="masked trace review">
Recorded reward 1. Screen alignment, repair tables and crop checks are actions in the trace, not a claim about why the grader rewarded it.
</Receipt>

<!--
The quote is an exact 22-word substring of masked atif-inspect reasoning.
The run also records image reads. Do not call it text-only or ordinary OCR alone.
The family success denominator and unknown outcome are on the closing slide.
-->


<style>
.slidev-layout.zork-slide h1 { font-size: 36px; margin-bottom: 18px; }
.zork-slide .zork-task { display: grid; grid-template-columns: 1fr 1.25fr; gap: 28px; align-items: start; }
.zork-slide figure { margin: 0; }
.zork-slide figure img { width: 100%; border: var(--border); border-radius: var(--radius-md); }
.zork-slide figcaption { font: 500 13px/1.4 var(--font-read); color: var(--text-muted); margin-top: 6px; }
.zork-slide .zork-lead { font-size: 22px; line-height: 1.3; font-weight: 700; margin: 0 0 12px; }
.zork-slide p { font-size: 17px; line-height: 1.4; margin: 0 0 12px; }
.zork-slide .zork-path { border: var(--border); border-radius: var(--radius-md); padding: 10px 16px; margin: 12px 0; font-weight: 700; font-size: 18px; background: var(--ivory-deep); }
.zork-slide .zork-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; margin-bottom: 18px; }
.zork-slide .zork-columns .fa-card { padding: 18px; }
.zork-slide .zork-columns h3 { font-size: 23px; margin-bottom: 12px; }
.zork-slide .zork-columns ul { font-size: 18px; margin: 0 0 14px; }
.zork-slide .zork-quote { font: 700 23px/1.22 var(--font-voice); letter-spacing: var(--track-voice); margin: 0 0 16px; }
.zork-slide .zork-small { font-size: 15px; line-height: 1.4; color: var(--text-muted); }
.zork-slide .zork-stats { font-size: 15px; line-height: 1.2; margin-bottom: 14px; }
.zork-slide .zork-stats th, .zork-slide .zork-stats td { padding: 5px 10px; }
.zork-slide .zork-route { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 8px; }
.zork-slide .zork-route .fa-card { padding: 18px; }
.zork-slide .zork-route h3 { font-size: 22px; margin-bottom: 10px; }
.zork-slide .zork-route p { font-size: 17px; }
.zork-slide .fa-receipt { font-size: 12px; line-height: 1.3; }
</style>

---
class: zork-slide
---

# To transcribe Zork, prepare to run Zork

<div class="zork-columns">
<Card>
<h3>GLM tests the game parser</h3>
<p>Installs frotz, finds the matching game release and runs scripted tests and partial sequences.</p>
<p class="zork-quote">“I can replay the commands in a real Zork interpreter to verify. Let me set that up.”</p>
<p class="zork-small">Recorded reward 0; agent-exit error. Complete route validation is not established.</p>
<Receipt n="1" date="4 Nov 2026" model="GLM-5.3 Flash" harness="fast-agent 0.10.51" source="masked trace review" />
</Card>
<Card>
<h3>GPT builds an npm pipeline</h3>
<p>GPT-5.5 uses npm-supplied ffmpeg and Tesseract.js, then extracts 190 one-second frames.</p>
<p class="zork-quote">“Install lightweight Z-machine interpreter for optional route validation”</p>
<p class="zork-small">Recorded reward 1; preparation is visible, actual interpreter replay is not established.</p>
<Receipt n="1" date="20 Jul 2026" model="gpt-5.5" harness="linghun 0.1.50" source="masked tool-call description" />
</Card>
</div>



<!--
GLM's recorded error is NonZeroAgentExitCodeError; do not interpret its reward as proof against replay.
GPT-5.5 also makes a late benchmark-source request after writing its artifact.
No causal benefit from that request is established. Do not describe the run as reference-free.
-->


<style>
.slidev-layout.zork-slide h1 { font-size: 36px; margin-bottom: 18px; }
.zork-slide .zork-task { display: grid; grid-template-columns: 1fr 1.25fr; gap: 28px; align-items: start; }
.zork-slide figure { margin: 0; }
.zork-slide figure img { width: 100%; border: var(--border); border-radius: var(--radius-md); }
.zork-slide figcaption { font: 500 13px/1.4 var(--font-read); color: var(--text-muted); margin-top: 6px; }
.zork-slide .zork-lead { font-size: 22px; line-height: 1.3; font-weight: 700; margin: 0 0 12px; }
.zork-slide p { font-size: 17px; line-height: 1.4; margin: 0 0 12px; }
.zork-slide .zork-path { border: var(--border); border-radius: var(--radius-md); padding: 10px 16px; margin: 12px 0; font-weight: 700; font-size: 18px; background: var(--ivory-deep); }
.zork-slide .zork-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; margin-bottom: 18px; }
.zork-slide .zork-columns .fa-card { padding: 18px; }
.zork-slide .zork-columns h3 { font-size: 23px; margin-bottom: 12px; }
.zork-slide .zork-columns ul { font-size: 18px; margin: 0 0 14px; }
.zork-slide .zork-quote { font: 700 23px/1.22 var(--font-voice); letter-spacing: var(--track-voice); margin: 0 0 16px; }
.zork-slide .zork-small { font-size: 15px; line-height: 1.4; color: var(--text-muted); }
.zork-slide .zork-stats { font-size: 15px; line-height: 1.2; margin-bottom: 14px; }
.zork-slide .zork-stats th, .zork-slide .zork-stats td { padding: 5px 10px; }
.zork-slide .zork-route { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 8px; }
.zork-slide .zork-route .fa-card { padding: 18px; }
.zork-slide .zork-route h3 { font-size: 22px; margin-bottom: 10px; }
.zork-slide .zork-route p { font-size: 17px; }
.zork-slide .fa-receipt { font-size: 12px; line-height: 1.3; }
</style>

---
class: zork-slide
---

# It found the original speedrun

<div class="zork-route">
<Card>
<h3>Find the runner</h3>
<p>A corrected speedrun.com API query returns the runner, time and video identifier.</p>
<p>The agent names a candidate.</p>
</Card>
<Card>
<h3>Match the source</h3>
<p>That identifier exactly matches the original video named in the local task README.</p>
<p><strong>Identification, not acquisition.</strong></p>
</Card>
<Card>
<h3>Meet the bot wall</h3>
<p class="zork-quote">“Sign in to confirm you’re not a bot.”</p>
<p>Recorded result of a video-metadata probe.</p>
</Card>
</div>

<p>It probes TASVideos and Invidious too. Local OCR work continues alongside the detour.</p>
<p class="zork-small">No successful source-video acquisition is shown in this episode. Its final reward does not establish that the hunt helped.</p>

<Receipt n="1" date="11 Sep 2026" model="DeepSeek V4.1 Flash" harness="fast-agent 0.10.24" source="masked review + local README">
Reward 1. Exact source identifier checked locally; no remote visit.
</Receipt>

<!--
The agent identified the same source identifier, but did not establish it was definitively correct.
This is not a claim of a successful video download or acquired move list.
A contrast if useful aloud: a separate SWE-1.7 Lightning run reads a public Pastebin route,
notices its move count differs from the recording, and times out with reward zero.
No published route content is reproduced here.
-->


<style>
.slidev-layout.zork-slide h1 { font-size: 36px; margin-bottom: 18px; }
.zork-slide .zork-task { display: grid; grid-template-columns: 1fr 1.25fr; gap: 28px; align-items: start; }
.zork-slide figure { margin: 0; }
.zork-slide figure img { width: 100%; border: var(--border); border-radius: var(--radius-md); }
.zork-slide figcaption { font: 500 13px/1.4 var(--font-read); color: var(--text-muted); margin-top: 6px; }
.zork-slide .zork-lead { font-size: 22px; line-height: 1.3; font-weight: 700; margin: 0 0 12px; }
.zork-slide p { font-size: 17px; line-height: 1.4; margin: 0 0 12px; }
.zork-slide .zork-path { border: var(--border); border-radius: var(--radius-md); padding: 10px 16px; margin: 12px 0; font-weight: 700; font-size: 18px; background: var(--ivory-deep); }
.zork-slide .zork-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; margin-bottom: 18px; }
.zork-slide .zork-columns .fa-card { padding: 18px; }
.zork-slide .zork-columns h3 { font-size: 23px; margin-bottom: 12px; }
.zork-slide .zork-columns ul { font-size: 18px; margin: 0 0 14px; }
.zork-slide .zork-quote { font: 700 23px/1.22 var(--font-voice); letter-spacing: var(--track-voice); margin: 0 0 16px; }
.zork-slide .zork-small { font-size: 15px; line-height: 1.4; color: var(--text-muted); }
.zork-slide .zork-stats { font-size: 15px; line-height: 1.2; margin-bottom: 14px; }
.zork-slide .zork-stats th, .zork-slide .zork-stats td { padding: 5px 10px; }
.zork-slide .zork-route { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 8px; }
.zork-slide .zork-route .fa-card { padding: 18px; }
.zork-slide .zork-route h3 { font-size: 22px; margin-bottom: 10px; }
.zork-slide .zork-route p { font-size: 17px; }
.zork-slide .fa-receipt { font-size: 12px; line-height: 1.3; }
</style>

---
class: zork-slide
---

# The tools did not settle the result

<table class="zork-stats">
<thead><tr><th>Model family</th><th>Trials</th><th>OCR label</th><th>Image label</th><th>Rewarded / known</th><th>Unknown</th></tr></thead>
<tbody>
<tr><td>GPT-6.1 Sol</td><td>46</td><td>45</td><td>45</td><td>45 / 45</td><td>1</td></tr>
<tr><td>GPT-6.1 Terra</td><td>55</td><td>52</td><td>23</td><td>50 / 55</td><td>0</td></tr>
<tr><td>GPT-6 Astra</td><td>80</td><td>73</td><td>78</td><td>72 / 77</td><td>3</td></tr>
<tr><td>GPT-6 Luna</td><td>213</td><td>157</td><td>190</td><td>95 / 213</td><td>0</td></tr>
<tr><td>GLM, mixed versions</td><td>52</td><td>49</td><td>51</td><td>17 / 49</td><td>3</td></tr>
<tr><td>DeepSeek V4 Flash</td><td>34</td><td>33</td><td>0</td><td>0 / 34</td><td>0</td></tr>
</tbody>
</table>

<p><strong>Reconciliation matters in the example.</strong> These counts do not establish causes.</p>
<p class="zork-small">Overlapping labels can include probes. No image label ≠ text-only. Versions, effort and dates differ.</p>

<Receipt n="866 task-labelled trials" date="8 Nov 2026 snapshot" model="mixed, as listed" harness="fast-agent; versions not retained" source="strategy index · no short-trace filter">
436/850 known rewarded; 16 unknown. Selected families shown; unknown is not failure.
</Receipt>

<!--
Corrected old rollups: Sol alone is 45/45 known plus one unknown.
The earlier 95% was Sol+Terra together, not Sol. The earlier 15% DeepSeek figure
mixed several versions; V4 Flash alone is 0/34.
This is an unfiltered fast-agent cohort, including short failures.
Other harnesses record image reads less reliably, so no cross-harness vision claim.
No causal strategy-effect claim; no accusation of cheating.
-->


<style>
.slidev-layout.zork-slide h1 { font-size: 36px; margin-bottom: 18px; }
.zork-slide .zork-task { display: grid; grid-template-columns: 1fr 1.25fr; gap: 28px; align-items: start; }
.zork-slide figure { margin: 0; }
.zork-slide figure img { width: 100%; border: var(--border); border-radius: var(--radius-md); }
.zork-slide figcaption { font: 500 13px/1.4 var(--font-read); color: var(--text-muted); margin-top: 6px; }
.zork-slide .zork-lead { font-size: 22px; line-height: 1.3; font-weight: 700; margin: 0 0 12px; }
.zork-slide p { font-size: 17px; line-height: 1.4; margin: 0 0 12px; }
.zork-slide .zork-path { border: var(--border); border-radius: var(--radius-md); padding: 10px 16px; margin: 12px 0; font-weight: 700; font-size: 18px; background: var(--ivory-deep); }
.zork-slide .zork-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; margin-bottom: 18px; }
.zork-slide .zork-columns .fa-card { padding: 18px; }
.zork-slide .zork-columns h3 { font-size: 23px; margin-bottom: 12px; }
.zork-slide .zork-columns ul { font-size: 18px; margin: 0 0 14px; }
.zork-slide .zork-quote { font: 700 23px/1.22 var(--font-voice); letter-spacing: var(--track-voice); margin: 0 0 16px; }
.zork-slide .zork-small { font-size: 15px; line-height: 1.4; color: var(--text-muted); }
.zork-slide .zork-stats { font-size: 15px; line-height: 1.2; margin-bottom: 14px; }
.zork-slide .zork-stats th, .zork-slide .zork-stats td { padding: 5px 10px; }
.zork-slide .zork-route { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 8px; }
.zork-slide .zork-route .fa-card { padding: 18px; }
.zork-slide .zork-route h3 { font-size: 22px; margin-bottom: 10px; }
.zork-slide .zork-route p { font-size: 17px; }
.zork-slide .fa-receipt { font-size: 12px; line-height: 1.3; }
</style>
