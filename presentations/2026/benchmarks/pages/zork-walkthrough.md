---
class: zork-slide
---

# Task: recover every command typed

<div class="zork-task">
<figure>
<video poster="/zork/opening-screen.png" autoplay muted loop playsinline aria-label="The supplied Zork task video, played at four times speed"><source src="/zork/zork-run-4x.webm" type="video/webm" /><source src="/zork/zork-run-4x.mp4" type="video/mp4" /></video>
<figcaption>The task video, 190 seconds, shown here at 4× speed.</figcaption>
</figure>
<div>
<p class="zork-lead">A screen recording of someone playing Zork.</p>
<p>The agent has to list each command the player typed, in order. The game's replies and repeated screens don't count.</p>
<p>The video file is already in the container.</p>
</div>
</div>

<div class="zork-path">Video → frames → OCR or image reading → reconcile → move list</div>

<Receipt source="TB 2.1 extract-moves-from-video, task video inspected 10 Nov 2026" />

<!--
The task supplies the video locally. An older task README described downloading it.
The frame shown is benchmark media, not a benchmark solution. No reference move list is reproduced.
The path line is a narrative map of what most runs do, not a census.
-->

---
class: zork-slide
---

# GPT-6.1 Sol: plain OCR, then reconcile

<div class="zork-columns">
<div>
<p class="zork-lead">Frames and Tesseract, like nearly everyone. The difference is what it does with the output.</p>
<ul>
<li>Aligns scrolling screens by fuzzy overlap.</li>
<li>Repairs OCR errors with explicit skip and fix tables.</li>
<li>Goes back to regions it can't read with denser crops.</li>
</ul>
<p>Reward 1. It reads some frames as images too.</p>
</div>
<Card>
<p class="zork-quote">“My plan is to do line fuzzy matching against the previous screen by selecting the longest contiguous approximate run.”</p>
<p class="zork-small">From its reasoning, masked.</p>
</Card>
</div>

<Receipt n="1 trial" date="3 Oct 2026" model="GPT-6.1 Sol" harness="fast-agent 0.10.42 · TB 2.1 extract-moves-from-video" />

<!--
The quote is an exact 22-word substring of masked atif-inspect reasoning, checked against the recorded tool calls.
The run also records image reads. Do not call it text-only or OCR-only.
Alignment, repair tables and crop checks are actions in the trace. That is not a claim about why the grader rewarded it.
The family's success denominator is on the last Zork slide.
-->

---
class: zork-slide
---

# To transcribe Zork, prepare to run Zork

<div class="zork-columns">
<Card>
<h3>GLM tests the game parser</h3>
<p>Installs frotz, finds the matching game release, and runs scripted tests and partial command sequences.</p>
<p class="zork-quote">“I can replay the commands in a real Zork interpreter to verify. Let me set that up.”</p>
<p class="zork-small">Reward 0: the agent exited with an error.</p>
<Receipt n="1 trial" date="4 Nov 2026" model="GLM-5.3 Flash" harness="fast-agent 0.10.51" />
</Card>
<Card>
<h3>GPT-5.5 builds an npm pipeline</h3>
<p>ffmpeg and Tesseract.js from npm, then 190 frames, one per second.</p>
<p class="zork-quote">“Install lightweight Z-machine interpreter for optional route validation”</p>
<p class="zork-small">Reward 1. It installs the interpreter; the trace doesn't show it replaying the route.</p>
<Receipt n="1 trial" date="20 Jul 2026" model="GPT-5.5" harness="linghun 0.1.50" />
</Card>
</div>

<!--
GLM's recorded error is NonZeroAgentExitCodeError. Don't read its reward 0 as evidence against replaying; full route validation isn't established either.
GPT-5.5's quote is a tool-call description. It also makes a late benchmark-source request after writing its artefact.
No causal benefit from that request is established. Do not describe the run as reference-free.
-->

---
class: zork-slide
---

# It found the original speedrun

<div class="zork-columns">
<ol>
<li>It queries the speedrun.com API and gets back a runner, a time and a video ID.</li>
<li>That ID is the source video named in the task's own README.</li>
<li>It asks for the video's metadata. It also tries TASVideos and Invidious.</li>
</ol>
<Card>
<p class="zork-quote">“Sign in to confirm you’re not a bot.”</p>
<p class="zork-small">What came back from the video-metadata request.</p>
</Card>
</div>

<p>It never gets the video. The OCR work carries on alongside, and the run is rewarded.</p>

<Receipt n="1 trial" date="11 Sep 2026" model="DeepSeek V4.1 Flash" harness="fast-agent 0.10.24 · TB 2.1 extract-moves-from-video">video ID checked against the local README, no remote visit</Receipt>

<!--
The agent identified the same source identifier, but did not establish it was definitively correct.
This is identification, not acquisition: no successful video download or acquired move list. The reward does not show the hunt helped.
A contrast if useful aloud: a separate SWE-1.7 Lightning run reads a public Pastebin route,
notices its move count differs from the recording, and times out with reward zero.
No published route content is reproduced here.
-->

---
class: zork-slide
---

# Same tools, very different results

<table class="zork-stats">
<thead><tr><th>Model family</th><th>Trials</th><th>Ran OCR</th><th>Read images</th><th>Rewarded</th><th>Reward unknown</th></tr></thead>
<tbody>
<tr><td>GPT-6.1 Sol</td><td>46</td><td>45</td><td>45</td><td>45 of 45</td><td>1</td></tr>
<tr><td>GPT-6.1 Terra</td><td>55</td><td>52</td><td>23</td><td>50 of 55</td><td>0</td></tr>
<tr><td>GPT-6 Astra</td><td>80</td><td>73</td><td>78</td><td>72 of 77</td><td>3</td></tr>
<tr><td>GPT-6 Luna</td><td>213</td><td>157</td><td>190</td><td>95 of 213</td><td>0</td></tr>
<tr><td>GLM, mixed versions</td><td>52</td><td>49</td><td>51</td><td>17 of 49</td><td>3</td></tr>
<tr><td>DeepSeek V4 Flash</td><td>34</td><td>33</td><td>0</td><td>0 of 34</td><td>0</td></tr>
</tbody>
</table>

<p>Almost every family runs OCR. Rewards run from 0 of 34 to 45 of 45.</p>
<p class="zork-small">Labels are pattern matches and can include probes. Versions, effort settings and dates differ, so this shows no cause.</p>

<Receipt n="866 trials, 850 with a known reward (436 rewarded)" date="snapshot 8 Nov 2026" model="6 families shown" harness="fast-agent, versions not kept in the index · TB 2.1 extract-moves-from-video" />

<!--
Corrected old rollups: Sol alone is 45/45 known plus one unknown.
The earlier 95% was Sol+Terra together, not Sol. The earlier 15% DeepSeek figure
mixed several versions; V4 Flash alone is 0/34.
This is an unfiltered fast-agent cohort, including short failures. Unknown reward is not failure.
"Read images" 0 for DeepSeek V4 Flash means no image read was recorded; other harnesses record image reads less reliably, so no cross-harness vision claim.
No causal strategy-effect claim; no accusation of cheating.
-->
