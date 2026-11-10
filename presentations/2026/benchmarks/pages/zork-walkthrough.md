---
class: zork-slide
---

# Task: list every command typed

<div class="zork-task">
<figure>
<video poster="/zork/opening-screen.png" autoplay muted loop playsinline aria-label="The supplied Zork task video, played at four times speed"><source src="/zork/zork-run-4x.webm" type="video/webm" /><source src="/zork/zork-run-4x.mp4" type="video/mp4" /></video>
</figure>
<div>
<p class="zork-lead">A 190-second recording of someone playing Zork.</p>
<p>List every command the player typed, in order.</p>
<p class="zork-small">Shown here at 4× speed.</p>
</div>
</div>

<!--
Talk over why this is hard: the screen scrolls, the same lines appear in many frames, the game's replies look like commands, and OCR on a DOS font makes small errors that change a command.
The task supplies the video locally; nothing needs downloading. The frame shown is benchmark media, not a solution, and no reference move list is reproduced.
Source: TB 2.1 extract-moves-from-video, task video inspected 10 Nov 2026, 190.10 s.
-->

---
class: zork-slide
---

# Detours are rare and don't clearly help

<div class="zork-columns">
<div>

<RateBars :rows="[
  { label: 'All trials', won: 436, n: 850, base: true },
  { label: 'OCR on video frames', won: 409, n: 755 },
  { label: 'Looks up a walkthrough', won: 29, n: 48 },
  { label: 'Runs Zork itself', won: 20, n: 41 },
  { label: 'Hunts the source video', won: 5, n: 14 },
]" />

<p class="zork-small">Share of trials rewarded. Almost everyone runs OCR. The three detours together are about one trial in eight.</p>

</div>
<Card>

### One run tracked down the original

<ol class="zork-story">
<li><b>Step 79</b> Asks speedrun.com for Zork runs. One is 173 seconds; the task's video is 190.</li>
<li><b>Step 86</b> Wonders whether the original comes with a list of inputs.</li>
<li><b>Step 88</b> Asks YouTube for it: <q>Sign in to confirm you’re not a bot.</q></li>
<li><b>Step 89</b> <q>YouTube download is blocked. Let me quickly probe TASVideos for a Zork movie, then refocus on reconstruction.</q></li>
</ol>

<p class="zork-small">DeepSeek V4.1 Flash. It had found the right video, never got in, went back to OCR and was rewarded.</p>

</Card>
</div>

<Receipt n="850 trials with a known reward, of 866" date="snapshot 8 Nov 2026" model="mixed" harness="fast-agent · TB 2.1 extract-moves-from-video" />

<!--
The point: the inventive strategies are real, but they are uncommon and they don't beat the ordinary route.
OCR: 755 of 850 trials run tesseract on extracted frames; 54% of those are rewarded, against 28% of the 95 that don't.
Walkthrough: 48 trials look one up (18 of them Grok). 60% rewarded, but on 48 trials that is not distinguishable from the 51% average. A published walkthrough is also not the route in the video.
Interpreter: 41 trials install frotz or similar (14 of them GLM) to replay or test commands. 48% rewarded.
Source video: 14 trials reach for yt-dlp (11 Grok), mostly as a probe. 5 rewarded.
The story on the right: DeepSeek V4.1 Flash, fast-agent 0.10.24, 11 Sep 2026, 439 steps, reward 1. Quotes are masked and verified at steps 88 and 89.
An interpreter example to tell aloud: GLM-5.3 Flash, fast-agent 0.10.51, 4 Nov 2026: "I can replay the commands in a real Zork interpreter to verify. Let me set that up." It installs frotz, finds the matching game release and runs partial sequences. Reward 0 with an agent-exit error, so don't read the 0 as evidence against replaying.
More on the speedrun hunt: it queried the speedrun.com API, got a runner, a time and a video ID, and that ID matches the source video named in the task's README. Why it looked: at step 86 its reasoning considers whether the source material might expose an input list. That is its stated intent, not something it obtained. It then asked for the video's metadata (a probe, not a download) and got the sign-in wall; it tried TASVideos (steps 89–99) and Invidious (step 101) too. Step 102: "Invidious instances blocked/disabled. Let me stop chasing YouTube." It never got the video. The OCR work carried on alongside and the run was rewarded: identification, not acquisition.
A GPT-5.5 run on another harness (linghun 0.1.50, 20 Jul 2026) builds the whole pipeline from npm, ffmpeg and Tesseract.js over 190 frames, and installs an interpreter "for optional route validation". Reward 1.
The detour groups overlap with OCR; a trial can be in several rows. Strategies are pattern matches on the agent's own tool calls, and the small groups are within noise of the average.
"The right video": the ID it found is the source video named in the task's own README.
-->

---
class: zork-slide
---

# Same OCR, from 0% to 100%

<div class="zork-columns">
<div style="--rates-label: 150px">

<RateBars :rows="[
  { label: 'GPT-6.1 Sol', won: 45, n: 45 },
  { label: 'GPT-6 Astra', won: 72, n: 77 },
  { label: 'GPT-6.1 Terra', won: 50, n: 55 },
  { label: 'GPT-6 Luna', won: 95, n: 213 },
  { label: 'GLM', won: 17, n: 49 },
  { label: 'DeepSeek V4 Flash', won: 0, n: 34 },
]" />

<p class="zork-small">Share of trials rewarded. All six run OCR in nearly every trial.</p>

</div>
<Card>

### What GPT-6.1 Sol does differently

Same tools as everyone else, plus bookkeeping:

1. lines up scrolling screens by fuzzy overlap
2. keeps explicit tables of OCR fixes
3. re-crops the regions it can't read

<p class="zork-quote-line"><q>My plan is to do line fuzzy matching against the previous screen by selecting the longest contiguous approximate run.</q></p>

</Card>
</div>

<Receipt n="866 trials, 6 families shown" date="snapshot 8 Nov 2026" model="as listed" harness="fast-agent · TB 2.1 extract-moves-from-video" />

<!--
The point of the segment: on this task the tool choice doesn't separate the models. Careful handling of the OCR output does.
OCR use: GPT-6.1 Sol 45 of 46 trials, Astra 73 of 80, Terra 52 of 55, Luna 157 of 213, GLM 49 of 52, DeepSeek V4 Flash 33 of 34.
DeepSeek V4 Flash can't read images (0 image reads recorded) and is rewarded 0 of 34 times. GPT-6 Luna reads images in 190 of 213 trials and is rewarded 95 times.
The Sol quote is an exact substring of masked reasoning (fast-agent 0.10.42, reward 1), checked against the recorded tool calls. That run also reads some frames as images, so don't call it OCR-only. Alignment, repair tables and re-crops are actions in the trace; that is not a claim about why the grader rewarded it.
Trials with an unknown reward are left out of each count; GLM is mixed versions.
Corrections to older rollups: Sol alone is 45 of 45 known plus one unknown; the earlier 95% was Sol and Terra together. The earlier 15% DeepSeek figure mixed versions; V4 Flash alone is 0 of 34.
Unfiltered fast-agent cohort, including short failures. Versions, effort settings and dates differ, so this is correlation.
-->
