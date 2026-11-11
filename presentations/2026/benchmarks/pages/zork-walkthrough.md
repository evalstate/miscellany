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

<ZorkMethods />

<Receipt source="850 fast-agent trials with a known reward, 9 models shown · TB 2.1 extract-moves-from-video · snapshot 8 Nov 2026 · method labels are pattern matches on the agent's own tool calls" />

<!--
The point of the segment: on this task the tool choice doesn't separate the models. All nine run OCR in nearly every trial.
The earlier version of this slide said what GPT-6.1 Sol does differently: it lines up the scrolling screens by fuzzy overlap, keeps tables of OCR fixes and re-crops what it can't read. That came from one trace. Checked across all 45 of its trials: fuzzy matching in 42, cropping in 42, a named fix table in 25. So nearly always for the first two, about half the time for the third.
It is not what earns the 100%. GPT-6.1 Terra has fuzzy matching in 2 of 55 trials, reads frames as images in only 23, takes a median 25 steps, and is rewarded 50 times. GPT-6 Astra: 32 of 77 with fuzzy matching, 72 rewarded.
Claude, which the earlier slide left out: Opus 5.5 10 of 10 and Sonnet 5.5 20 of 20, the same 100% as Sol. Opus has fuzzy matching in 8 of 10, Sonnet in 10 of 20. Haiku 5.5 is rewarded 2 of 25 times and Fable 5 1 of 6 (not on the slide: six trials).
Where the method does show: GPT-6 Luna, the largest group. With fuzzy matching 38 of 47 rewarded (80%); without it 57 of 166 (34%). With cropping 59 of 79 against 36 of 134. Across all 850 trials: 166 of 224 with fuzzy matching against 270 of 626 without.
So: the strong models pass however they do it, the method separates the middle of the table, and it doesn't rescue the bottom. GLM has fuzzy matching in 32 of 49 trials and is rewarded 17 times; DeepSeek V4 Flash in 7 of 34 and never.
Fuzzy matching means the trial's code uses difflib, SequenceMatcher, rapidfuzz, Levenshtein or similar. It is a pattern match on the code the agent wrote: it says the code is there, not that it was the reason for the reward. This is correlation: within Luna, the trials that write this code may also be the ones run at higher effort.
OCR use: GPT-6.1 Sol 45 of 46 trials, Astra 73 of 80, Terra 52 of 55, Luna 157 of 213, GLM 49 of 52, DeepSeek V4 Flash 33 of 34.
DeepSeek V4 Flash can't read images (0 image reads recorded) and is rewarded 0 of 34 times. GPT-6 Luna reads images in 190 of 213 trials and is rewarded 95 times.
The Sol quote is an exact substring of masked reasoning (fast-agent 0.10.42, reward 1), checked against the recorded tool calls. That run also reads some frames as images, so don't call it OCR-only. Alignment, repair tables and re-crops are actions in the trace; that is not a claim about why the grader rewarded it.
Trials with an unknown reward are left out of each count; GLM is mixed versions.
Corrections to older rollups: Sol alone is 45 of 45 known plus one unknown; the earlier 95% was Sol and Terra together. The earlier 15% DeepSeek figure mixed versions; V4 Flash alone is 0 of 34.
Unfiltered fast-agent cohort, including short failures. Versions, effort settings and dates differ, so this is correlation.
Source: data/zork-methods.json (analysis/zork-methods/extract.py).
-->
