# Calibration Finding: Opus 4.6's T=0 Noise Floor Is ~20× Higher Than the Project's Original Bet

**Date:** 2026-04-11
**Author:** Claude (Opus 4.6)
**Status:** Decision required before continuing work
**Cost spent so far:** $18.35 of API budget (probe-set signal check $1.05 + first calibration $17.30)

---

## TL;DR

We built a system to detect silent changes to Claude Opus 4.6 by fingerprinting its `temperature=0` outputs. The first real calibration ran successfully — 600 API calls in 114 seconds, producing a complete `baseline_v1.json` — but it surfaced a finding that materially changes what the project can credibly claim.

**The key finding:** repeated `temperature=0` calls to Opus 4.6 are far more nondeterministic than the original design assumed. On long-form prose prompts, the same call sent twice produces outputs that differ by ~20–40% of their characters. This is not a bug; it's confirmed by reproducing the effect with three fresh independent calls, and it's how Anthropic's serving stack actually behaves.

The fingerprint methodology still works at coarse resolution, but it cannot deliver on the most interesting promise of the project: detecting *subtle* quantization. It can detect dramatic changes (model swaps, heavy requantization, big checkpoint updates), but the kind of silent quantization that an attacker or a distracted ops team could plausibly slip past us is also subtle enough to hide inside the noise floor.

You need to decide between three paths before more API money is spent. My recommendation is **Path A** (ship the honest version), but the framing trade-off matters and is yours to make.

---

## 1. What this project is

A public, continuously-running watchdog website. Every day, a GitHub Actions cron calls the Anthropic API and runs 30 fixed prompts against `claude-opus-4-6` at `temperature=0`. We measure how much each prompt's output has drifted from a one-time pinned baseline, aggregate the result, and publish a green/yellow/red verdict on a static site at `<username>.github.io/quantization-canary` with per-prompt drill-down evidence.

The motivating question, as the visitor sees it: **has Opus 4.6 changed?**

The implicit promise was that we could catch quantization changes (Anthropic silently swapping in a more aggressively quantized version of the model), checkpoint swaps (a different model under the same ID), and server-side moderation/system-prompt rewrites. We were already explicit in the design that the three categories are not separable from black-box API access — we only commit to a single "something changed" verdict with drill-down evidence. That design choice is fine and stands. The new finding is about *how subtle* a change we can credibly detect.

---

## 2. The methodology and the assumption that broke

### How fingerprinting was supposed to work

For each of 30 hand-curated prompts, calibration calls Opus 4.6 **20 times at temperature=0**. From those 20 samples it picks the **medoid** — the sample whose mean character-level Levenshtein distance to the other 19 is smallest — and saves that medoid text as the "reference output" for that prompt. It then measures the **noise floor** for the prompt:

- `mu_i` = mean of the 20 distances from each sample to the medoid
- `sigma_i` = stddev of those same 20 distances

On a daily tick, the worker calls each prompt 3 times, computes the median distance from those new samples to the pinned reference, and computes a per-prompt z-score against `(mu_i, sigma_i)`. The aggregate over 30 prompts is the daily verdict.

### The assumption we made

The original probe-set design rested on the belief that `temperature=0` on a frontier model is *mostly* deterministic, and the only sample-to-sample drift comes from server-side batching and floating-point kernel nondeterminism. Under that belief, `mu_i` should be roughly **0.005–0.01** (0.5–1.0% of characters) for almost any prompt. The plan baked in a stability rule: any prompt with `mu_i > 0.05` is rejected and rewritten, on the theory that such cases are rare and indicate a poorly-chosen prompt.

The hypothesis behind picking long-form prose prompts was that they have many "near-margin token decisions" (places where top-1 and top-2 logits are close), giving the metric more chances to see a flip when the model genuinely changes. This was the right hypothesis for *sensitivity to drift*, but it interacts very badly with the assumption above.

### What actually happened

Of the 30 prompts, **23 violated the `mu > 0.05` stability rule.** The mean noise floor across all prompts is ~0.20 — meaning, on a typical prompt, repeated T=0 calls differ from the medoid by about 20% of their characters on average. The plan's stability rule was effectively impossible to satisfy on long-form prose prompts.

It is not a bug in the metric, the calibration loop, or the medoid computation. I confirmed this independently by making three additional fresh T=0 API calls to the noisiest prompt and observing the same effect.

---

## 3. Concrete numbers from the calibration run

### Run summary

| Metric | Value |
|---|---|
| Prompts calibrated | 30 |
| Samples per prompt | 20 |
| Total API calls | 600 |
| Wall time (50-worker pool) | 113.7 s |
| Total input tokens | 41,400 |
| Total output tokens | 222,336 |
| Total cost | **$17.30** (Opus 4.6 at $15 / $75 per M input/output) |
| Empirical yellow_z threshold | 0.2427 |
| Empirical red_z threshold | 0.3539 |
| CUSUM target_D | 0.1718 |
| CUSUM k | 0.0163 |
| CUSUM h | 0.1630 |

### Per-prompt noise floor (sorted by `mu`, ascending)

| ID | Category | mu | sigma | ref tokens | violates `mu>0.05`? |
|---|---|---|---|---|---|
| p030 | echo | 0.0000 | 0.0000 | 69 | clean |
| p026 | structured | 0.0130 | 0.0191 | 276 | clean |
| p020 | code | 0.0150 | 0.0337 | 500 | clean |
| p018 | code | 0.0231 | 0.0226 | 500 | clean |
| p019 | code | 0.0316 | 0.0761 | 500 | clean |
| p014 | reasoning | 0.0386 | 0.0801 | 370 | clean |
| p027 | structured | 0.0493 | 0.0748 | 500 | clean |
| p012 | reasoning | 0.0508 | 0.0904 | 370 | violates |
| p017 | code | 0.0531 | 0.0964 | 500 | violates |
| p028 | structured | 0.0578 | 0.1165 | 276 | violates |
| p011 | reasoning | 0.0809 | 0.1143 | 312 | violates |
| p009 | reasoning | 0.0887 | 0.1081 | 500 | violates |
| p015 | code | 0.1004 | 0.1286 | 500 | violates |
| p010 | reasoning | 0.1130 | 0.0830 | 391 | violates |
| p016 | code | 0.1654 | 0.1371 | 500 | violates |
| p001 | exposition | 0.1727 | 0.1968 | 406 | violates |
| p013 | reasoning | 0.1924 | 0.1585 | 500 | violates |
| p007 | exposition | 0.2103 | 0.1364 | 331 | violates |
| p025 | structured | 0.2255 | 0.1693 | 368 | violates |
| p002 | exposition | 0.2523 | 0.2315 | 330 | violates |
| p008 | exposition | 0.2653 | 0.1772 | 353 | violates |
| p006 | exposition | 0.2779 | 0.1848 | 340 | violates |
| p003 | exposition | 0.2809 | 0.2039 | 333 | violates |
| p023 | creative | 0.2820 | 0.2532 | 201 | violates |
| p024 | creative | 0.2882 | 0.1874 | 304 | violates |
| p029 | echo | 0.3017 | 0.2857 | 266 | violates |
| p004 | exposition | 0.3417 | 0.2458 | 364 | violates |
| p005 | exposition | 0.3565 | 0.2869 | 353 | violates |
| p022 | creative | 0.4111 | 0.2549 | 291 | violates |
| p021 | creative | 0.4133 | 0.2004 | 260 | violates |

**Stable count: 7 of 30. Violating count: 23 of 30.**

### Independent verification

To rule out a metric/code bug, I made three fresh `temperature=0` calls to the noisiest exposition prompt (p005, "Why is the sky blue?") *after* calibration finished, in parallel via three concurrent threads, and measured pairwise distances:

```
sample lengths: 1556, 1548, 1481
sample 0 hash: f3055c4900   first 40 chars: "# Why Is the Sky Blue?\n\nYou probably alr"
sample 1 hash: ebd320ed21   first 40 chars: "# Why Is the Sky Blue?\n\nYou probably alr"
sample 2 hash: 1907d4965f   first 40 chars: "# Why Is the Sky Blue?\n\nYou probably alr"

d(s0, s1) = 0.4737
d(s0, s2) = 0.4737
d(s1, s2) = 0.2603
```

All three samples open with the same heading and the same opening sentence, then diverge significantly. A representative diff between samples 0 and 1 (first dozen lines):

```diff
-You probably already know that sunlight looks white but is actually a mix
- of all the visible wavelengths — from red (longer wavelengths) through
- orange, yellow, green, and blue, all the way to violet (shorter
- wavelengths). So the real question is: what happens to that mix of
- wavelengths when sunlight enters our atmosphere?
+You probably already know that sunlight looks white but is actually a mix
+ of all the visible wavelengths — from red (longer wavelengths) through
+ orange, yellow, green, and blue, all the way to violet (shorter
+ wavelengths). So the real question is: what happens to that mix of
+ wavelengths as sunlight passes through our atmosphere?

-The atmosphere is filled with tiny molecules of nitrogen and oxygen that
- are far smaller than the wavelengths of visible light. When a beam of
- sunlight hits one of these molecules, the light can be **scattered**
- — redirected in a random direction. Here's the key: shorter wavelengths
- are scattered *much* more efficiently than longer ones. This process,
- called **Rayleigh scattering**, affects blue and violet light roughly
- 5–10 times more than red light.
+The atmosphere is made up of tiny molecules, mostly nitrogen and oxygen,
+ that are far smaller than the wavelengths of visible light. When a beam
+ of sunlight hits one of these molecules, the light can be absorbed and
+ quickly re-emitted in a random direction — a process called
+ **Rayleigh scattering**. Here's the key: this scattering doesn't treat
+ all wavelengths equally. Shorter wavelengths are scattered *much* more
+ strongly than longer ones. In fact, the intensity of scattering goes
+ roughly as the inverse fourth power of the wavelength, meaning that
+ blue light (~450 nm) is scattered nearly ten times more than red light
+ (~700 nm).
```

The samples are essentially the same essay, told three different ways. They share factual content and rough structure, but every paragraph is paraphrased independently. This is how Opus 4.6 actually responds to long-form `temperature=0` prompts.

### The pattern in the noise floor

The noise floor is sharply correlated with prompt category and output structure:

- **Echo, simple-format code, lists**: very stable (`mu` ≈ 0.00–0.05). These prompts produce outputs whose top-1 logits are dominant at every position — there is one obvious "right" structure and the model stays in it.
- **Code with non-trivial logic** (parsing, recursion): moderately stable (`mu` ≈ 0.05–0.20). The model picks one implementation strategy and sticks with it, but minor stylistic choices (variable names, comment placement, error handling) drift.
- **Long-form prose, creative writing, reasoning chains** (`mu` ≈ 0.15–0.45). Every sentence has a near-margin choice between multiple equally-fluent phrasings, and Opus 4.6's serving stack apparently does not produce the same choice across calls.

The exact mechanism for this nondeterminism is opaque to us, but the most likely cause is some combination of:
- Mixed-precision floating-point arithmetic on GPUs
- Variable batch sizes producing different attention layouts
- Speculative decoding that accepts/rejects draft tokens nondeterministically
- Possibly server-side load-based routing to different replicas with subtle differences

We can't fix it, we can only measure it and decide how to live with it.

### Why the stability rule was wrong about *which prompts* it would catch

I designed the probe set with the explicit hypothesis that long outputs with many near-margin decisions would be the most signal-rich. **They are.** They are also, for exactly the same reason, the most noise-rich. The hypothesis "near-margin decisions = high sensitivity to drift" is correct, but I missed the implication "near-margin decisions = also the things that flip on every call from sheer T=0 jitter." The most informative probes are also the most contaminated by intrinsic noise.

The 7 stable prompts that survived the rule are precisely the ones I would have expected to be **least** sensitive to subtle weight changes — short, structured, with one obvious answer. The probe set is essentially binary: either a prompt has high signal *and* high noise, or it has low signal *and* low noise. There is no middle ground.

---

## 4. What this means for the project

### What still works

1. **The empirical thresholds adapted automatically.** Because the calibration script computes `yellow_z` and `red_z` from percentiles of the actual calibration distribution (not from hardcoded values), the noisy baseline still produces usable detection thresholds. A "stable" day has aggregate `Z` near 0; the system flags drift relative to whatever the noise floor turned out to be.

2. **Coarse drift detection works.** Doing the math: a stable day has aggregate `D ≈ 0.17`. To trigger the red verdict via the abrupt detector, daily `D` would need to rise to roughly `0.27`. CUSUM would catch persistent shifts of about half that magnitude over 2–3 days. We can detect changes that move the average distance by ≥0.10 absolute. That is enough to catch:
   - A model swap (different checkpoint behind the same ID)
   - Heavy requantization (e.g., FP16 → INT4)
   - Major fine-tuning rounds
   - Server-side moderation rewrites that change refusal patterns

3. **The 7 stable prompts give crisp byte-level signal.** They could anchor a more sensitive sub-detector if we built one.

4. **Cost and reliability are fine.** $17.30 calibration was a one-time cost. Daily cost is still ~$2. The pipeline is parallelized to 50 workers, the worker tick takes about 30 seconds wall time, GitHub Actions cron will be reliable.

### What does NOT work as designed

1. **Subtle drift detection is dead in the water.** Anything that adds less than ~0.05 to the average distance is invisible — it's well inside the noise floor. The most plausible adversarial scenario for this project — Anthropic shipping a quantization update small enough to hope nobody notices — is also small enough to hide here.

2. **The drill-down "evidence" pane loses much of its rhetorical force.** The site's whole trust argument was *"don't trust our stats — look at the literal before/after diff."* But on stable days, Opus 4.6 already produces noticeably different output across calls. A visitor who clicks into the evidence pane will see dramatic-looking diffs even when nothing has changed. The "evidence" becomes ambiguous instead of damning.

3. **The original framing is misleading if we ship it as-is.** Calling the site "quantization canary" implies it can detect quantization. It cannot detect subtle quantization. It can detect *changes* of various kinds, conflated together, above a threshold that's set by Anthropic's serving stack rather than by us.

---

## 5. Three paths forward

### Path A — Ship the honest version (recommended)

Accept the noise floor. Update the methodology page on the site to be explicit about what we can and cannot detect. Land the calibration we already paid for, run one real daily tick (~$2), wire up GitHub Actions, ship the site, and let it run for a couple of weeks of real production data before deciding whether to iterate.

**Honest framing the site would make:** "This site detects when Opus 4.6 changes substantially — model swaps, heavy quantization, large fine-tuning passes. It cannot detect changes subtle enough to hide inside Opus 4.6's intrinsic temperature=0 nondeterminism, which on long-form prose is around ±20% of output characters per call. We chose long prompts on purpose: they're the most sensitive to drift, but also the noisiest. The project's value is converting folk anecdotes ('feels dumber this week') into systematic, time-stamped, publicly-auditable measurements — not catching every possible silent change."

**Costs:**
- API: ~$2 for one real daily tick today, then ~$2/day ongoing
- Time: 1–2 hours of work to wire Actions + Pages + ship
- No re-calibration needed

**Risks:**
- The site is less impressive than originally pitched
- People may push back on the framing if we're not very loud about the limitations
- We might miss the exact failure mode the project was named after

**Upsides:**
- The system runs immediately, generates real data, and informs whatever we build next
- Honest products age well; overclaiming doesn't
- We learn from production data what the actual behavior looks like
- The 7 stable prompts can be the seed of a Path C upgrade later, after we have data

### Path B — Rebuild the probe set around stable prompts only

Drop everything with `mu > 0.05` and add ~23 new prompts that we hope will calibrate clean. Lose categorical diversity (no more long prose, no more creative writing, no more multi-paragraph reasoning) but gain crisp drift signal. Recalibrate. Ship.

**Costs:**
- ~2–3 hours of prompt design + cross-model signal check (~$1) + recalibration (~$17)
- The recalibration is itself a gamble: many of our "candidate stable" prompts may also turn out noisy
- Potential additional rounds if calibration #2 still has many violations

**Risks:**
- We may replace the unstable prompts and find their replacements are *also* unstable, because the noise is intrinsic to long-output T=0 generation, not to specific prompt choices
- The new probe set would be heavily skewed toward short, structured outputs (code, lists, echoes) — losing exactly the kinds of prompts where subtle quantization would manifest most
- Counterintuitively, picking only stable prompts may make us *less* sensitive to subtle drift, not more, because stable prompts have huge top-1 margins that even significant quantization wouldn't flip

**Upsides:**
- A passing calibration that doesn't violate the original stability rule
- The rhetorical "evidence" pane becomes compelling again — diffs are real diffs, not noise
- The site can credibly claim "byte-level fingerprinting" again

### Path C — Two-channel hybrid

Keep the noisy long-form prompts as a "bulk drift" channel (low resolution but covers diverse capabilities) AND build a smaller stable-prompt channel as a "fine drift" detector. Two verdicts displayed side by side, each clearly scoped. The site shows "fine: stable / bulk: stable" as the happy path.

**Costs:**
- Most expensive of the three: we'd recalibrate both channels separately
- Methodology page becomes more complex
- ~3–4 hours of additional work + ~$30 of API spend

**Risks:**
- More complex methodology means more places for visitors to be confused
- Still doesn't solve the fundamental subtle-quantization problem on the bulk channel
- Two verdicts mean we have to communicate when they disagree, which is the most informative case but also the hardest to interpret

**Upsides:**
- Most complete coverage
- Each channel's claims are crisp on its own scale
- Most defensible against criticism

---

## 6. My recommendation

**Path A.** Three reasons:

1. **We have real data to work with already.** Path B and C both involve more guessing about what will calibrate clean, with no guarantee they will. Path A gets a working system into production immediately, where it can teach us things we can't learn from any amount of upfront re-design.

2. **The honest version is itself valuable.** "This site detects model swaps and heavy quantization, but not subtle quantization, on Claude Opus 4.6, with documented methodology and a public dataset" is a real and useful product. It's a smaller claim than the original pitch, but it's a true claim, and there is currently nothing else like it on the public internet.

3. **The 7 stable prompts already give us a Path C foothold for free.** If we ship Path A and run for 2–3 weeks, we will have real production data showing how those 7 prompts behave day to day — and we can decide whether to invest in Path C from a position of empirical knowledge instead of speculation.

The thing that pushes me away from Path B specifically is that I'm fairly confident a fresh set of "stable-looking" prompts would partially fail calibration too. The stability/sensitivity trade-off appears intrinsic to the API's behavior, not to my prompt picks. Spending another $17 to rediscover this would be the most expensive way to learn it.

If you reject Path A, I'd lean toward Path C over Path B, because Path C at least preserves the diverse-prompt coverage that Path B throws away.

---

## 7. What I need from you

A one-sentence answer:

- **"A"** → I run one real daily tick today, wire up GitHub Actions + Pages, push the public site, total additional spend ~$2, the site is live within an hour. The methodology page will be updated to be explicit about limitations.
- **"B"** → I don't run another tick. We discuss which prompts to drop, I draft replacements, we cross-model signal-check them, then I recalibrate. Estimated additional spend $18–35, estimated time ~3 hours of back-and-forth.
- **"C"** → Same as B but with two channels. Estimated additional spend $30+, more complex shipping.
- **"Something else"** → Tell me what.

---

## Appendix A — Costs to date

| Item | Cost |
|---|---|
| Probe-set cross-model signal check (Opus + Sonnet, 30 prompts) | $1.02 |
| p029 replacement re-verification | $0.03 |
| First real calibration (30 probes × 20 samples) | $17.30 |
| **Total** | **$18.35** |
| Daily budget ceiling (post-launch) | ~$2/day |

## Appendix B — Files written by this calibration

| File | Purpose |
|---|---|
| `baseline/baseline_v1.json` | The baseline itself: per-prompt medoid reference, mu, sigma, plus aggregate thresholds and CUSUM params |
| `logs/calibration-20260411-223959.log` | Full per-call log of the calibration run, with timestamps and token counts |

The baseline file is already on disk, ready for the worker to use. None of it has been committed to git yet.

## Appendix C — How to verify these findings yourself

```bash
# Re-read the calibration log
tail -100 logs/calibration-20260411-223959.log

# Inspect the baseline file directly
python -c "
import json
b = json.load(open('baseline/baseline_v1.json'))
for pid, p in sorted(b['prompts'].items(), key=lambda kv: kv[1]['mu']):
    print(f\"{pid} ({p.get('reference_output_tokens','?')} tok)  mu={p['mu']:.4f}  sigma={p['sigma']:.4f}\")
"

# Reproduce the noise on any single prompt (~$0.03 per run)
uv run python -c "
import anthropic
from src.env import load_env
from src.metrics import normalized_char_edit_distance
load_env()
client = anthropic.Anthropic()
prompt = 'YOUR PROMPT HERE'
samples = []
for i in range(3):
    r = client.messages.create(
        model='claude-opus-4-6', max_tokens=500, temperature=0,
        messages=[{'role': 'user', 'content': prompt}],
    )
    samples.append(r.content[0].text)
print(f'd(0,1)={normalized_char_edit_distance(samples[0], samples[1]):.4f}')
print(f'd(0,2)={normalized_char_edit_distance(samples[0], samples[2]):.4f}')
print(f'd(1,2)={normalized_char_edit_distance(samples[1], samples[2]):.4f}')
"
```

## Appendix D — What I will NOT do without your decision

- I will NOT run another daily tick (would cost $2 for nothing if you pick B or C)
- I will NOT commit anything to git
- I will NOT push to the GitHub remote
- I will NOT change the repository's visibility
- I will NOT enable GitHub Pages
- I will NOT set the GitHub Actions secret
- I will NOT trigger any GitHub Actions workflow
- I will NOT modify the probe set
- I will NOT recalibrate

The current state of the workspace is:
- Calibration finished, baseline written to `baseline/baseline_v1.json` (uncommitted)
- All source code, tests, probes, and the static site exist locally and are unchanged
- 54 unit tests still pass
- `docs/data.json` still contains the synthetic mock data from the earlier mocked end-to-end run, not real data
- The GitHub repo is still private with only the initial commit
