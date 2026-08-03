---
layout: post
title: "RL Teaches a Model When to Reason, Not How — and You Can Steer It Back In"
date: 2026-08-02
image: /assets/img/card-rl-when.png
---

*Part of [rl-reasoning-diff](https://github.com/yashsawant22/rl-reasoning-diff) — diffing a reasoning model across its RL step with a crosscoder.*

Here is the whole result in one figure. I took OLMo-2-1B **before** its final RL step — a checkpoint
that, handed a plain prompt, just tells a little story. I fed it one, then ran it again with a single
direction added mid-generation into its residual stream — a direction I read straight out of the
changes RL *would later* make to this model:

![Flipping reasoning on in a model that never learned it: the same pre-RL checkpoint on the plain prompt "After the rain stopped, the children went outside" continues with a story when steering is off (reasoning rate 8%), and invents a math word problem when a direction read from RL's changes is added in (reasoning rate 92%).](/assets/img/rl-when-hero.svg)

Same weights, same prompt. With the direction off, the children play. With it on, they stop playing
and start starring in a GSM8K word problem.

And it isn't a fluke of one prompt. Point the same steered model at other plain sentences and the
continuation pivots the same way — into rates, counting, and posed questions (reasoning text
highlighted, same as above):

![The same steering on the plain prompt "The train arrived at the station a few minutes late": with steering off the model continues a scene about passengers boarding; with steering on it turns into a rate problem — "the total time spent on the journey is 5 hours. Let's calculate the total number of hours…" (highlighted).](/assets/img/rl-when-train.svg)

![The same steering on the plain prompt "She poured a cup of coffee and looked out the window": with steering off the model continues an introspective scene; with steering on it becomes a counting word problem — "She had 5 cups of coffee and 2 cups of coffee. How many total cups of coffee did she have?" (highlighted).](/assets/img/rl-when-coffee.svg)

It's not always this tidy — push the strength too high and some prompts just repeat or break down —
but in the coherent range, adding this one direction reliably tips the pre-RL model toward counting,
enumerating, and posing questions. (The full dose-response, with controls, is [below](#the-aggregate-is-causal).)

That direction is the punchline of a longer question: **when RL turns a language model into a
reasoning model, what does it actually change inside?** Does it *install new reasoning machinery* —
new circuits the base model didn't have? Or does it just change *when* the model reaches for
machinery it already had? Those two stories look identical from the outside — both give you a model
that reasons — but they are completely different claims about what RL is.

This post is the second, mechanistic answer. The short version: RL's change is a **faint, diffuse
nudge** — no single feature moves more than ~6%, and it's smeared across thousands of features — but
the features it nudges are **overwhelmingly reasoning features**. Add that set back into the pre-RL
model and it starts doing math. **RL edits _when_ the model reasons, not _how_.**

## Why this is even a question

The framing isn't mine. A recent paper — ["Base models know *how*, RL learns
*when*"](https://arxiv.org/abs/2510.07364) — argues that RL post-training doesn't teach new
capabilities so much as it teaches the model to *deploy* capabilities it already has, on the right
inputs. It's a clean, falsifiable claim, and it has a mechanistic prediction hiding in it: if RL
only changes *when*, then the reasoning machinery should already exist in the pre-RL model, and RL's
edit should look like a re-weighting of existing features rather than the birth of new ones.

OLMo-2-1B is the perfect subject because AI2 released every post-training checkpoint: `base → SFT →
DPO → RLVR1`. That last hop, `DPO → RLVR1`, is *just* the RL step (RLVR = RL with verifiable
rewards, GRPO-style). So I can diff the model against itself across exactly one intervention and ask
what moved.

## First look: RL barely touches the weights, but aims what it touches

Before any fancy interpretability, the crudest possible diff — subtract the weight tensors — already
says something surprising.

The `DPO → RLVR1` weight change has a relative norm of about **0.002**. For comparison, `base →
RLVR1` is about **1.1**. RL is a roughly **500× smaller** weight edit than everything that came
before it. That alone is on-thesis: whatever RL does, it isn't rebuilding the model.

It's actually *so* small I couldn't measure its structure. The DPO and RLVR1 checkpoints are stored
in bf16, whose relative precision is ~0.4% — and RL's weight change (0.2%) sits *below* that floor.
The proof is a null experiment I almost didn't run: take the fp32 base model, round it to bf16 and
back — changing *nothing* — and the resulting "delta" has the same rank and the same relative norm
as the real RL delta. So the weight-space rank of the RL hop is simply **unmeasurable** from the
released checkpoints. (The bigger hops, `base→SFT` and `SFT→DPO`, sit well above the floor and stay
measurable — it's specifically RL that's too quiet to read in weight space.)

That's the methodological wall that pushed everything else in this project down to the **activation**
level, where there's real signal. And the signal points the right way: RL's per-token change to the
residual stream is **2.3× larger on reasoning text than on plain text** (6.0% vs 2.6%, controlling
for activation scale). A tiny edit — but aimed at reasoning.

"Aimed at reasoning," though, is a blunt instrument. *Which* reasoning? To answer that I needed to
turn the activation change into something with names. That's the crosscoder.

## The crosscoder, from scratch

A regular sparse autoencoder (SAE) takes one model's activations and learns an overcomplete
dictionary of features: a big, mostly-zero code where each active entry is a human-interpretable
concept. A **crosscoder** is the two-model version, and it's built for exactly the diffing question I
have.

The idea is one shared feature dictionary, but a **separate decoder per model**:

- One shared encoder reads both checkpoints' activations and produces one sparse feature vector `f`.
- Two decoders — `decoder₀` reconstructs the DPO stream from `f`, `decoder₁` reconstructs the RLVR1
  stream from the *same* `f`.

Because the features are shared but the decoders aren't, the **norm of each feature's decoder
column** tells you how much each model relies on that feature. Compare the two norms and you get,
per feature, a number in [0, 1]:

![A crosscoder diagram: DPO and RLVR1 layer-12 activations both feed one shared TopK feature dictionary of 8,192 latents, which is then reconstructed by two separate decoders — one per model. The ratio of a feature's two decoder norms (rel-norm) reads out which model leans on it more.](/assets/img/rl-crosscoder.svg)

```python
def relative_norms(self):
    # ‖dec_rlvr1‖ / (‖dec_dpo‖ + ‖dec_rlvr1‖), per feature.
    # ~0  → DPO-only feature   ~0.5 → shared   ~1 → RLVR1-only ("RL turned this up")
    n = self.decoder_norms()          # (2, d_hidden)
    return n[1] / (n[0] + n[1] + 1e-9)
```

A feature with `rel-norm ≈ 1` is one RL leans on far more than DPO did — the *when-to-reason*
candidates. `rel-norm ≈ 0.5` means both models use it equally. (One detail that matters: the
checkpoint is trained with TopK sparsity rather than an L1 penalty, which pins L0 at exactly `k` and
removes a decoder-norm gauge ambiguity the L1 version can't escape — so these norm ratios are
actually comparable.)

I didn't train the crosscoder in this repo — it's borrowed from a companion run and loaded from the
Hub. But I re-collected the activations independently and re-verified it before trusting a single
number: on my own activations it explains **87%** of the variance, realizes L0 = 64 exactly, and has
only 5.5% dead features. It's a faithful reconstruction, not a decorative one.

## What RL actually turned up

Now the examples. This is the part I find genuinely uncanny, so I'll just show you the features.

Sort all 8,192 features by rel-norm and look at the tail RL leaned on. Here are actual top features,
each with the real tokens it fires hardest on (`«token»` marks the firing position):

**Feature 3437** — "the model is at the end of a `Let's think step by step.`"
```text
average? Let's think step by step«.»
   ones? Let's think step by step«.»
   total? Let's think step by step«.»
```

**Feature 6884** — "a quantity question just ended; a reasoning cue is coming"
```text
How many shirts do they have in total«?» Let's think
  the total of college courses they both attended«?» Let's think
  tables did the carpenter make in total«?» Let's think
```

**Feature 8032** — "a word problem is stating amounts of money"
```text
20 less, she would have $80«.» If Jolly
 cousin Rene, and Rene received $300«,» how much money
's. If Niraj contributed $80«,» how much did
```

**Feature 197** — "a discount / percentage-off setup"
```text
A department store offers a 10%« discount» for the amount
 offering the entire set at 15%« off».  If
 when a certain website offers 30%« off» her entire purchase
```

These aren't cherry-picked oddities — they're what the RL-leaning tail *is*. Features for
`Let's think step by step`, for "how many … in total?", for stating dollar amounts, for
discount-and-percentage setups. RL didn't turn up a "multiply two numbers" circuit. It turned up the
features that recognize **"this is a moment to start reasoning."**

The aggregate confirms the eyeball read. Across the corpus, the baseline fraction of tokens that are
reasoning tokens is **0.29**. For the RL-up-weighted tail of features, the fraction of their firing
that lands on reasoning tokens is **0.82** — 2.8× the base rate. The features RL leans on are, by a
wide margin, the ones that fire when reasoning is about to happen.

Here's the twist I didn't expect, and it only makes sense to raise it now that you've seen the
features: you'd assume that if RL up-weighted reasoning features, there'd be a *discrete* set of them
— a handful of "RL features" you could point to. There isn't. The spread of the per-feature edit
across all 8,192 features has a standard deviation of **0.001**. There is no clean model-specific
feature; the entire `DPO → RLVR1` difference is **diffuse**, smeared thinly across thousands of
features that each barely move. The largest single-feature change RL made is +6.5%, and almost
everything moved less than 2%.

So the correlational story is: RL's edit is faint and diffuse, but it's diffuse *in the direction of
reasoning features*. Which sets up the only test that actually settles anything.

## Correlation isn't causation — so steer

Everything above is "these features moved, and they happen to be reasoning features." That's a
correlation. The whole point of interpretability is that you can do the intervention: take the
directions and *add them back*, into the model that hasn't had RL yet, and see if reasoning turns on.
If it does, the correlation was causal. If it doesn't, I was reading tea leaves.

Two experiments, and the first one **fails** — which turns out to be the more informative half.

### One feature is not a switch

The obvious first move: take the single most RL-up-weighted reasoning feature (6884, the
`How many…? Let's think` one), clamp it high in the pre-RL model on plain prompts, and watch
reasoning appear.

It doesn't. Across the whole coherent dose range, the reasoning rate stays flat at ~0 — a faint 0.17
right at the coefficient where the model starts to break down, versus 0.00 for both an equal-norm
random control and a shared-feature control. Here's a steered generation at a middle dose; it's just
a slightly different mundane story, no math:

```text
prompt   After the rain stopped, the children went outside
  +6884  to play. They found a small pond and decided to have a race.
         The race was between two children, and they...
```

At first this looks like a refutation. It's the opposite: it's exactly what a **diffuse** edit
predicts. If RL's change lives in one feature, clamping that feature should reproduce it. It doesn't,
because the change was never stored in any one place. No single feature carries the behavior — so to
test the thesis I have to add back the *whole distributed set at once.*

### The aggregate is causal

So I built the aggregate direction: sum the decoder columns of the RL-up-weighted reasoning features,
each weighted by how much RL leaned on it and how reasoning-selective it is. One vector, standing in
for the diffuse edit. Add it into the pre-RL model's residual stream on plain prompts, sweep the
strength, and measure the reasoning rate — with two controls and a reference:

![Dose-response curve: reasoning rate on plain prompts versus steering strength. The RL-up-weighted reasoning aggregate rises monotonically from 0.08 to 0.92 and stays coherent; a random direction and the RL-down-weighted set both stay flat on the floor; the raw reasoning-minus-plain direction peaks lower and degrades sooner.](/assets/img/rl-dose-response.svg)

| steering strength | RL reasoning set | random (ctrl) | RL-*down*-weighted (ctrl) | reasoning−plain (ref) |
|---|---|---|---|---|
| 0  | 0.08 | 0.08 | 0.08 | 0.08 |
| 10 | 0.33 | 0.00 | 0.08 | 0.17 |
| 15 | 0.50 | 0.00 | 0.00 | 0.58 |
| 20 | **0.83** | 0.00 | 0.00 | 0.50 |
| 25 | **0.92** | 0.00 | 0.00 | 0.42 |

The RL-up-weighted reasoning set drives a clean, monotonic dose-response from **0.08 → 0.92**, and
the generations stay coherent (100% coherent up to strength 20). The candy word problem from the top
of the post is one of these — a GSM8K-style question conjured on "After the rain stopped."

The controls are what make it mean something:

- **A random direction stays flat at 0.** So this isn't "any nudge at layer 12 makes the model
  ramble about numbers."
- **The RL-*down*-weighted reasoning aggregate also stays flat at 0** (and degrades coherence). This
  is the sharp one: it's still built from reasoning-ish features, but from the ones RL turned *down*.
  Adding those does nothing. The effect is specific to the features **RL up-weighted**, not to
  reasoning directions in general.
- It even beats the naive baseline — the raw `reasoning − plain` difference-in-means direction —
  which peaks around 0.58 and falls apart at higher strength. RL's specific set is a *cleaner*
  reasoning knob than the obvious contrast you'd reach for.

That's the causal claim closed. The distributed set of features RL up-weights, added back into the
model *before* RL ran, reproduces reasoning-launch behavior — and neither noise nor the down-weighted
features do.

## What it means

Put the two halves together and the thesis lands with a mechanism attached:

- The reasoning machinery is **already in the pre-RL model** — you can drive it with a direction, no
  new circuits required.
- RL's contribution is to **turn up the features that decide *when* to deploy it** — the
  `Let's think step by step`, the "how many in total?", the "here's a dollar amount" detectors.
- And it does this **diffusely**: not one switch but a thin re-weighting across thousands of
  features, which is why one feature does nothing and the aggregate does everything.

The nice part is that the messiest early finding — "the edit is frustratingly diffuse, there's no
clean RL feature" — stopped being a caveat and became the result. Diffuseness is *why* the
single-feature steer fails and the aggregate succeeds. It's the shape of the answer, not noise in it.

Now the honest limits, because they're real. This is a modest single-machine run: 33k tokens, one
layer (12), WikiText-2 for the plain corpus, greedy decoding, and a **keyword-based** reasoning
detector rather than a learned classifier. "Reasoning" here means GSM8K-style math, so some of that
0.82 selectivity is genuinely topical — a "does it start doing arithmetic word problems" signal, not
a pure "does it reason" one; RLVR was *trained* on exactly that kind of math, so this is on-thesis
but not a clean reasoning-vs-not contrast. The steering is a single constant vector at one layer.
I'd expect all of these numbers to sharpen with scale — a learned detector, more layers, more prompts
— rather than reverse, but that's a prediction, not a result.

What I'm confident in is the qualitative shape, because I can *watch* it: a plain prompt, a model
that has never done RL, a direction built entirely from RL's own changes — and the children stop
playing and start counting candies. RL didn't teach this model to reason. It taught it when to
start.

*Code, every number, and the reproduction scripts are in
[rl-reasoning-diff](https://github.com/yashsawant22/rl-reasoning-diff); the crosscoder itself is on
the [Hub](https://huggingface.co/Savianto/olmo2-1b-rl-crosscoder).*
