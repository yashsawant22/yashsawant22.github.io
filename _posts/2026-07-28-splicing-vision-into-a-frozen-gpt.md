---
layout: post
title: "Splicing Vision Into a Frozen GPT: Building a Minimal VLM From Scratch"
date: 2026-07-28
---

*Part of [embodied-gpt](https://github.com/yashsawant22/embodied-gpt) — Phase 2.*

By the end of the last post, CLIP could do something genuinely satisfying: hand it 8 real test
images and 8 real captions, and it matches all 8 correctly. Two independent towers, aligned into
one shared space, no cheating. But retrieval isn't generation. CLIP can tell you *which* caption
fits an image out of a fixed pool — it can't write you a new one. My GPT, meanwhile, writes fluent
sentences all day, but has never seen a pixel in its life. The obvious next question: can I get the
GPT to actually *describe* an image, using CLIP as the eyes?

![Two frozen models — a CLIP vision encoder and GPT-2 — joined by a single trained linear projection that maps CLIP's image vectors into GPT's embedding space, producing the caption "A dog is running through the woods."](/assets/img/vlm-hero.svg)

## The problem

The last post spent a whole section on why you can't just wire two independently-trained models
together — a vector only means something relative to the space it was shaped in, and CLIP's image
space and GPT's token-embedding space were shaped by two completely different objectives. CLIP
*fixed* exactly one version of that problem: it aligned an image tower and a text tower into a
*shared* space, purpose-built for cosine similarity comparisons between the two.

Here's the part I didn't expect going in: aligning two spaces for *comparison* doesn't make either
one *usable as input* to some third, unrelated model. My GPT was trained entirely on its own
vocabulary's token embeddings — it has never seen a CLIP vector, aligned or not, and there's no
reason its attention layers would know what to do with one dropped in raw. CLIP solved
"do these two things mean the same thing," not "can a frozen pretrained model read this." That's a
second, separate translation problem, and it's the actual subject of this post: how do you get a
vector from *any* space to be legible to a specific, already-trained, otherwise-untouched model?

## Historical context

The state of the field before this, once again a bespoke-architecture-per-task story: image
captioning meant a CNN encoder feeding an RNN or transformer decoder, trained end-to-end from
scratch on paired (image, caption) data. Every new captioning system was its own model, no reuse.
Meanwhile GPT-2/3 had already shown, in text, that one frozen pretrained model could be repurposed
across many tasks just by changing what you feed it.

**Frozen** (Tsimpoukelli et al., DeepMind, 2021 — "Multimodal Few-Shot Learning with Frozen
Language Models") asked whether that idea could extend to vision: keep a pretrained LLM's weights
completely untouched, and train *only* a vision encoder whose job is to map an image into a
handful of vectors that live in the LLM's existing embedding space — prepended in front of the text
prompt as if they were extra tokens. Nothing about the LLM changes. The vision encoder is trained
using the frozen LLM's own next-token loss as the only signal, and the LLM's language ability
transfers to images it was never trained on — because as far as its attention layers can tell,
those vectors are just more tokens.

**LLaVA** (Liu et al., 2023 — "Visual Instruction Tuning") is the version of this idea that
actually matches what I built: instead of training a vision encoder from scratch against that
weak, indirect signal, start from **CLIP's already-pretrained encoder** (frozen), and train only a
lightweight projection layer to translate its output into the LLM's space. I asked, at the time,
why you couldn't just skip CLIP and train the vision encoder from scratch the way Frozen did —
worth stating the actual answer here, because it's the reason this post starts from last post's
CLIP checkpoint instead of a blank vision encoder:

- **Signal density.** CLIP's contrastive loss gives every image a batch's worth of hard negatives
  to be discriminated against — a dense, forceful "what specifically makes this different from
  everything else here" pressure. Caption-prediction loss gives a much weaker, indirect signal:
  the gradient has to travel through "did the right *word* get predicted" to say anything about
  what's actually in the image.
- **Data scale.** CLIP: 400 million pairs (or, in my case, whatever signal 6,000 Flickr8k images
  can give a contrastive objective — still denser than the alternative). LLaVA's own alignment
  stage: ~558K captions. Not remotely enough to discover "vision" from a random initialization
  using only the weak signal above.
- **Bootstrapping.** A random-init vision encoder outputs meaningless vectors. The projection layer
  has nothing coherent to map, the LLM has no reason to attend to any of it, and no useful gradient
  reaches either one. Starting from CLIP means the projection layer's job is well-posed from step
  one: translate between two spaces that *already* mean something, instead of discovering meaning
  and translating it at the same time.

## Intuition: it's the same trick, one level up

`patch_embedding.py`, all the way back in Phase 1, was a linear projection of raw pixels into a
space the ViT's attention blocks already knew how to operate over. This is that exact move, run
twice: pixels → (frozen CLIP ViT) → CLIP's space → (**trained** projection layer) → GPT's space →
(frozen GPT) attends over it like any other token.

## A doubt worth naming: pooled vector, or the whole grid?

My first guess at the projection layer's shape was `nn.Linear(d_clip, d_gpt)` applied to something
shaped `(B, N, d_clip)` — right on the mechanism, but I hadn't checked it against my own code yet.
`ImageEncoder.forward()` (the one CLIP training actually uses) returns a single pooled vector per
image, `(B, proj_dim)` — exactly right for a contrastive loss that needs one embedding per image to
compare against one embedding per caption. But that pooled `[CLS]` vector is a *summary*,
compressed specifically for whole-image-vs-whole-caption matching. Feeding the GPT one blurry
gestalt vector throws away everything spatial. What the GPT actually needs is the full grid of
patch tokens — `x[:, 1:]`, shape `(B, 64, 192)` for a 128px image cut into 16px patches — so it gets
64 individually-attendable image tokens, the same way it gets one embedding per word instead of one
embedding for the whole sentence. I added `forward_patch_tokens()` alongside the existing method
rather than changing it, specifically so the already-trained CLIP checkpoint didn't need
retraining just to expose a different slice of the same computation.

## Same loss as always — just a new question about *where* it applies

The training objective here isn't new machinery: it's the exact next-token cross-entropy loss GPT
was always trained with. What's new is *which* tokens count. The combined input sequence is
`[image tokens] + [caption tokens]`, and every position's logits try to predict the next position's
token — but image tokens aren't drawn from a vocabulary, so most of them have no discrete target to
be scored against at all. Only one image position matters for the loss: the very last one, whose
"next token" is the caption's real first word — the actual image → text handoff. From there it's
ordinary shifted-label language modeling through the end of the caption. (Full LLaVA has a second
stage — unfreeze the LLM too, train on instruction-style data with the *question* also masked out
of the loss, not just the image. I built Stage 1 only: frozen CLIP, frozen GPT, a trained
projection, captioning rather than instruction-following. More on why that's the right ceiling to
hit first in Experiment results.)

## Architecture

```
Image (128x128x3)
   |  frozen CLIP ViT (Phase 2, part 1) -- patch embed, [CLS]+pos embed, transformer blocks
   v
Patch grid (64, 192)                       <-- NOT the pooled [CLS] vector
   |  projection layer: nn.Linear(192, 768) -- the ONLY trained weights in this whole post
   v
Image embeddings (64, 768) ---concat--- Caption token embeddings (T, 768)   <- frozen GPT's own wte
   |
   v
frozen GPT-2 (12 layers, 768-dim, causal self-attention -- ordinary, no cross-attention module)
   |
   v
logits (64+T, 50257) -- loss counted only from the last image position onward
```

The image tokens go *first*, and that's not arbitrary: GPT's attention is causal, unlike the ViT's
bidirectional attention, so every text token that follows can attend back over the *entire* image,
while nothing needs to attend forward into it.

## Math

Combined sequence length `N + T` (`N` = 64 image tokens, `T` = caption length). Position `i`'s
logits predict the token at position `i+1`:

```
labels[i] = -100                              for i in [0, N-2]      (no vocab target for an image position)
labels[N-1] = caption_ids[0]                  (the actual handoff: last image token -> first word)
labels[N + k] = caption_ids[k+1]  if k+1 < length   for k in [0, T-2]  (ordinary next-token target)
labels[N + k] = -100                          otherwise               (falls into padding, ignored)

loss = CrossEntropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100)
```

`-100` is PyTorch's `ignore_index` convention — those positions contribute zero loss and zero
gradient, same trick as masking the prompt out of the loss in full instruction-tuning, just applied
to image positions and padding here instead. I worked this out on a concrete toy example before
trusting it on real data — `num_image_tokens=3`, caption `[10, 20, 30, <eot>, <pad>]`, `length=4` —
by hand:

```
combined:  [img0, img1, img2,  10,   20,   30, <eot>, <pad>]
labels:    [-100, -100,   10,  20,   30, <eot>,  -100,      ]   (8 positions total; last has no target)
```

![Label-masking for the toy example: the input sequence img0 img1 img2 10 20 30 <eot> <pad>, and below it the target each position predicts — the two image positions and the <eot>/<pad> positions are masked to -100 (zero loss), while img2 predicts the caption's first token 10 (the image-to-text handoff, highlighted), and 10/20/30 predict 20/30/<eot>.](/assets/img/vlm-label-masking.svg)

Writing that out caught the off-by-one before it could hide in a shape-correct-but-silently-wrong
tensor operation — the same category of bug as Phase 1's attention-scaling mistake, where the
shapes never once complained.

## Code walkthrough

Everything new lives in `src/transformers/gpt.py` and `src/multimodal/vlm.py` /
`train_vlm.py` / `dataset.py`.

**`src/transformers/gpt.py` — the first shared building block the repo's `src/transformers/`
folder has ever actually held.** A hand-rolled, GPT-2-small-shaped decoder (causal attention +
pre-norm block, same pattern as `text_encoder.py`'s causal blocks from the CLIP post), plus one new
capability: `forward()` accepts either `input_ids` (does the normal embedding lookup) *or*
`inputs_embeds` (already-embedded vectors, used as-is) — the hook that lets the VLM splice
projected image tokens in front of real text embeddings.

**`export_gpt2_weights.py` — real pretrained weights, loaded into that hand-rolled module, and the
one genuine war story of this build.** HuggingFace's GPT-2 stores its linear layers as `Conv1D`,
weight shape `(in_features, out_features)`, used as `x @ W` — the opposite convention from
`nn.Linear`'s `(out_features, in_features)`, used as `x @ W.T`. Every `c_attn` / `c_proj` / `c_fc`
tensor needs an explicit transpose on the way in:

```python
c_attn_w = hf_state_dict[prefix + "attn.c_attn.weight"]  # (768, 2304) -- Conv1D layout
q_w, k_w, v_w = c_attn_w.split(embed_dim, dim=1)
sd["attn.q_proj.weight"] = q_w.T.contiguous()             # now nn.Linear layout
```

Get this wrong and nothing errors — shapes still match, the model still runs, and it produces
near-noise. I didn't trust the load just because it completed; I checked it against HF's own model
directly: max absolute logit difference of **0.000084** on real text, and 10 steps of greedy
generation, token-for-token identical to HF's model. My first version of that check asserted the
model would complete "The capital of France is" with "Paris" — it said "the" instead, which looked
like a failure until I checked HF's own base GPT-2 on the same prompt and got the same non-answer
("...the capital of the French Republic..."). The numeric diff was already the real proof; testing
for a specific fact was testing a property of GPT-2's training data, not of whether my weight remap
was correct.

**`vlm.py` — a near-miss with `no_grad()`.** The frozen image encoder's forward pass is correctly
wrapped in `torch.no_grad()` — nothing upstream of it needs gradients. The frozen GPT's forward
pass must **not** be, even though its own parameters don't get updated:

```python
with torch.no_grad():
    patch_tokens = self.image_encoder.forward_patch_tokens(images)   # correct: nothing needs this graph

image_embeds = self.projection(patch_tokens)      # the only trainable step
logits = self.gpt(inputs_embeds=...)              # must stay OUTSIDE no_grad() --
                                                   # gradients have to flow back through GPT
                                                   # to reach `projection`, even though GPT's
                                                   # own weights won't move.
```

Wrapping that second call in `no_grad()` too would run without a single error, produce correctly-
shaped output, and silently leave `projection.weight.grad == None` on every step — training that
does precisely nothing while looking like it's working. I wrote a real gradient-flow check into
`vlm.py`'s own smoke test specifically to catch this class of bug: forward, backward, then assert
`projection` has nonzero gradients while `image_encoder` and `gpt` have none.

**`dataset.py` — a different tokenizer, for a real reason.** CLIP's text tower used a from-scratch
word-level `WordTokenizer`. Captions here are tokenized with GPT-2's own BPE (`tiktoken`) instead —
not a style preference, a requirement: caption ids are targets for the *frozen* GPT's LM head, so
they have to already live in its vocabulary.

## Experiment results

**Setup:** frozen CLIP image encoder (last post's checkpoint, `embed_dim=192`), a single
`nn.Linear(192, 768)` projection (the only trained weights), frozen GPT-2-small (124M params, real
pretrained weights). 6,000 Flickr8k train images / 1,000 dev / 1,000 held-out test, batch size 32,
10 epochs, cosine LR + 1-epoch warmup, on an Apple Silicon GPU.

| epoch | train_loss | train_tok_acc | val_loss | val_tok_acc |
|---|---|---|---|---|
| 1 | 5.88 | 11.0% | 4.45 | 19.7% |
| 3 | 3.44 | 36.1% | 3.21 | 40.0% |
| 6 | 3.06 | 40.6% | 3.02 | 42.1% |
| 10 | 3.04 | 40.9% | 3.00 | 42.6% |

Two things stand out next to the CLIP post's own training curve. First, no overfitting gap
anywhere — val tracks train closely the whole way, even slightly ahead most epochs, a real contrast
with CLIP's run (which overfit hard by epoch 15-20). Makes sense: CLIP trained two whole towers;
this trains one 148,224-parameter linear layer sitting behind 126 million frozen parameters — there
just isn't enough capacity here to memorize anything. Second, the loss plateaus hard by epoch 6 and
barely moves for the next four — not underfitting (more epochs on this exact setup won't help,
same "underfitting is cheap to fix, this isn't that" lesson from Phase 1), but a real capacity
ceiling: a single linear map, with both the vision encoder and the language model frozen solid, can
only bridge the two spaces so well. That ceiling is precisely why LLaVA's own second stage exists —
unfreezing the LLM gives it room to adapt, instead of asking one linear layer to do all the work.

**The visual check**, same instinct as the last two posts — don't trust a number, look at real
generations. `experiments/vlm_flickr8k/eval.py` runs greedy autoregressive decoding (image tokens
in, generate one word at a time, feeding each prediction back in, until `<|endoftext|>` or a 30-
token cap) on 8 real held-out test images:

![VLM caption predictions](/assets/img/vlm-flickr8k-predictions.png)

All three actual dog photos in the sample — a close-up white dog, a jumping black Labrador, a
running tan puppy — got correctly dog/running-themed captions
(`"A dog is running through the woods."` / `"...along a grassy field."` / `"...on a trail."`),
despite nothing in training ever telling the model "this image contains a dog" directly — that
signal had to travel through CLIP's frozen features, through a single linear layer, into a frozen
GPT that's never seen an image, and come out as the right *word*. The five human/object images got
generically-phrased but broadly on-theme captions (mostly correctly identifying a person, wrong on
specifics) — one real miss, a running boy at the beach, got captioned as a dog, plausibly a motion-
blur/pose confusion inherited from CLIP's own features rather than something the projection layer
invented on its own. Exactly the shape I'd expect from Stage 1 alone: broad-category grounding
works, fine-grained detail doesn't, because the language model itself never got a chance to adapt —
the next real lever is upgrading either the frozen CLIP encoder underneath this (still trained on
just 6,000 images, with a fairly small backbone) or unfreezing the GPT for a real Stage 2, not just
running Stage 1 longer.

## What I learned

- A doubt worth repeating from the intuition section, because it cost me a wrong first guess: two
  spaces being *aligned for comparison* (what CLIP does) is a completely different property from
  one space being *usable as input* to some other, unrelated frozen model. CLIP solved the first
  problem last post; this post's whole job was the second one.
- "Does the model know a real-world fact" is the wrong test for verifying a from-scratch
  reimplementation of a pretrained model — it tests a property of the checkpoint's training data,
  not of whether your loading code is correct. Numeric parity against the reference implementation
  (max logit diff, token-for-token generation match) is the actual proof; I nearly mistook a correct
  implementation for a broken one because of this.
- The `no_grad()` scoping mistake is the sharpest new failure mode I've hit in this repo so far,
  specifically because it fails *silently* — no shape error, no crash, no NaN, just a trainable
  parameter that never moves while every log line looks completely normal. Worth a real gradient-
  flow assertion any time a forward pass mixes frozen and trainable submodules, not just a shape
  check.
- A loss plateau with no train/val gap is a different diagnosis than a loss plateau *with* a
  growing gap (Phase 1's ViT, and CLIP's own run, both showed the latter). No gap plus a plateau
  means the bottleneck is capacity, not data or regularization — the fix is a bigger/more expressive
  model somewhere in the chain, not more epochs or more images on the exact same setup.
- Same lesson as the last two posts, in a new form: three correctly-captioned real dog photos told
  me more about whether this actually works than the 42.6% token-accuracy number did by itself. A
  metric says how often; a real example says whether the thing that's supposed to be happening is
  actually happening.
