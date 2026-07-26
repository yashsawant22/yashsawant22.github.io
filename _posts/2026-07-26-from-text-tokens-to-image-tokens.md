---
layout: post
title: "From GPT Tokens to Image Tokens: Teaching Transformers to See"
date: 2026-07-26
---

*Part of [embodied-gpt](https://github.com/yashsawant22/embodied-gpt), my from-scratch series
working through LLMs → Vision Transformers → Vision-Language Models → Robot Learning →
Vision-Language-Action Models. This post covers Phase 1.*

I already had a GPT built from scratch — tokenization, embeddings, attention, the residual
stream, the training loop, all of it. So when I sat down to build a Vision Transformer, I
expected the hard part to be new math. It wasn't. The hard part was a much smaller, dumber-
sounding question that I kept getting stuck on: **what, exactly, is a token when the input is a
picture instead of a sentence?**

This post is the path I actually took through that question — including the points where I got
confused, asked a dumb-feeling question, and had to back up and re-explain something to myself
in plainer terms. If you've built a GPT and are eyeing a ViT next, I'd bet you hit the same
snags.

## The problem

In my GPT, a token is an integer ID, looked up in a table to get a vector. That machinery —
attention, residual stream, MLPs — doesn't care what the vectors *mean*, it just needs a
sequence of them. So the real question for images isn't "how does attention work on pixels,"
it's narrower: **what is the "token" of an image, and how do you turn it into a vector?**

The naive answer — one token per pixel — falls apart fast. A 32×32 CIFAR-10 image is already
1,024 pixels; a 224×224 image is ~50,000. Attention costs O(n²), so that's 2.5 billion pairwise
interactions per layer, per image, before you've even started. And a single pixel barely carries
any meaning on its own — you'd be asking attention to first discover local structure that a
cheap operation could hand it for free.

## Historical context

CNNs solved image understanding for a decade using convolutions — small filters slid across the
image, encoding an assumption baked directly into the architecture: nearby pixels are related,
and that relationship shouldn't have to be *learned*, it should be *assumed*. That assumption
is called an inductive bias, and it's why CNNs were so data-efficient early on.

ViT (Dosovitskiy et al., 2020 — "An Image is Worth 16x16 Words") made a different bet: strip
that assumption out, treat an image as a plain sequence, and let attention learn whatever
spatial structure it needs from data instead of having it hard-coded. The title is the whole
idea — a **patch** of a picture plays the same structural role a **word** plays in a sentence.

## Intuition: patches as words

Here's the pipeline, side by side with what I already knew:

```
TEXT (GPT)                              IMAGE (ViT)

"the cat sat"                            [picture]
      │                                       │
      ▼                                       ▼
["the","cat","sat"]  split into words    [patch1, patch2, ...]   split into squares
      │                                       │
      ▼                                       ▼
 [5, 812, 91]  look up each word's ID    [nums1, nums2, ...]     flatten each square
      │                                       │
      ▼                                       ▼
[vec1, vec2, vec3]   look up embedding   [vec1, vec2, ...]       run through a linear layer
      │                                       │
      ▼                                       ▼
   feed to transformer                    feed to transformer
```

Same shape of pipeline, one real difference: words come from a fixed dictionary, so you can
*look up* their vector in a table. Patches don't come from a dictionary — a patch is raw pixel
values, different every time — so instead of looking up a vector, you **compute** one, with a
linear layer. That linear layer is the direct analog of GPT's embedding table: same job (turn a
raw input unit into a vector), different mechanism because the input is continuous instead of
discrete.

I'll admit — the first pass I got at this intuition leaned hard on notation (`x_p ∈
R^(N × P²C)`, `E ∈ R^(P²C × D)`, etc.), and it didn't stick. What actually made it click was
dropping the symbols and just tracing concrete shapes through a concrete example, which is worth
showing here because I think it's a better way to hold the idea in your head than the equations:

| Step | Shape (CIFAR-10, 4×4 patches, embed dim 128) | What happened |
|---|---|---|
| Input image | `(B, 3, 32, 32)` | raw pixels |
| Cut into patches | `(B, 64, 4, 4, 3)` | 64 squares, 8×8 grid |
| Flatten each patch | `(B, 64, 48)` | each square → one list of 48 numbers |
| Linear projection | `(B, 64, 128)` | 48 numbers → 128-dim embedding |
| Prepend `[CLS]` | `(B, 65, 128)` | one extra learned token glued on front |
| Add positional embedding | `(B, 65, 128)` | shape unchanged — adds "where am I" info |
| Through transformer blocks | `(B, 65, 128)` | shape **never** changes here — same as GPT's `(B, T, D)` |
| Take `[CLS]` only | `(B, 128)` | throw away the other 64 |
| Classification head | `(B, 10)` | one linear layer down to 10 classes |

Two things jump out once you see it this way: the `64 → 65` bump happens exactly once, and after
that the shape is frozen through every single transformer block — identical to how `(B, T, D)`
never changes across a GPT's blocks either. The only step that changes the *amount of
information per token* is the `48 → 128` projection; everything else is rearranging or adding.

## A doubt worth naming: encoder or decoder?

Given I'd only ever built a decoder (GPT), I had to explicitly ask myself: is this the same
architecture with masking, or something structurally different?

It's structurally different, and the reason is worth being precise about, because it's not just
terminology — it changes what the attention layer's code looks like:

- **GPT (decoder-only):** each token can only look at tokens *before* it. You mask the future
  because the task is "predict the next token," and letting a token see its own answer would be
  cheating.
- **ViT (encoder-only):** every token looks at every other token, no masking at all. There's no
  "future" to hide — you're not predicting patch 17 from patches 1–16, you're letting all 65
  tokens exchange information freely so the model can build one global read of the whole image,
  then reading the answer off `[CLS]`.

Concretely: the ViT's attention layer is the exact same Q/K/V math as a GPT's attention layer,
minus the causal mask. That's the entire structural difference at the attention layer.

## The part I actually got stuck on: `[CLS]`

Everything above came reasonably fast. The thing that genuinely didn't click on the first, second,
or even third explanation was the `[CLS]` token. It's worth walking through the confusion
honestly, because I don't think I'm the only one who trips here.

My first pass at explaining it to myself was correct but useless: "it's a learnable token
prepended to the sequence whose final hidden state is used for classification." True. Meant
nothing to me. What actually worked was dropping to an analogy:

Imagine 64 people in a room, each of whom only looked at one small square of a photo and has an
opinion about their square. You never saw the photo. You walk around, listen to all 64 of them,
and form your own opinion by combining what they say. **You** are `[CLS]` — you start knowing
nothing, and everything you end up "knowing" is entirely built from listening to the real
tokens. At the end, *your* opinion — not any one person's — is what gets reported as the answer.

Once that landed, two follow-up questions came out of it that were sharper than the original
confusion:

**"Doesn't the position of `[CLS]` in the sequence not really matter?"** — correct instinct, and
worth being precise about *why*. Attention itself is permutation-agnostic; it only cares about
content, not array order. Position only enters via the positional embedding you add on top. So
`[CLS]` living at index 0 (vs. index 5, or anywhere else) is a convention, not a requirement —
what actually matters is *consistency*: whatever slot it lives in, it needs to get the same
positional embedding and be read from the same index every time. Contrast this with the *patch*
positional embeddings, which do encode something real — position 3 corresponds to an actual
grid location in the image.

**"In inference, does only `[CLS]` compute a query?"** — this one, I initially conflated with
something I already understood from GPT: KV-caching, where during generation only the *newest*
token computes a query, reusing cached K/V from every earlier token. It's a natural thing to
reach for, but the mechanism doesn't transfer, and pulling the two apart clarified both:

| | GPT + KV cache | ViT `[CLS]` |
|---|---|---|
| Why it's valid | causal mask freezes a past token's hidden state forever — nothing later can reach back and change it | you just know one output (patches' final layer) is never read |
| Applies to | every layer, every generation step | only the very last layer, in a single forward pass |
| What it saves | redundant recomputation across many sequential steps | one layer's worth of compute you'd throw away anyway |

There's no autoregression happening inside a ViT — it's one parallel forward pass over all 65
tokens, bidirectional, no cache. All 65 tokens (patches included) compute Q/K/V and attend fully
at *every* layer, because the patches need to keep refining each other for `[CLS]`'s final
summary to be worth anything. The only place you could legitimately skip patch computation is
the last layer, since nothing downstream reads the patches' output there — and even then, my
implementation doesn't bother, because I wanted every patch's attention weights available at
every layer for a possible future attention-map visualization.

## Architecture

```
Image (H x W x C)
   │  split into non-overlapping patches
   ▼
Patches (N x (P*P*C))
   │  linear projection
   ▼
Patch embeddings (N x D)  + prepend [CLS] + positional embeddings
   │
   ▼
Transformer block x L   (multi-head attention + MLP, pre-norm, residual)
   │
   ▼
take [CLS] output → classification head
```

## Math

Image `x ∈ R^(H×W×C)`, patch size `P`. Reshape into `N = (H/P)·(W/P)` patches, each flattened to
`R^(P²·C)`:

```
z_0 = [ x_cls ;  x_p^1 E ;  x_p^2 E ; ... ; x_p^N E ]  +  E_pos

E ∈ R^(P²C × D)        learned projection, analog of GPT's embedding table
x_cls ∈ R^D            learned [CLS] embedding
E_pos ∈ R^((N+1) × D)  learned positional embeddings
z_0 ∈ R^((N+1) × D)    input sequence to the transformer blocks
```

Attention, per head, no causal mask:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_head)) V
```

Everything downstream of `z_0` — multi-head attention, residual connections, MLP — is unchanged
from a GPT block.

## Code walkthrough

Full implementation is in
[`src/vision/`](https://github.com/yashsawant22/embodied-gpt/tree/main/src/vision):
`patch_embedding.py`, `positional_embedding.py`, `multi_head_attention.py`,
`transformer_block.py`, `vit.py`, `train.py`. Design write-up for each file is in
[`src/vision/README.md`](https://github.com/yashsawant22/embodied-gpt/blob/main/src/vision/README.md)
— I won't duplicate it here, but the one part worth calling out is `multi_head_attention.py`,
because I wrote a first draft myself and it had four real bugs, all instructive:

1. `nn.linear` instead of `nn.Linear` — a typo that just crashes.
2. **Missing the scaling** — `Q @ K.transpose(-1,-2)` with no `/ sqrt(head_dim)`. This is
   "scaled dot-product attention" with the scaling silently dropped; without it, dot products
   grow with `head_dim`, softmax saturates, gradients vanish.
3. **`softmax(dim=1)` instead of `dim=-1`** — softmax needs to normalize over the *key*
   dimension so each query's attention weights sum to 1; `dim=1` normalizes over the *heads*
   dimension instead, which is meaningless.
4. Incomplete head-merging logic after the attention output.

None of these broke the output *shape* — `(B, N, D)` in, `(B, N, D)` out, looked fine at a
glance. That's the real lesson: **shape checks catch structural bugs, not numerical ones.** A
model can run cleanly end to end and still be computing something wrong. The fix for #4 also
surfaced a small but genuinely useful PyTorch fact: `.transpose()` leaves a tensor non-contiguous
in memory, so merging the head dimension back afterward needs `.reshape()` (which copies if
needed) rather than `.view()` (which would just error).

## Experiment results

Trained on CIFAR-10, `embed_dim=128, depth=4, num_heads=4, patch_size=4`, on an Apple Silicon
GPU (`mps`).

**Getting the data was its own small detective story.** The official CIFAR-10 host
(`cs.toronto.edu`) downloaded at ~94KB/s in this environment — 28 minutes for a 170MB file. I
almost assumed the environment itself had a bandwidth cap, but a quick `curl` test against
GitHub hit ~12MB/s from the same machine, which ruled that out. The Toronto server itself is
just slow. Switching to CIFAR-10 hosted on HuggingFace (`uoft-cs/cifar10`) got ~40MB/s — roughly
400x faster, same official train/test split, no data-quality tradeoff.

**Run 1 (10 epochs, flat LR):**

| epoch | train_acc | test_acc |
|---|---|---|
| 1 | 35.7% | 43.8% |
| 5 | 59.7% | 61.8% |
| 10 | 68.7% | 68.0% |

Train and test tracked closely the whole way — no overfitting, but also clearly not finished
improving. That's the useful signal: when train and test accuracy are still moving together at
the end of a run, the fix is almost always just "run it longer," not "change the architecture."

**Run 2 (40 epochs, cosine LR schedule + 3-epoch warmup + label smoothing 0.1):** same
architecture, standard transformer-training recipe additions on top.

| epoch | train_acc | test_acc |
|---|---|---|
| 1 | 26.6% | 35.0% |
| 10 | 67.3% | 68.3% |
| 20 | 78.0% | 75.5% |
| 30 | 85.5% | 78.3% |
| 34 | — | **79.3% (best)** |
| 40 | 88.3% | 79.0% |

**68% → 79%**, ~11.4 minutes total. But the more interesting thing in this run is what happens
*after* epoch ~20: train accuracy keeps climbing (78% → 88%) while test accuracy basically
plateaus (75.5% → ~79%, with most of that gain arriving before epoch 30). That gap opening up is
overfitting starting, in real time, in the same run that started out underfit. Underfitting and
overfitting aren't opposite ends of one dial you're always somewhere on — they're two different
failure modes, and this run walked through both.

I also built an eval script as a visual sanity check beyond the accuracy number — it runs the
trained model on 16 real, held-out test images and saves a labeled grid (green = correct, red =
wrong):

![prediction grid](/assets/img/vit-cifar10-predictions.png)

Run 2's checkpoint gets all 16/16 here, including an image that Run 1's checkpoint got wrong — a
photo of a *sign with a frog printed on it*, which Run 1 called "bird." That's a genuinely
ambiguous image even for a human glancing quickly, which made it a satisfying one to watch flip
to correct.

68–79% isn't close to what a CNN gets on CIFAR-10 with similar compute — expected, since ViT
lacks convolution's built-in spatial assumptions and has to learn them from data instead, which
takes more of it. That was never the point of this experiment. The point was: does the
from-scratch implementation actually learn, cleanly, with no bug quietly capping it? It does.

## What I learned

- The `[CLS]` token stopped being confusing the moment I stopped trying to hold the linear-algebra
  description in my head and used an analogy instead (65 tokens, one of them starts knowing
  nothing and only ever holds a summary built from listening to the other 64).
- Two sharp-feeling questions — "does `[CLS]`'s position actually matter?" and "is this like
  KV-caching?" — turned out to be genuinely good questions with precise answers, not naive ones.
  Chasing down *why* an instinct is right (or where an analogy breaks) taught me more than either
  the original explanation or a flat "yes/no" would have.
- Shape-correctness is a weak test. My own first attempt at the attention layer ran, produced the
  right output shape, and was still wrong in three numerically meaningful ways (missing scale
  factor, wrong softmax axis, a typo). I'll be adding at least a values-level sanity check, not
  just a shape check, going forward.
- Don't trust that the "official" data source is the fast one — a two-line `curl` bandwidth test
  saved what would've been a 28-minute wait, and pointed at a genuinely better source
  (HuggingFace) rather than just working around a slow one.
- Underfitting and overfitting are different problems with different fixes, and a single training
  run can pass through both — the tell is simply whether train and test accuracy are still moving
  together or have started to diverge.

Full code, configs, and results: [github.com/yashsawant22/embodied-gpt](https://github.com/yashsawant22/embodied-gpt).
