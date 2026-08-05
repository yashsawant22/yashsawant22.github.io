---
layout: default
title: Writing
permalink: /writing/
---

<header class="page-intro">
  <h1>Writing</h1>
  <p>Notes from building and studying models.</p>
</header>

<section class="writing-group">
  <h2>Research notes</h2>
  <ul class="writing-list">
    <li><time datetime="2026-08-02">Aug 2, 2026</time><div><a href="{{ '/2026/08/02/rl-teaches-when-not-how.html' | relative_url }}">RL Teaches a Model When to Reason, Not How</a><p>Diffing an OLMo reasoning model across its final RL update, then causally steering the changed features.</p></div></li>
    <li><time datetime="2026-08-02">Aug 2, 2026</time><div><a href="{{ '/2026/08/02/the-sparsity-knob-that-did-nothing.html' | relative_url }}">The Sparsity Knob That Did Nothing</a><p>Why L1 sparsity failed while training a crosscoder from scratch, and what worked instead.</p></div></li>
    <li><time datetime="2026-05-09">May 9, 2026</time><div><a href="{{ '/2026/05/09/adaptive-lora-sft-vs-grpo.html' | relative_url }}">Adaptive LoRA Works for SFT. It Fails Under GRPO.</a><p>A controlled comparison of gradient-based rank allocation across supervised and reinforcement learning.</p></div></li>
    <li><time datetime="2026-05-08">May 8, 2026</time><div><a href="{{ '/2026/05/08/lora-grpo-rank-allocation.html' | relative_url }}">Gradient-Based LoRA Rank Allocation Fails in GRPO</a><p>The experiment and debugging trail behind an unexpected negative result.</p></div></li>
  </ul>
</section>

<section class="writing-group">
  <h2>Building from scratch</h2>
  <ul class="writing-list">
    <li><time datetime="2026-07-28">Jul 28, 2026</time><div><a href="{{ '/2026/07/28/splicing-vision-into-a-frozen-gpt.html' | relative_url }}">Splicing Vision Into a Frozen GPT</a><p>Joining a CLIP vision encoder and GPT-2 with one learned projection.</p></div></li>
    <li><time datetime="2026-07-27">Jul 27, 2026</time><div><a href="{{ '/2026/07/27/how-do-you-teach-an-llm-to-see.html' | relative_url }}">How Do You Teach an LLM to See?</a><p>Building CLIP and learning why independently trained vector spaces cannot simply be wired together.</p></div></li>
    <li><time datetime="2026-07-26">Jul 26, 2026</time><div><a href="{{ '/2026/07/26/from-text-tokens-to-image-tokens.html' | relative_url }}">From GPT Tokens to Image Tokens</a><p>Building a vision transformer by working out what a token means for an image.</p></div></li>
  </ul>
</section>

<section class="writing-group">
  <h2>Other work</h2>
  <ul class="writing-list">
    <li><time datetime="2026-05-09">May 9, 2026</time><div><a href="{{ '/2026/05/09/personalization-high-stakes-decisions.html' | relative_url }}">Personalizing LLMs for High-Stakes Decisions</a><p>Four lessons from using personalization where preferences and consequences can conflict.</p></div></li>
  </ul>
</section>

<p class="rss-subscribe">New posts are available <a href="{{ '/feed.xml' | relative_url }}">via RSS</a>.</p>
