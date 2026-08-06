---
layout: default
image: /assets/img/yash-sawant.png
---

<header class="research-intro">
  <h1>Research</h1>
  <p>I work on language-model training, evaluation, and interpretability. I’m particularly interested in what post-training changes inside models, when evaluations reward the wrong behavior, and how those changes can be tested causally.</p>
</header>

<main class="selected-work">
  <h2>Selected work</h2>

  <article class="work-item">
    <p class="artifact-meta">Interpretability · Research artifact</p>
    <h3><a href="{{ '/2026/08/02/rl-teaches-when-not-how.html' | relative_url }}">RL teaches a model when to reason, not how</a></h3>
    <p>Across OLMo-2-1B’s final RL step, the update preferentially changed reasoning features. Steering one RL-associated direction into the pre-RL model raised aggregate reasoning behavior from 8% to 92% without updating its weights.</p>
    <p class="artifact-links"><a href="{{ '/2026/08/02/rl-teaches-when-not-how.html' | relative_url }}">Result</a> · <a href="https://github.com/yashsawant22/rl-reasoning-diff">Experiment</a> · <a href="https://github.com/yashsawant22/rl-crosscoder">Crosscoder</a> · <a href="https://huggingface.co/Savianto">Models</a></p>
  </article>

  <article class="work-item">
    <p class="artifact-meta">Post-training · EMNLP 2026</p>
    <h3><a href="{{ '/2026/05/09/adaptive-lora-sft-vs-grpo.html' | relative_url }}">When Gradient Importance Lies</a></h3>
    <p>Gradient-based LoRA rank allocation succeeded under SFT but degraded under GRPO at the same parameter budget, where the gradients became flatter, noisier, and coupled to the rank they were meant to allocate.</p>
    <p class="artifact-links"><a href="https://arxiv.org/abs/2605.07366">Paper</a> · <a href="{{ '/2026/05/09/adaptive-lora-sft-vs-grpo.html' | relative_url }}">Analysis</a> · <a href="https://github.com/yashsawant22/adaptive-lora-rank-grpo">Code</a></p>
  </article>

  <article class="work-item">
    <p class="artifact-meta">Evaluation · ICML 2026</p>
    <h3><a href="https://arxiv.org/abs/2604.26460">Theory-Grounded Evaluation Exposes the Authorship Gap in LLM Personalization</a></h3>
    <p>PersonalBench tests whether personalized generations preserve an individual’s style, knowledge, preferences, and consistency—not merely whether they sound personal.</p>
    <p class="artifact-links"><a href="https://arxiv.org/abs/2604.26460">Paper</a> · <a href="https://github.com/yashsawant22/personalbench">Code and benchmark</a></p>
  </article>
</main>
