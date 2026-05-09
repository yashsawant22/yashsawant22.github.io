---
layout: post
title: "Personalizing LLMs for High-Stakes Decisions: Four Lessons"
date: 2026-05-09
---

Most LLM personalization research assumes the easy case: writing style, tone, topical preferences. What happens when personalization has to survive real consequences — when "getting the user what they want" and "getting the user what they need" actively diverge?

I spent the last several months building a personalized investment assistant for my own portfolio as a research testbed. The goal wasn't a product — it was to see which parts of the standard personalization stack (RAG over user history, preference modeling, instruction tuning) actually hold up when the downstream decision costs money and the user's stated preferences contradict their behavior.

Four things broke in ways I didn't expect. This post is about those four things, because I think they generalize to any domain where LLM personalization meets consequential decisions — healthcare, legal, career coaching, long-horizon planning.

---

## Background: A Thesis-Centric Architecture

Before the lessons, a quick mental model of the system, because the design choice matters for what follows.

Most LLM-over-finance setups start with price data and retrieve relevant news or filings. I inverted it: the primary unit of memory is *the user's stated reasoning for holding a position* — a structured "thesis" — and every downstream component (reports, alerts, evaluation) scores new evidence against that thesis, not against the market in the abstract.

```
DATA SOURCES
  Brokerage (Robinhood) · Market Data (yfinance)
  Earnings Calendar · Your Interactions (CLI/Web/Chat)
          │
          ▼
CORE ENGINE
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │ Living      │  │ Conviction  │  │ Behavioral  │
  │ Thesis      │  │ Scoring     │  │ Memory      │
  │ Manager     │  │ Engine      │  │ Extractor   │
  │             │  │             │  │             │
  │ Per-holding │  │ CONFIRMED   │  │ preferences │
  │ hypotheses  │  │ UNCHANGED   │  │ beliefs     │
  │ with break  │  │ WEAKENED    │  │ patterns    │
  │ conditions  │  │ BROKEN      │  │ rules       │
  └──────┬──────┘  └──────┬──────┘  │ risk toler. │
         │               │         └──────┬──────┘
         └───────┬────────┴───────────────┘
                 ▼
ANALYSIS LAYER
  Drift Detection · Pattern Matching · Position Grading
          │
          ▼
OUTPUT LAYER
  Daily Reports · Alerts · AI Chat · Web Dashboard
```

A thesis is not free-form text. It's a structured object with fields the LLM must evaluate against:

```python
@dataclass
class Thesis:
    ticker: str
    conviction_statement: str       # "AI infra capex is in a multi-year buildout"
    validation_triggers: list[str]  # signals that would strengthen the thesis
    break_conditions: list[str]     # signals that would falsify it
    catalysts: list[str]            # upcoming events that resolve uncertainty
    time_horizon: str               # "6-18 months"
    conviction_score: float         # 0-10, updated over time
```

When new evidence arrives (an earnings print, a macro data point, a news headline), the scoring prompt forces the model to map the evidence onto `validation_triggers` and `break_conditions` explicitly — not to produce a free-form "bullish / bearish" verdict. The structured format is what makes the four problems below tractable at all.

---

## Four Things That Broke

### 1. The user's "profile" contradicts itself — and the contradiction is the signal

Most personalization systems model the user as a stable preference vector. You extract preferences from history, you RAG over them, you condition generation on them. Done.

In a high-stakes domain, the user's stated rules and their behavior routinely disagree — and flattening that disagreement into a single "preference" throws away the most useful signal in the data.

Concretely: I have a stated rule of "never average down into a position whose thesis is weakening." My trade history shows I do exactly that during high-volatility stretches. A naive preference model either encodes the rule (and gives advice I'll ignore) or encodes the behavior (and endorses the mistake). Neither is useful.

The fix was to stop collapsing these into one profile. Behavioral memory is split into five typed stores with different decay rates:

```python
MEMORY_TYPES = {
    "preference":    {"decay_days": 180},  # "prefers dividend payers"
    "belief":        {"decay_days": 30},   # "thinks rates will cut in Q3"
    "pattern":       {"decay_days": 90},   # "buys dips during VIX spikes"
    "rule":          {"decay_days": 365},  # "never averages into broken theses"
    "risk_tolerance":{"decay_days": 365},  # stable over long horizons
}
```

At inference time, the system retrieves from all five and **explicitly surfaces conflicts** — "your stated rule R contradicts observed pattern P" — rather than resolving them silently. The contradiction is treated as first-class output, not an inconsistency to be smoothed away.

This one design change was the single biggest behavioral improvement. A flagged contradiction at the moment of decision is worth more than ten correct retrievals.

### 2. Retrieval isn't enough — you need evaluative consistency across time

This is the subtlest of the four. Vanilla RAG over a user's notes gives you *access* to their reasoning. It doesn't give you *consistency* in how that reasoning is applied to new evidence.

Suppose the user bought a semiconductor stock six weeks ago on the thesis "AI infrastructure capex is in a multi-year buildout." Earnings come in light. A stateless LLM, prompted to interpret the earnings, will produce a locally plausible "earnings miss means bearish" read. A RAG-augmented LLM will retrieve the original thesis note and produce something slightly more nuanced — but still anchored on whatever framing is most salient in the retrieved context.

The question the system *should* be answering is narrower: **does this specific evidence map to a `break_condition` the user wrote down at the time of purchase, or not?** If yes, downgrade conviction. If no, hold. That's a classification problem, not a generation problem — and the structured thesis format above is what makes it one.

The scoring prompt looks roughly like:

```
Given this thesis:
  conviction_statement: {conviction_statement}
  break_conditions:    {break_conditions}
  validation_triggers: {validation_triggers}

And this new evidence:
  {evidence}

For each break_condition and validation_trigger, answer:
  - Does the evidence directly address it? (yes/no)
  - If yes, does it trigger/validate/weaken it? (one-word label)
  - Cite the exact phrase from the evidence that supports your answer.

Return a JSON object with per-condition verdicts. Do NOT produce an
overall bullish/bearish judgment.
```

The key move is forbidding the model from producing a free-form verdict. Forcing it to commit to per-condition verdicts eliminates most of the drift you get when the same thesis is re-interpreted week after week.

### 3. The objective you actually want is the opposite of the objective personalization usually optimizes

In most personalization work, the north star is preference-matching: the system that best predicts what the user wants is the best system. In high-stakes domains, that objective inverts. A system that reliably tells you what you want to hear is a system that reliably confirms your biases.

This isn't a hypothetical. Sanz-Cruzado et al. (*Personalized Financial Advisors and LLM Personas*, 2025) found that users consistently preferred LLM financial advisors with more extroverted, confident personas — even when those advisors gave objectively worse advice. **User satisfaction and advice quality were negatively correlated.** Optimizing for the former actively hurts the latter.

The architectural response is to stop treating user satisfaction as a proxy for quality. Concretely, the report generator has two categories of content it is *required* to emit whenever relevant, regardless of whether the user wants to see it:

1. **Drift observations** — cases where recent actions contradict stated theses or rules.
2. **Pattern counterexamples** — cases where the user's stated reasoning for a current decision matches a historical pattern that previously led to a mistake.

These aren't generated by asking the model "is there anything the user should hear?" — that invites the sycophancy the paper above documents. They're generated by separate deterministic checks (comparing recent trades against thesis conviction scores, matching current justifications against a pattern store) and *injected* into the report as mandatory sections. The LLM writes them up, but it doesn't decide whether they appear.

The uncomfortable implication: the best version of this kind of system is one the user sometimes actively dislikes.

### 4. You cannot evaluate the system by whether the user's outcomes improved

This is the hardest one, and it's a general problem for any personalization work in a consequential domain: the outcome signal is too noisy, too delayed, and too confounded to use as a training or evaluation target.

In investing specifically, a position held on a sound thesis can lose money because of an unrelated macro shock. A reckless impulse trade can make money because the market went up that week. If you grade the system on P&L, you will eventually train it — or train the user it's advising — to gamble, because gambling and investing look identical on any individual trade and only diverge over hundreds of decisions.

The alternative is to grade **process**, not outcomes. When a position closes, the evaluator answers three questions independent of the return:

1. **Directionality** — was the thesis's causal claim about the world directionally supported by what actually happened? (Not: did the stock go up.)
2. **Timing** — did the entry and exit correspond to the catalysts the thesis specified, or to unrelated noise?
3. **Sizing consistency** — was position size proportional to the conviction score at entry?

A losing trade on a thesis that was directionally correct, entered on the right catalyst, and sized appropriately scores higher than a winning trade that violated all three. Over time, this grading signal is what the behavioral memory layer learns from — not the P&L.

This is an instance of a general pattern that shows up any time you want to evaluate an LLM system operating in a domain with noisy ground truth: **decompose the decision into intermediate artifacts that you can evaluate directly, and grade those, rather than waiting for the outcome.** The same move works in clinical decision support (grade whether the differential was complete, not whether the patient recovered), legal research (grade whether the cited precedents are on point, not whether the case was won), and long-horizon planning (grade the checkpoint milestones, not the final goal).

---

## Why This Generalizes Beyond Finance

The four problems above aren't really about investing. They show up whenever LLM personalization meets a domain with three properties:

- **Consequential decisions** — there's a real cost to being wrong, so "what the user prefers" and "what actually helps the user" can diverge.
- **Temporally extended commitments** — the user's state at decision time matters, and the system has to carry structured reasoning across weeks or months, not just across a conversation.
- **Noisy outcome signals** — you can't close the loop with a simple reward, so you have to evaluate the process, not the result.

Healthcare decision support, legal research assistants, long-horizon career and education coaching, and any agent that manages an ongoing plan all have these properties. Finance was useful as a testbed precisely because P&L is measurable enough to *seem* like ground truth while being noisy enough that using it as one quietly destroys the system.

The pattern I'd offer if you're building something in this space:

1. **Represent the user as multiple typed stores with different decay rates**, and surface contradictions between them as output rather than hiding them.
2. **Force structured evaluation against user-authored break conditions**, not free-form generation over retrieved context.
3. **Separate "content the user will like" from "content the user needs to see"**, and generate the second deterministically.
4. **Grade intermediate artifacts, not outcomes**, when outcomes are noisy or delayed.

None of these are new ideas on their own. What I underestimated going in was how aggressively the default LLM personalization stack — RAG over a flat profile, preference-conditioned generation, implicit reward modeling — resists each of them. Getting the four to work together required treating personalization less as a retrieval problem and more as a structured reasoning problem with the user's own prior commitments as the spec.

The easy version of LLM personalization is mostly solved. The version that survives contact with consequential decisions is, as far as I can tell, still wide open.

---

The full position paper is on arXiv: [*High-Stakes Personalization: Rethinking LLM Customization for Individual Investor Decision-Making*](https://arxiv.org/abs/2604.04300) — submitted to the [CustomNLP4U workshop at ACL 2026](https://customnlp4u-2026.github.io/).
