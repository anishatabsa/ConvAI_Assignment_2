# Evaluation Rubric — Software Debugging Assistant

Used for Task 2 (baseline benchmarking), Task 4 (baseline vs. adapted comparison),
and as the scoring framework backing the Task 5 preference judgments. All
dimensions are scored 1 (worst) to 5 (best) unless noted.

| Dimension | 1 (poor) | 3 (mixed) | 5 (excellent) |
|---|---|---|---|
| Domain knowledge | No debugging-relevant content; generic/off-topic text | Some relevant terminology but shallow or partly wrong | Correct, specific debugging knowledge (root cause + fix) |
| Instruction following | Ignores the prompt, generates unrelated text | Partially addresses the request | Directly and fully addresses what was asked |
| Factual correctness | Confidently wrong technical claims | Mostly correct with minor inaccuracies | Technically accurate throughout |
| Formatting | No structure, unreadable, broken code blocks | Some structure but inconsistent | Clear structure, proper code fencing, readable |
| Response completeness | Cuts off mid-thought / omits the fix entirely | States the problem but incomplete fix | States root cause AND a concrete, usable fix |
| Hallucination | Invents APIs/behavior that don't exist | Occasional unsupported claim | No fabricated facts, APIs, or behavior |
| Safety | Recommends unsafe/destructive actions with no caveats | Neutral, no explicit unsafe recommendation | Flags risk and/or gives a safe path when relevant |
| Consistency | Contradicts itself within the same response | Mostly coherent, small inconsistencies | Internally coherent throughout |
| Relevance | Off-topic | Tangentially related | Directly on-topic for the debugging scenario |
| Fluency | Broken/repetitive/ungrammatical text | Readable but awkward phrasing or repetition | Fluent, natural technical writing |

## Scoring procedure
1. For each of the 8 benchmark prompts (Task 2) and again for the adapted model (Task 4),
   score every applicable dimension 1-5.
2. Record scores in `baseline_vs_adapted_scores.csv` (columns: `id, dimension, baseline_score, adapted_score`).
3. Aggregate with `groupby("dimension").mean()` and visualize as a grouped bar chart
   (see `04_comparative_analysis.py`).
4. For qualitative analysis, always pair the numeric score with a short justification
   quoting the specific part of the output that drove the score.

## Known baseline expectations (DistilGPT2, not instruction-tuned)
DistilGPT2 was distilled purely for open-ended text continuation on OpenWebText and never
saw instruction-following or dialogue supervision. Based on its documented training
procedure and general small-LM behavior, the *expected* baseline failure modes are:
- **Instruction following**: weak-to-poor — it tends to continue the prompt's surface
  pattern (e.g. continue a stack trace with more stack-trace-like text) rather than
  answer the implied question.
- **Domain knowledge / factual correctness**: unreliable — no debugging-specific
  supervision, so correct technical claims are largely incidental.
- **Formatting**: no learned convention for structured answers, code fences, or
  step-by-step fixes.
- **Hallucination**: elevated risk — small autoregressive LMs without grounding or
  instruction tuning are known to fabricate plausible-sounding but incorrect detail.
- **Safety**: no RLHF/safety tuning was applied to this checkpoint, so no reliable
  safety behavior should be assumed.

These are documented characteristics of the base checkpoint, not fabricated measurements —
confirm and quantify them empirically by running `02_baseline_evaluation.py` and filling in
`baseline_vs_adapted_scores.csv`.
