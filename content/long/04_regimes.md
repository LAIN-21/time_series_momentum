# Long — Disagreement as a regime meter

## Idea
If four lookbacks disagree, the market is not in a clean trend. The multi-horizon score already shrinks size. Question: is that month *also* choppier ex post?

## Test
Lag the agreement label (high / mid / low |score|). Measure next-month |SPY return| and strategy behavior.

## Finding
High-agreement months: mean |SPY return| ~3.1%.  
Mid/low: ~4.0%.  
Strategy Sharpe is stronger in high-agreement buckets; low-agreement months are often near-flat by construction (score≈0).

## How to talk about it
Not “alpha from disagreement.” Better: “the dial is doing risk management that lines up with realized chop.” Falsifiable, useful, no hype.

## CTA
`research/03_disagreement.ipynb`
