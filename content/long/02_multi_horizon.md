# Long — Making trend following adaptive (without worshipping speed)

## Problem
12m TSMOM is robust and slow. When trends die, it is late. The temptation is a shorter lookback. Temptation has a bill: turnover.

## Experiment
On SPY, BIL, IEF, TLT, GLD, BTC-USD we compare sign rules at 1, 3, 6, 12 months, then the averaged score:

\[\mathrm{score}_t=\frac14\sum_{k\in\{1,3,6,12\}}\mathrm{sign}(R_{k,t}),\quad p_t=\mathrm{score}_{t-1}.\]

## Single-horizon lesson
1m turnover on SPY is an order of magnitude above 12m. Fast signals look “responsive” in a zero-cost world and expensive in a 10-bps world.

## Score lesson
The score is a **conviction dial**: ±1 when horizons agree, near 0 when they conflict. On SPY alone, Score Sharpe (~0.58) did **not** beat 12m (~0.71) — mean turnover rose from ~0.08 to ~0.36. Honest research: the mechanism is interesting; the SPY horse race is not a free lunch.

## Why keep it
1. Pedagogically clear regime meter.  
2. Natural input to portfolio risk budgeting.  
3. Sets up the disagreement study (chop when |score| is low).

## CTA
`strategy_design_2.ipynb` → then `application.ipynb` for the diversified punchline.
