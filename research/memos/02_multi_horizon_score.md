# Memo 02 — Multi-horizon score

**Claim:** Averaging 1/3/6/12m signs creates a conviction dial (full size when horizons agree). On single ETFs it often does **not** beat 12m Sharpe — turnover is the tax.

**Score:** \(\mathrm{score}_t=\frac14\sum_k\mathrm{sign}(R_{k,t})\), \(p_t=\mathrm{score}_{t-1}\).

**SPY snapshot:**

| | 12m | Score |
|--|----:|------:|
| Sharpe | ~0.71 | ~0.58 |
| Mean monthly turnover | ~0.08 | ~0.36 |

**Takeaway for content:** Sell the *mechanism* (agreement → size) and the *cost warning*, not “score beats 12m on SPY.” Portfolio chapter is where diversification helps.

**Content:** `content/short/02_fast_vs_slow.md`, `content/long/02_multi_horizon.md`
