# Memo 01 — SPY 12-month baseline

**Claim:** A lagged 12-month sign rule on SPY keeps most of buy-and-hold return with a materially smaller max drawdown.

**Rule:** \(s_t=\mathrm{sign}(r_{t-12:t})\), \(p_t=s_{t-1}\), \(r^{\mathrm{TSMOM}}_t=p_t\cdot r_t\).

**Snapshot (cached Yahoo SPY, monthly):**

| | Asset | 12m TSMOM |
|--|------:|----------:|
| CAGR | ~10.4% | ~9.9% |
| Sharpe | ~0.75 | ~0.71 |
| Max DD | ~−50.8% | ~−36.3% |

**Caveats:** No costs, financing, or futures rolls. One asset ≠ the paper’s diversified result.

**Content:** `content/short/01_spy_drawdown.md`, `content/long/01_tsmom_one_asset.md`
