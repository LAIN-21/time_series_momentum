# Memo 05 — Robustness

**Equal-weight multi-horizon score, 10 bps, Sharpe:**

| Spec | Sharpe |
|------|-------:|
| Full universe | ~0.74 |
| In-sample &lt;2015 | ~0.66 |
| Out-of-sample ≥2015 | ~0.88 |
| No BTC | ~0.57 |

**Cost ladder (full universe, score EW):**

| Cost | Sharpe | CAGR |
|-----:|-------:|-----:|
| 0 bps | ~0.79 | ~7.4% |
| 5 bps | ~0.77 | ~7.1% |
| 10 bps | ~0.74 | ~6.8% |
| 25 bps | ~0.67 | ~6.1% |

**Honest read:** OOS holds (even improves here). Dropping BTC hurts — crypto is a large part of the diversified punch in this ETF set. Edge decays with costs but does not instantly vanish at 10 bps.

**Content:** `content/short/05_what_survives.md`, `content/long/05_robustness.md`
