# Memo 06 — Combination grid (how to merge horizons)

**Question:** Is equal-weight average of 1/3/6/12 signs the best merge — or do weights, subsets, raw averages, or majority votes win?

**Answer (this lab):** **Plain 12m sign still wins net Sharpe** on SPY and on the equal-weight multi-asset book. Combos help mainly as *defense* (lower DD / vol), not as Sharpe upgrades.

## Rankings (10 bps net)

### SPY — top / key

| Spec | Sharpe net | Max DD | Turnover |
|------|----------:|-------:|---------:|
| **single_12m** | **0.71** | −37% | 0.08 |
| sign_eq_6_12 | 0.67 | −40% | 0.15 |
| sign_w_slower (5/15/30/50) | 0.66 | −34% | 0.20 |
| sign_eq_3_6_12 | 0.59 | −33% | 0.23 |
| sign_eq_1_3_6_12 (old score) | 0.55 | −28% | 0.36 |
| maj / thr 0.75 | ~0.44 | −25% | ~0.39 |
| single_1m | 0.06 | −53% | 0.87 |

### EW multi-asset — top / key

| Spec | Sharpe net | Max DD | Turnover |
|------|----------:|-------:|---------:|
| **single_12m** | **0.90** | −23% | 0.17 |
| sign_eq_6_12 | 0.86 | −21% | 0.22 |
| sign_w_slower | 0.85 | −16% | 0.26 |
| sign_w_slow | 0.82 | **−14%** | 0.30 |
| sign_eq_3_6_12 | 0.76 | −21% | 0.28 |
| sign_eq_1_3_6_12 | 0.74 | −18% | 0.40 |
| raw_eq_3_6_12 | 0.77 | −27% | 0.21 |

## What we learned

1. **Dropping 1m helps** vs full equal score (`3+6+12` and especially `6+12`), but still loses to pure 12m on Sharpe.
2. **Tilting weights toward 12m** recovers most of the way to the baseline; “slow blend” ≈ soft 12m with a bit more turnover.
3. **Raw-return average → sign** is not better than average-of-signs here; often similar or worse DD.
4. **Majority / hard thresholds** cut exposure and sometimes DD, but net Sharpe suffers (you’re flat too often or still turn over when you re-enter).
5. Best **defensive** EW combo in-grid: `sign_w_slow` (max DD ~−14%) — use that story if the post is about drawdowns, not Sharpe.

## Artifacts

- Notebook: `research/05_combo_grid.ipynb`
- Tables: `research/combo_grid_spy.csv`, `research/combo_grid_ew.csv`
- Summary JSON: `research/combo_grid_summary.json`

**Content:** `content/short/06_combo_grid.md`, `content/long/06_combo_grid.md`
