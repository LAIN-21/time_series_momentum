# Long — Merging momentum horizons: a small bake-off

## Setup
Same lab universe (SPY and EW of SPY/BIL/IEF/TLT/GLD/BTC). Lag-1 positions. 10 bps turnover cost.

Families tested:
- Single horizons (1, 3, 6, 12)
- Equal-weight **average of signs** on subsets
- **Weighted** sign averages (tilt slow / tilt fast)
- **Average trailing returns → sign**
- **Majority vote** (+ min-agree thresholds)
- **Hard threshold** on the equal score (|score| ≥ 0.5 / 0.75)

## Result
`single_12m` tops **net Sharpe** on both SPY (~0.71) and the EW book (~0.90).

The old equal 1/3/6/12 score sits mid-pack (~0.55 SPY / ~0.74 EW) — fine as a conviction dial, not a Sharpe upgrade.

## Useful runners-up
- `sign_eq_6_12` — best simple combo; almost 12m with a touch more responsiveness
- `sign_w_slower` — recovers toward 12m by overweighting the long lookback
- `sign_w_slow` on EW — **best left-tail** in the grid (max DD ~−14%) if the goal is defense

## What failed
- Anything heavy on **1m**
- Strict majority / high thresholds (too flat, or costly re-entry)
- Raw-return averaging as a magic alternative to averaging signs

## Post angle
“We didn’t just average four signals and declare victory. We grid-searched the merges. Slow still wins. Fast is a tax. Blends are for sizing and drawdowns.”

Notebook: `research/05_combo_grid.ipynb`
