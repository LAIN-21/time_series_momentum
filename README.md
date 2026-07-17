# Time-Series Momentum

Reproducible Python lab for **time-series momentum (TSMOM)** — trade an asset from its own past return (Moskowitz / Ooi / Pedersen style), then document findings for research notes and X.

## Notebooks

| Notebook | Claim |
|----------|--------|
| [`intro_1.ipynb`](intro_1.ipynb) | Single-asset 12m sign rule on SPY |
| [`strategy_design_2.ipynb`](strategy_design_2.ipynb) | Multi-horizon signs + averaged conviction score |
| [`application.ipynb`](application.ipynb) | Equal-weight multi-asset portfolio + costs + vol target |
| [`research/03_disagreement.ipynb`](research/03_disagreement.ipynb) | Does horizon disagreement forecast chop? |
| [`research/04_robustness.ipynb`](research/04_robustness.ipynb) | OOS split, no-BTC, cost ladder |
| [`research/05_combo_grid.ipynb`](research/05_combo_grid.ipynb) | Weights, subsets, raw avg, majority vote |

## Content (X)

- Short posts: [`content/short/`](content/short/)
- Long threads / articles: [`content/long/`](content/long/)
- Research memos: [`research/memos/`](research/memos/)
- Latest numbers: [`research/results_snapshot.json`](research/results_snapshot.json)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Data: Yahoo adjusted closes via `yfinance`, cached under `data/daily/` (gitignored).

## Pipeline (helpers)

- `helpers/data_utils.py` — load / month-end / returns
- `helpers/signal_utils.py` — trailing return, sign, multi-horizon score, agreement buckets
- `helpers/backtest_utils.py` — lag positions, stats, turnover, costs, EW portfolio, vol target
- `helpers/plot_utils.py` — diagnostics and cumulative charts
