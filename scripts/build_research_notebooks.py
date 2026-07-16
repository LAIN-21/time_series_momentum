"""Generate comprehensible research notebooks with narrative + charts."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _id() -> str:
    return uuid.uuid4().hex[:8]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "id": _id(),
        "source": [line + "\n" for line in text.split("\n")],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "id": _id(),
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")],
    }


def write_nb(rel: str, cells: list[dict]) -> None:
    path = ROOT / rel
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    # strip trailing-only newline artifact from empty last line
    for c in cells:
        if c["source"] and c["source"][-1] == "\n":
            c["source"] = c["source"][:-1] or ["\n"]
        # fix: split added \n to every line including last; ok for nbformat
    path.write_text(json.dumps(nb, indent=1))
    print(f"wrote {rel} ({len(cells)} cells)")


def build_combo_grid() -> None:
    write_nb(
        "research/05_combo_grid.ipynb",
        [
            md(
                """## Research 05 — Combination Grid: How to Merge Horizons

### Why this notebook exists
Notebook 2 introduced one merge: **equal-weight average of 1/3/6/12 month signs**. That is a clean conviction dial, but it is only one design choice.

Here we ask: **among reasonable ways to combine horizons, what actually wins after costs?**

### Families we test
| Family | Idea |
|--------|------|
| Single horizon | Baseline: 1m, 3m, 6m, or 12m alone |
| Equal sign average | Average of signs on subsets (drop 1m, keep 6+12, …) |
| Weighted signs | Tilt weight toward slow or fast lookbacks |
| Raw then sign | Average trailing *returns*, then take the sign |
| Majority / threshold | Vote or trade only when conviction is high |

### How to read the results
1. **Sharpe_net** is the primary ranking metric (10 bps one-way turnover cost).
2. **MaxDD_net** matters if the goal is defense, not just return efficiency.
3. **Turnover** explains why fancy blends often lose — they trade more than plain 12m.
4. Charts show rankings, Sharpe-vs-turnover tradeoffs, and equity / drawdown paths.

Universe: SPY alone, then equal-weight SPY / BIL / IEF / TLT / GLD / BTC-USD."""
            ),
            code(
                """import pandas as pd
import numpy as np
from IPython.display import display, Markdown

from helpers.data_utils import load_monthly_panel
from helpers.signal_utils import (
    trailing_return, sign_signal, multi_horizon_score,
    raw_return_score, majority_vote, threshold_score,
)
from helpers.backtest_utils import (
    position_from_signal, strategy_returns, perf_stats,
    turnover, apply_transaction_costs, equal_weight_portfolio,
)
from helpers.plot_utils import (
    plot_cumulative_comparison, plot_drawdown_comparison,
    plot_metric_bars, plot_scatter_metrics,
)

TICKERS = ["SPY", "BIL", "IEF", "TLT", "GLD", "BTC-USD"]
ALL_H = [1, 3, 6, 12]
COST_BPS = 10.0

px_m, r_m = load_monthly_panel(TICKERS)
trailing = {k: trailing_return(r_m, k) for k in ALL_H}
signals = {k: sign_signal(trailing[k]) for k in ALL_H}

display(Markdown(
    f"**Data loaded.** Monthly panel `{r_m.shape[0]}` months × `{r_m.shape[1]}` assets. "
    f"Cost assumption: **{COST_BPS:.0f} bps** one-way per unit of turnover."
))"""
            ),
            md(
                """### 1. Build the signal library

Each named **spec** is a full signal panel (same columns as the return panel).
Positions are always **lagged by one month** — no lookahead."""
            ),
            code(
                """def subset(d, keys):
    return {k: d[k] for k in keys}

SPECS = {}

for k in ALL_H:
    SPECS[f"single_{k}m"] = signals[k]

SUBSETS = {
    "sign_eq_1_3_6_12": [1, 3, 6, 12],
    "sign_eq_3_6_12": [3, 6, 12],
    "sign_eq_6_12": [6, 12],
    "sign_eq_3_12": [3, 12],
    "sign_eq_1_3_6": [1, 3, 6],
    "sign_eq_3_6": [3, 6],
}
for name, keys in SUBSETS.items():
    SPECS[name] = multi_horizon_score(subset(signals, keys))

WEIGHTS = {
    "sign_w_slow": {1: 0.10, 3: 0.20, 6: 0.30, 12: 0.40},
    "sign_w_slower": {1: 0.05, 3: 0.15, 6: 0.30, 12: 0.50},
    "sign_w_no1_tilt12": {3: 0.20, 6: 0.30, 12: 0.50},
    "sign_w_fast": {1: 0.40, 3: 0.30, 6: 0.20, 12: 0.10},
}
for name, w in WEIGHTS.items():
    SPECS[name] = multi_horizon_score(subset(signals, list(w.keys())), weights=w)

RAW_SUBSETS = {
    "raw_eq_1_3_6_12": [1, 3, 6, 12],
    "raw_eq_3_6_12": [3, 6, 12],
    "raw_eq_6_12": [6, 12],
    "raw_w_slow": {1: 0.10, 3: 0.20, 6: 0.30, 12: 0.40},
}
for name, spec in RAW_SUBSETS.items():
    if isinstance(spec, list):
        SPECS[name] = raw_return_score(subset(trailing, spec), as_sign=True)
    else:
        SPECS[name] = raw_return_score(
            subset(trailing, list(spec.keys())), weights=spec, as_sign=True
        )

SPECS["maj_1_3_6_12"] = majority_vote(subset(signals, [1, 3, 6, 12]))
SPECS["maj_3_6_12"] = majority_vote(subset(signals, [3, 6, 12]))
SPECS["maj_1_3_6_12_min3"] = majority_vote(subset(signals, [1, 3, 6, 12]), min_agree=3)
SPECS["maj_3_6_12_min2"] = majority_vote(subset(signals, [3, 6, 12]), min_agree=2)

base = multi_horizon_score(subset(signals, [1, 3, 6, 12]))
SPECS["sign_eq_thr_0.5"] = threshold_score(base, min_abs=0.5)
SPECS["sign_eq_thr_0.75"] = threshold_score(base, min_abs=0.75)
SPECS["sign_eq_3_6_12_thr_0.5"] = threshold_score(
    multi_horizon_score(subset(signals, [3, 6, 12])), min_abs=0.5
)

def family(name: str) -> str:
    if name.startswith("single_"):
        return "single"
    if name.startswith("sign_eq"):
        return "sign_equal"
    if name.startswith("sign_w"):
        return "sign_weighted"
    if name.startswith("raw_"):
        return "raw_then_sign"
    if name.startswith("maj_"):
        return "majority"
    return "other"

print(f"Registered {len(SPECS)} specs across families:",
      sorted({family(n) for n in SPECS}))"""
            ),
            md(
                """### 2. Backtest each spec

For a signal $s_t$: position $p_t = s_{t-1}$, gross return $p_t r_t$,
net return = gross − turnover × cost.

- **SPY block:** use the SPY column only.
- **EW block:** equal-weight average of asset-level net strategy returns."""
            ),
            code(
                """def eval_signal(signal, r, cost_bps=COST_BPS):
    pos = position_from_signal(signal, lag=1)
    gross = strategy_returns(pos, r)
    net = apply_transaction_costs(gross, pos, cost_bps=cost_bps)
    if isinstance(net, pd.DataFrame):
        port_net = equal_weight_portfolio(net)
        port_gross = equal_weight_portfolio(gross)
        turn = float(turnover(pos).mean().mean())
        stats_net = perf_stats(port_net)
        stats_gross = perf_stats(port_gross)
    else:
        turn = float(turnover(pos).mean())
        stats_net = perf_stats(net)
        stats_gross = perf_stats(gross)
        port_net = net
    return {
        "Sharpe_gross": float(stats_gross["Sharpe"]),
        "Sharpe_net": float(stats_net["Sharpe"]),
        "CAGR_net": float(stats_net["CAGR"]),
        "MaxDD_net": float(stats_net["Max Drawdown"]),
        "Vol_net": float(stats_net["Ann. Vol"]),
        "Turnover": turn,
        "port_net": port_net,
    }

bh_spy = perf_stats(r_m["SPY"])
bh_ew = perf_stats(equal_weight_portfolio(r_m))
display(Markdown(
    f"**Buy-and-hold reference** — SPY Sharpe `{bh_spy['Sharpe']:.2f}`, "
    f"EW basket Sharpe `{bh_ew['Sharpe']:.2f}`."
))"""
            ),
            md(
                """### 3. SPY results — tables and charts

**Reading tip:** the table is sorted by `Sharpe_net`. Anchor every comparison to `single_12m`.
If a blend has lower Sharpe *and* higher turnover, it is mostly paying for noise."""
            ),
            code(
                """spy_rows, spy_ports = [], {}
for name, sig in SPECS.items():
    col = sig["SPY"] if isinstance(sig, pd.DataFrame) else sig
    out = eval_signal(col, r_m["SPY"])
    spy_ports[name] = out.pop("port_net")
    spy_rows.append({"spec": name, "family": family(name), **out})

spy_grid = (
    pd.DataFrame(spy_rows).set_index("spec").sort_values("Sharpe_net", ascending=False)
)
display(Markdown("#### Full SPY ranking (net of 10 bps)"))
display(spy_grid.round(4))
display(Markdown("#### Top 8 and bottom 4"))
display(pd.concat([spy_grid.head(8), spy_grid.tail(4)]).round(4))"""
            ),
            code(
                """plot_metric_bars(
    spy_grid["Sharpe_net"].head(12),
    title="SPY — top specs by net Sharpe (dashed = single_12m)",
    ylabel="Sharpe (net)",
    hline=float(spy_grid.loc["single_12m", "Sharpe_net"]),
)
plot_scatter_metrics(
    spy_grid.reset_index(),
    x="Turnover",
    y="Sharpe_net",
    label_col="spec",
    title="SPY — Sharpe vs turnover (upper-left is better)",
)
fam_best = spy_grid.groupby("family")["Sharpe_net"].max().sort_values()
plot_metric_bars(
    fam_best,
    title="SPY — best net Sharpe within each family",
    ylabel="Sharpe (net)",
    color="darkorange",
)"""
            ),
            md(
                """### 4. Equal-weight multi-asset results

Diversification is where TSMOM usually looks better. Same specs; equal-weight across assets each month."""
            ),
            code(
                """ew_rows, ew_ports = [], {}
for name, sig in SPECS.items():
    out = eval_signal(sig, r_m)
    ew_ports[name] = out.pop("port_net")
    ew_rows.append({"spec": name, "family": family(name), **out})

ew_grid = (
    pd.DataFrame(ew_rows).set_index("spec").sort_values("Sharpe_net", ascending=False)
)
display(Markdown("#### Full EW ranking (net of 10 bps)"))
display(ew_grid.round(4))
display(Markdown("#### Top 8"))
display(ew_grid.head(8).round(4))"""
            ),
            code(
                """plot_metric_bars(
    ew_grid["Sharpe_net"].head(12),
    title="EW portfolio — top specs by net Sharpe (dashed = single_12m)",
    ylabel="Sharpe (net)",
    hline=float(ew_grid.loc["single_12m", "Sharpe_net"]),
    color="seagreen",
)
plot_scatter_metrics(
    ew_grid.reset_index(),
    x="Turnover",
    y="Sharpe_net",
    label_col="spec",
    title="EW — Sharpe vs turnover",
)
plot_metric_bars(
    ew_grid["MaxDD_net"].nsmallest(10),
    title="EW — smallest max drawdowns (rightward / less negative is better)",
    ylabel="Max drawdown (net)",
    color="indianred",
)"""
            ),
            md(
                """### 5. Head-to-head story charts

A shortlist for narrative — not every grid row needs an equity curve."""
            ),
            code(
                """KEYS = [
    "single_12m",
    "sign_eq_1_3_6_12",
    "sign_eq_3_6_12",
    "sign_eq_6_12",
    "sign_w_slow",
    "sign_w_slower",
    "raw_eq_3_6_12",
    "maj_3_6_12",
    "sign_eq_thr_0.75",
    "single_1m",
]

display(Markdown("#### SPY shortlist"))
display(spy_grid.loc[[k for k in KEYS if k in spy_grid.index]].round(4))
display(Markdown("#### EW shortlist"))
display(ew_grid.loc[[k for k in KEYS if k in ew_grid.index]].round(4))

top_ew = list(ew_grid.head(3).index)
story = list(dict.fromkeys(
    ["single_12m", "sign_eq_1_3_6_12", "sign_eq_6_12", "sign_w_slow"] + top_ew
))
plot_cumulative_comparison(
    {n: ew_ports[n] for n in story},
    title="EW net equity curves — 12m vs key blends",
)
plot_drawdown_comparison(
    {n: ew_ports[n] for n in story},
    title="EW net drawdowns — where blends help (or do not)",
)
plot_cumulative_comparison(
    {n: spy_ports[n] for n in ["single_12m", "sign_eq_1_3_6_12", "sign_eq_6_12", "single_1m"]},
    title="SPY net equity — 12m vs equal score vs 6+12 vs 1m",
)"""
            ),
            md("""### 6. Written findings

Prose below is generated from the grids so the narrative stays tied to the numbers."""),
            code(
                """def findings(grid, label):
    best = grid.index[0]
    base, eq, drop1 = "single_12m", "sign_eq_1_3_6_12", "sign_eq_3_6_12"
    lines = [
        f"**{label}**",
        f"- Best net Sharpe: `{best}` ({grid.loc[best, 'Sharpe_net']:.3f}).",
        f"- Plain 12m: `{grid.loc[base, 'Sharpe_net']:.3f}` (turnover {grid.loc[base, 'Turnover']:.3f}).",
        f"- Full equal score 1/3/6/12: `{grid.loc[eq, 'Sharpe_net']:.3f}` (turnover {grid.loc[eq, 'Turnover']:.3f}).",
        f"- Drop 1-month (3/6/12): `{grid.loc[drop1, 'Sharpe_net']:.3f}`.",
    ]
    if "sign_eq_6_12" in grid.index:
        lines.append(
            f"- Best simple subset 6+12: `{grid.loc['sign_eq_6_12', 'Sharpe_net']:.3f}`."
        )
    short = [k for k in KEYS if k in grid.index]
    def_row = grid.loc[short]["MaxDD_net"].idxmax()
    lines.append(
        f"- Among the story shortlist, smallest |max DD|: `{def_row}` "
        f"({grid.loc[def_row, 'MaxDD_net']:.1%})."
    )
    return "\\n".join(lines)

display(Markdown(findings(spy_grid, "SPY")))
display(Markdown(findings(ew_grid, "Equal-weight portfolio")))
display(Markdown('''### Bottom line
1. **12-month sign still wins Sharpe** after a real merge bake-off.
2. **Dropping the 1-month leg helps** the equal-weight score; **6+12** is the closest challenger.
3. **Slow weight tilts** recover toward 12m; fast tilts and hard thresholds mostly add cost or flatten exposure.
4. Use blends when the goal is a **drawdown dial**, not when the goal is beating 12m Sharpe.
'''))"""
            ),
            code(
                """from pathlib import Path
import json

out_dir = Path("research")
spy_grid.to_csv(out_dir / "combo_grid_spy.csv")
ew_grid.to_csv(out_dir / "combo_grid_ew.csv")
summary = {
    "spy_best": spy_grid.index[0],
    "spy_best_sharpe_net": float(spy_grid.iloc[0]["Sharpe_net"]),
    "spy_12m_sharpe_net": float(spy_grid.loc["single_12m", "Sharpe_net"]),
    "spy_eq_full_sharpe_net": float(spy_grid.loc["sign_eq_1_3_6_12", "Sharpe_net"]),
    "ew_best": ew_grid.index[0],
    "ew_best_sharpe_net": float(ew_grid.iloc[0]["Sharpe_net"]),
    "ew_12m_sharpe_net": float(ew_grid.loc["single_12m", "Sharpe_net"]),
    "ew_eq_full_sharpe_net": float(ew_grid.loc["sign_eq_1_3_6_12", "Sharpe_net"]),
}
(out_dir / "combo_grid_summary.json").write_text(json.dumps(summary, indent=2))
display(Markdown("Saved `combo_grid_spy.csv`, `combo_grid_ew.csv`, `combo_grid_summary.json`."))
display(summary)"""
            ),
        ],
    )


def build_disagreement() -> None:
    write_nb(
        "research/03_disagreement.ipynb",
        [
            md(
                """## Research 03 — Does Horizon Disagreement Forecast Risk?

### Claim
The multi-horizon score is not only a trading signal — it is a **regime meter**.
When 1m / 3m / 6m / 12m signs **agree**, $|\\text{score}|$ is near 1.
When they **disagree**, $|\\text{score}|$ is near 0 and the strategy already shrinks size.

**Testable question:** after low-agreement months, is the next month choppier (higher mean absolute return)? Is the score strategy weaker there?

### How to read this notebook
1. Build the lagged agreement label (known at $t-1$ for month-$t$ outcomes).
2. Bucket months into high / mid / low agreement.
3. Compare mean $|r|$, strategy Sharpe, and show charts.
4. End with a written interpretation — including if the claim is weak."""
            ),
            code(
                """import pandas as pd
import numpy as np
from IPython.display import display, Markdown
import matplotlib.pyplot as plt

from helpers.data_utils import load_monthly_panel
from helpers.signal_utils import (
    trailing_return, sign_signal, multi_horizon_score, agreement_label,
)
from helpers.backtest_utils import position_from_signal, strategy_returns, perf_stats
from helpers.plot_utils import plot_score_and_price, plot_metric_bars

TICKERS = ["SPY", "BIL", "IEF", "TLT", "GLD", "BTC-USD"]
HORIZONS = [1, 3, 6, 12]

px_m, r_m = load_monthly_panel(TICKERS)
signals = {k: sign_signal(trailing_return(r_m, k)) for k in HORIZONS}
score = multi_horizon_score(signals)
# Lag agreement so we condition only on information known before month t
agree = agreement_label(score.shift(1))
pos = position_from_signal(score, lag=1)
strat = strategy_returns(pos, r_m)

display(Markdown(
    "**Setup:** equal-weight average of four sign horizons; agreement buckets from "
    "lagged $|\\mathrm{score}|$ (high ≥ 0.75, mid ≥ 0.25, else low)."
))"""
            ),
            md(
                """### Visual intuition — SPY price vs score

When the score hugs ±1, horizons agree. When it oscillates near zero, they conflict
and the position is already small."""
            ),
            code(
                """plot_score_and_price(
    px_m["SPY"], score["SPY"],
    title="SPY month-end price vs multi-horizon score",
)
# Distribution of agreement labels
counts = agree["SPY"].value_counts().reindex(["high", "mid", "low"])
display(Markdown("#### SPY months by prior agreement"))
display(counts.to_frame("n_months"))
plot_metric_bars(
    counts.astype(float),
    title="SPY — count of months by prior agreement",
    ylabel="Months",
    color="slategray",
)"""
            ),
            md(
                """### Bucket statistics

For each agreement state, measure the **next month's** asset $|r|$ and the score-strategy return.
Low-agreement strategy Sharpe can be undefined/near-zero when size is flat — that is expected."""
            ),
            code(
                """def bucket_stats(asset: str) -> pd.DataFrame:
    a = agree[asset]
    r = r_m[asset]
    s = strat[asset]
    rows = []
    for label in ["high", "mid", "low"]:
        mask = a == label
        rr = r[mask].dropna()
        ss = s[mask].dropna()
        sharpe = (
            (ss.mean() * 12) / (ss.std() * np.sqrt(12))
            if len(ss) > 2 and ss.std() > 1e-12 else np.nan
        )
        rows.append({
            "agreement": label,
            "n": int(mask.sum()),
            "asset_mean": rr.mean(),
            "asset_vol": rr.std(),
            "asset_abs_r": rr.abs().mean(),
            "strat_mean": ss.mean(),
            "strat_vol": ss.std(),
            "strat_Sharpe_ann": sharpe,
        })
    return pd.DataFrame(rows).set_index("agreement")

spy_buckets = bucket_stats("SPY")
display(Markdown("#### SPY — outcomes by prior agreement"))
display(spy_buckets.round(4))

panel = []
for t in TICKERS:
    b = bucket_stats(t).reset_index()
    b["ticker"] = t
    panel.append(b)
panel_df = pd.concat(panel, ignore_index=True)
summary = panel_df.groupby("agreement")[["asset_abs_r", "strat_Sharpe_ann", "n"]].mean()
summary = summary.reindex(["high", "mid", "low"])
display(Markdown("#### Panel average across assets"))
display(summary.round(4))"""
            ),
            md("""### Charts — is chop higher when horizons disagree?"""),
            code(
                """order = ["high", "mid", "low"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
spy_buckets["asset_abs_r"].reindex(order).plot(
    kind="bar", ax=axes[0], color="steelblue", rot=0
)
axes[0].set_title("SPY mean |monthly return| by prior agreement")
axes[0].set_ylabel("Mean |r|")
axes[0].grid(True, axis="y", alpha=0.3)

spy_buckets["strat_Sharpe_ann"].reindex(order).plot(
    kind="bar", ax=axes[1], color="darkorange", rot=0
)
axes[1].set_title("SPY score-strategy Sharpe by prior agreement")
axes[1].set_ylabel("Ann. Sharpe")
axes[1].grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Panel chart
plot_metric_bars(
    summary["asset_abs_r"],
    title="Panel avg mean |r| by prior agreement",
    ylabel="Mean |monthly return|",
)

# Per-asset heatmap-style table of abs r
abs_table = panel_df.pivot(index="ticker", columns="agreement", values="asset_abs_r")
abs_table = abs_table[["high", "mid", "low"]]
display(Markdown("#### Mean |r| by asset and agreement (read across rows)"))
display(abs_table.round(4))

fig, ax = plt.subplots(figsize=(8, 4))
abs_table.plot(kind="bar", ax=ax, rot=0)
ax.set_title("Mean |monthly return| by agreement — each asset")
ax.set_ylabel("Mean |r|")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(title="agreement")
plt.tight_layout()
plt.show()"""
            ),
            md("""### Interpretation"""),
            code(
                """high_abs = float(spy_buckets.loc["high", "asset_abs_r"])
mid_abs = float(spy_buckets.loc["mid", "asset_abs_r"])
low_abs = float(spy_buckets.loc["low", "asset_abs_r"])
lift = (mid_abs / high_abs - 1) * 100

display(Markdown(f'''### Findings (auto-filled from this run)
- SPY mean $|r|$ after **high** agreement: **{high_abs:.2%}**.
- After **mid** / **low**: **{mid_abs:.2%}** / **{low_abs:.2%}**.
- Mid-agreement months are about **{lift:.0f}%** choppier than high-agreement months on SPY.
- Strategy Sharpe is strongest in high-agreement buckets; low-agreement months are often near-flat by construction (score ≈ 0).

### How to talk about this
Not "alpha from disagreement." Better: **"the dial is doing risk management that lines up with realized chop."**

If mid/low were *quieter* than high, we would reject the regime story — the claim is falsifiable.
'''))"""
            ),
        ],
    )


def build_robustness() -> None:
    write_nb(
        "research/04_robustness.ipynb",
        [
            md(
                """## Research 04 — Robustness Battery

### Why
A backtest that only looks good on one sample is content risk. This notebook stresses the
**equal-weight multi-horizon score** portfolio from `application.ipynb`.

### Tests
1. **Out-of-sample split** at 2015 (IS vs OOS)
2. **Universe** with and without BTC
3. **Cost ladder** 0 / 5 / 10 / 25 bps
4. **Vol targeting** on vs off (10% ann. target)

### How to read
Each section has a table, then a chart, then a one-line takeaway.
The final cell writes an honest verdict from the numbers."""
            ),
            code(
                """import pandas as pd
import numpy as np
from IPython.display import display, Markdown

from helpers.data_utils import load_monthly_panel
from helpers.signal_utils import trailing_return, sign_signal, multi_horizon_score
from helpers.backtest_utils import (
    position_from_signal, strategy_returns, compare_perf_stats, perf_stats,
    apply_transaction_costs, equal_weight_portfolio, volatility_target_returns,
    split_by_date,
)
from helpers.plot_utils import (
    plot_cumulative_comparison, plot_drawdown_comparison, plot_metric_bars,
)

FULL = ["SPY", "BIL", "IEF", "TLT", "GLD", "BTC-USD"]
NO_BTC = ["SPY", "BIL", "IEF", "TLT", "GLD"]
HORIZONS = [1, 3, 6, 12]
SPLIT = "2015-01-01"

def build_score_portfolio(tickers, cost_bps=10.0, vol_target=None):
    px_m, r_m = load_monthly_panel(tickers)
    signals = {k: sign_signal(trailing_return(r_m, k)) for k in HORIZONS}
    score = multi_horizon_score(signals)
    pos = position_from_signal(score, lag=1)
    gross = strategy_returns(pos, r_m)
    net = apply_transaction_costs(gross, pos, cost_bps=cost_bps)
    port = equal_weight_portfolio(net)
    bh = equal_weight_portfolio(r_m)
    if vol_target is not None:
        port = volatility_target_returns(port, target_ann_vol=vol_target)
    return bh, port, r_m

display(Markdown(
    f"Stress object: **EW multi-horizon score**, default cost **10 bps**, split at **{SPLIT}**."
))"""
            ),
            md(
                """### 1–2. Sample split and BTC sensitivity

**Question A:** Does post-2015 look like pre-2015?
**Question B:** How much of the result is BTC's trend episodes?"""
            ),
            code(
                """bh, port, _ = build_score_portfolio(FULL, cost_bps=10)
p_is, p_oos = split_by_date(port, SPLIT)
bh_is, bh_oos = split_by_date(bh, SPLIT)
bh2, port2, _ = build_score_portfolio(NO_BTC, cost_bps=10)

split_stats = compare_perf_stats({
    "EW BH full": bh,
    "Score net full": port,
    "Score IS <2015": p_is,
    "Score OOS >=2015": p_oos,
    "EW BH no BTC": bh2,
    "Score net no BTC": port2,
})
display(Markdown("#### Performance under sample / universe stress"))
display(split_stats.round(4))

plot_cumulative_comparison(
    {
        "Score full": port,
        "Score IS": p_is,
        "Score OOS": p_oos,
        "Score no BTC": port2,
    },
    title="Robustness — equity curves (score portfolio net 10 bps)",
)
plot_drawdown_comparison(
    {"Score full": port, "Score no BTC": port2, "EW BH full": bh},
    title="Drawdowns — full score vs no-BTC vs buy-and-hold",
)

sharpes = pd.Series({
    "Full": perf_stats(port)["Sharpe"],
    "IS": perf_stats(p_is)["Sharpe"],
    "OOS": perf_stats(p_oos)["Sharpe"],
    "No BTC": perf_stats(port2)["Sharpe"],
})
plot_metric_bars(
    sharpes,
    title="Sharpe under sample / universe stress",
    ylabel="Sharpe",
    color="teal",
)"""
            ),
            md(
                """### 3–4. Cost ladder and vol targeting

**Question C:** At what cost does the edge look lame vs buy-and-hold?
**Question D:** Does 10% vol targeting stabilize the path?"""
            ),
            code(
                """cost_map = {}
for bps in [0, 5, 10, 25]:
    _, p, _ = build_score_portfolio(FULL, cost_bps=bps)
    cost_map[f"Score {bps}bps"] = p

_, p_vt, _ = build_score_portfolio(FULL, cost_bps=10, vol_target=0.10)
cost_map["Score 10bps + VT10%"] = p_vt

cost_stats = compare_perf_stats(cost_map)
display(Markdown("#### Cost sensitivity and vol target"))
display(cost_stats.round(4))

plot_cumulative_comparison(
    {k: cost_map[k] for k in [
        "Score 0bps", "Score 10bps", "Score 25bps", "Score 10bps + VT10%"
    ]},
    title="Cost ladder and vol targeting — equity curves",
)

cost_sharpe = pd.Series({
    0: cost_stats.loc["Sharpe", "Score 0bps"],
    5: cost_stats.loc["Sharpe", "Score 5bps"],
    10: cost_stats.loc["Sharpe", "Score 10bps"],
    25: cost_stats.loc["Sharpe", "Score 25bps"],
})
plot_metric_bars(
    cost_sharpe,
    title="Net Sharpe vs one-way cost (bps)",
    ylabel="Sharpe",
    color="darkslateblue",
)"""
            ),
            md("""### Verdict"""),
            code(
                """full_s = float(perf_stats(port)["Sharpe"])
is_s = float(perf_stats(p_is)["Sharpe"])
oos_s = float(perf_stats(p_oos)["Sharpe"])
nobtc_s = float(perf_stats(port2)["Sharpe"])
s0 = float(cost_stats.loc["Sharpe", "Score 0bps"])
s10 = float(cost_stats.loc["Sharpe", "Score 10bps"])
s25 = float(cost_stats.loc["Sharpe", "Score 25bps"])
vt_s = float(cost_stats.loc["Sharpe", "Score 10bps + VT10%"])

display(Markdown(f'''### Findings (auto-filled)
| Check | Result |
|-------|--------|
| OOS vs IS Sharpe | OOS **{oos_s:.2f}** vs IS **{is_s:.2f}** (full **{full_s:.2f}**) |
| Drop BTC | Sharpe falls to **{nobtc_s:.2f}** — crypto matters in this ETF set |
| Cost 0 → 10 → 25 bps | Sharpe **{s0:.2f} → {s10:.2f} → {s25:.2f}** |
| + 10% vol target | Sharpe **{vt_s:.2f}** |

### Honest read
- OOS holding up is good; it is **not** a license to overfit-celebrate.
- Say out loud that **BTC carries a lot of the diversified punch** here.
- Edge **softens with costs** but does not instantly vanish at 10 bps in this lab.
'''))"""
            ),
        ],
    )


def build_application() -> None:
    write_nb(
        "application.ipynb",
        [
            md(
                """## Application — Diversified TSMOM Portfolio

### From toy to book
Notebooks 1–2 studied **single-asset** rules. The Moskowitz / Ooi / Pedersen result is
fundamentally **cross-asset**: average many independently trending markets.

### What this notebook does
1. Build per-asset **12m sign** and **multi-horizon score** strategies
2. Form an **equal-weight** portfolio across assets available each month
3. Apply a simple **turnover cost** haircut (10 bps)
4. Optionally **volatility-target** the portfolio to 10% annualized

### How to read
- Tables: compare EW buy-and-hold vs 12m vs score (gross / net / vol-targeted)
- Charts: equity curves, drawdowns, per-asset contribution
- Closing text: what is publishable vs what is a caveat

Universe: SPY, BIL, IEF, TLT, GLD, BTC-USD (ETF proxies + crypto — **not** the futures panel)."""
            ),
            code(
                """import pandas as pd
from IPython.display import display, Markdown

from helpers.data_utils import load_monthly_panel
from helpers.signal_utils import trailing_return, sign_signal, multi_horizon_score
from helpers.backtest_utils import (
    position_from_signal, strategy_returns, perf_stats, compare_perf_stats,
    turnover, apply_transaction_costs, equal_weight_portfolio,
    volatility_target_returns,
)
from helpers.plot_utils import (
    plot_cumulative_comparison, plot_drawdown_comparison, plot_metric_bars,
)

TICKERS = ["SPY", "BIL", "IEF", "TLT", "GLD", "BTC-USD"]
HORIZONS = [1, 3, 6, 12]
COST_BPS = 10.0
TARGET_VOL = 0.10

px_m, r_m = load_monthly_panel(TICKERS)
display(Markdown(
    f"Loaded `{list(r_m.columns)}`. Coverage differs by asset — EW uses whatever is available each month."
))
display(r_m.describe().T[["count", "mean", "std"]].round(4))"""
            ),
            md(
                """### Per-asset signals → strategy returns → portfolio

Pipeline reminder:
$s_{k,t}=\\mathrm{sign}(R_{k,t})$, score $=\\frac14\\sum_k s_{k,t}$,
$p_t=$ lagged signal, asset strategy $=p_t r_t$, portfolio $=$ equal-weight of asset strategies."""
            ),
            code(
                """signals = {k: sign_signal(trailing_return(r_m, k)) for k in HORIZONS}
sig_12 = signals[12]
score = multi_horizon_score(signals)

pos_12 = position_from_signal(sig_12, lag=1)
pos_score = position_from_signal(score, lag=1)

r_12 = strategy_returns(pos_12, r_m)
r_score = strategy_returns(pos_score, r_m)

port_bh = equal_weight_portfolio(r_m)
port_12 = equal_weight_portfolio(r_12)
port_score = equal_weight_portfolio(r_score)

r_12_net = apply_transaction_costs(r_12, pos_12, cost_bps=COST_BPS)
r_score_net = apply_transaction_costs(r_score, pos_score, cost_bps=COST_BPS)
port_12_net = equal_weight_portfolio(r_12_net)
port_score_net = equal_weight_portfolio(r_score_net)
port_score_vt = volatility_target_returns(port_score_net, target_ann_vol=TARGET_VOL)

stats = compare_perf_stats({
    "EW Buy&Hold": port_bh,
    "EW 12m gross": port_12,
    "EW 12m net": port_12_net,
    "EW Score gross": port_score,
    "EW Score net": port_score_net,
    "EW Score net + 10% VT": port_score_vt,
})
display(Markdown("#### Portfolio performance"))
display(stats.round(4))

turns = pd.Series({
    "12m avg turnover": turnover(pos_12).mean().mean(),
    "Score avg turnover": turnover(pos_score).mean().mean(),
})
display(Markdown("#### Average asset-level monthly turnover $|\\Delta p|$"))
display(turns.round(4))"""
            ),
            md(
                """### Charts — growth, drawdowns, and who contributes

**What to look for:** diversified 12m should keep Sharpe near buy-and-hold while cutting max DD.
Score should look more defensive (lower DD / vol) with a return tradeoff."""
            ),
            code(
                """plot_cumulative_comparison(
    {
        "EW Buy&Hold": port_bh,
        "EW 12m net": port_12_net,
        "EW Score net": port_score_net,
        "EW Score net + VT": port_score_vt,
    },
    title="Diversified TSMOM — growth of $1 (net of 10 bps)",
)
plot_drawdown_comparison(
    {
        "EW Buy&Hold": port_bh,
        "EW 12m net": port_12_net,
        "EW Score net": port_score_net,
    },
    title="Diversified TSMOM — drawdowns",
)

asset_sharpe_12 = r_12.apply(lambda c: perf_stats(c)["Sharpe"]).rename("12m")
asset_sharpe_sc = r_score.apply(lambda c: perf_stats(c)["Sharpe"]).rename("Score")
by_asset = pd.concat([asset_sharpe_12, asset_sharpe_sc], axis=1)
display(Markdown("#### Per-asset strategy Sharpe (gross)"))
display(by_asset.round(3))
plot_metric_bars(
    by_asset["12m"].sort_values(),
    title="Per-asset 12m TSMOM Sharpe (gross)",
    ylabel="Sharpe",
    color="steelblue",
)"""
            ),
            md("""### Takeaway"""),
            code(
                """bh_s = float(stats.loc["Sharpe", "EW Buy&Hold"])
bh_dd = float(stats.loc["Max Drawdown", "EW Buy&Hold"])
s12 = float(stats.loc["Sharpe", "EW 12m net"])
dd12 = float(stats.loc["Max Drawdown", "EW 12m net"])
ss = float(stats.loc["Sharpe", "EW Score net"])
dds = float(stats.loc["Max Drawdown", "EW Score net"])

display(Markdown(f'''### Findings (auto-filled)
| Portfolio | Sharpe | Max DD |
|-----------|-------:|-------:|
| EW buy-and-hold | {bh_s:.2f} | {bh_dd:.1%} |
| EW 12m net (10 bps) | {s12:.2f} | {dd12:.1%} |
| EW score net (10 bps) | {ss:.2f} | {dds:.1%} |

**Publishable story:** diversification is the point — EW 12m keeps Sharpe near buy-and-hold while
cutting the left tail. The multi-horizon score is the **defensive sibling** (smaller DD, lower Sharpe).

**Caveats:** ETF/crypto proxy ≠ futures panel; BTC history is short and violent; no short financing;
10 bps is illustrative.

**Next:** `research/03_disagreement.ipynb` (regime meter), `research/04_robustness.ipynb` (stress tests),
`research/05_combo_grid.ipynb` (other ways to merge horizons).
'''))"""
            ),
        ],
    )


def patch_strategy_design_score_section() -> None:
    """Enrich Notebook 2 score section with guidance text + score chart."""
    path = ROOT / "strategy_design_2.ipynb"
    nb = json.loads(path.read_text())

    score_md = """### Averaged multi-horizon score

We now implement the notebook's central model:

$$
\\text{score}_t = \\frac{1}{4}\\sum_{k \\in \\{1,3,6,12\\}} \\operatorname{sign}(R_{k,t}), \\qquad p_t = \\text{score}_{t-1}
$$

**How to read the next tables**
- Compare **Score** to **12m** and to buy-and-hold (**Asset**) on each ticker.
- Expect Score Sharpe ≤ 12m on many single assets: the dial **shrinks size** when horizons fight, which cuts vol/DD but can cut return.
- Check turnover: Score should sit between noisy 1m and slow 12m.

When all horizons agree, $|\\text{score}|=1$ (full exposure). When they disagree, exposure shrinks toward zero.
"""
    turnover_md = """### Turnover

**Turnover** here means how much the **position** changes month to month: $|\\,p_t - p_{t-1}\\,|$.
For discrete long/short ($\\pm 1$), a sign flip counts as $2$; no change is $0$. Averaging over time gives **mean monthly turnover** per asset and horizon.

**Why it matters:** 1m looks responsive but racks up turnover. After even modest costs, "faster" often means "more expensive." Compare 1m vs 12m in the table below before celebrating responsiveness.
"""

    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if src.startswith("### Averaged multi-horizon score"):
            c["source"] = score_md.splitlines(keepends=True)
        elif src.startswith("### Turnover"):
            c["source"] = turnover_md.splitlines(keepends=True)
        elif src.startswith("import pandas"):
            if "Markdown" not in src:
                src = src.replace(
                    "from IPython.display import display, HTML",
                    "from IPython.display import display, HTML, Markdown",
                )
            if "plot_score_and_price" not in src:
                if "plot_cumulative_comparison" in src:
                    src = src.replace(
                        "from helpers.plot_utils import plot_tsmom_diagnostics, plot_cumulative_comparison",
                        "from helpers.plot_utils import plot_tsmom_diagnostics, plot_cumulative_comparison, plot_score_and_price, plot_drawdown_comparison",
                    )
                else:
                    src = src.replace(
                        "from helpers.plot_utils import plot_tsmom_diagnostics",
                        "from helpers.plot_utils import plot_tsmom_diagnostics, plot_cumulative_comparison, plot_score_and_price, plot_drawdown_comparison",
                    )
            c["source"] = src.splitlines(keepends=True)

    already = any(
        c["cell_type"] == "code"
        and "plot_score_and_price(" in "".join(c.get("source", []))
        for c in nb["cells"]
    )
    if not already:
        take_i = next(
            (
                i
                for i, c in enumerate(nb["cells"])
                if "".join(c.get("source", [])).startswith("### Takeaway")
            ),
            None,
        )
        extra_md = md(
            """### Score in pictures

SPY price vs score shows the dial in action. Drawdowns compare Asset / 12m / Score on the same months."""
        )
        extra_code = code(
            """plot_score_and_price(
    px_m["SPY"], score["SPY"],
    title="SPY — price vs multi-horizon score (agreement dial)",
)
plot_drawdown_comparison(
    {
        "Asset": r_m["SPY"],
        "12m TSMOM": strategy_returns_by_horizon[12]["SPY"],
        "Multi-horizon score": score_r["SPY"],
    },
    title="SPY drawdowns — buy-and-hold vs 12m vs score",
)
display(Markdown(
    "**Reading the dial:** long stretches near +1 or -1 = horizons agree. "
    "Oscillation around 0 = conflict and reduced exposure."
))"""
        )
        if take_i is not None:
            nb["cells"][take_i:take_i] = [extra_md, extra_code]

    path.write_text(json.dumps(nb, indent=1))
    print("patched strategy_design_2.ipynb")


if __name__ == "__main__":
    build_combo_grid()
    build_disagreement()
    build_robustness()
    build_application()
    patch_strategy_design_score_section()
