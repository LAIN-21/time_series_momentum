"""Recompute research/results_snapshot.json from cached Yahoo data."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from helpers.backtest_utils import (
    apply_transaction_costs,
    compare_perf_stats,
    equal_weight_portfolio,
    perf_stats,
    position_from_signal,
    split_by_date,
    strategy_returns,
    turnover,
    volatility_target_returns,
)
from helpers.data_utils import load_monthly_panel
from helpers.signal_utils import (
    agreement_label,
    multi_horizon_score,
    sign_signal,
    trailing_return,
)

FULL = ["SPY", "BIL", "IEF", "TLT", "GLD", "BTC-USD"]
NO_BTC = ["SPY", "BIL", "IEF", "TLT", "GLD"]
HORIZONS = [1, 3, 6, 12]


def build_score_port(tickers, bps=10.0, vt=None):
    _, r = load_monthly_panel(tickers)
    sigs = {k: sign_signal(trailing_return(r, k)) for k in HORIZONS}
    score = multi_horizon_score(sigs)
    pos = position_from_signal(score, lag=1)
    net = apply_transaction_costs(strategy_returns(pos, r), pos, bps)
    port = equal_weight_portfolio(net)
    if vt is not None:
        port = volatility_target_returns(port, vt)
    return equal_weight_portfolio(r), port


def main():
    _, r_m = load_monthly_panel(FULL)
    signals = {k: sign_signal(trailing_return(r_m, k)) for k in HORIZONS}
    score = multi_horizon_score(signals)
    pos_12 = position_from_signal(signals[12], lag=1)
    pos_sc = position_from_signal(score, lag=1)
    r_12 = strategy_returns(pos_12, r_m)
    r_sc = strategy_returns(pos_sc, r_m)

    spy_stats = compare_perf_stats({"Asset": r_m["SPY"], "12m": r_12["SPY"]})
    port_bh = equal_weight_portfolio(r_m)
    port_12 = equal_weight_portfolio(apply_transaction_costs(r_12, pos_12, 10))
    port_sc = equal_weight_portfolio(apply_transaction_costs(r_sc, pos_sc, 10))
    port_vt = volatility_target_returns(port_sc, 0.10)
    ew = compare_perf_stats(
        {"BH": port_bh, "12m_net": port_12, "Score_net": port_sc, "Score_VT": port_vt}
    )

    agree = agreement_label(score.shift(1))
    disagree = {}
    for lab in ["high", "mid", "low"]:
        mask = agree["SPY"] == lab
        rr = r_m["SPY"][mask].dropna()
        ss = r_sc["SPY"][mask].dropna()
        sharpe = (
            (ss.mean() * 12) / (ss.std() * np.sqrt(12))
            if len(ss) > 2 and ss.std() > 1e-12
            else float("nan")
        )
        disagree[lab] = {
            "n": int(mask.sum()),
            "abs_r": float(rr.abs().mean()),
            "strat_sharpe": float(sharpe) if sharpe == sharpe else None,
        }

    _, p = build_score_port(FULL, 10)
    p_is, p_oos = split_by_date(p, "2015-01-01")
    _, p2 = build_score_port(NO_BTC, 10)
    costs = {}
    for bps in [0, 5, 10, 25]:
        _, pp = build_score_port(FULL, bps)
        costs[str(bps)] = {
            "Sharpe": float(perf_stats(pp)["Sharpe"]),
            "CAGR": float(perf_stats(pp)["CAGR"]),
        }

    out = {
        "spy_cagr_asset": float(spy_stats.loc["CAGR", "Asset"]),
        "spy_cagr_12m": float(spy_stats.loc["CAGR", "12m"]),
        "spy_sharpe_asset": float(spy_stats.loc["Sharpe", "Asset"]),
        "spy_sharpe_12m": float(spy_stats.loc["Sharpe", "12m"]),
        "spy_dd_asset": float(spy_stats.loc["Max Drawdown", "Asset"]),
        "spy_dd_12m": float(spy_stats.loc["Max Drawdown", "12m"]),
        "spy_turn_12m": float(turnover(pos_12["SPY"]).mean()),
        "spy_turn_1m": float(turnover(position_from_signal(signals[1]["SPY"], 1)).mean()),
        "spy_turn_score": float(turnover(pos_sc["SPY"]).mean()),
        "spy_score_sharpe": float(perf_stats(r_sc["SPY"])["Sharpe"]),
        "ew": {c: {m: float(ew.loc[m, c]) for m in ew.index} for c in ew.columns},
        "disagree": disagree,
        "robust_sharpe": {
            "full": float(perf_stats(p)["Sharpe"]),
            "IS": float(perf_stats(p_is)["Sharpe"]),
            "OOS": float(perf_stats(p_oos)["Sharpe"]),
            "noBTC": float(perf_stats(p2)["Sharpe"]),
        },
        "costs": costs,
    }
    dest = ROOT / "research" / "results_snapshot.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
