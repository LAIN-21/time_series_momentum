import pandas as pd
import numpy as np

def position_from_signal(signal: pd.Series, lag: int = 1) -> pd.Series:
    position = signal.shift(lag)
    position.name = "position"
    return position

def strategy_returns(position: pd.Series, asset_returns: pd.Series) -> pd.Series:
    strat_r = position * asset_returns
    strat_r.name = "strategy_return"
    return strat_r

def position_counts(position: pd.Series) -> pd.Series:
    return position.dropna().value_counts().sort_index()

def position_summary(position: pd.Series) -> pd.Series:
    position = position.dropna()
    return pd.Series({
        "long_frac": (position > 0).mean(),
        "short_frac": (position < 0).mean(),
        "flat_frac": (position == 0).mean()
    })

def cumulative_returns(returns: pd.Series, name: str = "cumulative_return") -> pd.Series:
    cum = (1 + returns).dropna().cumprod()
    cum.name = name
    return cum

def drawdown(cum_returns: pd.Series) -> pd.Series:
    running_max = cum_returns.cummax()
    dd = cum_returns / running_max - 1
    dd.name = "drawdown"
    return dd

def perf_stats(returns: pd.Series, periods_per_year: int = 12) -> pd.Series:
    returns = returns.dropna()

    if len(returns) == 0:
        return pd.Series({
            "CAGR": np.nan,
            "Ann. Mean": np.nan,
            "Ann. Vol": np.nan,
            "Sharpe": np.nan,
            "Max Drawdown": np.nan,
            "Hit Rate": np.nan
        })

    cagr = (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
    ann_mean = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_mean / ann_vol if ann_vol != 0 else np.nan
    max_dd = drawdown(cumulative_returns(returns)).min()
    hit_rate = (returns > 0).mean()

    return pd.Series({
        "CAGR": cagr,
        "Ann. Mean": ann_mean,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Hit Rate": hit_rate
    })

def compare_perf_stats(return_map: dict[str, pd.Series], periods_per_year: int = 12) -> pd.DataFrame:
    stats = {
        name: perf_stats(ret, periods_per_year=periods_per_year)
        for name, ret in return_map.items()
    }
    return pd.DataFrame(stats)