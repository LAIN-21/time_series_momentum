import pandas as pd
import numpy as np


def position_from_signal(
    signal: pd.Series | pd.DataFrame, lag: int = 1
) -> pd.Series | pd.DataFrame:
    if lag < 0:
        raise ValueError("lag must be >= 0")
    position = signal.shift(lag)
    if isinstance(position, pd.Series):
        position.name = "position"
    return position


def strategy_returns(
    position: pd.Series | pd.DataFrame,
    asset_returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    position, asset_returns = position.align(asset_returns, join="inner")
    strat_r = position * asset_returns
    if isinstance(strat_r, pd.Series):
        strat_r.name = "strategy_return"
    return strat_r


def position_counts(position: pd.Series) -> pd.Series:
    return position.dropna().value_counts().sort_index()


def position_summary(position: pd.Series) -> pd.Series:
    position = position.dropna()
    return pd.Series({
        "long_frac": (position > 0).mean(),
        "short_frac": (position < 0).mean(),
        "flat_frac": (position == 0).mean(),
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
            "Hit Rate": np.nan,
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
        "Hit Rate": hit_rate,
    })


def compare_perf_stats(
    return_map: dict[str, pd.Series], periods_per_year: int = 12
) -> pd.DataFrame:
    stats = {
        name: perf_stats(ret, periods_per_year=periods_per_year)
        for name, ret in return_map.items()
    }
    return pd.DataFrame(stats)


def turnover(position: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Absolute month-to-month change in position |p_t - p_{t-1}|."""
    return position.diff().abs()


def apply_transaction_costs(
    strategy_r: pd.Series | pd.DataFrame,
    position: pd.Series | pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.Series | pd.DataFrame:
    """
    Subtract a simple turnover cost: turnover * (cost_bps / 10000).

    cost_bps is one-way cost in basis points of notional traded.
    For a full flip from +1 to -1, turnover=2 so cost is 2 * bps.
    """
    costs = turnover(position) * (cost_bps / 10_000.0)
    strategy_r, costs = strategy_r.align(costs, join="inner")
    net = strategy_r - costs
    if isinstance(net, pd.Series):
        net.name = "strategy_return_net"
    return net


def equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight average across assets with non-NaN returns that month."""
    port = returns.mean(axis=1, skipna=True)
    port.name = "portfolio_return"
    return port


def volatility_target_returns(
    returns: pd.Series,
    target_ann_vol: float = 0.10,
    lookback: int = 12,
    periods_per_year: int = 12,
    max_leverage: float = 3.0,
) -> pd.Series:
    """
    Scale lagged trailing volatility so that ex-ante annualized vol ≈ target.

    Scale uses vol estimated through t-1 (shift 1) to avoid lookahead.
    """
    trailing_vol = returns.rolling(lookback).std() * np.sqrt(periods_per_year)
    scale = (target_ann_vol / trailing_vol).shift(1)
    scale = scale.clip(upper=max_leverage)
    scaled = returns * scale
    scaled.name = returns.name or "vol_targeted_return"
    return scaled


def split_by_date(
    returns: pd.Series, split_date: str
) -> tuple[pd.Series, pd.Series]:
    """Split a return series into in-sample / out-of-sample around split_date."""
    split = pd.Timestamp(split_date)
    return returns.loc[returns.index < split], returns.loc[returns.index >= split]
