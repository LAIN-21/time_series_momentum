import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers.backtest_utils import cumulative_returns, drawdown

def shade_positions(ax, position: pd.Series, long_alpha: float = 0.10, short_alpha: float = 0.10):
    """
    Shade positive and negative position regimes on a matplotlib axis.

    Positive positions are shaded green.
    Negative positions are shaded red.
    Flat periods are left unshaded.
    """
    pos = position.dropna()
    if pos.empty:
        return

    regime = np.sign(pos)
    start = regime.index[0]
    current = regime.iloc[0]

    for i in range(1, len(regime)):
        if regime.iloc[i] != current:
            end = regime.index[i]

            if current > 0:
                ax.axvspan(start, end, alpha=long_alpha, color="green")
            elif current < 0:
                ax.axvspan(start, end, alpha=short_alpha, color="red")

            start = regime.index[i]
            current = regime.iloc[i]

    end = regime.index[-1]
    if current > 0:
        ax.axvspan(start, end, alpha=long_alpha, color="green")
    elif current < 0:
        ax.axvspan(start, end, alpha=short_alpha, color="red")

def plot_tsmom_diagnostics(
    out: pd.DataFrame,
    ticker: str,
    trailing_col: str = "trailing_12m",
    position_col: str = "position",
    price_col: str = "month_end_price",
    asset_return_col: str = "asset_return",
    strategy_return_col: str = "strategy_return",
    figsize: tuple = (12, 16)
):
    asset_cum = cumulative_returns(out[asset_return_col], name="asset_cum")
    strategy_cum = cumulative_returns(out[strategy_return_col], name="strategy_cum")
    strategy_dd = drawdown(strategy_cum)

    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)

    axes[0].plot(out.index, out[price_col], label=f"{ticker} month-end price")
    shade_positions(axes[0], out[position_col])
    axes[0].set_title(f"{ticker} month-end price (green = long, red = short)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(out.index, out[trailing_col], label=trailing_col.replace("_", " "))
    axes[1].axhline(0, color="black", linewidth=1, alpha=0.6)
    axes[1].set_title(trailing_col.replace("_", " ").title())
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].step(out.index, out[position_col], where="post", label="Position")
    axes[2].axhline(0, color="black", linewidth=1, alpha=0.6)
    axes[2].set_title("Position")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    pos_abs_max = np.nanmax(np.abs(out[position_col].dropna()))
    axes[2].set_ylim(-max(1.2, pos_abs_max + 0.2), max(1.2, pos_abs_max + 0.2))

    axes[3].plot(asset_cum.index, asset_cum.values, label=f"{ticker} buy & hold")
    axes[3].plot(strategy_cum.index, strategy_cum.values, label="TSMOM strategy")
    axes[3].set_title("Cumulative return comparison")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    axes[4].plot(strategy_dd.index, strategy_dd.values, label="TSMOM drawdown")
    axes[4].fill_between(strategy_dd.index, strategy_dd.values, 0, alpha=0.2)
    axes[4].set_title("TSMOM drawdown")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    plt.tight_layout()
    plt.show()