# Long — Time-series momentum in one asset

## Hook
Most “momentum” tweets mean cross-sectional ranking. Time-series momentum is simpler and stranger: **ignore the other assets**. If *this* market’s trailing return is positive, go long; if negative, go short.

## The rule
At month-end \(t\), compute the trailing 12-month return \(r_{t-12:t}\). Signal \(s_t=\mathrm{sign}(\cdot)\). Position during month \(t\) uses \(s_{t-1}\) so there is no lookahead. Strategy return is \(p_t\cdot r_t\).

## What we ran
SPY adjusted closes → month-end → monthly returns → 12m trailing → sign → lag-1 → P&L. Helpers in this repo make that four function calls.

## Result (lab snapshot)
Buy-and-hold CAGR ~10.4%, Sharpe ~0.75, max DD ~−51%.  
TSMOM CAGR ~9.9%, Sharpe ~0.71, max DD ~−36%.

You keep most of the equity risk premium and cut a large chunk of the worst equity drawdowns — because you are short (or flat in other variants) when the prior year was negative.

## Why drawdowns shrink
Equity disasters cluster after weak trailing years. The sign rule flips you before the full multi-year scar finishes. It is not magic timing; it is a coarse filter correlated with crisis regimes.

## What this is not
No costs, no borrow, no futures roll. One ticker cannot reproduce Moskowitz–Ooi–Pedersen. The diversified notebook is the next chapter.

## CTA
Code: `intro_1.ipynb`. Next thread: why a 1-month lookback usually fails.
