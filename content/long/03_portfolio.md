# Long — From SPY toy to a diversified TSMOM book

## Thesis
The academic TSMOM result is **cross-asset**. One ETF understates it: you need many independently trending markets so that whipsaws cancel.

## Lab universe
Equal-weight across SPY, BIL, IEF, TLT, GLD, BTC-USD. ETF/crypto proxies — not the futures panel — but good enough to show the mechanic.

## Construction
1. Per asset: 12m sign **or** multi-horizon score.  
2. Strategy return = lagged position × asset return.  
3. Portfolio = equal-weight of available assets each month.  
4. Subtract turnover × 10 bps.  
5. Optional: vol-target portfolio to 10% ann. vol.

## Snapshot
EW buy-and-hold: Sharpe ~0.94, max DD ~−39%.  
EW 12m net: Sharpe ~0.90, max DD ~−23%.  
EW Score net: Sharpe ~0.74, max DD ~−18%.  
Score + VT: Sharpe ~0.84, CAGR closer to BH.

**Punchline for X:** diversified 12m keeps Sharpe, cuts the left tail roughly in half. Score is the defensive sibling.

## Caveats worth saying out loud
BTC’s short, violent history boosts the panel. No financing on shorts. 10 bps is a cartoon of real costs. Still: the *direction* of the result matches the literature’s spirit.

## CTA
`application.ipynb`. Robustness thread next.
