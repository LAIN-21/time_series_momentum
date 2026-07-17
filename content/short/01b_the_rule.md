# Short — The rule in one equation

Time-series momentum is not “rank stocks.”

It’s: trade *this* asset from *its own* past return.

$$s_t = \mathrm{sign}(r_{t-12:t}),\quad r^{TSMOM}_t = s_{t-1}\cdot r_t$$

That’s it. Sign of last year’s return. Lag one month. Done.

Next: why drawdowns shrink without killing CAGR.
