# Long-Only Equity Momentum — Cross-Sectional & Time-Series

Two long-only S&P 500 momentum strategies, built, IC-tested, vol-targeted,
cost-aware, and validated out-of-sample with walk-forward analysis.

* **`momentum_strategies.py`** — the central module: *all* classes and functions
  (data layer, signals, IC, cost models, backtester, walk-forward validator,
  metrics, dashboards).
* **`momentum_research.ipynb`** — runs the module end-to-end and shows results
  (already executed; outputs embedded).
* **`results/`** — saved `figures/`, `metrics/` (CSV + JSON), `figure_data/`
  (the data behind every chart panel), and `weights/` (current + full-history
  portfolio weights).

## Data

iShares Core S&P 500 ETF (IVV) **monthly holdings snapshots** from
[riazarbi/sp500-scraper](https://github.com/riazarbi/sp500-scraper). Each
snapshot lists every holding with a market `price`, so stacking ~20 years of
snapshots (Oct-2006 → Jun-2026, ~502 names/month) yields a **point-in-time price
panel**. Using the *actual* historical constituents makes the universe
**survivorship-bias-free**. Only `asset_class == "Equity"` rows are kept; the two
supplementary scraper sources (`tidyquant`, `wikipedia`) are not needed.

```python
# one-time download (≈120 MB) into data/ishares/...
ms.DataLoader("data").download()
md = ms.DataLoader("data").load()     # parses, cleans splits, caches to data/ishares_panels_ME.pkl
```

> Prices are **un-adjusted**, so splits are detected via the `shares` column (a
> clean share-count jump with an offsetting price move) and de-split. Returns are
> **price-return only** (no dividends → ~1.5–2%/yr understated vs total return).

## Run it

```bash
pip install pandas numpy scipy matplotlib           # (jupyter to view/edit)
jupyter nbconvert --to notebook --execute --inplace momentum_research.ipynb
# …or just open momentum_research.ipynb and Run All
```

## The two strategies (both long-only, monthly rebalance, ~15% vol target)

| | Cross-sectional (XS) | Time-series (TS) |
|---|---|---|
| Idea | *relative* strength | *absolute* / trend |
| Selection | top `top_pct` by risk-adj momentum | names with own momentum > 0 |
| Sizing | equal-weight, fully invested | breadth-scaled (cash when few trend up) |
| Risk | vol-targeted to 15% (≤2× leverage) | vol-targeted + automatic de-risking |

Both use **risk-adjusted momentum** (trailing 12-1 return ÷ trailing vol).

## Proving the IC

Raw price momentum has *weak* cross-sectional IC in large-cap S&P names (mean
rank-IC ≈ 0.010, t ≈ 0.8) — a well-documented post-2008 phenomenon.
**Risk-adjusting roughly doubles it** (IC ≈ 0.015, t ≈ 1.2 at 1m), it peaks at the
3–6 month horizon (t ≈ 1.8), and a **blended 3/6/12m signal is significant at 6m
(t ≈ 2.1)**. The decile sort is near-monotonic (losers ≈ 5.7%/yr → winners ≈ 11%),
with extreme winners reverting — the edge is mostly *avoiding losers*. See
`results/figures/ic_analysis.png`.

## Headline results (net of realistic costs)

| | XS full | TS full | **XS OOS** | **TS OOS** | Benchmark |
|---|---|---|---|---|---|
| Sharpe | 0.36 | 0.47 | **0.68** | **0.63** | 0.60 |
| Ann. vol | 16.0% | 13.6% | 15.2% | 13.4% | 15.6% |
| CAGR | 4.6% | 5.6% | 9.5% | 7.9% | 8.5% |
| Max drawdown | −52% | −28% | −24% | −24% | −53% |
| Calmar | 0.09 | 0.20 | 0.40 | 0.33 | 0.16 |

*Net of the realistic cost model (§3 below: FX 15bps gross + data-estimated spread
+ square-root impact, $100M AUM). OOS = stitched walk-forward out-of-sample
(adaptive params; period ~2011→ as the 2008–09 crash sits in the first training
window).* The validated strategies still beat the cap-weighted index on
risk-adjusted return, mainly through **drawdown control** — the TS book de-risks
into cash when trend breadth collapses. (On a held USD balance / net-FX, the
full-sample XS Sharpe recovers to ~0.41.)

Full metric set per strategy (Sharpe, Sortino, Calmar, max drawdown, win rate,
profit factor, avg win/loss, CAGR, vol, skewness, kurtosis, leverage, turnover,
transaction cost) — for the full sample, the **turnover-penalty-enhanced**
variant, and the **walk-forward OOS** — is in `results/metrics/strategy_metrics.csv`.

## Portfolio weights output

The actual held book is tracked every period. `save_weights(res, prefix)` writes
the **current (latest-rebalance) portfolio** (`results/weights/{strat}_latest_weights.csv`
— ticker, weight, % of invested, sector) and the **full history**
(`..._weights_history.csv`). Weights are post-vol-target, so they sum to the
current gross exposure. `res.current_weights()` returns the live book in code.

## Transaction-cost-aware turnover penalty

Three cost-aware rebalancing controls in `StrategyConfig`, applied in the
sequential backtester:

* `no_trade_band` — skip per-name trades smaller than *X× the average position*
  (auto-scales to book breadth; full entries/exits always clear it).
* `rank_buffer` — XS hysteresis: keep a held name until it falls past the wider
  `top_pct·(1+buffer)` rank (kills the name-churn that dominates momentum turnover).
* `trade_rate` — execute only a fraction of the gap toward target (partial step).

`turnover_penalty_sweep(md, sig, cfg)` quantifies the trade-off. Monthly: a 50%
rank buffer **improves net Sharpe 0.42→0.43 while cutting turnover ~27%**; band +
buffer cuts turnover ~44% at roughly neutral net Sharpe. The benefit grows with
frequency — see below.

## Weekly rebalance (§8 in the notebook)

The scraper switched to **daily** snapshots in late-2022, so weekly rebalancing is
studied on ~2022-11 → 2026-06. `DataLoader(freq="W")` builds a weekly panel (it
auto-restricts to the dense period); the `Backtester(periods_per_year=52)` makes
the whole engine frequency-agnostic. On a fair **same-window** comparison, weekly
XS (Sharpe 1.41, CAGR 20%) beat monthly (0.77, 12%) at ~2× turnover — higher
frequency captured momentum's faster rotations. There the turnover penalty cuts
weekly turnover ~62% **and lifts net Sharpe 1.41→1.49**. *(Short, momentum-friendly
sample — indicative, not comparable to the 20-year base case.)*

## Transaction costs (realistic, data-driven)

`RealisticCostModel` = **FX 15 bps** (per conversion) + **commission 0** +
**half-spread** + **square-root impact**, all per unit of traded notional.
Spread and impact are estimated from the data: half-spread = `0.01 × daily_vol`
(the spread-∝-vol shape is supported by the Roll estimator's ~0.8 cross-sectional
correlation with vol; level calibrated to ~1–3 bps as Roll over-states the level
at daily frequency), and impact is the Almgren square-root model with ADV proxied
from `market_value`. `cost_breakdown(...)` decomposes the annual drag.

For monthly XS at $100M the all-in drag is **~1.3%/yr**, of which **FX is ~83%**
(spread ~9%, impact ~8%). The dominant lever is FX accounting: charged on every
trade (`fx_on="gross"`, default) it costs ~0.06 Sharpe; charged only on net flows
when a **USD cash balance is held** (`fx_on="net"`) it nearly vanishes for a
rotation. Higher-frequency books (weekly/daily) pay proportionally more, so the
turnover penalty is essential there. See `results/figures/cost_model.png`.

## Dashboards (2×3 per strategy)

`results/figures/{xs,ts}_momentum_dashboard.png` — growth of \$1, drawdowns,
rolling Sharpe, per-asset Sharpe distribution, Sharpe-vs-transaction-cost, and
capacity (square-root market-impact model swept over AUM). Underlying data for
each panel is in `results/figure_data/{slug}_panel{1..6}_*.csv`.

## Module API (quick reference)

```python
import momentum_strategies as ms
md   = ms.DataLoader("data").load()                       # or freq="W" for weekly
sig  = ms.build_signal(md, ms.DEFAULT_XS)                 # risk-adjusted momentum
ic   = ms.information_coefficient(sig, md.returns)        # mean IC, t-stat, hit rate
cost = ms.RealisticCostModel(fx_bps=15.0, aum=1e8)        # FX + spread + impact
bt   = ms.Backtester(md, cost)                            # periods_per_year=52 for weekly
res  = bt.run(sig, ms.DEFAULT_XS)
m    = ms.metrics_from_result(res)                        # full metric Series
bd   = ms.cost_breakdown(md, sig, ms.DEFAULT_XS, fx_bps=15)  # fx/spread/impact drag
cur  = ms.save_weights(res, "results/weights/xs_momentum", sectors=md.sectors)  # current book
sweep = ms.turnover_penalty_sweep(md, sig, ms.DEFAULT_XS)  # turnover/Sharpe trade-off
wf   = ms.WalkForwardValidator(bt, ms.DEFAULT_XS, {"top_pct":[0.1,0.2,0.3]}).run()
fig, data = ms.plot_strategy_dashboard(res, md, sig, "XS", data_dir="results/figure_data")
```

Cost models: `LinearCostModel(bps)`, `SquareRootImpactModel` (Almgren, for
capacity), and `RealisticCostModel` (FX + commission + data-estimated spread +
impact; `fx_on="gross"|"net"`).
Turnover controls live in `StrategyConfig`: `no_trade_band`, `rank_buffer`,
`trade_rate`; lag control: `exec_lag`. Frequency presets: `DEFAULT_*_WEEKLY`,
`DEFAULT_*_DAILY`. `lag_sweep(md, cfg, param="exec_lag"|"gap", periods_per_year=…)`
stress-tests implementation lag.

## Implementation & publishing lag

No look-ahead. iShares publishes holdings at ~T+1–2 days; the signal at month-end
*t* uses only data through the **prior** snapshot (the `gap=1` skip), so positions
are formed with a full period of buffer (publishing lag absorbed many times over)
and vol-target leverage is causal (`.shift(1)`). `StrategyConfig.exec_lag` adds
*extra* whole-period lag to stress-test timing: net Sharpe is essentially flat for
the slow XS book (0.42 → 0.41 → 0.45 at +0/1/2 months) and decays only gracefully
for the faster TS/weekly books — the signature of no timing artefact. See the
notebook appendix and `results/figures/implementation_lag_robustness.png`.

**Running daily.** `DataLoader(freq="D")` builds a trading-day panel (snapshot
dates are used directly — no weekend/holiday gap rows; dense data is 2022-11→).
The scraper updates around midday with a close that is **~2 trading sessions old**
(latency stretches around weekends/holidays), so the daily presets
(`DEFAULT_XS_DAILY` / `DEFAULT_TS_DAILY`, lookbacks in trading days) default to
**`exec_lag=2`**. `lag_sweep(md, cfg, param="exec_lag", periods_per_year=252)`
validates it on the real daily data: net Sharpe is near-flat from 0→10 trading
days of extra lag, so the 2–3 day latency is immaterial for this slow signal. The
real cost of going daily is **turnover** (~12×/yr vs ~3.5× monthly) — the
turnover penalty cuts it ~75% and *improves* net Sharpe, so daily is viable only
with the penalty on. Drive the lag off the latest file `date`, not the calendar.
See `results/figures/daily_lag_validation.png`.

## Caveats

Price-return only (no dividends); ETF-holding value used as a liquidity proxy (not
true ADV); risk-free rate assumed 0; leverage up to 2× to hit the vol target.
Framework-level research results — not a live trading recommendation.
