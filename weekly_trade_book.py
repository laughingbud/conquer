"""Weekly trade book — trend-filtered momentum (cross-sectional XS & time-series TS).

Two modes (pass as CLI arg; default walkforward):

  walkforward  -- TRADE THE VALIDATED PROCESS. Each run, the hyper-parameters
                  (lookback, gap, SMA window, and selection cutoff/threshold) are
                  re-chosen from the most recent `TRAIN` weeks using only past data
                  -- exactly what the walk-forward does. No parameter is fixed with
                  full-sample hindsight.
  static       -- use the FIXED parameters in STATIC_XS / STATIC_TS below (e.g. the
                  95th-percentile XS = top 5%). Fully under your control.

Run weekly AFTER refreshing data (`ms.DataLoader("data").download()`):
    python weekly_trade_book.py            # walk-forward selected params (recommended)
    python weekly_trade_book.py static     # fixed params

Writes results/live/weekly_trend_{xs,ts}_latest_weights.csv and prints both books.

Note: XS = top-quantile, equal-weight, fully invested (relative strength). TS =
hold every above-SMA + positive-momentum name, invested fraction = breadth (cash
when few trend; de-risking). The selection cutoff (percentile) is an XS lever; TS's
analogue is its momentum threshold.
"""
import os
import sys
import numpy as np
import pandas as pd
import momentum_strategies as ms

MODE = (sys.argv[1] if len(sys.argv) > 1 else "walkforward").lower()
assert MODE in ("walkforward", "static"), "mode must be 'walkforward' or 'static'"
PPY, TRAIN, AUM = 52, 104, 1e8                      # weekly; select on the last ~2 years

# structural base (a-priori): weekly 52-4 momentum, ~39-week (~9-month) SMA trend filter
XS_BASE = ms.StrategyConfig(kind="xs", lookback=52, gap=4, sma_window=39, weighting="equal",
                            top_pct=0.20, target_vol=0.20, max_leverage=1.0,
                            vol_window=52, vol_lookback=26, vol_halflife=17.0)
TS_BASE = ms.StrategyConfig(kind="ts", lookback=52, gap=4, sma_window=39, ts_threshold=0.0,
                            target_vol=0.20, max_leverage=1.0,
                            vol_window=52, vol_lookback=26, vol_halflife=17.0)
STATIC_XS = {"top_pct": 0.05}                        # your fixed choice: 95th percentile (top 5%)
STATIC_TS = {"ts_threshold": 0.0}
XS_GRID = {"lookback": [26, 52], "gap": [0, 4], "sma_window": [26, 39], "top_pct": [0.25, 0.20, 0.10, 0.05]}
TS_GRID = {"lookback": [26, 52], "gap": [0, 4], "sma_window": [26, 39], "ts_threshold": [0.0, 0.25, 0.5]}

md = ms.DataLoader("data", freq="W").load(rebuild=True)
bt = ms.Backtester(md, ms.RealisticCostModel(fx_bps=15.0, aum=AUM), periods_per_year=PPY)
os.makedirs("results/live", exist_ok=True)
TUNABLE = ("lookback", "gap", "sma_window", "top_pct", "ts_threshold", "weighting")


def make_book(name, base, static_override, grid):
    if MODE == "static":
        cfg, note = ms.StrategyConfig(**{**base.__dict__, **static_override}), "STATIC (fixed) params"
    else:
        cfg, isval, _ = ms.select_latest_params(bt, base, grid, train_periods=TRAIN)
        note = f"WALK-FORWARD selected on last {TRAIN} wks (in-sample Sharpe {isval:.2f})"
    res = bt.run(ms.build_signal(md, cfg), cfg)
    cur = ms.save_weights(res, f"results/live/weekly_trend_{name}", sectors=md.sectors)
    params = {k: getattr(cfg, k) for k in TUNABLE}
    return cfg, cur, res, note, params


xs_cfg, xs_book, xs_res, xs_note, xs_p = make_book("xs", XS_BASE, STATIC_XS, XS_GRID)
ts_cfg, ts_book, ts_res, ts_note, ts_p = make_book("ts", TS_BASE, STATIC_TS, TS_GRID)
asof = xs_res.weights[(xs_res.weights.abs() > 1e-9).any(axis=1)].index[-1]

print(f"=== Weekly trade book  |  mode={MODE.upper()}  |  as-of {asof.date()} ===\n")
for tag, cur, note, params in [("XS", xs_book, xs_note, xs_p), ("TS", ts_book, ts_note, ts_p)]:
    g = cur.sum()
    print(f"[{tag}] {note}")
    print(f"     params: {params}")
    print(f"     {len(cur)} names, gross {g:.0%} invested ({1-g:.0%} cash)")
    print("     " + ", ".join(f"{t} {w:.1%}" for t, w in cur.head(15).items())
          + (" ..." if len(cur) > 15 else ""))
    print()
print("Saved -> results/live/weekly_trend_xs_latest_weights.csv")
print("Saved -> results/live/weekly_trend_ts_latest_weights.csv")
if MODE == "walkforward":
    print("\nRealistic expectation: see the monthly walk-forward OOS (~0.74 XS / ~0.70 TS).")
    print("The weekly sample (~2022-11+) is too short for a robust OOS estimate.")
