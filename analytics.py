"""analytics.py -- sweeps, per-asset/rolling analytics, dashboards, result I/O."""
from __future__ import annotations
import json, os
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from data import MarketData
from costs import LinearCostModel, SquareRootImpactModel, RealisticCostModel
from signals import (build_signal, information_coefficient, ic_by_horizon,
                     decile_forward_returns)
from backtester import (StrategyConfig, BacktestResult, Backtester,
                        WalkForwardResult, compute_metrics, metrics_from_result)

MONTHS_PER_YEAR = 12


# =====================================================================
# 7. ANALYTICS FOR CHARTS  (each returns the underlying data)
# =====================================================================

def rolling_sharpe(returns: pd.Series, window: int = 24,
                   periods_per_year: int = MONTHS_PER_YEAR) -> pd.Series:
    mu = returns.rolling(window).mean() * periods_per_year
    sd = returns.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    return (mu / sd).rename("rolling_sharpe")


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = (1 + returns.fillna(0)).cumprod()
    return (eq / eq.cummax() - 1).rename("drawdown")


def per_asset_sharpe(res: BacktestResult, min_periods: int = 6) -> pd.DataFrame:
    """Standalone annualised Sharpe of each name's return contribution,
    measured only over the periods the name was actually held. Reveals whether
    performance is broad-based or driven by a handful of names."""
    ppy = res.periods_per_year
    contrib = res.contributions
    held = res.weights.abs() > 1e-9
    rows = []
    for name in contrib.columns:
        c = contrib[name][held[name]].dropna()
        if len(c) < min_periods:
            continue
        vol = c.std(ddof=1) * np.sqrt(ppy)
        rows.append({
            "asset": name,
            "periods_held": int(len(c)),
            "total_contribution": float(c.sum()),
            "mean_per_period": float(c.mean()),
            "sharpe": float(c.mean() * ppy / vol) if vol > 0 else np.nan,
        })
    df = pd.DataFrame(rows).set_index("asset")
    return df.sort_values("total_contribution", ascending=False)


def sharpe_vs_tcost(md: MarketData, signal: pd.DataFrame, cfg: StrategyConfig,
                    cost_bps_grid: Sequence[float] = (0, 2.5, 5, 10, 20, 35, 50, 75, 100),
                    periods_per_year: int = MONTHS_PER_YEAR) -> pd.DataFrame:
    """Net Sharpe and CAGR as the flat per-trade cost is swept up."""
    rows = []
    for bps in cost_bps_grid:
        bt = Backtester(md, LinearCostModel(bps), periods_per_year)
        res = bt.run(signal, cfg)
        m = metrics_from_result(res, "net")
        rows.append({"cost_bps": bps, "net_sharpe": m["Sharpe"],
                     "net_cagr": m["CAGR"], "ann_tcost": m["AnnTCost"]})
    return pd.DataFrame(rows)


def capacity_curve(md: MarketData, signal: pd.DataFrame, cfg: StrategyConfig,
                   aum_grid: Sequence[float] = (1e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10, 5e10, 1e11),
                   periods_per_year: int = MONTHS_PER_YEAR,
                   **impact_kwargs) -> pd.DataFrame:
    """Net Sharpe / CAGR / cost drag as a function of strategy AUM, using the
    square-root market-impact model. The point where net Sharpe rolls over is
    the strategy's practical capacity."""
    rows = []
    for aum in aum_grid:
        bt = Backtester(md, SquareRootImpactModel(aum=aum, **impact_kwargs), periods_per_year)
        res = bt.run(signal, cfg)
        m = metrics_from_result(res, "net")
        rows.append({"aum": aum, "net_sharpe": m["Sharpe"], "net_cagr": m["CAGR"],
                     "ann_tcost": m["AnnTCost"], "gross_sharpe": metrics_from_result(res, "gross")["Sharpe"]})
    return pd.DataFrame(rows)


def turnover_penalty_sweep(
    md: MarketData, signal: pd.DataFrame, cfg: StrategyConfig,
    settings: Optional[Sequence[dict]] = None,
    cost_bps: float = 5.0,
    periods_per_year: int = MONTHS_PER_YEAR,
    cost_model: Optional[object] = None,
) -> pd.DataFrame:
    """Sweep cost-aware rebalancing settings (no-trade band / rank buffer /
    trade rate) and report turnover, cost drag and net Sharpe for each. Shows
    that penalising turnover cuts cost with little (or positive) net impact.
    Pass ``cost_model`` (e.g. a RealisticCostModel) to override the flat bps."""
    if settings is None:
        # no_trade_band is in units of "average positions"
        settings = [
            {"label": "full rebalance", "no_trade_band": 0.0, "rank_buffer": 0.0, "trade_rate": 1.0},
            {"label": "band 0.25x pos", "no_trade_band": 0.25, "rank_buffer": 0.0, "trade_rate": 1.0},
            {"label": "band 0.5x pos", "no_trade_band": 0.5, "rank_buffer": 0.0, "trade_rate": 1.0},
            {"label": "rank buffer 50%", "no_trade_band": 0.0, "rank_buffer": 0.5, "trade_rate": 1.0},
            {"label": "band 0.5x + buffer 50%", "no_trade_band": 0.5, "rank_buffer": 0.5, "trade_rate": 1.0},
            {"label": "trade rate 50%", "no_trade_band": 0.0, "rank_buffer": 0.0, "trade_rate": 0.5},
        ]
    bt = Backtester(md, cost_model or LinearCostModel(cost_bps), periods_per_year)
    rows = []
    for st in settings:
        c = StrategyConfig(**{**cfg.__dict__,
                              **{k: st[k] for k in ("no_trade_band", "rank_buffer", "trade_rate") if k in st}})
        res = bt.run(signal, c)
        mg = metrics_from_result(res, "gross")
        mn = metrics_from_result(res, "net")
        rows.append({
            "setting": st["label"],
            "ann_turnover": mn["AnnTurnover"], "ann_tcost": mn["AnnTCost"],
            "gross_sharpe": mg["Sharpe"], "net_sharpe": mn["Sharpe"],
            "net_cagr": mn["CAGR"], "net_maxDD": mn["MaxDrawdown"],
        })
    return pd.DataFrame(rows)


def lag_sweep(
    md: MarketData, base_cfg: StrategyConfig,
    lags: Sequence[int] = (0, 1, 2, 3, 5, 10),
    param: str = "exec_lag",
    cost_bps: float = 5.0,
    periods_per_year: int = MONTHS_PER_YEAR,
    cost_model: Optional[object] = None,
) -> pd.DataFrame:
    """Sweep an implementation-lag parameter and report performance.

    ``param`` is either ``"exec_lag"`` (act on a signal lagged this many extra
    periods -- models publishing/implementation delay) or ``"gap"`` (the
    momentum skip itself). For a daily panel the lag unit is *trading days*, so
    e.g. ``exec_lag=2`` reproduces "the freshest close I can read is 2 sessions
    old". A flat Sharpe across lags is evidence the edge is not a timing
    artefact."""
    bt = Backtester(md, cost_model or LinearCostModel(cost_bps), periods_per_year)
    rows = []
    for L in lags:
        cfg = StrategyConfig(**{**base_cfg.__dict__, param: int(L)})
        sig = build_signal(md, cfg)        # rebuilt per gap; identical across exec_lag
        m = metrics_from_result(bt.run(sig, cfg))
        rows.append({param: int(L), "sharpe": m["Sharpe"], "cagr": m["CAGR"],
                     "ann_vol": m["Ann.Vol"], "maxDD": m["MaxDrawdown"],
                     "ann_turnover": m["AnnTurnover"]})
    return pd.DataFrame(rows)


def cost_breakdown(md: MarketData, signal: pd.DataFrame, cfg: StrategyConfig,
                   periods_per_year: int = MONTHS_PER_YEAR,
                   **model_kwargs) -> pd.Series:
    """Annualised cost drag (fraction/yr) of a :class:`RealisticCostModel` split
    by component. Costs don't affect the trades, so components are exactly
    additive -- we isolate each by zeroing the others' rates."""
    isolations = {
        "fx":         {"commission_bps": 0, "spread_vol_coef": 0, "impact_coef": 0},
        "commission": {"fx_bps": 0, "spread_vol_coef": 0, "impact_coef": 0},
        "spread":     {"fx_bps": 0, "commission_bps": 0, "impact_coef": 0},
        "impact":     {"fx_bps": 0, "commission_bps": 0, "spread_vol_coef": 0},
    }
    out = {}
    for comp, override in isolations.items():
        m = RealisticCostModel(**{**model_kwargs, **override})
        out[comp] = metrics_from_result(
            Backtester(md, m, periods_per_year).run(signal, cfg))["AnnTCost"]
    out["total"] = float(sum(out.values()))
    return pd.Series(out, name="ann_cost_drag")


# =====================================================================
# 8. DASHBOARD  (2x3 figure + saved underlying data)
# =====================================================================

def plot_strategy_dashboard(
    res: BacktestResult,
    md: MarketData,
    signal: pd.DataFrame,
    title: str,
    fig_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    slug: Optional[str] = None,
):
    """Render the 2x3 dashboard and (optionally) save the figure plus the data
    behind every panel.

    Panels: (1) growth of $1, (2) drawdowns, (3) rolling Sharpe,
    (4) Sharpe by asset, (5) Sharpe degradation vs t-cost, (6) capacity.
    """
    import matplotlib.pyplot as plt

    slug = slug or title.lower().replace(" ", "_")
    cfg = res.config

    # ---- compute panel data ----
    eq_net = res.equity("net")
    eq_gross = res.equity("gross")
    bench_eq = (1 + res.benchmark.reindex(eq_net.index).fillna(0)).cumprod()
    growth = pd.DataFrame({"net": eq_net, "gross": eq_gross, "benchmark": bench_eq})

    ppy = res.periods_per_year
    rs_window = 2 * ppy
    dd = drawdown_series(res.net_returns)
    rs = rolling_sharpe(res.net_returns, rs_window, ppy)
    pa = per_asset_sharpe(res)
    tc = sharpe_vs_tcost(md, signal, cfg, periods_per_year=ppy)
    cap = capacity_curve(md, signal, cfg, periods_per_year=ppy)

    # ---- save underlying data ----
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        growth.to_csv(os.path.join(data_dir, f"{slug}_panel1_growth_of_1.csv"))
        dd.to_frame().to_csv(os.path.join(data_dir, f"{slug}_panel2_drawdowns.csv"))
        rs.to_frame().to_csv(os.path.join(data_dir, f"{slug}_panel3_rolling_sharpe.csv"))
        pa.to_csv(os.path.join(data_dir, f"{slug}_panel4_sharpe_by_asset.csv"))
        tc.to_csv(os.path.join(data_dir, f"{slug}_panel5_sharpe_vs_tcost.csv"), index=False)
        cap.to_csv(os.path.join(data_dir, f"{slug}_panel6_capacity.csv"), index=False)

    # ---- plot ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(growth.index, growth["net"], label="Net", lw=1.8)
    ax.plot(growth.index, growth["gross"], label="Gross", lw=1.0, alpha=0.7)
    ax.plot(growth.index, growth["benchmark"], label="Cap-wt benchmark", lw=1.0, alpha=0.7, color="grey")
    ax.set_yscale("log")
    ax.set_title("Growth of $1 (log scale)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.4)
    ax.set_title(f"Drawdowns (max {dd.min():.1%})")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(rs.index, rs.values, color="navy", lw=1.2)
    ax.axhline(0, color="black", lw=0.6)
    full_sharpe = metrics_from_result(res)["Sharpe"]
    ax.axhline(full_sharpe, color="green", ls="--", lw=1, label=f"full-sample {full_sharpe:.2f}")
    units = "m" if ppy == 12 else ("w" if ppy == 52 else "p")
    ax.set_title(f"Rolling {rs_window}{units} Sharpe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if len(pa):
        ax.hist(pa["sharpe"].dropna(), bins=30, color="steelblue", alpha=0.8)
        ax.axvline(pa["sharpe"].median(), color="red", ls="--",
                   label=f"median {pa['sharpe'].median():.2f}")
        ax.legend(fontsize=8)
    ax.set_title(f"Per-asset Sharpe distribution (n={len(pa)})")
    ax.set_xlabel("annualised Sharpe (held periods)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(tc["cost_bps"], tc["net_sharpe"], marker="o", color="darkorange")
    ax.set_title("Sharpe degradation vs transaction cost")
    ax.set_xlabel("per-trade cost (bps, one-way)")
    ax.set_ylabel("net Sharpe")
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(cap["aum"], cap["net_sharpe"], marker="o", color="purple", label="net")
    ax.plot(cap["aum"], cap["gross_sharpe"], ls="--", color="grey", label="gross")
    ax.set_xscale("log")
    ax.set_title("Capacity: net Sharpe vs AUM")
    ax.set_xlabel("strategy AUM ($)")
    ax.set_ylabel("net Sharpe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if fig_path:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    return fig, {
        "growth": growth, "drawdown": dd, "rolling_sharpe": rs,
        "per_asset_sharpe": pa, "sharpe_vs_tcost": tc, "capacity": cap,
    }


def plot_ic_analysis(
    md: MarketData,
    signals: Dict[str, pd.DataFrame],
    primary: str,
    horizons: Sequence[int] = (1, 3, 6, 12),
    fig_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    slug: str = "ic_analysis",
):
    """Proof-of-signal dashboard: cumulative IC, IC by horizon, decile sort.

    ``signals`` maps a label -> signal panel (e.g. raw vs risk-adjusted vs
    blended). Saves the per-signal IC time series, the horizon table and the
    decile table behind the charts."""
    import matplotlib.pyplot as plt

    ic_objs = {k: information_coefficient(v, md.returns) for k, v in signals.items()}
    horizon_tabs = {k: ic_by_horizon(v, md.returns, horizons) for k, v in signals.items()}
    deciles = decile_forward_returns(signals[primary], md.returns)

    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        ic_df = pd.DataFrame({k: o["ic_series"] for k, o in ic_objs.items()})
        ic_df.to_csv(os.path.join(data_dir, f"{slug}_ic_timeseries.csv"))
        hz = pd.concat({k: t.set_index("horizon_m") for k, t in horizon_tabs.items()})
        hz.to_csv(os.path.join(data_dir, f"{slug}_ic_by_horizon.csv"))
        deciles.to_csv(os.path.join(data_dir, f"{slug}_decile_returns.csv"), index=False)
        pd.DataFrame({k: {kk: o[kk] for kk in
                          ("mean_ic", "std_ic", "ic_ir", "t_stat", "p_value", "hit_rate", "n_months")}
                      for k, o in ic_objs.items()}).to_csv(
            os.path.join(data_dir, f"{slug}_ic_summary.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Information Coefficient -- proof of signal", fontsize=14, fontweight="bold")

    ax = axes[0]
    for k, o in ic_objs.items():
        ax.plot(o["ic_series"].index, o["ic_series"].cumsum(), label=k, lw=1.4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Cumulative monthly rank-IC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    width = 0.8 / max(1, len(horizon_tabs))
    for i, (k, t) in enumerate(horizon_tabs.items()):
        ax.bar(np.arange(len(t)) + i * width, t["t_stat"], width=width, label=k)
    ax.set_xticks(np.arange(len(horizons)) + width * (len(horizon_tabs) - 1) / 2)
    ax.set_xticklabels([f"{h}m" for h in horizons])
    ax.axhline(2.0, color="red", ls="--", lw=1, label="t = 2")
    ax.axhline(-2.0, color="red", ls="--", lw=1)
    ax.set_title("IC t-stat by forward horizon")
    ax.set_xlabel("forward horizon")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.bar(deciles["decile"], deciles["ann_fwd_return"], color="seagreen", alpha=0.85)
    ax.set_title(f"Forward return by signal decile\n({primary})")
    ax.set_xlabel("decile (1=losers, 10=winners)")
    ax.set_ylabel("annualised forward return")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if fig_path:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    return fig, {"ic": ic_objs, "horizon": horizon_tabs, "deciles": deciles}


def plot_walkforward(
    wf: WalkForwardResult,
    full_returns: pd.Series,
    title: str,
    fig_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    slug: str = "walkforward",
):
    """Compare stitched out-of-sample equity to the in-sample (full-period,
    fixed-default) equity, plus per-fold IS vs OOS Sharpe. Saves the fold
    table and the OOS return series."""
    import matplotlib.pyplot as plt

    oos = wf.oos_returns
    full = full_returns.reindex(oos.index)
    eq_oos = (1 + oos.fillna(0)).cumprod()
    eq_full = (1 + full.fillna(0)).cumprod()

    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        wf.fold_table.to_csv(os.path.join(data_dir, f"{slug}_folds.csv"), index=False)
        pd.DataFrame({"oos_net_return": oos, "oos_equity": eq_oos,
                      "fixed_default_equity": eq_full}).to_csv(
            os.path.join(data_dir, f"{slug}_oos_returns.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(eq_oos.index, eq_oos.values, label="Walk-forward OOS (adaptive params)", lw=1.8)
    ax.plot(eq_full.index, eq_full.values, label="Fixed default params", lw=1.2, alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("Growth of $1 -- OOS vs fixed (same window)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ft = wf.fold_table
    x = np.arange(len(ft))
    ax.bar(x - 0.2, ft["is_sharpe"], width=0.4, label="in-sample (train)")
    ax.bar(x + 0.2, ft["oos_sharpe"], width=0.4, label="out-of-sample (test)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Per-fold Sharpe: in-sample vs out-of-sample")
    ax.set_xlabel("walk-forward fold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if fig_path:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    return fig, {"oos_equity": eq_oos, "fold_table": wf.fold_table}


# =====================================================================
# 9. CONVENIENCE ORCHESTRATION
# =====================================================================

def save_metrics_table(metrics: Dict[str, pd.Series], path: str) -> pd.DataFrame:
    """Combine per-strategy metric Series into one table and save CSV+JSON."""
    table = pd.DataFrame(metrics)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_csv(path)
    with open(path.replace(".csv", ".json"), "w") as fh:
        json.dump({k: {kk: (None if pd.isna(vv) else float(vv))
                       for kk, vv in v.items()} for k, v in metrics.items()},
                  fh, indent=2)
    return table


def save_weights(res: BacktestResult, path_prefix: str,
                 sectors: Optional[pd.Series] = None) -> pd.Series:
    """Save the strategy's weights and return the *current* (latest) portfolio.

    Writes ``{prefix}_weights_history.csv`` (full date x ticker held-weight
    panel, zeros dropped to sparse long form) and ``{prefix}_latest_weights.csv``
    (the actionable current book: ticker, weight, % of invested, sector)."""
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)

    # full history (sparse long form keeps the file small and readable)
    hist = res.weights.copy()
    long = hist.stack()
    long = long[long.abs() > 1e-9].rename("weight").reset_index()
    long.columns = ["date", "ticker", "weight"]
    long.to_csv(f"{path_prefix}_weights_history.csv", index=False)

    # current portfolio
    cur = res.current_weights()
    out = cur.rename("weight").to_frame()
    out["pct_of_invested"] = out["weight"] / out["weight"].sum()
    if sectors is not None:
        out["sector"] = sectors.reindex(out.index)
    out.index.name = "ticker"
    out.to_csv(f"{path_prefix}_latest_weights.csv")
    return cur


