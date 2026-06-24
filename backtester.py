"""backtester.py -- config, portfolio construction, vol targeting, metrics, walk-forward."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from costs import LinearCostModel
from signals import build_signal, momentum_signal

MONTHS_PER_YEAR = 12


# =====================================================================
# 4. BACKTESTER  (portfolio construction + vol targeting + costs)
# =====================================================================

@dataclass
class StrategyConfig:
    kind: str = "xs"                 # "xs" (cross-sectional) or "ts" (time-series)
    lookback: int = 12               # momentum formation window (periods)
    gap: int = 1                     # skip most-recent periods
    risk_adjusted: bool = True       # divide trailing return by trailing vol
    vol_window: Optional[int] = None  # window for the risk-adjustment vol
    blend_lookbacks: Optional[Tuple[int, ...]] = None  # if set, blended signal
    sma_window: Optional[int] = None  # if set, trend-filter to price > SMA(sma_window)
    top_pct: float = 0.20            # XS: fraction of universe held long
    ts_threshold: float = 0.0        # TS: hold names with signal above this
    ts_weighting: str = "breadth"    # TS: "breadth" (cash when few trend) or "selected"
    weighting: str = "equal"         # within-book: "equal" | "signal" (rank-tilt) | "sweet_spot"
    sweet_peak: Optional[float] = None  # sweet_spot: percentile above which to taper
                                     # (down-weight extreme winners); default 1 - top_pct/2
    target_vol: float = 0.20         # annualised vol target
    vol_lookback: int = 6            # min periods for ex-ante vol estimate
    vol_halflife: float = 4.0        # EWMA halflife (periods) for vol estimate
    max_leverage: float = 2.0        # cap on gross leverage
    min_names: int = 10              # don't trade if universe smaller than this
    exec_lag: int = 0                # extra execution lag (periods) on top of the
                                     # built-in `gap` skip -- for publishing/implementation
                                     # lag robustness (signal is lagged this many periods)
    # --- transaction-cost-aware rebalancing (turnover penalty) ---
    no_trade_band: float = 0.0       # skip per-name trades smaller than this MANY average positions
    trade_rate: float = 1.0          # execute only this fraction of the gap to target (0,1]
    rank_buffer: float = 0.0         # XS: keep held names until they fall past top_pct*(1+buffer)


@dataclass
class BacktestResult:
    config: StrategyConfig
    gross_returns: pd.Series         # levered, before costs
    net_returns: pd.Series           # levered, after costs
    base_returns: pd.Series          # unlevered (ideal) book return
    leverage: pd.Series              # realised gross exposure (sum of held weights)
    turnover: pd.Series              # one-way turnover per period
    tcost: pd.Series                 # cost drag per period (fraction)
    weights: pd.DataFrame            # actual held (post-leverage, post-trade) weights
    contributions: pd.DataFrame      # per-name return contribution (gross)
    benchmark: pd.Series
    periods_per_year: int = MONTHS_PER_YEAR

    def equity(self, which: str = "net") -> pd.Series:
        r = getattr(self, f"{which}_returns")
        return (1 + r.fillna(0)).cumprod()

    def current_weights(self, top_n: Optional[int] = None) -> pd.Series:
        """Latest rebalance's held weights (the actionable portfolio)."""
        nonzero = self.weights[(self.weights.abs() > 1e-9).any(axis=1)]
        if nonzero.empty:
            return pd.Series(dtype=float)
        w = nonzero.iloc[-1]
        w = w[w.abs() > 1e-9].sort_values(ascending=False)
        return w.head(top_n) if top_n else w


class Backtester:
    """Run a long-only momentum strategy on :class:`MarketData`.

    ``periods_per_year`` makes the engine frequency-agnostic (12 for monthly,
    52 for weekly): it scales the vol-targeting windows, the per-name vol used
    by the impact model, and all annualisation.
    """

    def __init__(self, md: MarketData, cost_model: Optional[object] = None,
                 periods_per_year: int = MONTHS_PER_YEAR):
        self.md = md
        self.cost_model = cost_model or LinearCostModel(5.0)
        self.periods_per_year = periods_per_year
        ppy = periods_per_year
        # rolling per-name vol (annualised) for the impact model, causal
        self._name_vol = (
            md.returns.rolling(ppy, min_periods=max(2, ppy // 2)).std() * np.sqrt(ppy)
        )

    # ---- name selection (shared by ideal book and sequential pass) ----
    def _select_names(self, s: pd.Series, cfg: StrategyConfig,
                      held: Optional[set]) -> Tuple[pd.Series, set]:
        """Unlevered target weights for one period. With ``rank_buffer`` and a
        prior ``held`` set, XS selection uses hysteresis: a held name is kept
        until it falls past the wider ``top_pct*(1+buffer)`` rank, cutting the
        name churn that dominates momentum turnover."""
        if cfg.kind == "xs":
            n = len(s)
            k = max(1, int(round(n * cfg.top_pct)))
            if cfg.rank_buffer > 0 and held:
                ranked = s.rank(ascending=False, method="first")   # 1 = best
                keep_k = max(k, int(round(n * cfg.top_pct * (1 + cfg.rank_buffer))))
                sel = set(ranked.index[ranked <= k])               # fresh entrants
                sel |= {x for x in held if x in ranked.index and ranked[x] <= keep_k}
                if len(sel) > keep_k:                              # cap to best keep_k
                    sel = set(ranked[ranked.index.isin(sel)].nsmallest(keep_k).index)
                chosen = s[s.index.isin(sel)]
            else:
                chosen = s.nlargest(k)
            invest = 1.0
        elif cfg.kind == "ts":
            chosen = s[s > cfg.ts_threshold]
            if len(chosen) < 1:
                return pd.Series(dtype=float), set()
            invest = len(chosen) / len(s) if cfg.ts_weighting == "breadth" else 1.0
        elif cfg.kind == "ls":
            # long-short dollar-neutral: long the top top_pct, short the bottom
            # top_pct, each leg summing to +/-1 (gross 2, net 0). The vol-target
            # multiplier then scales this book; gross leverage = 2 * multiplier.
            n = len(s)
            k = max(1, int(round(n * cfg.top_pct)))
            longs, shorts = s.nlargest(k).index, s.nsmallest(k).index
            w = pd.Series(0.0, index=s.index)
            if cfg.weighting == "signal":
                wl = s[longs].rank(); wl /= wl.sum()
                ws = (-s[shorts]).rank(); ws /= ws.sum()
                w[longs], w[shorts] = wl.values, -ws.values
            else:
                w[longs], w[shorts] = 1.0 / k, -1.0 / k
            return w, set(longs) | set(shorts)
        else:
            raise ValueError(cfg.kind)
        if cfg.weighting == "signal":
            # cross-sectional tilt: weight proportional to the name's momentum
            # rank *within the selected book* (strongest momentum gets the
            # largest position, the cutoff name the smallest) -- a real
            # cross-sectional bet, not equal weight. Effective breadth ~0.75*k.
            w = chosen.rank()
            w = w / w.sum()
        elif cfg.weighting == "sweet_spot":
            # full weight to the bulk of the winners, then taper the extreme top
            # (the decile-10 names that mean-revert): w = 1 below the peak
            # percentile, falling linearly to 0 at the very top of the universe.
            peak = cfg.sweet_peak if cfg.sweet_peak is not None else 1.0 - cfg.top_pct / 2.0
            p = s.rank(pct=True).reindex(chosen.index)
            w = ((1.0 - p) / max(1e-6, 1.0 - peak)).clip(lower=0.0, upper=1.0)
            w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(chosen), index=chosen.index)
        else:
            w = pd.Series(1.0 / len(chosen), index=chosen.index)
        return w * invest, set(chosen.index)

    def _target_weights(self, signal: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
        """Full-rebalance (non-hysteresis) target panel -- used to size leverage."""
        prices = self.md.prices
        W = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
        for t in signal.index:
            s = signal.loc[t]
            s = s[s.notna() & prices.loc[t].notna()]
            if len(s) < cfg.min_names:
                continue
            w, _ = self._select_names(s, cfg, held=None)
            if len(w):
                W.loc[t, w.index] = w.values
        return W

    # ---- ex-ante leverage for vol targeting -------------------------
    def _vol_target_leverage(self, base_ret: pd.Series, cfg: StrategyConfig) -> pd.Series:
        # Ex-ante vol = max(EWMA, short realised): responsive to vol spikes so
        # the book de-levers quickly (reduces realised-vol overshoot). shift(1)
        # keeps it causal -- only past returns size the current position.
        ppy = self.periods_per_year
        ew = base_ret.ewm(halflife=cfg.vol_halflife, min_periods=cfg.vol_lookback).std()
        rw = base_ret.rolling(max(3, ppy // 4), min_periods=2).std()
        vol = pd.concat([ew, rw], axis=1).max(axis=1) * np.sqrt(ppy)
        lev = (cfg.target_vol / vol.shift(1)).clip(upper=cfg.max_leverage)
        return lev.fillna(1.0).clip(lower=0.0)

    def run(self, signal: pd.DataFrame, cfg: StrategyConfig,
            date_slice: Optional[Tuple] = None) -> BacktestResult:
        md = self.md
        prices = md.prices
        fwd = md.returns.shift(-1)                          # realised t -> t+1
        # explicit publishing/implementation lag: act on a signal that is this
        # many periods stale (on top of the built-in `gap` skip). The signal
        # already only uses data through t-gap, so this just stress-tests timing.
        if cfg.exec_lag:
            signal = signal.shift(cfg.exec_lag)
        idx = signal.index

        # leverage sized off the ideal (full-rebalance) book -- causal vol est.
        base_ideal = (self._target_weights(signal, cfg) * fwd).sum(axis=1, min_count=1)
        lev = self._vol_target_leverage(base_ideal, cfg)

        cols = signal.columns
        prev_held = pd.Series(0.0, index=cols)              # actual levered holdings
        held_names: set = set()
        gross = pd.Series(np.nan, index=idx)
        turnover = pd.Series(0.0, index=idx)
        tcost = pd.Series(0.0, index=idx)
        real_lev = pd.Series(0.0, index=idx)
        held_panel: Dict = {}

        for t in idx:
            s = signal.loc[t]
            s = s[s.notna() & prices.loc[t].notna()]
            L = lev.loc[t] if (t in lev.index and np.isfinite(lev.loc[t])) else 1.0
            target = pd.Series(0.0, index=cols)
            if len(s) >= cfg.min_names:
                w, held_names = self._select_names(s, cfg, held_names)
                if len(w):
                    target[w.index] = (w.values * L)
            # --- cost-aware rebalancing: no-trade band + partial trade rate ---
            # band auto-scales to the average position so it works at any
            # breadth (full entries/exits always clear it; only small drift
            # rebalances among holders are suppressed)
            gap = target - prev_held
            if cfg.no_trade_band > 0:
                nz = target[target.abs() > 1e-12]
                thresh = cfg.no_trade_band * (nz.abs().mean() if len(nz) else 0.0)
                gap = gap.where(gap.abs() >= thresh, 0.0)
            gap = gap * cfg.trade_rate
            held = prev_held + gap
            held_names = set(held.index[held.abs() > 1e-12])

            turnover.loc[t] = 0.5 * gap.abs().sum()
            tcost.loc[t] = self.cost_model.cost(
                gap,
                mktcap=md.mktcap.loc[t] if t in md.mktcap.index else None,
                name_vol=self._name_vol.loc[t] if t in self._name_vol.index else None,
            )
            real_lev.loc[t] = held.abs().sum()   # gross exposure (= net for long-only; ~2x for L/S)
            held_panel[t] = held

            r = fwd.loc[t].reindex(held.index).fillna(0.0)
            port_ret = float((held * r).sum())
            gross.loc[t] = port_ret
            nav_growth = 1.0 + port_ret                     # cash bucket earns 0
            prev_held = (held * (1 + r)) / nav_growth if nav_growth != 0 else held * 0.0

        # drop trailing period with no realised forward return
        valid = fwd.reindex(idx).notna().any(axis=1)
        gross[~valid] = np.nan
        net = gross - tcost
        held_df = pd.DataFrame(held_panel).T.reindex(columns=cols).fillna(0.0)
        contrib = held_df.mul(fwd.reindex(held_df.index))

        if date_slice is not None:
            lo, hi = date_slice
            mask = (idx >= lo) & (idx <= hi)
            ms_ = pd.Series(mask, index=idx)
            gross, net = gross[ms_], net[ms_]
            base_ideal, real_lev = base_ideal[ms_], real_lev[ms_]
            turnover, tcost = turnover[ms_], tcost[ms_]
            held_df, contrib = held_df[mask], contrib[mask]

        return BacktestResult(
            config=cfg, gross_returns=gross, net_returns=net, base_returns=base_ideal,
            leverage=real_lev, turnover=turnover, tcost=tcost, weights=held_df,
            contributions=contrib, benchmark=md.benchmark.reindex(gross.index),
            periods_per_year=self.periods_per_year,
        )


# =====================================================================
# 5. PERFORMANCE METRICS
# =====================================================================

def compute_metrics(
    returns: pd.Series,
    leverage: Optional[pd.Series] = None,
    turnover: Optional[pd.Series] = None,
    tcost: Optional[pd.Series] = None,
    periods_per_year: int = MONTHS_PER_YEAR,
) -> pd.Series:
    """Full performance metric set for a (net) monthly return series."""
    r = returns.dropna()
    if len(r) == 0:
        return pd.Series(dtype=float)
    n = len(r)
    yrs = n / periods_per_year
    eq = (1 + r).cumprod()

    ann_ret = r.mean() * periods_per_year
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else np.nan

    downside = r[r < 0]
    downside_dev = downside.std(ddof=1) * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan

    dd = eq / eq.cummax() - 1
    max_dd = dd.min()

    wins, losses = r[r > 0], r[r < 0]
    gross_win, gross_loss = wins.sum(), losses.sum()

    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    sortino = ann_ret / downside_dev if downside_dev and downside_dev > 0 else np.nan
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    m = {
        "CAGR": cagr,
        "Ann.Return": ann_ret,
        "Ann.Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "MaxDrawdown": max_dd,
        "WinRate": (r > 0).mean(),
        "ProfitFactor": (gross_win / abs(gross_loss)) if gross_loss != 0 else np.nan,
        "AvgWin": wins.mean() if len(wins) else np.nan,
        "AvgLoss": losses.mean() if len(losses) else np.nan,
        "Skewness": stats.skew(r, bias=False),
        "Kurtosis": stats.kurtosis(r, fisher=True, bias=False),  # excess kurtosis
        "Months": n,
    }
    m["AvgLeverage"] = float(leverage.reindex(r.index).mean()) if leverage is not None else np.nan
    m["AnnTurnover"] = float(turnover.reindex(r.index).mean() * periods_per_year) if turnover is not None else np.nan
    m["AnnTCost"] = float(tcost.reindex(r.index).mean() * periods_per_year) if tcost is not None else np.nan
    return pd.Series(m)


def metrics_from_result(res: BacktestResult, which: str = "net") -> pd.Series:
    return compute_metrics(
        getattr(res, f"{which}_returns"),
        leverage=res.leverage, turnover=res.turnover, tcost=res.tcost,
        periods_per_year=res.periods_per_year,
    )


# =====================================================================
# 6. WALK-FORWARD VALIDATION
# =====================================================================

@dataclass
class WalkForwardResult:
    oos_returns: pd.Series           # stitched out-of-sample net returns
    fold_table: pd.DataFrame         # chosen params + IS/OOS sharpe per fold
    param_grid: List[dict]
    periods_per_year: int = MONTHS_PER_YEAR

    def metrics(self) -> pd.Series:
        return compute_metrics(self.oos_returns, periods_per_year=self.periods_per_year)


class WalkForwardValidator:
    """Rolling-window walk-forward parameter selection.

    For each fold we optimise the strategy hyper-parameters on a trailing
    *train* window (objective: in-sample net Sharpe), then apply the chosen
    parameters to the next, untouched *test* window. Stitching the test-window
    returns gives a genuine out-of-sample track record that exposes
    over-fitting: if OOS performance collapses relative to in-sample, the edge
    was a fit artefact.
    """

    def __init__(
        self,
        backtester: Backtester,
        base_config: StrategyConfig,
        param_grid: Dict[str, Sequence],
        train_months: int = 60,
        test_months: int = 12,
        objective: str = "Sharpe",
    ):
        self.bt = backtester
        self.base = base_config
        self.param_grid = param_grid
        self.train_months = train_months
        self.test_months = test_months
        self.objective = objective

    def _expand_grid(self) -> List[dict]:
        keys = list(self.param_grid)
        out = [{}]
        for k in keys:
            out = [dict(o, **{k: v}) for o in out for v in self.param_grid[k]]
        return out

    def _cfg_with(self, overrides: dict) -> StrategyConfig:
        return StrategyConfig(**{**self.base.__dict__, **overrides})

    def run(self) -> WalkForwardResult:
        md = self.bt.md
        dates = md.returns.index
        grid = self._expand_grid()

        # precompute signals + per-combo full-sample net returns once
        sig_cache: Dict[int, pd.DataFrame] = {}
        ret_cache: Dict[int, pd.Series] = {}
        for gi, combo in enumerate(grid):
            cfg = self._cfg_with(combo)
            key = (cfg.lookback, cfg.gap, cfg.risk_adjusted, cfg.vol_window,
                   cfg.blend_lookbacks, cfg.sma_window)
            if key not in sig_cache:
                sig_cache[key] = build_signal(md, cfg)
            res = self.bt.run(sig_cache[key], cfg)
            ret_cache[gi] = res.net_returns

        oos_parts, rows = [], []
        start = self.train_months
        while start + self.test_months <= len(dates):
            tr_lo, tr_hi = dates[start - self.train_months], dates[start - 1]
            te_lo, te_hi = dates[start], dates[min(start + self.test_months - 1, len(dates) - 1)]

            ppy = self.bt.periods_per_year
            best_gi, best_obj = None, -np.inf
            for gi in range(len(grid)):
                seg = ret_cache[gi].loc[tr_lo:tr_hi]
                if seg.dropna().shape[0] < self.train_months // 2:
                    continue
                obj = compute_metrics(seg, periods_per_year=ppy).get(self.objective, np.nan)
                if np.isfinite(obj) and obj > best_obj:
                    best_obj, best_gi = obj, gi
            if best_gi is None:
                start += self.test_months
                continue

            oos_seg = ret_cache[best_gi].loc[te_lo:te_hi]
            oos_parts.append(oos_seg)
            rows.append({
                "train_start": tr_lo, "train_end": tr_hi,
                "test_start": te_lo, "test_end": te_hi,
                "is_sharpe": best_obj,
                "oos_sharpe": compute_metrics(oos_seg, periods_per_year=ppy).get("Sharpe", np.nan),
                **grid[best_gi],
            })
            start += self.test_months

        oos = pd.concat(oos_parts).sort_index() if oos_parts else pd.Series(dtype=float)
        oos = oos[~oos.index.duplicated()]
        return WalkForwardResult(oos_returns=oos, fold_table=pd.DataFrame(rows),
                                 param_grid=grid, periods_per_year=self.bt.periods_per_year)


def select_latest_params(bt: "Backtester", base_cfg: StrategyConfig,
                         param_grid: Dict[str, Sequence], train_periods: Optional[int] = None,
                         objective: str = "Sharpe") -> Tuple[StrategyConfig, float, pd.DataFrame]:
    """Choose the hyper-parameter combo that maximises ``objective`` over the most
    recent ``train_periods`` of (causal) net returns -- i.e. what a walk-forward
    would pick for the *next* live period, using only past data (no look-ahead).
    This lets the live book *trade the validated process* rather than fixed params.
    Returns ``(best_cfg, best_value, ranking)``; ``train_periods=None`` uses all history."""
    keys = list(param_grid)
    combos = [{}]
    for k in keys:
        combos = [dict(o, **{k: v}) for o in combos for v in param_grid[k]]
    vals = []
    for combo in combos:
        cfg = StrategyConfig(**{**base_cfg.__dict__, **combo})
        net = bt.run(build_signal(bt.md, cfg), cfg).net_returns.dropna()
        seg = net.iloc[-train_periods:] if train_periods else net
        vals.append(compute_metrics(seg, periods_per_year=bt.periods_per_year).get(objective, np.nan))
    ranking = pd.DataFrame([{**c, objective: v} for c, v in zip(combos, vals)]
                           ).sort_values(objective, ascending=False, ignore_index=True)
    if not np.isfinite(vals).any():
        return base_cfg, float("nan"), ranking
    best = combos[int(np.nanargmax(vals))]
    return StrategyConfig(**{**base_cfg.__dict__, **best}), float(np.nanmax(vals)), ranking


# Long-only => max_leverage=1.0 (never borrow; vol-targeting only de-risks to cash,
# so realised vol sits at or below the 20% target). XS is equal-weighted by default
# (signal/sweet_spot tilts available); TS is breadth/equal-weighted.
DEFAULT_XS = StrategyConfig(kind="xs", lookback=12, gap=1, top_pct=0.20,
                            weighting="equal", target_vol=0.20, max_leverage=1.0)
DEFAULT_TS = StrategyConfig(kind="ts", lookback=12, gap=1, ts_threshold=0.0,
                            target_vol=0.20, max_leverage=1.0)

# Weekly-frequency presets (lookbacks/vol windows expressed in weeks;
# ~52-4 weeks approximates the monthly 12-1 momentum window).
DEFAULT_XS_WEEKLY = StrategyConfig(kind="xs", lookback=52, gap=4, top_pct=0.20,
                                   weighting="equal", target_vol=0.20, max_leverage=1.0,
                                   vol_window=52, vol_lookback=26, vol_halflife=17.0)
DEFAULT_TS_WEEKLY = StrategyConfig(kind="ts", lookback=52, gap=4, ts_threshold=0.0,
                                   vol_window=52, target_vol=0.20, max_leverage=1.0,
                                   vol_lookback=26, vol_halflife=17.0)

# Daily presets (lookbacks/windows in TRADING DAYS; ~252-21 ≈ the 12-1 month
# window). exec_lag is then the publishing/implementation lag in trading days.
DEFAULT_XS_DAILY = StrategyConfig(kind="xs", lookback=252, gap=21, top_pct=0.20,
                                  weighting="equal", target_vol=0.20, max_leverage=1.0,
                                  vol_window=126, vol_lookback=21, vol_halflife=21.0, exec_lag=2)
DEFAULT_TS_DAILY = StrategyConfig(kind="ts", lookback=252, gap=21, ts_threshold=0.0,
                                  vol_window=126, target_vol=0.20, max_leverage=1.0,
                                  vol_lookback=21, vol_halflife=21.0, exec_lag=2)
