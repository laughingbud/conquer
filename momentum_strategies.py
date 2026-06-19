"""
momentum_strategies.py
======================

Central module for two long-only equity momentum strategies on the S&P 500:

    1. Cross-sectional momentum (XS) -- rank stocks by trailing 12-1 return,
       hold the top quantile (relative strength).
    2. Time-series momentum (TS)     -- hold each stock whose own trailing
       12-1 return is positive (absolute / trend momentum).

Data source
-----------
iShares Core S&P 500 ETF (IVV) monthly holdings snapshots scraped by
riazarbi/sp500-scraper (https://github.com/riazarbi/sp500-scraper). Each
snapshot lists every holding with its market `price`, so stacking snapshots
yields a point-in-time price panel. Using the *actual* historical holdings
makes the investable universe survivorship-bias-free: names that were dropped
from the index simply disappear from later snapshots.

Everything (data, signals, costs, backtester, walk-forward validation,
metrics, plotting) lives in this one module; the companion notebook only
orchestrates and displays.

Design notes
------------
* Frequency: monthly. Multiple intra-month snapshots (the scraper went daily
  in recent years) are collapsed to month-end (last observation).
* No look-ahead: a signal formed at month-end ``t`` uses prices up to ``t``
  only and is rewarded with the return realised over ``t -> t+1``.
* Split handling: iShares reports raw (un-adjusted) prices, so a 2:1 split
  shows up as a spurious -50% return. We detect splits using the `shares`
  column (shares jump by ~an integer factor while price moves inversely) and
  de-split the return series. A light winsorisation is a backstop.
* Risk: returns are treated as excess returns (rf = 0) throughout, which is
  the standard convention for Sharpe/Sortino on a self-financing long book.
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# scipy is only needed for distribution stats / rank correlation
from scipy import stats

MONTHS_PER_YEAR = 12


# =====================================================================
# 1. DATA LAYER
# =====================================================================

@dataclass
class MarketData:
    """Container for the monthly market panels (index = month-end dates)."""

    prices: pd.DataFrame      # raw last price per name (un-adjusted)
    returns: pd.DataFrame     # split-cleaned monthly simple returns
    mktcap: pd.DataFrame      # company market-cap proxy ($) per name
    weights: pd.DataFrame     # iShares index weight (fraction, sums ~1)
    sectors: pd.Series        # last-known GICS sector per name
    benchmark: pd.Series      # cap-weighted index return (price return)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.returns.index

    def summary(self) -> Dict[str, object]:
        n_per_month = self.prices.notna().sum(axis=1)
        return {
            "start": str(self.returns.index.min().date()),
            "end": str(self.returns.index.max().date()),
            "n_months": int(len(self.returns)),
            "n_unique_names": int(self.prices.shape[1]),
            "avg_names_per_month": float(n_per_month.mean()),
            "min_names_per_month": int(n_per_month.min()),
            "max_names_per_month": int(n_per_month.max()),
        }


class DataLoader:
    """Load iShares holdings snapshots into clean monthly panels (cached)."""

    REPO_TARBALL = (
        "https://codeload.github.com/riazarbi/sp500-scraper/tar.gz/refs/heads/main"
    )

    # pandas Period code for each supported resample frequency
    _PERIOD = {"ME": "M", "M": "M", "W": "W-FRI", "W-FRI": "W-FRI"}

    def __init__(
        self,
        data_dir: str,
        asset_class: str = "Equity",
        etf_ownership_frac: float = 0.011,
        winsorize: Tuple[float, float] = (-0.75, 2.0),
        freq: str = "ME",
        start: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        data_dir : folder containing ``ishares/sp500/csv/*.csv``.
        asset_class : keep only rows with this ``asset_class`` (default
            "Equity"; drops Cash / Money Market / Futures lines).
        etf_ownership_frac : IVV's approximate ownership fraction of each
            constituent's market cap, used to scale the ETF's holding value
            up to a company market-cap proxy for the capacity model.
            (IVV AUM ~ $0.5T vs S&P 500 cap ~ $45T => ~1.1%.)
        winsorize : (low, high) clip applied to single-name returns as a
            backstop against residual data errors / un-caught splits.
        freq : resample frequency -- "ME" (month-end, default) or "W"/"W-FRI"
            (weekly). The scraper only provides sub-monthly snapshots from
            ~late-2022, so for weekly we auto-restrict to that dense period.
        start : optional explicit start date (overrides dense auto-detection).
        """
        self.data_dir = data_dir
        self.csv_dir = os.path.join(data_dir, "ishares", "sp500", "csv")
        self.asset_class = asset_class
        self.etf_ownership_frac = etf_ownership_frac
        self.winsorize = winsorize
        self.freq = freq
        self.start = start
        suffix = freq.replace("-", "") + (f"_{start}" if start else "")
        self.cache_path = os.path.join(data_dir, f"ishares_panels_{suffix}.pkl")

    # ---- raw ingest -------------------------------------------------
    def _read_all_snapshots(self) -> pd.DataFrame:
        files = sorted(glob.glob(os.path.join(self.csv_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(
                f"No CSVs in {self.csv_dir}. Run DataLoader.download() first."
            )
        usecols = [
            "symbol", "asset_class", "price", "market_value",
            "weight_pct", "shares", "sector", "date",
        ]
        frames = []
        for f in files:
            try:
                df = pd.read_csv(f, usecols=lambda c: c in usecols)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"skipping {os.path.basename(f)}: {exc}")
                continue
            frames.append(df)
        raw = pd.concat(frames, ignore_index=True)
        raw = raw[raw["asset_class"] == self.asset_class].copy()
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        for col in ["price", "market_value", "weight_pct", "shares"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw = raw.dropna(subset=["date", "symbol", "price"])
        raw = raw[raw["price"] > 0]
        return raw

    # ---- split cleaning --------------------------------------------
    @staticmethod
    def _desplit_returns(prices: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
        """Return split-cleaned monthly returns.

        A split is flagged for (name, t) when the share count jumps by ~a
        clean factor *and* price moves inversely by ~the same factor. The
        return is then de-split: r = price_t * f / price_{t-1} - 1, which
        strips the mechanical jump while keeping any genuine drift.
        """
        raw_ret = prices.pct_change(fill_method=None)
        share_ratio = shares / shares.shift(1)
        price_ratio = prices / prices.shift(1)

        candidates = np.array([2, 3, 4, 5, 6, 7, 8, 10, 1.5,
                               1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10])
        adj = raw_ret.copy()
        sr = share_ratio.values
        pr = price_ratio.values
        out = adj.values
        for i in range(sr.shape[0]):
            for j in range(sr.shape[1]):
                s = sr[i, j]
                p = pr[i, j]
                if not np.isfinite(s) or not np.isfinite(p):
                    continue
                # nearest clean split factor to the share-count change
                k = candidates[np.argmin(np.abs(candidates - s))]
                if abs(s - k) / k < 0.05 and abs(p - 1.0 / k) < 0.12 and k != 1:
                    out[i, j] = p * k - 1.0  # de-split return
        return pd.DataFrame(out, index=prices.index, columns=prices.columns)

    # ---- public API -------------------------------------------------
    def load(self, rebuild: bool = False) -> MarketData:
        if (not rebuild) and os.path.exists(self.cache_path):
            d = pd.read_pickle(self.cache_path)
            return MarketData(**d)

        raw = self._read_all_snapshots()
        raw = raw.sort_values("date")

        # an explicit start applies at any frequency; otherwise, for sub-monthly
        # frequencies auto-restrict to the period where the scraper actually
        # provides sub-monthly snapshots (else "weekly" returns would span the
        # monthly-only era and be meaningless)
        if self.start is not None:
            raw = raw[raw["date"] >= pd.Timestamp(self.start)]
        elif self.freq not in ("ME", "M"):
            per_month = raw.drop_duplicates("date").set_index("date").resample("ME").size()
            dense = per_month[per_month >= 3]
            if len(dense):
                raw = raw[raw["date"] >= dense.index.min().to_period("M").to_timestamp()]

        # collapse to the period end: keep the last snapshot within each period.
        # For daily ("D"/"B") the snapshot dates *are* the trading days, so we
        # group on the date itself -- no calendar grid, hence no weekend/holiday
        # gap rows.
        if self.freq in ("D", "B"):
            raw["period"] = raw["date"].dt.normalize()
        else:
            pcode = self._PERIOD.get(self.freq, self.freq)
            raw["period"] = raw["date"].dt.to_period(pcode).dt.to_timestamp(how="end").dt.normalize()
        last = raw.groupby(["period", "symbol"], as_index=False).last()

        piv = lambda v: last.pivot(index="period", columns="symbol", values=v).sort_index()
        prices, shares, mv, wpct = piv("price"), piv("shares"), piv("market_value"), piv("weight_pct")

        # drop sparse boundary periods (e.g. half-populated first/last week)
        if self.freq not in ("ME", "M"):
            cnt = prices.notna().sum(axis=1)
            keep = cnt >= 0.5 * cnt.median()
            prices, shares, mv, wpct = prices[keep], shares[keep], mv[keep], wpct[keep]

        # split-cleaned returns, then winsorise as a backstop
        returns = self._desplit_returns(prices, shares)
        lo, hi = self.winsorize
        returns = returns.clip(lower=lo, upper=hi)

        # company market-cap proxy ($) for the capacity model
        mktcap = mv / self.etf_ownership_frac

        # index weights as fractions (snapshot weight_pct is in %)
        weights = wpct.div(wpct.sum(axis=1), axis=0)

        # cap-weighted benchmark: w_t (at t) * realised return (t -> t+1)
        fwd = returns.shift(-1)
        bench = (weights * fwd).sum(axis=1, min_count=1)
        bench.name = "benchmark"

        # last-known sector per name (older snapshots store "-")
        sec = last.copy()
        sec = sec[sec["sector"].notna() & (sec["sector"] != "-")]
        sectors = (
            sec.sort_values("period").groupby("symbol")["sector"].last()
            if len(sec) else pd.Series(dtype=object)
        )

        md = MarketData(
            prices=prices, returns=returns, mktcap=mktcap,
            weights=weights, sectors=sectors, benchmark=bench,
        )
        pd.to_pickle(md.__dict__, self.cache_path)
        return md

    def download(self) -> None:  # pragma: no cover - network side effect
        """Download + extract the scraper tarball into ``data_dir``."""
        import io
        import tarfile
        import urllib.request

        os.makedirs(self.data_dir, exist_ok=True)
        with urllib.request.urlopen(self.REPO_TARBALL, timeout=900) as resp:
            buf = io.BytesIO(resp.read())
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            members = [m for m in tar.getmembers() if "/ishares/" in m.name]
            root = members[0].name.split("/")[0]
            tar.extractall(self.data_dir, members=members)
        src = os.path.join(self.data_dir, root, "ishares")
        dst = os.path.join(self.data_dir, "ishares")
        if os.path.abspath(src) != os.path.abspath(dst):
            os.replace(src, dst)


# =====================================================================
# 2. SIGNALS & INFORMATION COEFFICIENT
# =====================================================================

def momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 12,
    gap: int = 1,
    risk_adjusted: bool = False,
    returns: Optional[pd.DataFrame] = None,
    vol_window: Optional[int] = None,
) -> pd.DataFrame:
    """Trailing total return over a ``lookback``-month window, skipping the
    most recent ``gap`` months (classic 12-1 momentum).

    At row ``t`` the value is ``price_{t-gap} / price_{t-lookback} - 1`` --
    i.e. it only uses information available at month-end ``t``. The same raw
    signal drives both the cross-sectional and the time-series strategies;
    they differ only in how the signal is turned into positions.

    If ``risk_adjusted`` is True the raw return is divided by the trailing
    return volatility (a Sharpe-style / "frog-in-the-pan" momentum). On
    large-cap US equities the risk-adjusted variant has materially higher
    Information Coefficient than raw price momentum -- see the IC notebook.
    """
    if lookback <= gap:
        raise ValueError("lookback must exceed gap")
    raw = prices.shift(gap) / prices.shift(lookback) - 1.0
    if not risk_adjusted:
        return raw
    if returns is None:
        raise ValueError("risk_adjusted=True requires the returns panel")
    w = vol_window or max(lookback, 6)
    vol = returns.rolling(w, min_periods=min(6, w)).std().shift(gap)
    return raw / vol.replace(0.0, np.nan)


def _zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional (row-wise) z-score."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def blended_momentum_signal(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    lookbacks: Sequence[int] = (3, 6, 12),
    gap: int = 1,
    risk_adjusted: bool = True,
) -> pd.DataFrame:
    """Average of cross-sectionally standardised momentum signals computed at
    several lookbacks -- a more robust composite than any single horizon."""
    parts = [
        _zscore_xs(momentum_signal(prices, lb, gap, risk_adjusted, returns))
        for lb in lookbacks
    ]
    return sum(parts) / len(parts)


def build_signal(md: "MarketData", cfg: "StrategyConfig") -> pd.DataFrame:
    """Construct the signal panel implied by a :class:`StrategyConfig`."""
    if cfg.blend_lookbacks:
        return blended_momentum_signal(
            md.prices, md.returns, cfg.blend_lookbacks, cfg.gap, cfg.risk_adjusted
        )
    return momentum_signal(
        md.prices, cfg.lookback, cfg.gap, cfg.risk_adjusted, md.returns, cfg.vol_window
    )


def ic_by_horizon(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    horizons: Sequence[int] = (1, 3, 6, 12),
    method: str = "spearman",
) -> pd.DataFrame:
    """Mean IC / t-stat / hit-rate of a signal against forward cumulative
    returns at several horizons. Momentum's IC typically peaks at 3-6 months
    and decays (or reverses) by 12 -- the data behind that statement."""
    rows = []
    for h in horizons:
        if h == 1:
            fwd = returns.shift(-1)                       # return over (t, t+1]
        else:
            # cumulative return over (t, t+h], aligned to t
            fwd = (1 + returns).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
        ic = []
        for t in signal.index:
            d = pd.concat([signal.loc[t], fwd.loc[t]], axis=1).dropna()
            if len(d) < 20:
                continue
            if method == "spearman":
                c = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1]).correlation
            else:
                c = d.iloc[:, 0].corr(d.iloc[:, 1])
            ic.append(c)
        ic = pd.Series(ic).dropna()
        n = len(ic)
        ir = ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else np.nan
        rows.append({
            "horizon_m": h, "mean_ic": ic.mean(), "ic_ir": ir,
            "t_stat": ir * np.sqrt(n) if np.isfinite(ir) else np.nan,
            "hit_rate": float((ic > 0).mean()), "n_months": n,
        })
    return pd.DataFrame(rows)


def decile_forward_returns(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    n_bins: int = 10,
    min_names: int = 50,
) -> pd.DataFrame:
    """Average annualised forward 1-month return per signal decile.

    A monotonic increase from low to high deciles is direct visual proof that
    the signal sorts the cross-section of returns."""
    fwd = returns.shift(-1)
    rows = {}
    for t in signal.index:
        d = pd.concat([signal.loc[t], fwd.loc[t]], axis=1).dropna()
        d.columns = ["sig", "fwd"]
        if len(d) < min_names:
            continue
        try:
            d["bin"] = pd.qcut(d["sig"], n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        rows[t] = d.groupby("bin")["fwd"].mean()
    panel = pd.DataFrame(rows).T
    out = pd.DataFrame({
        "decile": panel.columns + 1,
        "ann_fwd_return": panel.mean().values * MONTHS_PER_YEAR,
        "monthly_fwd_return": panel.mean().values,
        "n_months": panel.notna().sum().values,
    })
    return out


def information_coefficient(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    method: str = "spearman",
    min_names: int = 20,
) -> Dict[str, object]:
    """Cross-sectional IC of ``signal_t`` vs the *forward* return ``t->t+1``.

    Returns a dict with the per-month IC series and summary statistics
    (mean, std, information ratio, Newey-West-free t-stat, hit rate). A
    significantly positive mean IC is the evidence that the signal forecasts
    the cross-section of returns.
    """
    fwd = returns.shift(-1)
    common = signal.index.intersection(fwd.index)
    ic = pd.Series(index=common, dtype=float)
    n_obs = pd.Series(index=common, dtype=float)
    for t in common:
        s = signal.loc[t]
        r = fwd.loc[t]
        df = pd.concat([s, r], axis=1).dropna()
        if len(df) < min_names:
            continue
        if method == "spearman":
            c = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1]).correlation
        else:
            c = df.iloc[:, 0].corr(df.iloc[:, 1])
        ic.loc[t] = c
        n_obs.loc[t] = len(df)
    ic = ic.dropna()
    mean = ic.mean()
    std = ic.std(ddof=1)
    n = len(ic)
    ir = mean / std if std > 0 else np.nan          # IC information ratio
    tstat = ir * np.sqrt(n) if np.isfinite(ir) else np.nan
    pval = 2 * (1 - stats.t.cdf(abs(tstat), df=n - 1)) if np.isfinite(tstat) else np.nan
    return {
        "ic_series": ic,
        "n_obs": n_obs.reindex(ic.index),
        "mean_ic": mean,
        "std_ic": std,
        "ic_ir": ir,
        "t_stat": tstat,
        "p_value": pval,
        "hit_rate": float((ic > 0).mean()),
        "n_months": n,
        "method": method,
    }


# =====================================================================
# 3. TRANSACTION-COST / MARKET-IMPACT MODELS
# =====================================================================

class LinearCostModel:
    """Flat per-unit-traded cost. ``cost = bps * sum_i |dw_i|``.

    ``cost_bps`` is charged on one-way traded notional as a fraction of NAV;
    buys and sells both count, so a full round-trip of the book costs
    ``2 * cost_bps``. Default 5 bps matches the brief's fallback.
    """

    def __init__(self, cost_bps: float = 5.0):
        self.cost_bps = cost_bps

    def cost(self, dweights: pd.Series, **_) -> float:
        return float(dweights.abs().sum() * self.cost_bps * 1e-4)


class SquareRootImpactModel:
    """Almgren-style square-root market-impact model used for capacity work.

    For each name the cost (as a fraction of the traded notional) is

        c_i = half_spread + impact_coef * sigma_daily_i * sqrt(Q_i / ADV_i)

    where ``Q_i`` is the dollar trade in name ``i`` (= |dw_i| * AUM), ADV_i is
    the name's dollar average daily volume (proxied from its market cap) and
    ``sigma_daily_i`` is its daily return volatility. Total portfolio cost is
    the notional-weighted average, expressed as a fraction of NAV. Because
    ``Q_i`` scales with AUM, cost grows ~sqrt(AUM) -> this is what bends the
    capacity curve.
    """

    def __init__(
        self,
        aum: float = 1e8,
        half_spread_bps: float = 2.5,
        impact_coef: float = 0.1,
        adv_turnover: float = 0.005,   # daily $ volume as % of market cap
        exec_days: float = 5.0,        # days to work the rebalance trade
    ):
        self.aum = aum
        self.half_spread_bps = half_spread_bps
        self.impact_coef = impact_coef
        self.adv_turnover = adv_turnover
        self.exec_days = exec_days

    def cost(
        self,
        dweights: pd.Series,
        mktcap: Optional[pd.Series] = None,
        name_vol: Optional[pd.Series] = None,
        **_,
    ) -> float:
        traded = dweights.abs()
        traded = traded[traded > 0]
        if traded.empty or mktcap is None:
            return float(traded.sum() * self.half_spread_bps * 1e-4)
        q = traded * self.aum                       # $ traded per name
        adv = (mktcap.reindex(traded.index) * self.adv_turnover * self.exec_days)
        adv = adv.where(adv > 0)
        participation = (q / adv).clip(upper=1.0).fillna(0.5)
        if name_vol is not None:
            sig = name_vol.reindex(traded.index).fillna(name_vol.median())
            sig_daily = sig / np.sqrt(21.0)
        else:
            sig_daily = pd.Series(0.02, index=traded.index)
        half_spread = self.half_spread_bps * 1e-4
        c_i = half_spread + self.impact_coef * sig_daily * np.sqrt(participation)
        # notional-weighted average cost as a fraction of NAV
        return float((traded * c_i).sum())


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
    top_pct: float = 0.20            # XS: fraction of universe held long
    ts_threshold: float = 0.0        # TS: hold names with signal above this
    ts_weighting: str = "breadth"    # TS: "breadth" (cash when few trend) or "selected"
    weighting: str = "equal"         # within-book: "equal" or "signal" (rank-tilt)
    target_vol: float = 0.15         # annualised vol target
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
        else:
            raise ValueError(cfg.kind)
        if cfg.weighting == "signal":
            w = chosen.rank()
            w = w / w.sum()
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
            real_lev.loc[t] = held.sum()
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
                   cfg.blend_lookbacks)
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
) -> pd.DataFrame:
    """Sweep cost-aware rebalancing settings (no-trade band / rank buffer /
    trade rate) and report turnover, cost drag and net Sharpe for each. Shows
    that penalising turnover cuts cost with little (or positive) net impact."""
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
    bt = Backtester(md, LinearCostModel(cost_bps), periods_per_year)
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
) -> pd.DataFrame:
    """Sweep an implementation-lag parameter and report performance.

    ``param`` is either ``"exec_lag"`` (act on a signal lagged this many extra
    periods -- models publishing/implementation delay) or ``"gap"`` (the
    momentum skip itself). For a daily panel the lag unit is *trading days*, so
    e.g. ``exec_lag=2`` reproduces "the freshest close I can read is 2 sessions
    old". A flat Sharpe across lags is evidence the edge is not a timing
    artefact."""
    bt = Backtester(md, LinearCostModel(cost_bps), periods_per_year)
    rows = []
    for L in lags:
        cfg = StrategyConfig(**{**base_cfg.__dict__, param: int(L)})
        sig = build_signal(md, cfg)        # rebuilt per gap; identical across exec_lag
        m = metrics_from_result(bt.run(sig, cfg))
        rows.append({param: int(L), "sharpe": m["Sharpe"], "cagr": m["CAGR"],
                     "ann_vol": m["Ann.Vol"], "maxDD": m["MaxDrawdown"],
                     "ann_turnover": m["AnnTurnover"]})
    return pd.DataFrame(rows)


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


DEFAULT_XS = StrategyConfig(kind="xs", lookback=12, gap=1, top_pct=0.20,
                            target_vol=0.15, max_leverage=2.0)
DEFAULT_TS = StrategyConfig(kind="ts", lookback=12, gap=1, ts_threshold=0.0,
                            target_vol=0.15, max_leverage=2.0)

# Weekly-frequency presets (lookbacks/vol windows expressed in weeks;
# ~52-4 weeks approximates the monthly 12-1 momentum window).
DEFAULT_XS_WEEKLY = StrategyConfig(kind="xs", lookback=52, gap=4, top_pct=0.20,
                                   vol_window=52, target_vol=0.15, max_leverage=2.0,
                                   vol_lookback=26, vol_halflife=17.0)
DEFAULT_TS_WEEKLY = StrategyConfig(kind="ts", lookback=52, gap=4, ts_threshold=0.0,
                                   vol_window=52, target_vol=0.15, max_leverage=2.0,
                                   vol_lookback=26, vol_halflife=17.0)

# Daily presets (lookbacks/windows in TRADING DAYS; ~252-21 ≈ the 12-1 month
# window). exec_lag is then the publishing/implementation lag in trading days.
DEFAULT_XS_DAILY = StrategyConfig(kind="xs", lookback=252, gap=21, top_pct=0.20,
                                  vol_window=126, target_vol=0.15, max_leverage=2.0,
                                  vol_lookback=21, vol_halflife=21.0, exec_lag=2)
DEFAULT_TS_DAILY = StrategyConfig(kind="ts", lookback=252, gap=21, ts_threshold=0.0,
                                  vol_window=126, target_vol=0.15, max_leverage=2.0,
                                  vol_lookback=21, vol_halflife=21.0, exec_lag=2)
