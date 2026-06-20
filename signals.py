"""signals.py -- momentum signal construction and the Information Coefficient."""
from __future__ import annotations
from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd
from scipy import stats

MONTHS_PER_YEAR = 12


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


def trend_filter(prices: pd.DataFrame, sma_window: int, gap: int = 1) -> pd.DataFrame:
    """Boolean panel: True where price is above its ``sma_window`` simple moving
    average, lagged by ``gap`` to match the momentum signal's information set
    (no look-ahead). ``sma_window`` is in panel periods (months/weeks/days)."""
    if sma_window <= 0:
        raise ValueError("sma_window must be positive")
    sma = prices.rolling(sma_window, min_periods=max(2, sma_window // 2)).mean()
    return (prices > sma).shift(gap).fillna(False)


def trend_filtered_momentum(
    prices: pd.DataFrame,
    returns: Optional[pd.DataFrame] = None,
    lookback: int = 12,
    gap: int = 1,
    sma_window: int = 9,
    risk_adjusted: bool = True,
    vol_window: Optional[int] = None,
) -> pd.DataFrame:
    """Cross-sectional momentum restricted to up-trending names (price above
    their ``sma_window`` SMA). Momentum is NaN for non-trending names, so the XS
    selection ranks only trenders -- and the backtester goes to cash when too
    few names trend (built-in de-risking)."""
    mom = momentum_signal(prices, lookback, gap, risk_adjusted, returns, vol_window)
    return mom.where(trend_filter(prices, sma_window, gap))


def build_signal(md: "MarketData", cfg: "StrategyConfig") -> pd.DataFrame:
    """Construct the signal panel implied by a :class:`StrategyConfig`.

    If ``cfg.sma_window`` is set, the momentum signal is masked to names trading
    above their SMA (trend-filtered momentum)."""
    if cfg.blend_lookbacks:
        sig = blended_momentum_signal(
            md.prices, md.returns, cfg.blend_lookbacks, cfg.gap, cfg.risk_adjusted)
    else:
        sig = momentum_signal(
            md.prices, cfg.lookback, cfg.gap, cfg.risk_adjusted, md.returns, cfg.vol_window)
    sma_window = cfg.sma_window
    if sma_window is not None:
        sig = sig.where(trend_filter(md.prices, sma_window, cfg.gap))
    return sig


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


