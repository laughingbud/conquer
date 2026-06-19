"""costs.py -- transaction-cost / market-impact models."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd


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
            # name_vol is ANNUALISED -> daily vol = annualised / sqrt(252)
            sig = name_vol.reindex(traded.index).fillna(name_vol.median())
            sig_daily = sig / np.sqrt(252.0)
        else:
            sig_daily = pd.Series(0.015, index=traded.index)
        half_spread = self.half_spread_bps * 1e-4
        c_i = half_spread + self.impact_coef * sig_daily * np.sqrt(participation)
        # notional-weighted average cost as a fraction of NAV
        return float((traded * c_i).sum())


class RealisticCostModel:
    """All-in transaction cost = FX + commission + half-spread + market impact,
    calibrated to the brief and estimated from the provided data.

    Components, each charged per unit of traded notional and summed over names:

    * **FX** (``fx_bps``, default 15) -- currency conversion for a non-USD
      investor trading USD-denominated S&P names. By default charged on *gross*
      trades (broker auto-converts each trade); ``fx_on="net"`` charges only the
      net currency flow (a USD cash balance is held, so rotations need no
      conversion -- usually far cheaper, since for a fully-invested rotation
      buys ~ sells).
    * **commission** (``commission_bps``, default 0).
    * **half-spread** = ``spread_vol_coef * daily_vol_i`` clipped to
      ``[min_hs_bps, max_hs_bps]``. The proportional-to-volatility form is
      strongly supported by the data (cross-sectional corr of the Roll spread
      estimator with daily vol ~0.8); the *level* is calibrated to realistic
      S&P large-cap effective spreads (~1-3 bps half-spread), because the Roll
      estimator over-states the absolute level at daily frequency.
    * **impact** -- Almgren square-root model (``SquareRootImpactModel``),
      proxying ADV from market cap and scaling with ``aum``.

    All cost components are subtracted from returns; they do not feed back into
    position sizing, so the per-component drag is exactly additive
    (see :func:`cost_breakdown`).
    """

    def __init__(self, fx_bps: float = 15.0, commission_bps: float = 0.0,
                 fx_on: str = "gross", spread_vol_coef: float = 0.01,
                 min_hs_bps: float = 0.5, max_hs_bps: float = 10.0,
                 aum: float = 1e8, impact_coef: float = 0.1,
                 adv_turnover: float = 0.005, exec_days: float = 5.0):
        self.fx_bps = fx_bps
        self.commission_bps = commission_bps
        self.fx_on = fx_on
        self.spread_vol_coef = spread_vol_coef
        self.min_hs = min_hs_bps * 1e-4
        self.max_hs = max_hs_bps * 1e-4
        self.aum = aum
        self._impact = SquareRootImpactModel(
            aum=aum, half_spread_bps=0.0, impact_coef=impact_coef,
            adv_turnover=adv_turnover, exec_days=exec_days)

    def components(self, dweights: pd.Series, mktcap=None, name_vol=None) -> Dict[str, float]:
        traded = dweights.abs()
        traded = traded[traded > 0]
        if traded.empty:
            return {"fx": 0.0, "commission": 0.0, "spread": 0.0, "impact": 0.0}
        gross = float(traded.sum())
        net = abs(float(dweights.sum()))
        fx_base = net if self.fx_on == "net" else gross
        if self.spread_vol_coef > 0:
            if name_vol is not None:
                sig_d = name_vol.reindex(traded.index).fillna(name_vol.median()) / np.sqrt(252.0)
            else:
                sig_d = pd.Series(0.015, index=traded.index)
            hs = (self.spread_vol_coef * sig_d).clip(lower=self.min_hs, upper=self.max_hs)
            spread = float((traded * hs).sum())
        else:
            spread = 0.0                      # spread explicitly off (e.g. decomposition)
        return {
            "fx": self.fx_bps * 1e-4 * fx_base,
            "commission": self.commission_bps * 1e-4 * gross,
            "spread": spread,
            "impact": self._impact.cost(dweights, mktcap=mktcap, name_vol=name_vol),
        }

    def cost(self, dweights: pd.Series, mktcap=None, name_vol=None, **_) -> float:
        return float(sum(self.components(dweights, mktcap, name_vol).values()))


