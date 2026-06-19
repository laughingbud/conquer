"""data.py -- market-data layer: point-in-time panels from iShares snapshots."""
from __future__ import annotations
import glob, os, warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


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


