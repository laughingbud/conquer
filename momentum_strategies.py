"""momentum_strategies.py -- aggregator re-exporting the specialised modules.

Implementation now lives in:
  data.py        -- DataLoader, MarketData (point-in-time panels)
  signals.py     -- momentum signals + Information Coefficient
  costs.py       -- LinearCostModel, SquareRootImpactModel, RealisticCostModel
  backtester.py  -- StrategyConfig, Backtester, metrics, walk-forward, presets
  analytics.py   -- sweeps, dashboards, weights/metrics I/O

`import momentum_strategies as ms` still exposes the full public API.
"""
from data import *          # noqa: F401,F403
from signals import *       # noqa: F401,F403
from costs import *         # noqa: F401,F403
from backtester import *    # noqa: F401,F403
from analytics import *     # noqa: F401,F403
