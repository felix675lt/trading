"""Backtesting infrastructure — DSR, CPCV, overfit defense."""
from core.backtest.dsr_cpcv import (
    CombinatorialPurgedCV,
    DeflatedSharpe,
    DSRResult,
)

__all__ = ["DeflatedSharpe", "DSRResult", "CombinatorialPurgedCV"]
