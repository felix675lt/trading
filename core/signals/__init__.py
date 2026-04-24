"""Alternative quant signal modules (OFI, liquidation cluster, etc)."""
from core.signals.liquidation_cluster import LiquidationClusterDetector
from core.signals.ofi import OFISignal

__all__ = ["OFISignal", "LiquidationClusterDetector"]
