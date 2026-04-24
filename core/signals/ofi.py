"""Order Flow Imbalance (OFI) — Cont-Kukanov-Stoikov (2014).

OFI는 오더북 L1 snapshot 변화를 누적한 "공격적 매수 vs 매도" 불균형 척도.
실전에서 단기(초~분) 가격 방향의 선행 지표로 널리 쓰이며, 특히 마켓메이커의
passive liquidity를 유지할 타이밍을 잡는다.

완전한 OFI는 bid/ask 레벨 1 snapshot 스트림이 필요하지만, 15m OHLCV 기준
트레이더를 위해 아래 두 **근사치(proxy)**를 제공한다.

1) **VWAP-based OFI** (Kyle 1985 lambda와 유사):
   OFI_t = sign(close_t - vwap_t) × volume_t
   ; close가 vwap보다 높으면 주도적 매수가 지배적이었다는 근사.

2) **Tick-Rule OFI** (Lee-Ready 1991):
   bar_direction = +1 if close_t > close_{t-1} else -1
   OFI_t = bar_direction × volume_t

두 지표를 과거 N봉 표준화(z-score)해 "공격적 유입/유출 극값"을 판정.

Usage:
    sig = OFISignal(window=20, z_threshold=1.5)
    out = sig.compute(df)   # df: [close, volume, high, low] 최근 N봉
    # out: {"ofi_z": float, "direction": "long|short|neutral", "strength": 0~1}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class OFISignal:
    """Order Flow Imbalance proxy from OHLCV (L1-free).

    Designed as a 5th vote source in StrategyManager.decide(), complementing:
      ML, RL, MOM, EXT, BREAKOUT → +OFI.

    Integration notes:
      - 15m/5m 타임프레임에서 가장 의미 있음 (HFT 수준은 tick data 필요).
      - window=20 봉 (15m → 5시간) z-score 기반으로 노이즈 제거.
      - z > +z_threshold → LONG vote, z < -z_threshold → SHORT vote.
    """

    def __init__(
        self,
        window: int = 20,
        z_threshold: float = 1.5,
        use_vwap: bool = True,
        vote_weight: float = 0.45,
    ):
        self.window = window
        self.z_threshold = z_threshold
        self.use_vwap = use_vwap
        self.vote_weight = vote_weight

    @staticmethod
    def _vwap(df: pd.DataFrame) -> pd.Series:
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_pv = (typical * df["volume"]).cumsum()
        cum_v = df["volume"].cumsum().replace(0, np.nan)
        return cum_pv / cum_v

    def compute(self, df: pd.DataFrame) -> dict:
        """최신 봉 OFI z-score + 방향 판정.

        Args:
            df: 최소 window+2 봉의 OHLCV DataFrame. 최신이 마지막 행.
        Returns:
            {"ofi_z": float, "direction": str, "strength": float, "reason": str}
        """
        if len(df) < self.window + 2:
            return {"ofi_z": 0.0, "direction": "neutral", "strength": 0.0,
                    "reason": f"표본부족 n={len(df)}"}
        try:
            d = df.iloc[-(self.window + 1):].copy()
            if self.use_vwap:
                vw = self._vwap(d)
                # sign(close - vwap) × volume
                flow = np.sign(d["close"] - vw).fillna(0) * d["volume"]
            else:
                diff = d["close"].diff().fillna(0)
                flow = np.sign(diff) * d["volume"]

            # z-score of most recent bar within rolling window
            recent = flow.iloc[-1]
            hist = flow.iloc[:-1]
            mu = float(hist.mean())
            sd = float(hist.std(ddof=1)) + 1e-9
            z = (float(recent) - mu) / sd

            if z > self.z_threshold:
                direction = "long"
                strength = min(self.vote_weight + min((z - self.z_threshold) / 3.0, 1.0) * 0.15,
                               1.0)
            elif z < -self.z_threshold:
                direction = "short"
                strength = min(self.vote_weight + min((abs(z) - self.z_threshold) / 3.0, 1.0) * 0.15,
                               1.0)
            else:
                direction = "neutral"
                strength = 0.0
            return {
                "ofi_z": round(z, 3),
                "direction": direction,
                "strength": round(strength, 3),
                "reason": f"OFI z={z:.2f} (window={self.window}, threshold={self.z_threshold})",
            }
        except Exception as e:
            logger.debug(f"[OFI] compute 실패: {e}")
            return {"ofi_z": 0.0, "direction": "neutral", "strength": 0.0,
                    "reason": f"error:{e}"}
