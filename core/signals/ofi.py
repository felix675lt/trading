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
        toxicity_window: int = 20,
        toxicity_high: float = 0.7,
    ):
        self.window = window
        self.z_threshold = z_threshold
        self.use_vwap = use_vwap
        self.vote_weight = vote_weight
        # Trade Flow Toxicity (Easley-style) — 같은 방향 연속 체결 비율 + 방향 편향
        self.toxicity_window = toxicity_window
        self.toxicity_high = toxicity_high

    @staticmethod
    def _vwap(df: pd.DataFrame) -> pd.Series:
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_pv = (typical * df["volume"]).cumsum()
        cum_v = df["volume"].cumsum().replace(0, np.nan)
        return cum_pv / cum_v

    def compute_trade_toxicity(self, df: pd.DataFrame) -> dict:
        """Trade Flow Toxicity (스마트머니 압력 감지)

        주문흐름 독성 = 0.6 × 연속 같은방향비율 + 0.4 × 방향편향(abs(mean(side))).
        높을수록 한 방향 강한 압력 → 정보 비대칭 가능성 ↑.

        Lee-Ready (1991) 룰로 봉 방향 sign(close_t - close_{t-1}) 사용:
          +1 = 매수 주도, -1 = 매도 주도.

        Args:
            df: 최소 toxicity_window+2 봉의 OHLCV DataFrame.
        Returns:
            {"toxicity": float, "direction_bias": float, "same_direction_ratio": float,
             "level": "low|normal|high", "is_toxic": bool}
        """
        if len(df) < max(10, self.toxicity_window):
            return {
                "toxicity": 0.0, "direction_bias": 0.0,
                "same_direction_ratio": 0.0, "level": "low", "is_toxic": False,
            }
        try:
            tail = df.iloc[-self.toxicity_window:]
            diffs = tail["close"].diff().fillna(0).values
            sides = np.where(diffs > 0, 1.0, np.where(diffs < 0, -1.0, 0.0))
            non_zero = sides[sides != 0]
            if len(non_zero) < 5:
                return {
                    "toxicity": 0.0, "direction_bias": 0.0,
                    "same_direction_ratio": 0.0, "level": "low", "is_toxic": False,
                }
            # 연속 같은 방향 비율
            same_dir = float(np.sum(non_zero[1:] == non_zero[:-1]) / max(1, len(non_zero) - 1))
            # 방향 편향 (1: 100% 한 방향)
            dir_bias = float(abs(non_zero.mean()))
            tox = 0.6 * same_dir + 0.4 * dir_bias
            level = "high" if tox >= self.toxicity_high else (
                "normal" if tox >= 0.4 else "low"
            )
            return {
                "toxicity": round(float(tox), 3),
                "direction_bias": round(dir_bias, 3),
                "same_direction_ratio": round(same_dir, 3),
                "dominant_side": (
                    "buy" if non_zero.mean() > 0.1 else
                    ("sell" if non_zero.mean() < -0.1 else "neutral")
                ),
                "level": level,
                "is_toxic": bool(tox >= self.toxicity_high),
            }
        except Exception as e:
            logger.debug(f"[Toxicity] compute 실패: {e}")
            return {
                "toxicity": 0.0, "direction_bias": 0.0,
                "same_direction_ratio": 0.0, "level": "low", "is_toxic": False,
            }

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
            # Trade Toxicity (스마트머니 압력) — OFI 보강
            tox = self.compute_trade_toxicity(df)

            # 토큰 토스 보너스: OFI 방향과 toxicity dominant_side 일치 시 강도 +0.1
            if direction != "neutral" and tox.get("is_toxic"):
                ds = tox.get("dominant_side")
                if (direction == "long" and ds == "buy") or (direction == "short" and ds == "sell"):
                    strength = float(min(strength + 0.10, 1.0))

            return {
                "ofi_z": round(z, 3),
                "direction": direction,
                "strength": round(strength, 3),
                "toxicity": tox,
                "reason": (
                    f"OFI z={z:.2f} (w={self.window}, thr={self.z_threshold})"
                    f" | tox={tox.get('toxicity'):.2f}({tox.get('level')})"
                ),
            }
        except Exception as e:
            logger.debug(f"[OFI] compute 실패: {e}")
            return {"ofi_z": 0.0, "direction": "neutral", "strength": 0.0,
                    "toxicity": {"toxicity": 0.0, "level": "low", "is_toxic": False},
                    "reason": f"error:{e}"}
