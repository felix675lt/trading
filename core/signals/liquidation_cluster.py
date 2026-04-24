"""청산 클러스터 탐지 — 돌파매매 승수용.

배경:
- 무기한 선물은 마진 레버리지 기반이라 가격이 특정 레벨에 도달하면 자동 청산.
- 대량의 롱 포지션이 가격 P_L에서 청산 트리거되면 → 대량 시장 매도 발생 → 가격 급락
  (cascade). 숏도 대칭.
- "청산 히트맵(liquidation heatmap)" = 예상 청산 유동성이 밀집된 가격대.
- 돌파 직후 반대편 liq cluster까지 빨려들어가는 패턴 — 돌파매매의 엣지 증폭 요인.

데이터 소스 제약:
- Binance는 최근 청산 실데이터(forceOrder)를 WebSocket으로 제공 (ccxt: watchTrades 일부
  거래소만). 본 모듈은 **OI(Open Interest) 변화 + funding rate + 가격 위치**로 근사한다.
- 실제 완벽한 heatmap은 별도 데이터 공급(Hyblock, Coinalyze 등) 필요.

근사 알고리즘:
  1. OHLCV에서 급격한 price drop + volume spike 지점을 "liquidation proxy"로 식별
  2. 최근 N봉에서 이런 지점들을 모아 가격대별 밀도 계산
  3. 현재 가격에서 가장 가까운 상/하 liq cluster까지 거리 반환

Usage:
    lc = LiquidationClusterDetector(lookback=200, min_spike_z=2.5)
    result = lc.detect(df, side="long")
    # → {"cluster_above": 50000, "cluster_below": 47500,
    #    "distance_to_target": 0.024, "amplify_breakout": 1.3}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class LiquidationClusterDetector:
    """OHLCV 기반 청산 클러스터 근사 탐지기.

    돌파매매에서 사용:
      - LONG 돌파 시 위쪽에 숏 liq cluster 있으면 → amplify (스퀴즈 가속)
      - SHORT 돌파 시 아래쪽에 롱 liq cluster 있으면 → amplify
    """

    def __init__(
        self,
        lookback: int = 200,
        min_spike_z: float = 2.5,          # volume z-score for "liq-like" candle
        min_price_move_pct: float = 0.01,  # 1% 이상 급락/급등
        bucket_pct: float = 0.003,          # 0.3% 가격 bucket
    ):
        self.lookback = lookback
        self.min_spike_z = min_spike_z
        self.min_price_move_pct = min_price_move_pct
        self.bucket_pct = bucket_pct

    def _find_liq_candles(self, df: pd.DataFrame) -> list[tuple[float, float, str]]:
        """급격한 price move + volume spike 바 찾기.

        Returns:
            [(price_of_event, volume, side), ...]
            side: "long_liq" (급락+volume) or "short_liq" (급등+volume)
        """
        d = df.iloc[-self.lookback:]
        if len(d) < 50:
            return []
        vol_mean = d["volume"].rolling(20).mean()
        vol_std = d["volume"].rolling(20).std() + 1e-9
        vol_z = (d["volume"] - vol_mean) / vol_std

        # bar direction by (close-open)/open
        o = d["open"].values
        c = d["close"].values
        move = (c - o) / (o + 1e-9)

        events = []
        for i in range(len(d)):
            if np.isnan(vol_z.iloc[i]):
                continue
            if vol_z.iloc[i] < self.min_spike_z:
                continue
            if abs(move[i]) < self.min_price_move_pct:
                continue
            # 급락 + volume spike → 롱 청산 이벤트로 추정
            side = "long_liq" if move[i] < 0 else "short_liq"
            # 이벤트 "가격"은 low (롱 liq는 낮은 가격에서 체결) or high
            event_price = float(d["low"].iloc[i]) if side == "long_liq" else float(d["high"].iloc[i])
            events.append((event_price, float(d["volume"].iloc[i]), side))
        return events

    def _bucket_density(
        self, events: list[tuple[float, float, str]], side_filter: str
    ) -> dict[float, float]:
        """이벤트 → 가격 bucket별 volume 총합."""
        buckets: dict[float, float] = {}
        for price, vol, side in events:
            if side != side_filter:
                continue
            # bucket by quantized price
            if price <= 0:
                continue
            key = round(price / (price * self.bucket_pct)) * (price * self.bucket_pct)
            buckets[key] = buckets.get(key, 0.0) + vol
        return buckets

    def detect(self, df: pd.DataFrame, side: str = "long") -> dict:
        """현재 진입 방향(side)에서 반대편 liq cluster까지의 거리 계산.

        Args:
            df: OHLCV 최근 lookback+ 봉
            side: "long" 진입 → 위쪽 숏liq cluster가 있으면 breakout 증폭
                  "short" 진입 → 아래쪽 롱liq cluster가 있으면 breakout 증폭
        Returns:
            {
              "cluster_price": float | None,
              "cluster_volume": float,
              "distance_pct": float,
              "amplify_breakout": float (1.0 = no amplify, 1.3 = +30%)
            }
        """
        if df is None or len(df) < 50:
            return {"cluster_price": None, "cluster_volume": 0.0,
                    "distance_pct": 0.0, "amplify_breakout": 1.0,
                    "reason": "표본부족"}
        events = self._find_liq_candles(df)
        if not events:
            return {"cluster_price": None, "cluster_volume": 0.0,
                    "distance_pct": 0.0, "amplify_breakout": 1.0,
                    "reason": "liq 이벤트 없음"}

        last_price = float(df["close"].iloc[-1])
        # 반대편 clusters — LONG 진입 시 위쪽의 숏-liq cluster (short_liq)를 찾음
        opposite_side = "short_liq" if side == "long" else "long_liq"
        buckets = self._bucket_density(events, opposite_side)
        if not buckets:
            return {"cluster_price": None, "cluster_volume": 0.0,
                    "distance_pct": 0.0, "amplify_breakout": 1.0,
                    "reason": f"{opposite_side} cluster 없음"}

        # 진입 방향 기준 가장 가까운 cluster
        if side == "long":
            # 현재가 위쪽 clusters
            candidates = {p: v for p, v in buckets.items() if p > last_price}
        else:
            candidates = {p: v for p, v in buckets.items() if p < last_price}
        if not candidates:
            return {"cluster_price": None, "cluster_volume": 0.0,
                    "distance_pct": 0.0, "amplify_breakout": 1.0,
                    "reason": "반대편 cluster 없음"}

        # 가장 가까운 cluster
        nearest_price = min(candidates.keys(), key=lambda p: abs(p - last_price))
        nearest_vol = candidates[nearest_price]
        dist_pct = abs(nearest_price - last_price) / max(last_price, 1e-9)

        # 증폭 계산: cluster이 가까울수록, volume이 클수록 amplify ↑
        # dist 5% 이내 + 충분한 volume → +30% amplify cap
        total_vol = sum(buckets.values())
        vol_share = nearest_vol / max(total_vol, 1e-9)
        if dist_pct < 0.05 and vol_share > 0.15:
            amplify = min(1.0 + (0.05 - dist_pct) * 6.0, 1.30)
        else:
            amplify = 1.0
        return {
            "cluster_price": round(nearest_price, 4),
            "cluster_volume": round(nearest_vol, 2),
            "distance_pct": round(dist_pct, 4),
            "amplify_breakout": round(amplify, 3),
            "reason": f"{opposite_side} cluster @${nearest_price:.2f} "
                      f"dist={dist_pct*100:.2f}% vol_share={vol_share:.1%} → amp={amplify:.2f}",
        }
