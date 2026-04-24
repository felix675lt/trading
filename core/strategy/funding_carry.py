"""Funding Rate Carry + Cash-and-Carry Basis 차익 전략.

학계/업계 근거:
- Perpetual funding rates typically oscillate around the short-term cost of carry,
  with persistent deviations earnable via market-neutral hedge.
  (Refs: Bitmex "Perpetual Swap Primer"; Alexander-Heck 2023 survey; Cascade/Paradigm
   FT arb reports 2023-2024)
- Historical Binance BTC/ETH perp funding 8h rates avg ~0.01% → ~11%/yr annualized,
  with spikes to 0.1%+ (bull mania). Capturable via spot-long + perp-short market-neutral
  position sized to keep net delta ≈ 0.

왜 대자본 전용인가 (사용자 직접 요청 반영):
  1. 선물 수수료+슬리피지: 체결당 ~0.08% (ccxt market+taker) → 8h funding 0.01% × 3cycle/day
     = 일 0.03% 리턴 vs 체결 왕복 0.16% → 최소 5일+ 보유 필요.
  2. 현물 + 선물 동시 헷지 → 자본의 2배 필요 (spot-notional + futures-margin).
  3. USDT/BTC 스팟 잔고 유지 수수료 + lending opportunity cost도 있음.
  4. 소시드($500~$10K)에서는 체결비·슬리피지가 edge를 잠식 → tier=pro($50K+) +
     추가 min_equity($100K) 이상에서만 의미.

작동 방식 (실제 진입은 아직 스텁):
  1. Binance에서 perpetual funding rate 스트림 조회
  2. 상위 5개 심볼 중 |funding_rate|이 임계값(기본 0.03% = 연 33%) 이상이면 기회
  3. 양funding → spot LONG + perp SHORT (롱이 숏에게 funding 지급 받음)
     음funding → spot SHORT(대출) + perp LONG (숏이 롱에게 funding 지급 받음)
  4. 델타 중립 유지, funding 3사이클 받으면 청산

현재는 `detect_opportunity()`만 구현. 실제 주문 실행은 main.py에서
`self.funding_carry.enabled`를 체크한 후에만 동작.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger


@dataclass
class CarryOpportunity:
    symbol: str
    funding_rate_8h: float      # e.g. 0.0003 = 0.03%
    annualized: float            # funding_rate_8h * 3 * 365
    direction: str               # "spot_long_perp_short" or "spot_short_perp_long"
    est_entry_cost_pct: float    # spot+perp 체결+슬리피지 예상
    est_net_edge_per_cycle: float
    reason: str
    ts: datetime = field(default_factory=datetime.utcnow)


class FundingCarryEngine:
    """현물 + 퍼프 델타-중립 funding 차익.

    활성화 조건 (둘 다 true):
      - tier_manager.get_tier('live').name == 'pro'  (= $50K+)
      - equity >= min_equity_usd (기본 $100,000) — 사용자 명시 조건

    둘 다 만족 전까지는 `enabled=False`로 남고 진단 로그만 남긴다.
    """

    def __init__(
        self,
        min_equity_usd: float = 100_000.0,
        funding_threshold_8h: float = 0.0003,   # 0.03% ≒ 연33%
        min_hold_cycles: int = 3,                # 3 × 8h = 24h 최소 보유
        est_round_trip_cost_pct: float = 0.0016, # 0.16% (spot+perp 왕복 taker)
        target_symbols: list[str] | None = None,
    ):
        self.min_equity_usd = min_equity_usd
        self.funding_threshold = funding_threshold_8h
        self.min_hold_cycles = min_hold_cycles
        self.est_cost = est_round_trip_cost_pct
        self.target_symbols = target_symbols or [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
            "BNB/USDT:USDT", "XRP/USDT:USDT",
        ]
        self.enabled = False
        self.positions: dict[str, dict[str, Any]] = {}
        logger.info(
            f"[FundingCarry] 초기화 — min_equity=${min_equity_usd:,.0f} "
            f"threshold={funding_threshold_8h*100:.3f}% min_hold={min_hold_cycles}cycles "
            f"est_cost={est_round_trip_cost_pct*100:.2f}%"
        )

    def update_gate(self, tier_name: str, equity_usd: float) -> bool:
        """tier + equity 체크 → enabled 갱신."""
        tier_ok = tier_name == "pro"
        equity_ok = equity_usd >= self.min_equity_usd
        self.enabled = tier_ok and equity_ok
        if not self.enabled:
            logger.debug(
                f"[FundingCarry] 비활성 — tier={tier_name} equity=${equity_usd:,.0f} "
                f"(need pro+${self.min_equity_usd:,.0f})"
            )
        return self.enabled

    def detect_opportunity(
        self, funding_snapshot: dict[str, float], prices: dict[str, float] | None = None
    ) -> list[CarryOpportunity]:
        """현재 퍼프 funding rate 스냅샷에서 기회 감지.

        Args:
            funding_snapshot: {symbol: funding_rate_8h} — e.g. {"BTC/USDT:USDT": 0.0005}
            prices: {symbol: last_price} (로그용)
        Returns:
            기회 리스트 (연환산 > threshold만)
        """
        if not self.enabled:
            return []
        ops: list[CarryOpportunity] = []
        for sym in self.target_symbols:
            fr = float(funding_snapshot.get(sym, 0.0))
            if abs(fr) < self.funding_threshold:
                continue
            annual = fr * 3 * 365  # 3 cycles/day × 365
            direction = "spot_long_perp_short" if fr > 0 else "spot_short_perp_long"
            # Net edge = funding * cycles - round_trip_cost (둘 다 %)
            gross = abs(fr) * self.min_hold_cycles
            net = gross - self.est_cost
            if net <= 0:
                continue
            ops.append(CarryOpportunity(
                symbol=sym,
                funding_rate_8h=fr,
                annualized=annual,
                direction=direction,
                est_entry_cost_pct=self.est_cost,
                est_net_edge_per_cycle=net / self.min_hold_cycles,
                reason=(
                    f"funding={fr*100:.3f}%/8h annual≈{annual*100:.1f}% → "
                    f"{self.min_hold_cycles}cycles gross={gross*100:.3f}% "
                    f"-cost={self.est_cost*100:.3f}% ⇒ net={net*100:.3f}%"
                ),
            ))
        return sorted(ops, key=lambda o: o.est_net_edge_per_cycle, reverse=True)

    def report(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_equity_usd": self.min_equity_usd,
            "funding_threshold_8h": self.funding_threshold,
            "positions_open": len(self.positions),
            "target_symbols": self.target_symbols,
        }
