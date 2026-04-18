"""Statistical Arbitrage / Pairs Trading.

전제: 공적분 페어의 spread는 평균 회귀한다.
전략:
  - spread z-score > +threshold → spread overpriced → SELL a, BUY b·β (spread 숏)
  - spread z-score < -threshold → spread underpriced → BUY a, SELL b·β (spread 롱)
  - |z| < exit_threshold → 청산

구현:
  - CointegrationTester로 페어 자동 탐색 (daily rebalance 권장)
  - 동시 open/close 체결 (delta-neutral)
  - Kelly/CVaR 위험 관리와 독립 트랙으로 운영

사용:
    strategy = PairsTradingStrategy(exchange, coint_tester)
    await strategy.discover_pairs(["BTC/USDT:USDT", "ETH/USDT:USDT", ...])
    await strategy.scan_and_trade(equity=10000)
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from core.strategy.cointegration import CointegrationTester


class PairsTradingStrategy:
    """통계차익 전략 — spread mean-reversion."""

    def __init__(
        self,
        exchange,
        coint_tester: CointegrationTester | None = None,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        zscore_window: int = 30,
        max_pairs: int = 3,
    ):
        self.exchange = exchange
        self.coint = coint_tester or CointegrationTester()
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.zscore_window = zscore_window
        self.max_pairs = max_pairs

        # 활성 페어 {"(a,b)": {hedge_ratio, spread_mean, std, opened_at, side, ...}}
        self.active_pairs: dict[str, dict] = {}
        self.candidates: list[dict] = []
        self.last_discovery: datetime | None = None

    async def discover_pairs(
        self,
        symbols: list[str],
        fetch_prices_fn,
    ) -> list[dict]:
        """공적분 페어 탐색 — fetch_prices_fn(symbol) → pd.Series.

        Args:
            fetch_prices_fn: 심볼별 가격 시계열 반환 함수 (async 또는 sync)
        """
        price_dict = {}
        for s in symbols:
            try:
                if hasattr(fetch_prices_fn, "__await__") or hasattr(fetch_prices_fn, "__call__"):
                    data = fetch_prices_fn(s)
                    if hasattr(data, "__await__"):
                        data = await data
                    price_dict[s] = pd.Series(data)
            except Exception as e:
                logger.debug(f"[Pairs] {s} 가격 로드 실패: {e}")

        if len(price_dict) < 2:
            logger.warning("[Pairs] 심볼 2개 미만 — 페어 탐색 불가")
            return []

        pairs = self.coint.find_pairs(price_dict, max_pairs=self.max_pairs)
        self.candidates = pairs
        self.last_discovery = datetime.utcnow()
        for p in pairs:
            logger.info(
                f"[Pairs] 공적분 페어 발견: {p['a']}~{p['b']} "
                f"β={p['hedge_ratio']} p={p['p_value']:.4f} half_life={p['half_life']}"
            )
        return pairs

    def compute_zscore(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        hedge_ratio: float,
    ) -> float:
        """현재 spread z-score 반환."""
        spread = self.coint.compute_spread(price_a.values, price_b.values, hedge_ratio)
        if len(spread) < self.zscore_window:
            return 0.0
        window = spread[-self.zscore_window:]
        mean = float(np.mean(window))
        std = float(np.std(window))
        if std < 1e-12:
            return 0.0
        return (spread[-1] - mean) / std

    def signal(
        self,
        pair_info: dict,
        price_a: pd.Series,
        price_b: pd.Series,
    ) -> dict:
        """현재 시점의 진입/청산 시그널.

        Returns:
            {"action": "enter_long"|"enter_short"|"exit"|"hold", "z": float, "side_a": str, "side_b": str}
        """
        z = self.compute_zscore(price_a, price_b, pair_info["hedge_ratio"])
        pair_key = f"{pair_info['a']}~{pair_info['b']}"
        active = self.active_pairs.get(pair_key)

        if active:
            # 청산 조건: |z| < exit_z 또는 stop_z 초과 (blowup)
            if abs(z) < self.exit_z:
                return {"action": "exit", "z": z, "reason": "mean_reverted"}
            if abs(z) > self.stop_z:
                return {"action": "exit", "z": z, "reason": "stop_loss"}
            return {"action": "hold", "z": z}

        # 진입 조건
        if z > self.entry_z:
            # spread overpriced: a/β·b 초과 → a 숏, b 롱
            return {
                "action": "enter_short", "z": z,
                "side_a": "short", "side_b": "long",
                "pair_key": pair_key,
            }
        if z < -self.entry_z:
            return {
                "action": "enter_long", "z": z,
                "side_a": "long", "side_b": "short",
                "pair_key": pair_key,
            }
        return {"action": "hold", "z": z}

    async def scan_and_trade(
        self,
        equity: float,
        fetch_prices_fn,
        order_manager=None,
        notional_per_pair: float | None = None,
    ) -> list[dict]:
        """모든 후보 페어를 스캔하여 시그널 기반 체결.

        Args:
            equity: 총 자본
            fetch_prices_fn: (symbol) → pd.Series
            order_manager: OrderManager 인스턴스 (실주문용). None이면 dry-run.
            notional_per_pair: 페어당 한쪽 다리 notional (기본 equity × 5%)
        """
        if not self.candidates:
            logger.debug("[Pairs] 후보 페어 없음 — discover_pairs 먼저 실행")
            return []

        notional = notional_per_pair or (equity * 0.05)
        results = []

        for pair in self.candidates:
            try:
                price_a = pd.Series(fetch_prices_fn(pair["a"]))
                price_b = pd.Series(fetch_prices_fn(pair["b"]))
                if hasattr(price_a, "__await__"):
                    price_a = await price_a
                if hasattr(price_b, "__await__"):
                    price_b = await price_b
            except Exception as e:
                logger.debug(f"[Pairs] {pair['a']}~{pair['b']} 가격 로드 실패: {e}")
                continue

            sig = self.signal(pair, price_a, price_b)
            log_msg = (
                f"[Pairs] {pair['a']}~{pair['b']} z={sig['z']:+.2f} → {sig['action']}"
            )
            logger.info(log_msg)

            if sig["action"] in ("enter_long", "enter_short") and order_manager:
                # 두 다리 동시 체결 — delta-neutral
                await self._open_pair(pair, sig, price_a.iloc[-1], price_b.iloc[-1], notional, order_manager)
            elif sig["action"] == "exit" and order_manager:
                await self._close_pair(pair, order_manager)

            results.append({"pair": f"{pair['a']}~{pair['b']}", **sig})

        return results

    async def _open_pair(
        self,
        pair: dict,
        sig: dict,
        price_a: float,
        price_b: float,
        notional: float,
        om,
    ):
        """페어 양쪽 다리 체결."""
        hedge_ratio = pair["hedge_ratio"]
        # a 다리: notional → amount_a
        # b 다리: notional × hedge_ratio (대략 delta-neutral)
        amount_a = notional / price_a
        amount_b = (notional * abs(hedge_ratio)) / price_b

        try:
            pos_a = await om.open_position(
                pair["a"], sig["side_a"], notional, leverage=1, trade_type="pair"
            )
            pos_b = await om.open_position(
                pair["b"], sig["side_b"], notional * abs(hedge_ratio), leverage=1, trade_type="pair"
            )
            self.active_pairs[sig["pair_key"]] = {
                "hedge_ratio": hedge_ratio,
                "opened_at": datetime.utcnow().isoformat(),
                "side_a": sig["side_a"],
                "side_b": sig["side_b"],
                "entry_z": sig["z"],
            }
            logger.info(f"[Pairs] OPEN {sig['pair_key']} @ z={sig['z']:+.2f}")
        except Exception as e:
            logger.error(f"[Pairs] open 실패: {e}")

    async def _close_pair(self, pair: dict, om):
        """페어 양쪽 다리 청산."""
        key = f"{pair['a']}~{pair['b']}"
        try:
            await om.close_position(pair["a"], reason="pair_exit")
            await om.close_position(pair["b"], reason="pair_exit")
            self.active_pairs.pop(key, None)
            logger.info(f"[Pairs] CLOSE {key}")
        except Exception as e:
            logger.error(f"[Pairs] close 실패: {e}")

    def status(self) -> dict:
        return {
            "candidates": len(self.candidates),
            "active": len(self.active_pairs),
            "active_pairs": list(self.active_pairs.keys()),
            "last_discovery": self.last_discovery.isoformat() if self.last_discovery else None,
        }
