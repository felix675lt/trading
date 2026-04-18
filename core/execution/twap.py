"""TWAP (Time-Weighted Average Price) Execution — 대형 주문 분할 체결.

tier=large+ 에서 활성화. 단일 시장가로 넣으면 slippage 커지므로
주문을 N개로 쪼개 일정 간격으로 나눠 낸다. 각 청크는 limit-first 시도 후
미체결 시 market fallback.

사용:
    twap = TWAPExecutor(order_manager)
    order = await twap.execute(symbol, "buy", total_amount=0.5, n_slices=5, duration_seconds=60)
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime

from loguru import logger


class TWAPExecutor:
    """N-slice TWAP 체결기 — OrderManager에 위임."""

    def __init__(
        self,
        order_manager,
        default_slices: int = 5,
        default_duration_s: int = 60,
    ):
        self.om = order_manager
        self.default_slices = default_slices
        self.default_duration_s = default_duration_s

    async def execute(
        self,
        symbol: str,
        order_side: str,  # "buy" or "sell"
        total_amount: float,
        n_slices: int | None = None,
        duration_seconds: int | None = None,
    ) -> dict:
        """TWAP 실행.

        Returns:
            {"total_filled": float, "avg_price": float, "slices": [...], "full_fill": bool}
        """
        n_slices = n_slices or self.default_slices
        duration = duration_seconds or self.default_duration_s
        slice_amount = total_amount / n_slices
        interval = max(1.0, duration / n_slices)

        # 최소 주문 수량 체크 — step 크기 이하면 슬라이스 수 줄임
        try:
            step = await self.om.exchange.get_amount_precision(symbol) or 0.001
            if slice_amount < step:
                n_slices = max(1, int(total_amount / step))
                slice_amount = total_amount / n_slices
                interval = max(1.0, duration / n_slices)
        except Exception:
            pass

        logger.info(
            f"[TWAP] {order_side} {total_amount:.6f} {symbol} "
            f"→ {n_slices} slices × {slice_amount:.6f} (interval={interval:.1f}s)"
        )

        slices_result = []
        total_filled = 0.0
        total_notional = 0.0

        for i in range(n_slices):
            started = datetime.utcnow()
            order = None
            # 1) limit-first 우선
            if self.om.limit_first_enabled:
                try:
                    order = await self.om._try_limit_first(symbol, order_side, slice_amount)
                except Exception as e:
                    logger.debug(f"[TWAP] slice {i+1} limit 실패: {e}")

            # 2) fallback market
            if order is None:
                try:
                    order = await self.om.exchange.create_market_order(symbol, order_side, slice_amount)
                except Exception as e:
                    logger.error(f"[TWAP] slice {i+1} market 실패: {e}")
                    slices_result.append({"slice": i + 1, "filled": 0, "error": str(e)})
                    continue

            filled = float(order.get("filled", slice_amount) or slice_amount)
            price = float(order.get("average", 0) or order.get("price", 0) or 0)
            total_filled += filled
            total_notional += filled * price
            slices_result.append({
                "slice": i + 1,
                "filled": filled,
                "price": price,
                "elapsed_s": (datetime.utcnow() - started).total_seconds(),
            })
            logger.info(
                f"[TWAP] slice {i+1}/{n_slices} filled={filled:.6f} @ {price:.4f}"
            )

            # 마지막 슬라이스 아니면 대기
            if i < n_slices - 1:
                await asyncio.sleep(interval)

        avg_price = total_notional / total_filled if total_filled > 0 else 0.0
        full_fill = math.isclose(total_filled, total_amount, rel_tol=0.02)
        logger.info(
            f"[TWAP] 완료 | filled={total_filled:.6f}/{total_amount:.6f} "
            f"avg=${avg_price:.4f} full_fill={full_fill}"
        )
        return {
            "total_filled": total_filled,
            "avg_price": avg_price,
            "slices": slices_result,
            "full_fill": full_fill,
        }
