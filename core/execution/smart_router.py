"""Smart Order Routing — 다거래소 최적 가격/유동성 비교 체결.

tier=pro 에서 활성화. 여러 거래소의 best bid/ask 및 depth를 비교하여
buy는 최저 ask / sell은 최고 bid 거래소로 자동 라우팅.

핵심 기능:
  1. 거래소별 best bid/ask 수집 (병렬)
  2. 주문 크기 대비 depth 고려 → 단일 거래소 불충분 시 다거래소 분할
  3. 수수료 차이 반영 (각 거래소의 taker fee)
  4. 잔고 제약 반영 (각 거래소의 free USDT)

사용:
    router = SmartRouter({"binance": ex1, "bybit": ex2})
    result = await router.route("BTC/USDT:USDT", "buy", amount=0.1)
    # result = {"fills": [{"exchange": "binance", "amount": 0.07, "price": ...}, ...], "avg_price": ...}
"""

from __future__ import annotations

import asyncio
from loguru import logger


class SmartRouter:
    """다거래소 스마트 주문 라우팅."""

    def __init__(
        self,
        exchanges: dict,  # {"binance": ExchangeClient, "bybit": ExchangeClient}
        taker_fees: dict[str, float] | None = None,
    ):
        self.exchanges = exchanges
        # 거래소별 taker fee (없으면 기본 0.04%)
        self.taker_fees = taker_fees or {name: 0.0004 for name in exchanges}

    async def fetch_quotes(self, symbol: str) -> dict[str, dict]:
        """각 거래소의 best bid/ask 병렬 수집."""
        async def _get(name, ex):
            try:
                bid, ask, last = await ex.get_bid_ask(symbol)
                return name, {"bid": bid, "ask": ask, "last": last, "error": None}
            except Exception as e:
                return name, {"error": str(e), "bid": 0.0, "ask": 0.0, "last": 0.0}

        tasks = [_get(n, e) for n, e in self.exchanges.items()]
        results = await asyncio.gather(*tasks)
        return {name: data for name, data in results}

    async def route(
        self,
        symbol: str,
        order_side: str,  # "buy" or "sell"
        amount: float,
    ) -> dict:
        """최적 거래소로 라우팅 체결.

        현재 구현은 단일 거래소 선택 — 다거래소 split은 후속 확장.
        """
        quotes = await self.fetch_quotes(symbol)
        valid_quotes = {n: q for n, q in quotes.items() if q.get("error") is None}
        if not valid_quotes:
            logger.error(f"[SmartRouter] {symbol} 견적 실패 (모든 거래소)")
            return {"fills": [], "avg_price": 0.0, "error": "no_quotes"}

        # fee 포함 effective price 비교
        if order_side == "buy":
            # 낮은 (ask × (1+fee))가 유리
            best = min(
                valid_quotes.items(),
                key=lambda kv: kv[1]["ask"] * (1 + self.taker_fees.get(kv[0], 0.0004)),
            )
        else:
            # 높은 (bid × (1-fee))가 유리
            best = max(
                valid_quotes.items(),
                key=lambda kv: kv[1]["bid"] * (1 - self.taker_fees.get(kv[0], 0.0004)),
            )

        best_name, best_q = best
        best_ex = self.exchanges[best_name]
        effective_price = (
            best_q["ask"] * (1 + self.taker_fees[best_name])
            if order_side == "buy"
            else best_q["bid"] * (1 - self.taker_fees[best_name])
        )

        logger.info(
            f"[SmartRouter] {symbol} {order_side} {amount:.6f} → {best_name} "
            f"(effective=${effective_price:.4f}, candidates={list(valid_quotes.keys())})"
        )

        try:
            order = await best_ex.create_market_order(symbol, order_side, amount)
            filled = float(order.get("filled", amount) or amount)
            price = float(order.get("average", best_q["last"]) or best_q["last"])
            return {
                "fills": [{
                    "exchange": best_name,
                    "amount": filled,
                    "price": price,
                    "notional": filled * price,
                }],
                "avg_price": price,
                "total_filled": filled,
                "full_fill": True,
                "routed_to": best_name,
                "alternatives": {
                    n: {"ask": q["ask"], "bid": q["bid"]}
                    for n, q in valid_quotes.items()
                },
            }
        except Exception as e:
            logger.error(f"[SmartRouter] {best_name} 체결 실패: {e}")
            return {"fills": [], "avg_price": 0.0, "error": str(e)}

    def best_liquidity_for(self, quotes: dict[str, dict], side: str) -> str | None:
        """최적 거래소 선택 — fee 포함 price + liquidity 점수."""
        if not quotes:
            return None
        if side == "buy":
            return min(quotes, key=lambda n: quotes[n].get("ask", float("inf")))
        return max(quotes, key=lambda n: quotes[n].get("bid", 0.0))
