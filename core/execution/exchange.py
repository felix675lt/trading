"""거래소 추상화 레이어 - ccxt 기반 통합 인터페이스"""

import asyncio
from typing import Optional

import ccxt.async_support as ccxt
from loguru import logger


class ExchangeClient:
    """ccxt 래퍼 - 거래소 주문 및 포지션 관리"""

    def __init__(self, exchange_name: str, config: dict):
        self.name = exchange_name
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            "apiKey": config.get("api_key", ""),
            "secret": config.get("secret", ""),
            "options": config.get("options", {}),
            "enableRateLimit": True,
        })
        if config.get("testnet"):
            self.exchange.set_sandbox_mode(True)
            logger.info(f"{exchange_name}: 테스트넷 모드")

    async def close(self):
        await self.exchange.close()

    async def get_balance(self) -> dict:
        balance = await self.exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        return {
            "total": usdt.get("total", 0),
            "free": usdt.get("free", 0),
            "used": usdt.get("used", 0),
        }

    async def set_leverage(self, symbol: str, leverage: int):
        try:
            await self.exchange.set_leverage(leverage, symbol)
            logger.info(f"레버리지 설정: {symbol} x{leverage}")
        except Exception as e:
            logger.warning(f"레버리지 설정 실패: {e}")

    async def create_market_order(self, symbol: str, side: str, amount: float) -> dict:
        """시장가 주문"""
        try:
            order = await self.exchange.create_order(symbol, "market", side, amount)
            logger.info(f"시장가 주문 체결: {side} {amount} {symbol} @ {order.get('average', 'N/A')}")
            return order
        except Exception as e:
            logger.error(f"주문 실패: {e}")
            raise

    async def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> dict:
        """지정가 주문"""
        try:
            order = await self.exchange.create_order(symbol, "limit", side, amount, price)
            logger.info(f"지정가 주문: {side} {amount} {symbol} @ {price}")
            return order
        except Exception as e:
            logger.error(f"주문 실패: {e}")
            raise

    async def create_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float) -> dict:
        """스탑로스 주문"""
        try:
            params = {"stopPrice": stop_price, "reduceOnly": True}
            order = await self.exchange.create_order(symbol, "stop_market", side, amount, None, params)
            logger.info(f"스탑로스 설정: {side} {amount} {symbol} @ {stop_price}")
            return order
        except Exception as e:
            logger.warning(f"스탑로스 설정 실패: {e}")
            return {}

    async def get_position(self, symbol: str) -> dict:
        """현재 포지션 조회"""
        try:
            positions = await self.exchange.fetch_positions([symbol])
            for pos in positions:
                contracts = pos.get("contracts", 0) or 0
                if float(contracts) > 0:
                    lev = pos.get("leverage")
                    leverage = int(lev) if lev is not None else 1
                    entry = pos.get("entryPrice") or 0
                    upnl = pos.get("unrealizedPnl") or 0
                    return {
                        "symbol": symbol,
                        "side": pos.get("side", ""),
                        "size": float(contracts),
                        "entry_price": float(entry),
                        "unrealized_pnl": float(upnl),
                        "leverage": leverage,
                    }
            return {"symbol": symbol, "side": "", "size": 0, "entry_price": 0, "unrealized_pnl": 0}
        except Exception as e:
            logger.error(f"포지션 조회 실패: {e}")
            return {}

    async def close_position(self, symbol: str, fallback_side: str = "", fallback_size: float = 0) -> dict:
        """포지션 전체 청산 — get_position 실패 시 fallback 사용"""
        pos = await self.get_position(symbol)
        size = pos.get("size", 0)
        side_raw = pos.get("side", "")

        # get_position 실패 또는 빈 결과 시 fallback 사용
        if size == 0 and fallback_size > 0:
            size = fallback_size
            side_raw = fallback_side
            logger.warning(f"포지션 조회 실패 → fallback 사용: {fallback_side} {fallback_size}")

        if size == 0:
            return {}

        close_side = "sell" if side_raw == "long" else "buy"
        return await self.create_market_order(symbol, close_side, size)

    async def cancel_all_orders(self, symbol: str):
        """모든 미체결 주문 취소"""
        try:
            orders = await self.exchange.fetch_open_orders(symbol)
            for order in orders:
                await self.exchange.cancel_order(order["id"], symbol)
            logger.info(f"{symbol} 미체결 주문 {len(orders)}건 취소")
        except Exception as e:
            logger.warning(f"주문 취소 실패: {e}")

    async def get_ticker_price(self, symbol: str) -> float:
        ticker = await self.exchange.fetch_ticker(symbol)
        return float(ticker["last"])
