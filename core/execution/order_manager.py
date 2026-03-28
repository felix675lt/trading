"""주문 관리자 - 포지션 추적, 주문 실행, 스탑로스/TP 관리"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from core.execution.exchange import ExchangeClient


@dataclass
class Position:
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    opened_at: str = ""
    unrealized_pnl: float = 0.0

    def __post_init__(self):
        if not self.opened_at:
            self.opened_at = str(datetime.utcnow())


class OrderManager:
    """주문 실행 및 포지션 라이프사이클 관리"""

    def __init__(self, exchange: ExchangeClient, risk_config: dict):
        self.exchange = exchange
        self.risk_config = risk_config
        self.positions: dict[str, Position] = {}
        self._failed_attempts: dict[str, int] = {}  # 연속 실패 횟수
        self._on_sl_callback = None  # SL 소멸 감지 콜백

    def set_sl_callback(self, callback):
        """SL/자동청산 감지 시 호출할 콜백 등록 (피드백 학습용)"""
        self._on_sl_callback = callback

    async def open_position(
        self,
        symbol: str,
        side: str,
        size_usdt: float,
        leverage: int = 5,
    ) -> Position | None:
        """포지션 개시"""
        if symbol in self.positions:
            logger.warning(f"{symbol} 이미 포지션 보유 중")
            return None

        # 연속 3회 실패 시 10루프 동안 스킵
        fails = self._failed_attempts.get(symbol, 0)
        if fails >= 3 and fails < 13:
            self._failed_attempts[symbol] = fails + 1
            return None
        elif fails >= 13:
            self._failed_attempts[symbol] = 0  # 쿨다운 끝

        try:
            await self.exchange.set_leverage(symbol, leverage)
            price = await self.exchange.get_ticker_price(symbol)

            # 거래소에서 심볼별 수량 정밀도 및 최소 notional 조회
            import math
            precision = await self.exchange.get_amount_precision(symbol)
            step = precision if precision > 0 else 0.001
            raw_amount = (size_usdt * leverage) / price
            # step size 단위로 올림
            amount = math.ceil(raw_amount / step) * step

            # 최소 notional 미달 시 수량 증가
            min_notional = await self.exchange.get_min_notional(symbol)
            notional = amount * price
            if notional < min_notional:
                amount = math.ceil(min_notional / price / step) * step

            order = await self.exchange.create_market_order(symbol, "buy" if side == "long" else "sell", amount)
            fill_price = float(order.get("average", price))

            # 스탑로스/TP 계산
            sl_pct = self.risk_config.get("stop_loss_pct", 0.02)
            tp_pct = self.risk_config.get("take_profit_pct", 0.04)

            if side == "long":
                stop_loss = fill_price * (1 - sl_pct)
                take_profit = fill_price * (1 + tp_pct)
            else:
                stop_loss = fill_price * (1 + sl_pct)
                take_profit = fill_price * (1 - tp_pct)

            # 스탑로스 주문
            sl_side = "sell" if side == "long" else "buy"
            await self.exchange.create_stop_loss(symbol, sl_side, amount, stop_loss)

            position = Position(
                symbol=symbol,
                side=side,
                size=amount,
                entry_price=fill_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            self.positions[symbol] = position
            self._failed_attempts.pop(symbol, None)
            logger.info(f"포지션 개시: {side} {amount:.6f} {symbol} @ {fill_price:.2f} | SL: {stop_loss:.2f} TP: {take_profit:.2f}")
            return position

        except Exception as e:
            self._failed_attempts[symbol] = self._failed_attempts.get(symbol, 0) + 1
            logger.error(f"포지션 개시 실패 (시도 {self._failed_attempts[symbol]}): {e}")
            return None

    async def close_position(self, symbol: str, reason: str = "") -> dict:
        """포지션 청산 — 실패 시 포지션 유지, 재시도 가능
        청산 완료 시 잔여 오픈 오더(SL/TP) 전부 취소
        """
        if symbol not in self.positions:
            return {}

        pos = self.positions[symbol]
        try:
            # 1. 먼저 대기 주문(SL/TP) 전부 취소
            await self.exchange.cancel_all_orders(symbol)

            # 2. 포지션 청산 (fallback으로 내부 포지션 정보 전달)
            order = await self.exchange.close_position(
                symbol,
                fallback_side=pos.side,
                fallback_size=pos.size,
            )

            if not order:
                logger.error(f"포지션 청산 주문 미실행 ({symbol}) — 포지션 유지, 다음 루프에서 재시도")
                return {}

            fill_price = float(order.get("average", 0))

            pnl = 0.0
            if fill_price > 0:
                if pos.side == "long":
                    pnl = (fill_price - pos.entry_price) / pos.entry_price * pos.size * pos.entry_price
                else:
                    pnl = (pos.entry_price - fill_price) / pos.entry_price * pos.size * pos.entry_price

            del self.positions[symbol]

            # 3. 청산 후 혹시 남아있는 잔여 주문 한 번 더 정리
            try:
                await self.exchange.cancel_all_orders(symbol)
            except Exception:
                pass

            logger.info(f"포지션 청산: {symbol} | PnL: {pnl:.2f} USDT | 사유: {reason}")

            return {
                "symbol": symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": fill_price,
                "size": pos.size,
                "pnl": pnl,
                "reason": reason,
            }
        except Exception as e:
            logger.error(f"포지션 청산 실패 ({symbol}): {e} — 포지션 유지, 다음 루프에서 재시도")
            return {}

    async def update_positions(self):
        """모든 포지션의 미실현 PnL 업데이트 및 TP 확인
        + 거래소에서 이미 청산된 포지션(SL 체결 등) 감지 → 내부 정리 + 잔여 주문 취소
        """
        for symbol, pos in list(self.positions.items()):
            try:
                # 1. 거래소 실제 포지션 확인
                exchange_pos = await self.exchange.get_position(symbol)
                exchange_size = exchange_pos.get("size", 0) if exchange_pos else 0

                # 2. 거래소에 포지션이 없음 → SL 체결 등으로 이미 청산됨
                if exchange_size == 0:
                    # SL PnL 추정 (SL가격 기준)
                    if pos.side == "long":
                        sl_pnl = (pos.stop_loss - pos.entry_price) / pos.entry_price * pos.size * pos.entry_price
                    else:
                        sl_pnl = (pos.entry_price - pos.stop_loss) / pos.entry_price * pos.size * pos.entry_price

                    logger.info(
                        f"[OrderManager] {symbol} 거래소에서 포지션 소멸 감지 "
                        f"(SL/청산 체결) → 추정 PnL: ${sl_pnl:.2f} | 내부 정리 + 잔여 주문 취소"
                    )

                    # 피드백 콜백 호출 (SL 패턴 학습용)
                    if self._on_sl_callback:
                        try:
                            self._on_sl_callback({
                                "symbol": symbol,
                                "side": pos.side,
                                "entry_price": pos.entry_price,
                                "exit_price": pos.stop_loss,
                                "size": pos.size,
                                "pnl": sl_pnl,
                                "reason": "SL 체결 (거래소 자동)",
                            })
                        except Exception as cb_e:
                            logger.debug(f"SL 콜백 실패: {cb_e}")

                    # 잔여 오픈 오더 전부 취소
                    await self.exchange.cancel_all_orders(symbol)
                    del self.positions[symbol]
                    continue

                # 3. 현재가 조회 및 PnL 업데이트
                price = await self.exchange.get_ticker_price(symbol)

                if pos.side == "long":
                    pos.unrealized_pnl = (price - pos.entry_price) / pos.entry_price
                    if price >= pos.take_profit:
                        await self.close_position(symbol, "TP 도달")
                else:
                    pos.unrealized_pnl = (pos.entry_price - price) / pos.entry_price
                    if price <= pos.take_profit:
                        await self.close_position(symbol, "TP 도달")
            except Exception as e:
                logger.warning(f"포지션 업데이트 실패 ({symbol}): {e}")

    def get_all_positions(self) -> list[dict]:
        return [
            {
                "symbol": p.symbol, "side": p.side, "size": p.size,
                "entry_price": p.entry_price, "unrealized_pnl": p.unrealized_pnl,
                "stop_loss": p.stop_loss, "take_profit": p.take_profit,
            }
            for p in self.positions.values()
        ]
