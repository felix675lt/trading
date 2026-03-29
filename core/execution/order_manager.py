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
    # 트레일링 스탑 상태
    highest_price: float = 0.0   # 진입 후 최고가 (롱)
    lowest_price: float = 0.0    # 진입 후 최저가 (숏)
    trailing_activated: bool = False

    def __post_init__(self):
        if not self.opened_at:
            self.opened_at = str(datetime.utcnow())
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
        if self.lowest_price == 0.0:
            self.lowest_price = self.entry_price


class OrderManager:
    """주문 실행 및 포지션 라이프사이클 관리"""

    def __init__(self, exchange: ExchangeClient, risk_config: dict,
                 trailing_config: dict | None = None):
        self.exchange = exchange
        self.risk_config = risk_config
        self.positions: dict[str, Position] = {}
        self._failed_attempts: dict[str, int] = {}  # 연속 실패 횟수
        self._on_sl_callback = None  # SL 소멸 감지 콜백
        self._on_tp_callback = None  # TP 청산 콜백
        self._last_close_time: dict[str, datetime] = {}  # 심볼별 마지막 청산 시각
        self._last_close_side: dict[str, str] = {}  # 심볼별 마지막 포지션 방향
        self._consecutive_sl: dict[str, int] = {}  # 심볼별 연속 SL 횟수
        self._min_reentry_minutes = 5  # SL 후 최소 재진입 대기 시간

        # 트레일링 스탑 설정 (PaperTrader와 동일)
        tc = trailing_config or {}
        self.trailing_activate_pct = tc.get("activate_pct", 0.015)
        self.trailing_distance_pct = tc.get("distance_pct", 0.008)
        self.trailing_step_pct = tc.get("step_pct", 0.004)

    def set_sl_callback(self, callback):
        """SL/자동청산 감지 시 호출할 콜백 등록 (피드백 학습용)"""
        self._on_sl_callback = callback

    def set_tp_callback(self, callback):
        """TP 청산 시 호출할 콜백 등록 (피드백 학습용)"""
        self._on_tp_callback = callback

    def can_reenter(self, symbol: str, side: str) -> tuple[bool, str]:
        """재진입 가능 여부 확인
        - SL 직후: 최소 5분 × 연속SL횟수 대기
        - TP 직후: 최소 3분 대기 (같은 가격대 재진입 방지)
        - 같은 방향 2연속 SL: 방향 전환 요구
        """
        last_close = self._last_close_time.get(symbol)
        if not last_close:
            return True, ""

        elapsed = (datetime.utcnow() - last_close).total_seconds() / 60
        consecutive = self._consecutive_sl.get(symbol, 0)

        if consecutive > 0:
            # SL 후 대기 (연속 SL일수록 길어짐)
            wait_minutes = self._min_reentry_minutes * max(1, consecutive)
            if elapsed < wait_minutes:
                remaining = wait_minutes - elapsed
                return False, f"SL 후 대기: {remaining:.0f}분 남음 ({consecutive}연속SL)"

            # 같은 방향 2연속 SL → 방향 전환 요구
            last_side = self._last_close_side.get(symbol, "")
            if consecutive >= 2 and side == last_side:
                return False, f"같은방향 {side} {consecutive}연속SL → 방향전환 필요"
        else:
            # TP 후에도 최소 3분 대기 (같은 가격대 재진입 방지 + 수수료 절약)
            if elapsed < 3:
                remaining = 3 - elapsed
                return False, f"TP 후 대기: {remaining:.1f}분 남음 (같은가격대 재진입 방지)"

        return True, ""

    def record_close(self, symbol: str, side: str, was_sl: bool):
        """청산 기록 (재진입 제어용)"""
        self._last_close_time[symbol] = datetime.utcnow()
        self._last_close_side[symbol] = side
        if was_sl:
            self._consecutive_sl[symbol] = self._consecutive_sl.get(symbol, 0) + 1
        else:
            self._consecutive_sl[symbol] = 0  # TP 시 리셋

    async def recover_positions(self, symbols: list[str]) -> list[Position]:
        """시스템 재시작 시 거래소의 기존 포지션 복구 + SL 재설정"""
        recovered = []
        for symbol in symbols:
            try:
                exchange_pos = await self.exchange.get_position(symbol)
                size = exchange_pos.get("size", 0)
                if size == 0:
                    continue

                side = exchange_pos.get("side", "")
                entry_price = exchange_pos.get("entry_price", 0)
                leverage = exchange_pos.get("leverage", 3)

                if not side or entry_price == 0:
                    continue

                # SL/TP 계산
                sl_pct = self.risk_config.get("stop_loss_pct", 0.008)
                tp_pct = self.risk_config.get("take_profit_pct", 0.012)

                if side == "long":
                    stop_loss = entry_price * (1 - sl_pct)
                    take_profit = entry_price * (1 + tp_pct)
                else:
                    stop_loss = entry_price * (1 + sl_pct)
                    take_profit = entry_price * (1 - tp_pct)

                # 기존 Algo 주문 확인 (SL/TP)
                algo_orders = await self.exchange.get_algo_orders(symbol)
                has_sl = any(
                    o.get("orderType", "").upper() in ("STOP_MARKET", "STOP")
                    for o in algo_orders
                )
                has_tp = any(
                    o.get("orderType", "").upper() in ("TAKE_PROFIT_MARKET", "TAKE_PROFIT")
                    for o in algo_orders
                )

                # 기존 주문 전부 취소 후 깨끗하게 재설정
                if algo_orders:
                    await self.exchange.cancel_all_orders(symbol)
                    has_sl = False
                    has_tp = False

                close_side = "sell" if side == "long" else "buy"

                # SL 설정
                if not has_sl:
                    await self.exchange.create_stop_loss(symbol, close_side, size, stop_loss)
                    logger.warning(f"[복구] {symbol} SL 설정: {stop_loss:.4f}")

                # TP 설정
                if not has_tp:
                    await self.exchange.create_take_profit(symbol, close_side, size, take_profit)
                    logger.warning(f"[복구] {symbol} TP 설정: {take_profit:.4f}")

                position = Position(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )
                self.positions[symbol] = position
                recovered.append(position)
                logger.info(
                    f"[복구] 포지션 복구: {side} {size} {symbol} @ {entry_price:.4f} "
                    f"| SL: {stop_loss:.4f} TP: {take_profit:.4f}"
                )

            except Exception as e:
                logger.warning(f"[복구] {symbol} 포지션 확인 실패: {e}")

        return recovered

    async def open_position(
        self,
        symbol: str,
        side: str,
        size_usdt: float,
        leverage: int = 5,
        sl_pct: float | None = None,
        tp_pct: float | None = None,
        atr_pct: float | None = None,
    ) -> Position | None:
        """포지션 개시 — ATR 기반 동적 SL/TP 지원"""
        if symbol in self.positions:
            logger.warning(f"{symbol} 이미 포지션 보유 중")
            return None

        # SL 후 재진입 체크
        can, reason = self.can_reenter(symbol, side)
        if not can:
            logger.info(f"[재진입차단] {symbol} {side}: {reason}")
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

            # === ATR 기반 동적 SL/TP 계산 ===
            sl_floor = self.risk_config.get("sl_floor_pct", 0.005)
            sl_cap = self.risk_config.get("sl_cap_pct", 0.030)

            if atr_pct and atr_pct > 0 and not (atr_pct != atr_pct):  # NaN 체크
                # ATR 기반: 실제 시장 변동성에 맞춤
                atr_sl_mult = self.risk_config.get("atr_sl_multiplier", 2.0)
                atr_tp_mult = self.risk_config.get("atr_tp_multiplier", 3.5)
                final_sl_pct = max(sl_floor, min(sl_cap, atr_pct * atr_sl_mult))
                final_tp_pct = max(final_sl_pct * 1.5, atr_pct * atr_tp_mult)  # 최소 RR 1.5:1
                logger.info(
                    f"[ATR-SL/TP] {symbol} ATR={atr_pct*100:.2f}% → "
                    f"SL={final_sl_pct*100:.2f}% TP={final_tp_pct*100:.2f}% "
                    f"(RR {final_tp_pct/final_sl_pct:.1f}:1)"
                )
            else:
                # fallback: 전달된 값 또는 config 기본값
                final_sl_pct = sl_pct or self.risk_config.get("stop_loss_pct", 0.015)
                final_tp_pct = tp_pct or self.risk_config.get("take_profit_pct", 0.025)
                final_sl_pct = max(sl_floor, min(sl_cap, final_sl_pct))

            if side == "long":
                stop_loss = fill_price * (1 - final_sl_pct)
                take_profit = fill_price * (1 + final_tp_pct)
            else:
                stop_loss = fill_price * (1 + final_sl_pct)
                take_profit = fill_price * (1 - final_tp_pct)

            # 스탑로스 + 테이크프로핏 주문 (거래소 Algo Order)
            sl_side = "sell" if side == "long" else "buy"
            await self.exchange.create_stop_loss(symbol, sl_side, amount, stop_loss)
            await self.exchange.create_take_profit(symbol, sl_side, amount, take_profit)

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

            # 4. 재진입 제어용 기록 (SL인지 TP인지 판별)
            was_sl = pnl < 0
            self.record_close(symbol, pos.side, was_sl)

            logger.info(f"포지션 청산: {symbol} | PnL: {pnl:.2f} USDT | 사유: {reason} | {'SL' if was_sl else 'TP'}")

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

                # 2. 거래소에 포지션이 없음 → SL/TP 체결로 이미 청산됨
                if exchange_size == 0:
                    # 마지막 가격으로 SL인지 TP인지 판별
                    try:
                        last_price = await self.exchange.get_ticker_price(symbol)
                    except Exception:
                        last_price = pos.stop_loss  # 조회 실패 시 SL로 간주

                    # TP 방향으로 청산됐는지 판별
                    if pos.side == "long":
                        was_tp = last_price >= pos.take_profit * 0.998  # TP 근처
                        exit_price = pos.take_profit if was_tp else pos.stop_loss
                        pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.size * pos.entry_price
                    else:
                        was_tp = last_price <= pos.take_profit * 1.002
                        exit_price = pos.take_profit if was_tp else pos.stop_loss
                        pnl = (pos.entry_price - exit_price) / pos.entry_price * pos.size * pos.entry_price

                    close_type = "TP" if was_tp else "SL"
                    was_sl = not was_tp

                    logger.info(
                        f"[OrderManager] {symbol} 거래소에서 포지션 소멸 감지 "
                        f"({close_type} 체결) → 추정 PnL: ${pnl:.2f} | 내부 정리 + 잔여 주문 취소"
                    )

                    # 재진입 제어 기록
                    self.record_close(symbol, pos.side, was_sl=was_sl)

                    # 콜백 호출 (학습용)
                    callback_data = {
                        "symbol": symbol,
                        "side": pos.side,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "size": pos.size,
                        "pnl": pnl,
                        "reason": f"{close_type} 체결 (거래소 자동)",
                        "close_type": close_type,
                    }

                    if was_sl and self._on_sl_callback:
                        try:
                            self._on_sl_callback(callback_data)
                        except Exception as cb_e:
                            logger.debug(f"SL 콜백 실패: {cb_e}")
                    elif was_tp and self._on_tp_callback:
                        try:
                            self._on_tp_callback(callback_data)
                        except Exception as cb_e:
                            logger.debug(f"TP 콜백 실패: {cb_e}")

                    # 잔여 오픈 오더 전부 취소
                    await self.exchange.cancel_all_orders(symbol)
                    del self.positions[symbol]
                    continue

                # 3. 현재가 조회 및 PnL 업데이트 + 트레일링 스탑
                price = await self.exchange.get_ticker_price(symbol)

                if pos.side == "long":
                    pos.unrealized_pnl = (price - pos.entry_price) / pos.entry_price
                    pos.highest_price = max(pos.highest_price, price)
                    profit_pct = (price - pos.entry_price) / pos.entry_price

                    # 트레일링 활성화 체크
                    if profit_pct >= self.trailing_activate_pct and not pos.trailing_activated:
                        pos.trailing_activated = True
                        new_sl = pos.highest_price * (1 - self.trailing_distance_pct)
                        if new_sl > pos.stop_loss:
                            old_sl = pos.stop_loss
                            pos.stop_loss = new_sl
                            await self._update_exchange_sl(symbol, pos)
                            logger.info(
                                f"[Trailing-LIVE] {symbol} 롱 트레일링 활성화 | "
                                f"수익 {profit_pct:.2%} | SL {old_sl:.4f} → {pos.stop_loss:.4f}"
                            )

                    # 트레일링 활성 중: 최고가 갱신 시 SL 끌어올림
                    if pos.trailing_activated:
                        new_sl = pos.highest_price * (1 - self.trailing_distance_pct)
                        if new_sl > pos.stop_loss + (pos.entry_price * self.trailing_step_pct):
                            old_sl = pos.stop_loss
                            pos.stop_loss = new_sl
                            await self._update_exchange_sl(symbol, pos)
                            logger.info(
                                f"[Trailing-LIVE] {symbol} 롱 SL 상향 | "
                                f"{old_sl:.4f} → {pos.stop_loss:.4f} | 최고가: {pos.highest_price:.4f}"
                            )

                    # TP 도달 → 트레일링으로 전환 (즉시 청산 안 함)
                    if price >= pos.take_profit and not pos.trailing_activated:
                        pos.trailing_activated = True
                        old_sl = pos.stop_loss
                        pos.stop_loss = pos.entry_price * (1 + self.trailing_activate_pct)
                        pos.take_profit = pos.entry_price * (1 + 0.20)  # TP를 아주 높게 → 실질 무한
                        await self._update_exchange_sl_tp(symbol, pos)
                        logger.info(
                            f"[Trailing-LIVE] {symbol} 롱 TP 도달 → 트레일링 전환 | "
                            f"수익확보선: {pos.stop_loss:.4f} (이전SL: {old_sl:.4f})"
                        )

                else:  # short
                    pos.unrealized_pnl = (pos.entry_price - price) / pos.entry_price
                    pos.lowest_price = min(pos.lowest_price, price) if pos.lowest_price > 0 else price
                    profit_pct = (pos.entry_price - price) / pos.entry_price

                    # 트레일링 활성화 체크
                    if profit_pct >= self.trailing_activate_pct and not pos.trailing_activated:
                        pos.trailing_activated = True
                        new_sl = pos.lowest_price * (1 + self.trailing_distance_pct)
                        if new_sl < pos.stop_loss:
                            old_sl = pos.stop_loss
                            pos.stop_loss = new_sl
                            await self._update_exchange_sl(symbol, pos)
                            logger.info(
                                f"[Trailing-LIVE] {symbol} 숏 트레일링 활성화 | "
                                f"수익 {profit_pct:.2%} | SL {old_sl:.4f} → {pos.stop_loss:.4f}"
                            )

                    # 트레일링 활성 중: 최저가 갱신 시 SL 끌어내림
                    if pos.trailing_activated:
                        new_sl = pos.lowest_price * (1 + self.trailing_distance_pct)
                        if new_sl < pos.stop_loss - (pos.entry_price * self.trailing_step_pct):
                            old_sl = pos.stop_loss
                            pos.stop_loss = new_sl
                            await self._update_exchange_sl(symbol, pos)
                            logger.info(
                                f"[Trailing-LIVE] {symbol} 숏 SL 하향 | "
                                f"{old_sl:.4f} → {pos.stop_loss:.4f} | 최저가: {pos.lowest_price:.4f}"
                            )

                    # TP 도달 → 트레일링으로 전환
                    if price <= pos.take_profit and not pos.trailing_activated:
                        pos.trailing_activated = True
                        old_sl = pos.stop_loss
                        pos.stop_loss = pos.entry_price * (1 - self.trailing_activate_pct)
                        pos.take_profit = pos.entry_price * (1 - 0.20)  # TP를 아주 낮게 → 실질 무한
                        await self._update_exchange_sl_tp(symbol, pos)
                        logger.info(
                            f"[Trailing-LIVE] {symbol} 숏 TP 도달 → 트레일링 전환 | "
                            f"수익확보선: {pos.stop_loss:.4f} (이전SL: {old_sl:.4f})"
                        )

            except Exception as e:
                logger.warning(f"포지션 업데이트 실패 ({symbol}): {e}")

    async def _update_exchange_sl(self, symbol: str, pos: Position):
        """거래소 SL 주문을 취소 후 새 가격으로 재설정"""
        try:
            close_side = "sell" if pos.side == "long" else "buy"
            # 기존 주문 전부 취소
            await self.exchange.cancel_all_orders(symbol)
            # SL 재설정
            await self.exchange.create_stop_loss(symbol, close_side, pos.size, pos.stop_loss)
            # TP 유지 (take_profit이 inf가 아닌 경우)
            if pos.take_profit > 0 and pos.take_profit < pos.entry_price * 5:
                await self.exchange.create_take_profit(symbol, close_side, pos.size, pos.take_profit)
        except Exception as e:
            logger.error(f"[Trailing-LIVE] {symbol} SL 거래소 업데이트 실패: {e}")

    async def _update_exchange_sl_tp(self, symbol: str, pos: Position):
        """거래소 SL + TP 모두 취소 후 새 값으로 재설정"""
        try:
            close_side = "sell" if pos.side == "long" else "buy"
            await self.exchange.cancel_all_orders(symbol)
            await self.exchange.create_stop_loss(symbol, close_side, pos.size, pos.stop_loss)
            # TP 전환된 경우에도 아주 먼 TP를 설정 (안전장치)
            await self.exchange.create_take_profit(symbol, close_side, pos.size, pos.take_profit)
        except Exception as e:
            logger.error(f"[Trailing-LIVE] {symbol} SL/TP 거래소 업데이트 실패: {e}")

    def get_all_positions(self) -> list[dict]:
        return [
            {
                "symbol": p.symbol, "side": p.side, "size": p.size,
                "entry_price": p.entry_price, "unrealized_pnl": p.unrealized_pnl,
                "stop_loss": p.stop_loss, "take_profit": p.take_profit,
            }
            for p in self.positions.values()
        ]
