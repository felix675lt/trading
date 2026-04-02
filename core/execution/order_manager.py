"""주문 관리자 - 포지션 추적, 주문 실행, 내부 SL/TP 모니터링

거래소 Algo Order(SL/TP) 사용하지 않음.
봇이 직접 가격을 감시하고 시장가로 청산 — PAPER와 동일한 방식.
거래소에 SL 위치를 노출하지 않아 스탑 헌팅 회피.
"""

import math
from dataclasses import dataclass
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
    trade_type: str = "scalp"    # "scalp" or "swing"
    leverage: int = 1

    def __post_init__(self):
        if not self.opened_at:
            self.opened_at = str(datetime.utcnow())
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
        if self.lowest_price == 0.0:
            self.lowest_price = self.entry_price


class OrderManager:
    """주문 실행 및 포지션 라이프사이클 관리 — 내부 SL/TP 모니터링"""

    SL_COOLDOWN_SECONDS = 300  # SL 청산 후 5분간 같은 심볼 재진입 차단

    def __init__(self, exchange: ExchangeClient, risk_config: dict,
                 trailing_config: dict | None = None,
                 trade_profiles: dict | None = None):
        self.exchange = exchange
        self.risk_config = risk_config
        self.trade_profiles = trade_profiles or {}
        self.positions: dict[str, Position] = {}
        self._failed_attempts: dict[str, int] = {}
        self._on_sl_callback = None
        self._on_tp_callback = None
        self._sl_cooldown: dict[str, datetime] = {}  # symbol → SL 청산 시각

        # 트레일링 스탑 기본값
        tc = trailing_config or {}
        self.trailing_activate_pct = tc.get("activate_pct", 0.015)
        self.trailing_distance_pct = tc.get("distance_pct", 0.008)
        self.trailing_step_pct = tc.get("step_pct", 0.004)

    def _get_profile(self, trade_type: str) -> dict:
        return self.trade_profiles.get(trade_type, {})

    def _get_trailing_params(self, trade_type: str) -> tuple[float, float, float]:
        profile = self._get_profile(trade_type)
        tc = profile.get("trailing", {})
        return (
            tc.get("activate_pct", self.trailing_activate_pct),
            tc.get("distance_pct", self.trailing_distance_pct),
            tc.get("step_pct", self.trailing_step_pct),
        )

    def set_sl_callback(self, callback):
        self._on_sl_callback = callback

    def set_tp_callback(self, callback):
        self._on_tp_callback = callback

    async def recover_positions(self, symbols: list[str]) -> list[Position]:
        """시스템 재시작 시 거래소의 기존 포지션 복구 (Algo Order 없이)"""
        recovered = []
        for symbol in symbols:
            try:
                exchange_pos = await self.exchange.get_position(symbol)
                size = exchange_pos.get("size", 0) if exchange_pos else 0
                if size == 0:
                    continue

                side = exchange_pos.get("side", "")
                entry_price = exchange_pos.get("entry_price", 0)
                leverage = exchange_pos.get("leverage", 3)

                if not side or entry_price == 0:
                    continue

                # SL/TP 계산 (내부 관리용)
                sl_pct = self.risk_config.get("stop_loss_pct", 0.015)
                tp_pct = self.risk_config.get("take_profit_pct", 0.025)

                if side == "long":
                    stop_loss = entry_price * (1 - sl_pct)
                    take_profit = entry_price * (1 + tp_pct)
                else:
                    stop_loss = entry_price * (1 + sl_pct)
                    take_profit = entry_price * (1 - tp_pct)

                # 거래소에 걸린 Algo Order 전부 취소 (내부 모니터링으로 전환)
                try:
                    await self.exchange.cancel_all_orders(symbol)
                except Exception:
                    pass

                position = Position(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=leverage,
                )
                self.positions[symbol] = position
                recovered.append(position)
                logger.info(
                    f"[복구] 포지션 복구: {side} {size} {symbol} @ {entry_price:.4f} | "
                    f"내부 SL: {stop_loss:.4f} TP: {take_profit:.4f} (거래소 Algo 없음)"
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
        trade_type: str = "scalp",
    ) -> Position | None:
        """포지션 개시 — 시장가 진입만, SL/TP는 내부 모니터링"""
        if symbol in self.positions:
            logger.warning(f"{symbol} 이미 포지션 보유 중")
            return None

        # SL 쿨다운: 최근 SL 청산 후 5분간 재진입 차단
        sl_time = self._sl_cooldown.get(symbol)
        if sl_time:
            elapsed = (datetime.utcnow() - sl_time).total_seconds()
            if elapsed < self.SL_COOLDOWN_SECONDS:
                remaining = self.SL_COOLDOWN_SECONDS - elapsed
                logger.info(
                    f"[쿨다운] {symbol} SL 후 재진입 차단 | "
                    f"{elapsed:.0f}초 경과 / {self.SL_COOLDOWN_SECONDS}초 필요 "
                    f"(남은: {remaining:.0f}초)"
                )
                return None
            else:
                del self._sl_cooldown[symbol]

        # 연속 실패 쿨다운
        fails = self._failed_attempts.get(symbol, 0)
        if fails >= 3 and fails < 13:
            self._failed_attempts[symbol] = fails + 1
            return None
        elif fails >= 13:
            self._failed_attempts[symbol] = 0

        try:
            await self.exchange.set_leverage(symbol, leverage)
            price = await self.exchange.get_ticker_price(symbol)

            # 수량 정밀도 및 최소 notional 처리
            precision = await self.exchange.get_amount_precision(symbol)
            step = precision if precision > 0 else 0.001
            raw_amount = (size_usdt * leverage) / price
            amount = math.ceil(raw_amount / step) * step

            min_notional = await self.exchange.get_min_notional(symbol)
            notional = amount * price
            if notional < min_notional:
                amount = math.ceil(min_notional / price / step) * step

            # 시장가 진입만 — Algo Order 없음
            order = await self.exchange.create_market_order(
                symbol, "buy" if side == "long" else "sell", amount
            )
            fill_price = float(order.get("average", price))

            # SL/TP 계산 (내부 모니터링용)
            profile = self._get_profile(trade_type)
            sl_floor = profile.get("sl_floor_pct", self.risk_config.get("sl_floor_pct", 0.005))
            sl_cap = profile.get("sl_cap_pct", self.risk_config.get("sl_cap_pct", 0.030))

            if atr_pct and atr_pct > 0 and not (atr_pct != atr_pct):
                atr_sl_mult = profile.get("atr_sl_multiplier", self.risk_config.get("atr_sl_multiplier", 2.0))
                atr_tp_mult = profile.get("atr_tp_multiplier", self.risk_config.get("atr_tp_multiplier", 3.5))
                final_sl_pct = max(sl_floor, min(sl_cap, atr_pct * atr_sl_mult))
                final_tp_pct = max(final_sl_pct * 1.5, atr_pct * atr_tp_mult)
                logger.info(
                    f"[ATR-SL/TP] {symbol} ATR={atr_pct*100:.2f}% → "
                    f"SL={final_sl_pct*100:.2f}% TP={final_tp_pct*100:.2f}% "
                    f"(RR {final_tp_pct/final_sl_pct:.1f}:1)"
                )
            else:
                final_sl_pct = sl_pct or profile.get("sl_pct", self.risk_config.get("stop_loss_pct", 0.015))
                final_tp_pct = tp_pct or profile.get("tp_pct", self.risk_config.get("take_profit_pct", 0.025))
                final_sl_pct = max(sl_floor, min(sl_cap, final_sl_pct))

            if side == "long":
                stop_loss = fill_price * (1 - final_sl_pct)
                take_profit = fill_price * (1 + final_tp_pct)
            else:
                stop_loss = fill_price * (1 + final_sl_pct)
                take_profit = fill_price * (1 - final_tp_pct)

            position = Position(
                symbol=symbol,
                side=side,
                size=amount,
                entry_price=fill_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trade_type=trade_type,
                leverage=leverage,
            )
            self.positions[symbol] = position
            self._failed_attempts.pop(symbol, None)
            logger.info(
                f"포지션 개시: {side} {amount:.6f} {symbol} @ {fill_price:.2f} | "
                f"내부 SL: {stop_loss:.2f} TP: {take_profit:.2f} | 타입: {trade_type} "
                f"(거래소 Algo 없음)"
            )
            return position

        except Exception as e:
            self._failed_attempts[symbol] = self._failed_attempts.get(symbol, 0) + 1
            logger.error(f"포지션 개시 실패 (시도 {self._failed_attempts[symbol]}): {e}")
            return None

    async def close_position(self, symbol: str, reason: str = "") -> dict:
        """포지션 전량 청산 — 거래소 실제 잔여 수량 확인 후 시장가 청산

        버그 방지:
        - 거래소에서 실제 포지션 사이즈를 조회하여 전량 청산
        - 내부 추적 사이즈와 거래소 실제 사이즈 불일치 시 거래소 기준
        - SL 청산 시 쿨다운 등록 (5분간 같은 심볼 재진입 차단)
        """
        if symbol not in self.positions:
            return {}

        pos = self.positions[symbol]
        try:
            # 혹시 남아있는 거래소 주문 정리
            try:
                await self.exchange.cancel_all_orders(symbol)
            except Exception:
                pass

            # 거래소 실제 포지션 사이즈 확인 — 전량 청산 보장
            actual_size = pos.size
            try:
                exchange_pos = await self.exchange.get_position(symbol)
                ex_size = exchange_pos.get("size", 0) if exchange_pos else 0
                if ex_size > 0:
                    if abs(ex_size - pos.size) / max(pos.size, 1) > 0.01:
                        logger.warning(
                            f"[청산] {symbol} 사이즈 불일치: 내부={pos.size:.6f} 거래소={ex_size:.6f} "
                            f"→ 거래소 기준으로 전량 청산"
                        )
                    actual_size = ex_size
            except Exception:
                pass  # 조회 실패 시 내부 추적값 사용

            # 시장가 전량 청산
            order = await self.exchange.close_position(
                symbol,
                fallback_side=pos.side,
                fallback_size=actual_size,
            )

            if not order:
                logger.error(f"포지션 청산 주문 미실행 ({symbol}) — 다음 루프에서 재시도")
                return {}

            fill_price = float(order.get("average", 0))

            pnl = 0.0
            if fill_price > 0:
                if pos.side == "long":
                    pnl = (fill_price - pos.entry_price) / pos.entry_price * actual_size * pos.entry_price
                else:
                    pnl = (pos.entry_price - fill_price) / pos.entry_price * actual_size * pos.entry_price

            # 내부 포지션 삭제
            del self.positions[symbol]

            # SL 청산이면 쿨다운 등록 (5분간 재진입 차단)
            if "SL" in reason.upper() or "sl" in reason:
                self._sl_cooldown[symbol] = datetime.utcnow()
                logger.info(f"[쿨다운] {symbol} SL 청산 → {self.SL_COOLDOWN_SECONDS}초 재진입 차단 시작")

            logger.info(f"포지션 청산: {symbol} | PnL: {pnl:.2f} USDT | 사유: {reason}")

            return {
                "symbol": symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": fill_price,
                "size": actual_size,
                "pnl": pnl,
                "reason": reason,
            }
        except Exception as e:
            logger.error(f"포지션 청산 실패 ({symbol}): {e} — 다음 루프에서 재시도")
            return {}

    async def update_positions(self):
        """내부 SL/TP/트레일링 모니터링 — PAPER와 동일한 방식

        거래소에 SL/TP 주문이 없으므로 봇이 직접 가격 체크 후 시장가 청산.
        거래소에 SL 위치가 노출되지 않음.
        """
        auto_closed: list[tuple[str, float, str]] = []

        for symbol, pos in list(self.positions.items()):
            try:
                price = await self.exchange.get_ticker_price(symbol)
                t_activate, t_distance, t_step = self._get_trailing_params(pos.trade_type)

                if pos.side == "long":
                    pos.unrealized_pnl = (price - pos.entry_price) / pos.entry_price
                    pos.highest_price = max(pos.highest_price, price)
                    profit_pct = (price - pos.entry_price) / pos.entry_price

                    # 트레일링 활성화
                    if profit_pct >= t_activate and not pos.trailing_activated:
                        pos.trailing_activated = True
                        new_sl = pos.highest_price * (1 - t_distance)
                        pos.stop_loss = max(pos.stop_loss, new_sl)
                        logger.info(
                            f"[Trailing-LIVE] {symbol} 롱({pos.trade_type}) 트레일링 활성화 | "
                            f"수익 {profit_pct:.2%} | SL → {pos.stop_loss:.2f}"
                        )

                    # 트레일링 활성 중: SL 끌어올림
                    if pos.trailing_activated:
                        new_sl = pos.highest_price * (1 - t_distance)
                        if new_sl > pos.stop_loss + (pos.entry_price * t_step):
                            old_sl = pos.stop_loss
                            pos.stop_loss = new_sl
                            logger.info(
                                f"[Trailing-LIVE] {symbol} 롱 SL 상향 | "
                                f"{old_sl:.4f} → {pos.stop_loss:.4f}"
                            )

                    # SL 체크
                    if price <= pos.stop_loss:
                        reason = "트레일링 SL 도달" if pos.trailing_activated else "SL 도달"
                        auto_closed.append((symbol, price, reason))
                    # TP 도달 → 트레일링 전환
                    elif price >= pos.take_profit and not pos.trailing_activated:
                        pos.trailing_activated = True
                        pos.stop_loss = pos.entry_price * (1 + t_activate)
                        pos.take_profit = float("inf")
                        logger.info(
                            f"[Trailing-LIVE] {symbol} 롱 TP 도달 → 트레일링 전환 | "
                            f"수익 확보선: {pos.stop_loss:.2f}"
                        )

                else:  # short
                    pos.unrealized_pnl = (pos.entry_price - price) / pos.entry_price
                    pos.lowest_price = min(pos.lowest_price, price) if pos.lowest_price > 0 else price
                    profit_pct = (pos.entry_price - price) / pos.entry_price

                    # 트레일링 활성화
                    if profit_pct >= t_activate and not pos.trailing_activated:
                        pos.trailing_activated = True
                        new_sl = pos.lowest_price * (1 + t_distance)
                        pos.stop_loss = min(pos.stop_loss, new_sl)
                        logger.info(
                            f"[Trailing-LIVE] {symbol} 숏({pos.trade_type}) 트레일링 활성화 | "
                            f"수익 {profit_pct:.2%} | SL → {pos.stop_loss:.2f}"
                        )

                    # 트레일링 활성 중: SL 끌어내림
                    if pos.trailing_activated:
                        new_sl = pos.lowest_price * (1 + t_distance)
                        if new_sl < pos.stop_loss - (pos.entry_price * t_step):
                            old_sl = pos.stop_loss
                            pos.stop_loss = new_sl
                            logger.info(
                                f"[Trailing-LIVE] {symbol} 숏 SL 하향 | "
                                f"{old_sl:.4f} → {pos.stop_loss:.4f}"
                            )

                    # SL 체크
                    if price >= pos.stop_loss:
                        reason = "트레일링 SL 도달" if pos.trailing_activated else "SL 도달"
                        auto_closed.append((symbol, price, reason))
                    # TP 도달 → 트레일링 전환
                    elif price <= pos.take_profit and not pos.trailing_activated:
                        pos.trailing_activated = True
                        pos.stop_loss = pos.entry_price * (1 - t_activate)
                        pos.take_profit = 0.0
                        logger.info(
                            f"[Trailing-LIVE] {symbol} 숏 TP 도달 → 트레일링 전환 | "
                            f"수익 확보선: {pos.stop_loss:.2f}"
                        )

                logger.info(
                    f"[Monitor] {symbol} {pos.side}({pos.trade_type}) | "
                    f"현재: {price:.5f} | 진입: {pos.entry_price:.5f} | "
                    f"내부SL: {pos.stop_loss:.5f} | trailing: {pos.trailing_activated}"
                )

            except Exception as e:
                import traceback
                logger.error(f"포지션 업데이트 실패 ({symbol}): {e}\n{traceback.format_exc()}")

        # 루프 밖에서 청산 처리 — 시장가 주문으로 직접 청산
        for symbol, close_price, reason in auto_closed:
            pos = self.positions.get(symbol)
            if not pos:
                continue

            was_sl = "SL" in reason
            pnl = 0.0
            if pos.side == "long":
                pnl = (close_price - pos.entry_price) / pos.entry_price * pos.size * pos.entry_price
            else:
                pnl = (pos.entry_price - close_price) / pos.entry_price * pos.size * pos.entry_price

            logger.info(f"[LIVE-Auto] {reason} {symbol} {pos.side} | 추정 PnL: ${pnl:.2f} → 시장가 청산")

            result = await self.close_position(symbol, reason)

            callback_data = {
                "symbol": symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": result.get("exit_price", close_price),
                "size": pos.size,
                "pnl": result.get("pnl", pnl),
                "reason": reason,
                "close_type": "SL" if was_sl else "TP",
            }

            if was_sl and self._on_sl_callback:
                try:
                    self._on_sl_callback(callback_data)
                except Exception as cb_e:
                    logger.debug(f"SL 콜백 실패: {cb_e}")
            elif not was_sl and self._on_tp_callback:
                try:
                    self._on_tp_callback(callback_data)
                except Exception as cb_e:
                    logger.debug(f"TP 콜백 실패: {cb_e}")

    def get_all_positions(self) -> list[dict]:
        return [
            {
                "symbol": p.symbol, "side": p.side, "size": p.size,
                "entry_price": p.entry_price, "unrealized_pnl": p.unrealized_pnl,
                "stop_loss": p.stop_loss, "take_profit": p.take_profit,
            }
            for p in self.positions.values()
        ]
