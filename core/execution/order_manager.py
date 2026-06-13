"""주문 관리자 - 포지션 추적, 주문 실행, 내부 SL/TP 모니터링

거래소 Algo Order(SL/TP) 사용하지 않음.
봇이 직접 가격을 감시하고 시장가로 청산 — PAPER와 동일한 방식.
거래소에 SL 위치를 노출하지 않아 스탑 헌팅 회피.

v5 — Limit-first order routing (tier=small+ 활성화):
- 진입 시 best-bid/ask에 post-only limit 주문 → maker fee 0.02% 활용
- limit_wait_seconds (기본 20초) 대기 → 미체결 시 취소하고 market fallback
- taker fee 0.04% vs maker fee 0.02% → 편도 50% 수수료 절감
"""

import asyncio
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

        # 수수료 (왕복: 진입 + 청산)
        self.commission_pct = risk_config.get("commission_pct", 0.0004)  # 편도 0.04%

        # Order routing modes (tier 연동):
        #   market_only (micro): 시장가 단일 체결
        #   limit_first (small+): maker fee 우선, 미체결 시 market fallback
        #   twap (large+): N-slice TWAP (내부적으로 limit-first 사용)
        #   smart (pro): 다거래소 비교 라우팅 (SmartRouter 주입 시)
        self.routing_mode: str = "market_only"
        self.limit_first_enabled = False  # main.py에서 tier 기반으로 set
        self.limit_wait_seconds = 20      # 지정가 대기 시간
        self.limit_offset_pct = 0.0001    # best-bid/ask에서 소폭 offset (체결 확률 ↑)

        # TWAP 설정 (tier=large)
        self.twap_slices = 5
        self.twap_duration_s = 60
        self._twap_threshold_usdt = 500  # 이 notional 이상이면 TWAP 적용

        # Smart routing (tier=pro) — SmartRouter 인스턴스 주입
        self.smart_router = None

        # 지정가 체결 통계 — Paper 피드백용 (maker fill rate 실측)
        self.limit_fill_stats = {"attempts": 0, "filled": 0, "fallback_market": 0}

        # === 슬리피지 실측 (2026-04-24 A: Paper-Live 괴리 제거) ===
        # 각 체결마다 (requested_price vs fill_price)를 bps로 기록.
        # sync_from_live_execution()이 이걸 중앙값으로 집계해서 Paper에 주입.
        # rolling 50건만 유지 — 시장 레짐 변화에 따라잡기 좋은 윈도우.
        from collections import deque
        self._slip_entry: deque = deque(maxlen=50)
        self._slip_exit: deque = deque(maxlen=50)
        self._slip_exit_sl: deque = deque(maxlen=50)

    def get_maker_fill_rate(self) -> float | None:
        """Limit-first 시도 대비 maker 체결 성공률 (부분체결 수용 포함).
        샘플 수가 너무 적으면 None 반환 → 호출부가 기본값 유지.
        """
        attempts = self.limit_fill_stats.get("attempts", 0)
        if attempts < 5:
            return None
        return self.limit_fill_stats.get("filled", 0) / attempts

    def get_execution_stats(self) -> dict:
        """LIVE 실측 체결 통계 — Paper 피드백 동기화용 (2026-04-24).

        PAPER와 LIVE의 체결 모델 괴리를 제거하기 위해, 실전 LIVE 체결 결과를
        집계해 Paper의 슬리피지/maker 체결률을 갱신하는 데 쓴다.

        반환 필드:
            entry_slip_bps_med: 진입 슬리피지 중앙값 (bps, 방향불리=+)
            exit_slip_bps_med:  일반 청산 슬리피지 중앙값
            sl_slip_bps_med:    SL/liquidation 청산 슬리피지 중앙값
            maker_fill_rate:    limit-first 체결률 (None이면 샘플부족)
            n_samples:          집계에 쓰인 총 관측 수

        호출부가 n_samples<5면 무시해야 (초기 노이즈 방어).
        """
        import statistics as _st
        def _median_safe(arr) -> float | None:
            vals = [float(x) for x in arr if isinstance(x, (int, float))]
            if not vals:
                return None
            try:
                return float(_st.median(vals))
            except Exception:
                return None

        entry_med = _median_safe(self._slip_entry)
        exit_med = _median_safe(self._slip_exit)
        sl_med = _median_safe(self._slip_exit_sl)
        maker_rate = self.get_maker_fill_rate()
        n = len(self._slip_entry) + len(self._slip_exit) + len(self._slip_exit_sl)
        return {
            "entry_slip_bps_med": round(entry_med, 2) if entry_med is not None else None,
            "exit_slip_bps_med": round(exit_med, 2) if exit_med is not None else None,
            "sl_slip_bps_med": round(sl_med, 2) if sl_med is not None else None,
            "maker_fill_rate": round(maker_rate, 3) if maker_rate is not None else None,
            "n_samples": int(n),
            "n_entry": len(self._slip_entry),
            "n_exit": len(self._slip_exit),
            "n_exit_sl": len(self._slip_exit_sl),
        }

    def set_routing(
        self,
        limit_first: bool = False,
        wait_seconds: int = 20,
        offset_pct: float = 0.0001,
        mode: str = "limit_first",
        twap_slices: int = 5,
        twap_duration_s: int = 60,
        smart_router=None,
    ):
        """Capital Tier에 따라 주문 라우팅 방식 설정 (main.py에서 호출).

        Args:
            mode: market_only / limit_first / twap / smart
            limit_first: legacy 파라미터 — mode로 통합 판단
            twap_slices: TWAP 분할 수
            twap_duration_s: TWAP 총 지속시간 (초)
            smart_router: SmartRouter 인스턴스 (mode=smart일 때 사용)
        """
        self.routing_mode = mode
        self.limit_first_enabled = limit_first or mode in ("limit_first", "twap", "smart")
        self.limit_wait_seconds = wait_seconds
        self.limit_offset_pct = offset_pct
        self.twap_slices = twap_slices
        self.twap_duration_s = twap_duration_s
        self.smart_router = smart_router

        logger.info(
            f"[Routing] mode={mode} | limit_first={self.limit_first_enabled} "
            f"(wait={wait_seconds}s, offset={offset_pct*100:.3f}%) | "
            f"twap_slices={twap_slices} | smart_router={'ON' if smart_router else 'OFF'}"
        )

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

    def set_profit_callback(self, callback):
        """포지션 청산 시 호출될 추가 콜백 (BTC Reserve 적립 등).

        close_position이 성공적으로 체결된 후 PnL과 상관없이 호출된다.
        콜백은 async이며 trade dict를 받는다: callback(trade: dict).
        PnL>0 필터링은 콜백 내부에서 수행.
        """
        self._profit_callback = callback

    async def _try_limit_first(
        self,
        symbol: str,
        order_side: str,  # "buy" or "sell"
        amount: float,
    ) -> dict | None:
        """Limit-first 진입 시도 — 실패 시 None 반환하여 호출부가 market fallback

        전략:
        1. best bid(buy)/ask(sell)에서 offset 만큼 유리한 가격에 limit 주문
        2. limit_wait_seconds 동안 체결 대기 (1초마다 상태 확인)
        3. 체결됨 → order 반환 / 미체결 → 취소하고 None 반환
        """
        self.limit_fill_stats["attempts"] += 1
        try:
            bid, ask, _ = await self.exchange.get_bid_ask(symbol)
            # maker 체결 우선 — bid에 buy, ask에 sell (크로스 방지)
            if order_side == "buy":
                limit_price = bid * (1 - self.limit_offset_pct)
            else:
                limit_price = ask * (1 + self.limit_offset_pct)

            logger.info(
                f"[Limit-first] {order_side} {amount:.6f} {symbol} @ {limit_price:.4f} "
                f"(bid={bid:.4f} ask={ask:.4f}, wait={self.limit_wait_seconds}s)"
            )
            order = await self.exchange.create_limit_order(
                symbol, order_side, amount, limit_price
            )
            order_id = order.get("id")
            if not order_id:
                logger.warning(f"[Limit-first] {symbol} 주문 ID 없음 → market fallback")
                return None

            # 폴링 (1초 간격)
            for elapsed in range(self.limit_wait_seconds):
                await asyncio.sleep(1)
                status = await self.exchange.fetch_order_status(order_id, symbol)
                if not status:
                    continue
                st = (status.get("status") or "").lower()
                filled = float(status.get("filled", 0) or 0)
                if st in ("closed", "filled") or filled >= amount * 0.99:
                    self.limit_fill_stats["filled"] += 1
                    logger.info(
                        f"[Limit-first] {symbol} 체결 성공 ({elapsed+1}s) "
                        f"filled={filled:.6f}/{amount:.6f} @ {status.get('average', limit_price)}"
                    )
                    return status
                if st in ("canceled", "cancelled", "expired"):
                    self.limit_fill_stats["fallback_market"] += 1
                    logger.info(f"[Limit-first] {symbol} 주문 취소됨 ({st}) → market fallback")
                    return None

            # 시간 초과 → 취소
            logger.info(
                f"[Limit-first] {symbol} {self.limit_wait_seconds}s 미체결 → 취소 후 market fallback"
            )
            await self.exchange.cancel_order(order_id, symbol)
            # 취소 후 부분 체결 있으면 반영
            final = await self.exchange.fetch_order_status(order_id, symbol)
            final_filled = float(final.get("filled", 0) or 0) if final else 0
            if final_filled >= amount * 0.5:
                # 절반 이상 체결됐으면 limit 결과 수용 (나머지는 포기)
                self.limit_fill_stats["filled"] += 1
                logger.warning(
                    f"[Limit-first] {symbol} 부분체결 {final_filled:.6f} ({final_filled/amount*100:.0f}%) 수용"
                )
                return final
            self.limit_fill_stats["fallback_market"] += 1
            return None

        except Exception as e:
            self.limit_fill_stats["fallback_market"] += 1
            logger.warning(f"[Limit-first] {symbol} 실패: {e} → market fallback")
            return None

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

            # === 주문 라우팅: market_only / limit_first / twap / smart ===
            order_side = "buy" if side == "long" else "sell"
            order = None
            notional_est = amount * price

            # [1] SMART — 다거래소 라우팅 (tier=pro)
            if self.routing_mode == "smart" and self.smart_router is not None:
                try:
                    smart_result = await self.smart_router.route(symbol, order_side, amount)
                    if smart_result.get("total_filled", 0) > 0:
                        order = {
                            "average": smart_result["avg_price"],
                            "filled": smart_result["total_filled"],
                            "routed_to": smart_result.get("routed_to"),
                        }
                        logger.info(f"[Smart] {symbol} routed to {smart_result.get('routed_to')}")
                except Exception as e:
                    logger.warning(f"[Smart] 실패 → TWAP/limit fallback: {e}")

            # [2] TWAP — 분할 체결 (tier=large, 주문 크기 threshold 이상일 때)
            if order is None and self.routing_mode == "twap" and notional_est >= self._twap_threshold_usdt:
                try:
                    from core.execution.twap import TWAPExecutor
                    twap = TWAPExecutor(self, default_slices=self.twap_slices, default_duration_s=self.twap_duration_s)
                    twap_result = await twap.execute(
                        symbol, order_side, amount,
                        n_slices=self.twap_slices,
                        duration_seconds=self.twap_duration_s,
                    )
                    if twap_result.get("total_filled", 0) > 0:
                        order = {
                            "average": twap_result["avg_price"],
                            "filled": twap_result["total_filled"],
                            "routing": "twap",
                            "slices": len(twap_result.get("slices", [])),
                        }
                except Exception as e:
                    logger.warning(f"[TWAP] 실패 → limit/market fallback: {e}")

            # [3] LIMIT-FIRST — maker fee 우선 (tier=small+)
            if order is None and self.limit_first_enabled:
                order = await self._try_limit_first(symbol, order_side, amount)

            # [4] MARKET — 최종 fallback
            if order is None:
                order = await self.exchange.create_market_order(symbol, order_side, amount)

            fill_price = float(order.get("average", price) or price)

            # === 진입 슬리피지 실측 기록 (Paper-Live 피드백용) ===
            # long buy면 체결가가 요청가보다 높을수록 불리(+bps),
            # short sell이면 체결가가 요청가보다 낮을수록 불리(+bps).
            # price는 주문 직전 ticker_price이므로 비교 기준으로 적합.
            try:
                if price > 0 and fill_price > 0:
                    if side == "long":
                        slip_bps = (fill_price - price) / price * 10000.0
                    else:
                        slip_bps = (price - fill_price) / price * 10000.0
                    # 비정상 outlier 방어 — ±200bp 초과면 기록 제외
                    if -200.0 <= slip_bps <= 200.0:
                        self._slip_entry.append(float(slip_bps))
            except Exception:
                pass

            # 실제 체결 수량 확인 — 부분 체결 대응
            filled = float(order.get("filled", 0))
            if filled > 0:
                if abs(filled - amount) / max(amount, 1) > 0.05:
                    logger.warning(
                        f"[부분체결] {symbol} 요청={amount:.2f} 체결={filled:.2f} "
                        f"({filled/amount*100:.1f}%)"
                    )
                amount = filled
            else:
                # filled 정보 없으면 거래소에서 직접 확인
                try:
                    exchange_pos = await self.exchange.get_position(symbol)
                    ex_size = exchange_pos.get("size", 0) if exchange_pos else 0
                    if ex_size > 0:
                        amount = ex_size
                except Exception:
                    pass  # 실패 시 원래 amount 사용

            # SL/TP 계산 (내부 모니터링용) — 수학적 RR 최소 강제 (2026-04-20)
            # BE_WR = SL / (SL+TP). RR=2.5:1이면 BE_WR=28.6% (실적 34% 대비 여유)
            profile = self._get_profile(trade_type)
            sl_floor = profile.get("sl_floor_pct", self.risk_config.get("sl_floor_pct", 0.020))
            sl_cap = profile.get("sl_cap_pct", self.risk_config.get("sl_cap_pct", 0.045))
            min_rr = self.risk_config.get("min_rr_ratio", 2.5)  # EV 양수 조건

            if atr_pct and atr_pct > 0 and not (atr_pct != atr_pct):
                atr_sl_mult = profile.get("atr_sl_multiplier", self.risk_config.get("atr_sl_multiplier", 3.0))
                atr_tp_mult = profile.get("atr_tp_multiplier", self.risk_config.get("atr_tp_multiplier", 7.5))
                final_sl_pct = max(sl_floor, min(sl_cap, atr_pct * atr_sl_mult))
                # RR 최소 min_rr 강제: TP ≥ SL × min_rr (수학적 EV 양수 조건)
                final_tp_pct = max(final_sl_pct * min_rr, atr_pct * atr_tp_mult)
                logger.info(
                    f"[ATR-SL/TP] {symbol} ATR={atr_pct*100:.2f}% → "
                    f"SL={final_sl_pct*100:.2f}% TP={final_tp_pct*100:.2f}% "
                    f"(RR {final_tp_pct/final_sl_pct:.1f}:1, BE_WR={final_sl_pct/(final_sl_pct+final_tp_pct)*100:.1f}%)"
                )
            else:
                final_sl_pct = sl_pct or profile.get("sl_pct", self.risk_config.get("stop_loss_pct", 0.020))
                final_tp_pct = tp_pct or profile.get("tp_pct", self.risk_config.get("take_profit_pct", 0.050))
                final_sl_pct = max(sl_floor, min(sl_cap, final_sl_pct))
                # RR 최소 min_rr 강제
                final_tp_pct = max(final_sl_pct * min_rr, final_tp_pct)

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

            # 더스트 포지션 체크 — 최소 주문금액($5) 미달 시 거래소 청산 불가
            ticker_price = None
            try:
                ticker_price = await self.exchange.get_ticker_price(symbol)
            except Exception:
                pass
            notional = actual_size * (ticker_price or pos.entry_price)
            if notional < 5.0:
                logger.warning(
                    f"[더스트] {symbol} notional=${notional:.2f} < $5 — 거래소 청산 불가, 내부 포지션 정리"
                )
                del self.positions[symbol]
                if "SL" in reason.upper() or "sl" in reason:
                    self._sl_cooldown[symbol] = datetime.utcnow()
                return {
                    "symbol": symbol, "side": pos.side,
                    "entry_price": pos.entry_price, "exit_price": ticker_price or pos.entry_price,
                    "size": actual_size, "pnl": 0.0, "reason": f"{reason} (더스트 정리)",
                }

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

            # === 청산 슬리피지 실측 기록 (Paper-Live 피드백용) ===
            # 청산은 반대 방향: long close = sell, short close = buy.
            # ticker_price (주문 직전 시장가)와 fill_price 차이를 방향불리로 계산.
            try:
                ref_price = float(ticker_price) if ticker_price else 0.0
                if ref_price > 0 and fill_price > 0:
                    if pos.side == "long":  # 청산=매도, 체결가가 낮을수록 불리(+bps)
                        slip_bps = (ref_price - fill_price) / ref_price * 10000.0
                    else:  # 청산=매수, 체결가가 높을수록 불리(+bps)
                        slip_bps = (fill_price - ref_price) / ref_price * 10000.0
                    if -200.0 <= slip_bps <= 200.0:
                        # SL/liquidation 여부로 분리 저장 (Paper의 SL_SLIPPAGE_EXTRA 재보정용)
                        is_sl_hit = ("SL" in reason.upper()) or ("liq" in reason.lower())
                        if is_sl_hit:
                            self._slip_exit_sl.append(float(slip_bps))
                        else:
                            self._slip_exit.append(float(slip_bps))
            except Exception:
                pass

            pnl = 0.0
            if fill_price > 0:
                notional = actual_size * pos.entry_price
                if pos.side == "long":
                    pnl = (fill_price - pos.entry_price) / pos.entry_price * notional
                else:
                    pnl = (pos.entry_price - fill_price) / pos.entry_price * notional
                # 왕복 수수료 차감 (진입 + 청산)
                fee = notional * self.commission_pct * 2
                pnl -= fee

            # 내부 포지션 삭제
            del self.positions[symbol]

            # SL 청산이면 쿨다운 등록 (5분간 재진입 차단)
            if "SL" in reason.upper() or "sl" in reason:
                self._sl_cooldown[symbol] = datetime.utcnow()
                logger.info(f"[쿨다운] {symbol} SL 청산 → {self.SL_COOLDOWN_SECONDS}초 재진입 차단 시작")

            logger.info(f"포지션 청산: {symbol} | PnL: {pnl:.2f} USDT (수수료 {fee:.4f}) | 사유: {reason}")

            trade_result = {
                "symbol": symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": fill_price,
                "fee": fee,
                "size": actual_size,
                "pnl": pnl,
                "reason": reason,
            }

            # === BTC Reserve 적립 콜백 (async) — 실패해도 청산 결과는 유지 ===
            cb = getattr(self, "_profit_callback", None)
            if cb is not None:
                try:
                    result = cb(trade_result)
                    # coroutine이면 fire-and-forget (청산 반환을 블로킹하지 않음)
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
                except Exception as e:
                    logger.debug(f"[Live] profit_callback 실패 (무시): {e}")

            return trade_result
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
                        pos.stop_loss = new_sl
                        logger.info(
                            f"[Trailing-LIVE] {symbol} 숏({pos.trade_type}) 트레일링 활성화 | "
                            f"수익 {profit_pct:.2%} | SL → {pos.stop_loss:.2f}"
                        )

                    # 트레일링 활성 중: 가격 하락 시 SL도 따라 내림 (수익 보호)
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

                # === [Patch T, 2026-06-13] stale 자기수정 — PAPER 동일 정책을 LIVE에 적용 ===
                # PAPER는 main.py에서 240분+손실 포지션을 강제청산(auto_close)하지만
                # LIVE엔 시간 청산이 없어 scalp 포지션이 SL 직전에서 수일간 표류하며
                # 슬롯(max_concurrent_live=1)을 점유 → LIVE 신규진입 전체 정지 (6/9 ETH 97h 사례).
                if not any(s == symbol for s, _, _ in auto_closed):
                    try:
                        age_min = (datetime.utcnow() - datetime.fromisoformat(pos.opened_at)).total_seconds() / 60
                    except Exception:
                        age_min = 0.0
                    if age_min > 240 and profit_pct < 0:
                        logger.info(
                            f"[LIVE-Stale] {symbol} {pos.side} {age_min:.0f}분 보유 + "
                            f"손실 {profit_pct:.2%} → 자기수정 청산"
                        )
                        # reason에 'SL' 포함 → SL 콜백 경로로 기록(손실 학습+리포트)
                        auto_closed.append((symbol, price, f"자기수정 SL ({age_min:.0f}분 보유+손실)"))

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

            # 청산 실패 시 콜백 호출하지 않음 (가짜 알림 방지)
            if not result:
                logger.warning(f"[청산실패] {symbol} — 콜백 스킵, 다음 루프 재시도")
                continue

            callback_data = {
                "symbol": symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": result.get("exit_price", close_price),
                "size": result.get("size", pos.size),
                "pnl": result.get("pnl", pnl),
                "reason": result.get("reason", reason),
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
