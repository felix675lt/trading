"""페이퍼 트레이딩 엔진 - 실시간 시뮬레이션"""

from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger


@dataclass
class PaperPosition:
    symbol: str
    side: str
    size: float
    entry_price: float
    leverage: int = 5
    stop_loss: float = 0.0
    take_profit: float = 0.0
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0   # 진입 후 최고가 (롱)
    lowest_price: float = 0.0    # 진입 후 최저가 (숏)
    trailing_activated: bool = False  # 트레일링 스탑 활성화 여부
    trade_type: str = "scalp"    # "scalp" or "swing"


class PaperTrader:
    """실거래 없이 시뮬레이션하는 페이퍼 트레이딩 엔진"""

    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.0004,
                 trailing_config: dict | None = None,
                 trade_profiles: dict | None = None):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.commission = commission
        self.trade_profiles = trade_profiles or {}
        self.positions: dict[str, PaperPosition] = {}
        self.trade_history: list[dict] = []
        self.equity_history: list[dict] = []

        # 트레일링 스탑 기본값 (fallback)
        tc = trailing_config or {}
        self.trailing_activate_pct = tc.get("activate_pct", 0.02)
        self.trailing_distance_pct = tc.get("distance_pct", 0.015)
        self.trailing_step_pct = tc.get("step_pct", 0.005)

        # 자동 청산 콜백 (SL/TP/트레일링으로 청산 시 호출)
        self._on_auto_close_callback = None

    def set_auto_close_callback(self, callback):
        """SL/TP/트레일링 자동 청산 시 호출할 콜백 등록
        callback(trade: dict) — trade는 close_position 반환값과 동일
        """
        self._on_auto_close_callback = callback

    def _get_trailing_params(self, trade_type: str) -> tuple[float, float, float]:
        """trade_type별 트레일링 파라미터"""
        profile = self.trade_profiles.get(trade_type, {})
        tc = profile.get("trailing", {})
        return (
            tc.get("activate_pct", self.trailing_activate_pct),
            tc.get("distance_pct", self.trailing_distance_pct),
            tc.get("step_pct", self.trailing_step_pct),
        )

    def open_position(self, symbol: str, side: str, size_usdt: float,
                      price: float, leverage: int = 5, sl_pct: float = 0.02,
                      tp_pct: float = 0.04, atr_pct: float = 0.0,
                      trade_type: str = "scalp") -> PaperPosition | None:
        if symbol in self.positions:
            logger.warning(f"[Paper] {symbol} 이미 포지션 보유")
            return None

        amount = (size_usdt * leverage) / price
        fee = size_usdt * self.commission
        self.equity -= fee

        # trade_type 프로파일에서 기본값 가져오기
        profile = self.trade_profiles.get(trade_type, {})

        # ATR 기반 동적 SL/TP (atr_pct가 유효하면 사용)
        if atr_pct and atr_pct > 0 and atr_pct == atr_pct:  # NaN 체크
            atr_sl_mult = profile.get("atr_sl_multiplier", 2.0)
            atr_tp_mult = profile.get("atr_tp_multiplier", 3.5)
            sl_floor = profile.get("sl_floor_pct", 0.005)
            sl_cap = profile.get("sl_cap_pct", 0.030)
            final_sl = max(sl_floor, min(sl_cap, atr_pct * atr_sl_mult))
            final_tp = max(final_sl * 1.5, atr_pct * atr_tp_mult)
        else:
            final_sl = sl_pct if sl_pct != 0.02 else profile.get("sl_pct", sl_pct)
            final_tp = tp_pct if tp_pct != 0.04 else profile.get("tp_pct", tp_pct)

        if side == "long":
            sl = price * (1 - final_sl)
            tp = price * (1 + final_tp)
        else:
            sl = price * (1 + final_sl)
            tp = price * (1 - final_tp)

        pos = PaperPosition(
            symbol=symbol, side=side, size=amount,
            entry_price=price, leverage=leverage,
            stop_loss=sl, take_profit=tp,
            highest_price=price, lowest_price=price,
            trade_type=trade_type,
        )
        self.positions[symbol] = pos
        logger.info(
            f"[Paper] 포지션 개시: {side} {amount:.6f} {symbol} @ {price:.2f} | "
            f"타입: {trade_type} | SL: {final_sl*100:.1f}% TP: {final_tp*100:.1f}%"
        )
        return pos

    def close_position(self, symbol: str, price: float, reason: str = "") -> dict:
        if symbol not in self.positions:
            return {}

        pos = self.positions[symbol]
        if pos.side == "long":
            pnl = (price - pos.entry_price) / pos.entry_price * pos.size * pos.entry_price * pos.leverage
        else:
            pnl = (pos.entry_price - price) / pos.entry_price * pos.size * pos.entry_price * pos.leverage

        fee = pos.size * price * self.commission
        net_pnl = pnl - fee
        self.equity += net_pnl

        trade = {
            "timestamp": str(datetime.utcnow()),
            "symbol": symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "size": pos.size,
            "pnl": net_pnl,
            "fee": fee,
            "reason": reason,
        }
        self.trade_history.append(trade)
        del self.positions[symbol]

        logger.info(f"[Paper] 포지션 청산: {symbol} | PnL: {net_pnl:.2f} | 사유: {reason}")
        return trade

    def update_prices(self, prices: dict[str, float]):
        """현재가 업데이트 + 트레일링 스탑 + SL/TP 체크"""
        auto_closed = []  # (symbol, close_price, reason) — 루프 후 처리

        for symbol, price in prices.items():
            if symbol not in self.positions:
                continue
            pos = self.positions[symbol]
            t_activate, t_distance, t_step = self._get_trailing_params(pos.trade_type)

            if pos.side == "long":
                pos.unrealized_pnl = (price - pos.entry_price) / pos.entry_price * pos.leverage
                pos.highest_price = max(pos.highest_price, price)
                profit_pct = (price - pos.entry_price) / pos.entry_price

                # 트레일링 스탑 로직
                if profit_pct >= t_activate and not pos.trailing_activated:
                    pos.trailing_activated = True
                    new_sl = pos.highest_price * (1 - t_distance)
                    pos.stop_loss = max(pos.stop_loss, new_sl)
                    logger.info(f"[Trailing] {symbol}({pos.trade_type}) 트레일링 활성화 | 수익 {profit_pct:.2%} | SL → {pos.stop_loss:.2f}")

                if pos.trailing_activated:
                    new_sl = pos.highest_price * (1 - t_distance)
                    if new_sl > pos.stop_loss + (pos.entry_price * t_step):
                        old_sl = pos.stop_loss
                        pos.stop_loss = new_sl
                        logger.info(f"[Trailing] {symbol}({pos.trade_type}) SL 상향 | {old_sl:.2f} → {pos.stop_loss:.2f}")

                # SL/TP 체크
                if price <= pos.stop_loss:
                    reason = "트레일링 SL 도달" if pos.trailing_activated else "SL 도달"
                    auto_closed.append((symbol, pos.stop_loss, reason))
                elif price >= pos.take_profit and not pos.trailing_activated:
                    pos.trailing_activated = True
                    pos.stop_loss = pos.entry_price * (1 + t_activate)
                    pos.take_profit = float("inf")
                    logger.info(f"[Trailing] {symbol}({pos.trade_type}) TP 도달 → 트레일링 전환 | 수익 확보선: {pos.stop_loss:.2f}")

            else:  # short
                pos.unrealized_pnl = (pos.entry_price - price) / pos.entry_price * pos.leverage
                pos.lowest_price = min(pos.lowest_price, price) if pos.lowest_price > 0 else price
                profit_pct = (pos.entry_price - price) / pos.entry_price

                if profit_pct >= t_activate and not pos.trailing_activated:
                    pos.trailing_activated = True
                    new_sl = pos.lowest_price * (1 + t_distance)
                    pos.stop_loss = min(pos.stop_loss, new_sl)
                    logger.info(f"[Trailing] {symbol}({pos.trade_type}) 숏 트레일링 활성화 | 수익 {profit_pct:.2%} | SL → {pos.stop_loss:.2f}")

                if pos.trailing_activated:
                    new_sl = pos.lowest_price * (1 + t_distance)
                    if new_sl < pos.stop_loss - (pos.entry_price * t_step):
                        old_sl = pos.stop_loss
                        pos.stop_loss = new_sl
                        logger.info(f"[Trailing] {symbol}({pos.trade_type}) 숏 SL 하향 | {old_sl:.2f} → {pos.stop_loss:.2f}")

                if price >= pos.stop_loss:
                    reason = "트레일링 SL 도달" if pos.trailing_activated else "SL 도달"
                    auto_closed.append((symbol, pos.stop_loss, reason))
                elif price <= pos.take_profit and not pos.trailing_activated:
                    pos.trailing_activated = True
                    pos.stop_loss = pos.entry_price * (1 - t_activate)
                    pos.take_profit = 0.0
                    logger.info(f"[Trailing] {symbol}({pos.trade_type}) 숏 TP 도달 → 트레일링 전환 | 수익 확보선: {pos.stop_loss:.2f}")

        # 루프 밖에서 청산 처리 (dict 변경 안전)
        for symbol, close_price, reason in auto_closed:
            trade = self.close_position(symbol, close_price, reason)
            if trade and self._on_auto_close_callback:
                try:
                    self._on_auto_close_callback(trade)
                except Exception as e:
                    logger.error(f"[Paper] 자동청산 콜백 실패 {symbol}: {e}")

        self.equity_history.append({
            "timestamp": str(datetime.utcnow()),
            "equity": self.equity,
        })

    def get_stats(self) -> dict:
        if not self.trade_history:
            return {"total_trades": 0, "equity": self.equity}
        pnls = [t["pnl"] for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        return {
            "total_trades": len(self.trade_history),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / len(pnls),
            "max_win": max(pnls) if pnls else 0,
            "max_loss": min(pnls) if pnls else 0,
            "equity": self.equity,
            "return_pct": (self.equity - self.initial_capital) / self.initial_capital,
        }
