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


class PaperTrader:
    """실거래 없이 시뮬레이션하는 페이퍼 트레이딩 엔진"""

    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.0004):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.commission = commission
        self.positions: dict[str, PaperPosition] = {}
        self.trade_history: list[dict] = []
        self.equity_history: list[dict] = []

    def open_position(self, symbol: str, side: str, size_usdt: float,
                      price: float, leverage: int = 5, sl_pct: float = 0.02,
                      tp_pct: float = 0.04) -> PaperPosition | None:
        if symbol in self.positions:
            logger.warning(f"[Paper] {symbol} 이미 포지션 보유")
            return None

        amount = (size_usdt * leverage) / price
        fee = size_usdt * self.commission
        self.equity -= fee

        if side == "long":
            sl = price * (1 - sl_pct)
            tp = price * (1 + tp_pct)
        else:
            sl = price * (1 + sl_pct)
            tp = price * (1 - tp_pct)

        pos = PaperPosition(
            symbol=symbol, side=side, size=amount,
            entry_price=price, leverage=leverage,
            stop_loss=sl, take_profit=tp,
        )
        self.positions[symbol] = pos
        logger.info(f"[Paper] 포지션 개시: {side} {amount:.6f} {symbol} @ {price:.2f}")
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
        """현재가 업데이트 + SL/TP 체크"""
        for symbol, price in prices.items():
            if symbol not in self.positions:
                continue
            pos = self.positions[symbol]

            # 미실현 PnL 업데이트
            if pos.side == "long":
                pos.unrealized_pnl = (price - pos.entry_price) / pos.entry_price * pos.leverage
                if price <= pos.stop_loss:
                    self.close_position(symbol, pos.stop_loss, "SL 도달")
                elif price >= pos.take_profit:
                    self.close_position(symbol, pos.take_profit, "TP 도달")
            else:
                pos.unrealized_pnl = (pos.entry_price - price) / pos.entry_price * pos.leverage
                if price >= pos.stop_loss:
                    self.close_position(symbol, pos.stop_loss, "SL 도달")
                elif price <= pos.take_profit:
                    self.close_position(symbol, pos.take_profit, "TP 도달")

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
