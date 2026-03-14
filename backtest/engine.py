"""백테스트 엔진 - 과거 데이터로 전략 검증"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class BacktestResult:
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    avg_trade_pnl: float = 0.0
    profit_factor: float = 0.0
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)


class BacktestEngine:
    """과거 데이터 기반 전략 시뮬레이션"""

    def __init__(self, config: dict):
        self.initial_capital = config.get("initial_capital", 10000)
        self.commission = config.get("commission_pct", 0.0004)
        self.slippage = config.get("slippage_pct", 0.0001)
        self.leverage = config.get("leverage", 5)

    def run(
        self,
        df: pd.DataFrame,
        signals: list[dict],
    ) -> BacktestResult:
        """
        백테스트 실행

        Args:
            df: OHLCV 데이터프레임
            signals: 각 바에 대한 시그널 리스트 [{"action": "long"/"short"/"close"/"hold", "size": 0~1}]
        """
        equity = self.initial_capital
        position = 0.0  # 양수=롱, 음수=숏
        entry_price = 0.0
        equity_curve = [equity]
        trades = []
        peak_equity = equity

        prices = df["close"].values

        for i in range(min(len(prices), len(signals))):
            price = prices[i]
            signal = signals[i]
            action = signal.get("action", "hold")
            size = signal.get("size", 1.0)

            # 슬리피지 적용
            exec_price = price * (1 + self.slippage if action == "long" else 1 - self.slippage)

            if action == "long" and position <= 0:
                # 기존 숏 청산
                if position < 0:
                    pnl = -position * (entry_price - exec_price) / entry_price * self.leverage * equity
                    fee = abs(position) * self.commission * equity
                    equity += pnl - fee
                    trades.append({"type": "close_short", "price": exec_price, "pnl": pnl - fee, "bar": i})

                # 롱 진입
                position = size
                entry_price = exec_price
                fee = size * self.commission * equity
                equity -= fee

            elif action == "short" and position >= 0:
                # 기존 롱 청산
                if position > 0:
                    pnl = position * (exec_price - entry_price) / entry_price * self.leverage * equity
                    fee = abs(position) * self.commission * equity
                    equity += pnl - fee
                    trades.append({"type": "close_long", "price": exec_price, "pnl": pnl - fee, "bar": i})

                # 숏 진입
                position = -size
                entry_price = exec_price
                fee = size * self.commission * equity
                equity -= fee

            elif action == "close" and position != 0:
                if position > 0:
                    pnl = position * (exec_price - entry_price) / entry_price * self.leverage * equity
                else:
                    pnl = -position * (entry_price - exec_price) / entry_price * self.leverage * equity
                fee = abs(position) * self.commission * equity
                equity += pnl - fee
                trades.append({"type": "close", "price": exec_price, "pnl": pnl - fee, "bar": i})
                position = 0.0

            # 미실현 PnL 반영
            if position != 0:
                if position > 0:
                    unrealized = position * (price - entry_price) / entry_price * self.leverage
                else:
                    unrealized = -position * (entry_price - price) / entry_price * self.leverage
                current_equity = equity + unrealized * equity
            else:
                current_equity = equity

            peak_equity = max(peak_equity, current_equity)
            equity_curve.append(current_equity)

        # 결과 계산
        equity_arr = np.array(equity_curve)
        returns = np.diff(equity_arr) / (equity_arr[:-1] + 1e-8)

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        result = BacktestResult(
            total_return=(equity_curve[-1] - self.initial_capital) / self.initial_capital,
            sharpe_ratio=float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)),
            max_drawdown=float(np.max(np.maximum.accumulate(equity_arr) - equity_arr) / (peak_equity + 1e-8)),
            win_rate=len(wins) / len(pnls) if pnls else 0,
            total_trades=len(trades),
            avg_trade_pnl=np.mean(pnls) if pnls else 0,
            profit_factor=abs(sum(wins)) / (abs(sum(losses)) + 1e-8) if losses else float("inf"),
            equity_curve=equity_curve,
            trades=trades,
        )

        logger.info(
            f"백테스트 결과 - 수익률: {result.total_return:.2%}, Sharpe: {result.sharpe_ratio:.2f}, "
            f"MDD: {result.max_drawdown:.2%}, 승률: {result.win_rate:.2%}, 거래수: {result.total_trades}"
        )
        return result
