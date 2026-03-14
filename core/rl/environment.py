"""Gymnasium 호환 트레이딩 환경"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnvironment(gym.Env):
    """
    강화학습 트레이딩 환경

    관찰 공간: [시장 피처들, ML 시그널, 현재 포지션, 미실현 PnL, ...]
    행동 공간: 0=홀드, 1=롱 진입, 2=숏 진입, 3=청산
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: np.ndarray,
        feature_dim: int,
        initial_capital: float = 10000.0,
        commission: float = 0.0004,
        leverage: int = 5,
        max_position: float = 1.0,
    ):
        super().__init__()
        self.df = df  # shape: (timesteps, features + OHLCV)
        self.feature_dim = feature_dim
        self.initial_capital = initial_capital
        self.commission = commission
        self.leverage = leverage
        self.max_position = max_position

        # 관찰: 시장 피처 + 포지션 정보 (4개: position, unrealized_pnl, equity_ratio, holding_time)
        obs_dim = feature_dim + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        # 행동: 홀드, 롱, 숏, 청산
        self.action_space = spaces.Discrete(4)

        self._reset_state()

    def _reset_state(self):
        self.current_step = 0
        self.equity = self.initial_capital
        self.position = 0.0  # -1 ~ 1 (음수=숏, 양수=롱)
        self.entry_price = 0.0
        self.holding_time = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.equity_history = [self.initial_capital]
        self.peak_equity = self.initial_capital

    def _get_price(self, step: int | None = None) -> float:
        """현재 close 가격 (피처 뒤 4번째 = close)"""
        s = step if step is not None else self.current_step
        # df 구조: [features..., open, high, low, close, volume]
        return float(self.df[s, self.feature_dim + 3])

    def _get_obs(self) -> np.ndarray:
        features = self.df[self.current_step, :self.feature_dim].astype(np.float32)
        price = self._get_price()
        unrealized_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl = self.position * (price - self.entry_price) / self.entry_price * self.leverage

        equity_ratio = self.equity / self.initial_capital
        pos_info = np.array([
            self.position,
            unrealized_pnl,
            equity_ratio,
            min(self.holding_time / 100.0, 1.0),
        ], dtype=np.float32)

        return np.concatenate([features, pos_info])

    def _calculate_reward(self, old_equity: float, action: int) -> float:
        """보상 계산: 수익률 기반 + 페널티"""
        equity_change = (self.equity - old_equity) / self.initial_capital

        # 기본 보상: 자산 변화율
        reward = equity_change * 100

        # 거래 비용 페널티
        if action in [1, 2, 3] and self.total_trades > 0:
            reward -= 0.01

        # 드로다운 페널티
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        if drawdown > 0.1:
            reward -= drawdown * 2

        # 장기 홀딩 페널티 (손실 포지션)
        if self.holding_time > 50 and self.position != 0:
            price = self._get_price()
            if self.position > 0 and price < self.entry_price:
                reward -= 0.02
            elif self.position < 0 and price > self.entry_price:
                reward -= 0.02

        return float(reward)

    def step(self, action: int):
        old_equity = self.equity
        price = self._get_price()

        # 행동 실행
        if action == 1 and self.position <= 0:  # 롱 진입
            if self.position < 0:
                self._close_position(price)
            self._open_position(price, 1.0)
        elif action == 2 and self.position >= 0:  # 숏 진입
            if self.position > 0:
                self._close_position(price)
            self._open_position(price, -1.0)
        elif action == 3 and self.position != 0:  # 청산
            self._close_position(price)

        # 미실현 PnL 업데이트
        if self.position != 0:
            self.holding_time += 1
            unrealized = self.position * (price - self.entry_price) / self.entry_price * self.leverage
            current_equity = self.initial_capital + self.total_pnl + unrealized * self.initial_capital
            self.equity = max(current_equity, 0)
        else:
            self.equity = self.initial_capital + self.total_pnl

        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_history.append(self.equity)

        reward = self._calculate_reward(old_equity, action)

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = self.equity <= self.initial_capital * 0.5  # 50% 이상 손실 시 종료

        return self._get_obs(), reward, terminated, truncated, {
            "equity": self.equity,
            "position": self.position,
            "total_trades": self.total_trades,
            "total_pnl": self.total_pnl,
        }

    def _open_position(self, price: float, direction: float):
        fee = abs(direction) * self.commission * self.equity
        self.equity -= fee
        self.position = direction * self.max_position
        self.entry_price = price
        self.holding_time = 0
        self.total_trades += 1

    def _close_position(self, price: float):
        if self.entry_price <= 0:
            return
        pnl = self.position * (price - self.entry_price) / self.entry_price * self.leverage * self.initial_capital
        fee = abs(self.position) * self.commission * self.equity
        self.total_pnl += pnl - fee
        if pnl > 0:
            self.winning_trades += 1
        self.position = 0.0
        self.entry_price = 0.0
        self.holding_time = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def get_metrics(self) -> dict:
        equity_arr = np.array(self.equity_history)
        returns = np.diff(equity_arr) / equity_arr[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)
        max_dd = np.max(np.maximum.accumulate(equity_arr) - equity_arr) / self.peak_equity

        return {
            "total_return": (self.equity - self.initial_capital) / self.initial_capital,
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "final_equity": self.equity,
        }
