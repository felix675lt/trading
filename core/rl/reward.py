"""보상 함수 모듈 - 다양한 보상 전략"""

import numpy as np


class RewardCalculator:
    """다양한 보상 함수를 제공하는 모듈"""

    def __init__(self, reward_type: str = "sharpe"):
        self.reward_type = reward_type
        self.returns_buffer: list[float] = []

    def calculate(self, equity_change_pct: float, position: float,
                  holding_time: int, drawdown: float) -> float:
        if self.reward_type == "sharpe":
            return self._sharpe_reward(equity_change_pct)
        elif self.reward_type == "sortino":
            return self._sortino_reward(equity_change_pct)
        elif self.reward_type == "calmar":
            return self._calmar_reward(equity_change_pct, drawdown)
        else:
            return self._simple_reward(equity_change_pct)

    def _sharpe_reward(self, ret: float) -> float:
        self.returns_buffer.append(ret)
        if len(self.returns_buffer) < 20:
            return ret * 100
        recent = self.returns_buffer[-100:]
        mean_r = np.mean(recent)
        std_r = np.std(recent) + 1e-8
        return float(mean_r / std_r * 10)

    def _sortino_reward(self, ret: float) -> float:
        self.returns_buffer.append(ret)
        if len(self.returns_buffer) < 20:
            return ret * 100
        recent = self.returns_buffer[-100:]
        mean_r = np.mean(recent)
        downside = np.std([r for r in recent if r < 0]) + 1e-8
        return float(mean_r / downside * 10)

    def _calmar_reward(self, ret: float, drawdown: float) -> float:
        self.returns_buffer.append(ret)
        reward = ret * 100
        if drawdown > 0.05:
            reward -= drawdown * 50
        return float(reward)

    def _simple_reward(self, ret: float) -> float:
        return float(ret * 100)

    def reset(self):
        self.returns_buffer.clear()
