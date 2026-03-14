"""PPO 기반 강화학습 트레이딩 에이전트"""

from pathlib import Path

import numpy as np
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from core.rl.environment import TradingEnvironment


class TradingCallback(BaseCallback):
    """학습 중 성과 로깅 콜백"""

    def __init__(self, eval_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            infos = self.locals.get("infos", [{}])
            if infos and "equity" in infos[-1]:
                info = infos[-1]
                logger.info(
                    f"Step {self.n_calls} - Equity: {info['equity']:.2f}, "
                    f"Trades: {info['total_trades']}, PnL: {info['total_pnl']:.2f}"
                )
        return True


class RLAgent:
    """PPO 트레이딩 에이전트"""

    def __init__(self, config: dict, model_dir: str = "models_saved"):
        self.config = config
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: PPO | None = None
        self.env: TradingEnvironment | None = None

    def create_env(self, data: np.ndarray, feature_dim: int, **kwargs) -> TradingEnvironment:
        """학습/평가용 환경 생성"""
        self.env = TradingEnvironment(
            df=data,
            feature_dim=feature_dim,
            initial_capital=kwargs.get("initial_capital", 10000),
            commission=kwargs.get("commission", 0.0004),
            leverage=kwargs.get("leverage", 5),
        )
        return self.env

    def train(self, env: TradingEnvironment, total_timesteps: int | None = None):
        """PPO 에이전트 학습"""
        timesteps = total_timesteps or self.config.get("total_timesteps", 100000)

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config.get("learning_rate", 3e-4),
            n_steps=self.config.get("n_steps", 2048),
            batch_size=self.config.get("batch_size", 64),
            gamma=self.config.get("gamma", 0.99),
            verbose=0,
            policy_kwargs={"net_arch": [256, 256, 128]},
        )

        callback = TradingCallback(eval_freq=10000)
        logger.info(f"PPO 학습 시작 - {timesteps} steps")
        self.model.learn(total_timesteps=timesteps, callback=callback)
        logger.info("PPO 학습 완료")

        metrics = env.get_metrics()
        logger.info(
            f"학습 결과 - 수익률: {metrics['total_return']:.2%}, "
            f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
            f"MDD: {metrics['max_drawdown']:.2%}, "
            f"승률: {metrics['win_rate']:.2%}"
        )
        return metrics

    def predict(self, obs: np.ndarray) -> tuple[int, float]:
        """관찰값으로 행동 결정"""
        if self.model is None:
            return 0, 0.0  # 모델 없으면 홀드

        action, _ = self.model.predict(obs, deterministic=True)
        # 행동 확률 계산
        obs_tensor = self.model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
        dist = self.model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy()[0]
        confidence = float(probs[action])

        return int(action), confidence

    def save(self, name: str = "ppo_trader"):
        if self.model:
            path = self.model_dir / name
            self.model.save(str(path))
            logger.info(f"PPO 모델 저장: {path}")

    def load(self, name: str = "ppo_trader") -> bool:
        path = self.model_dir / f"{name}.zip"
        if not path.exists():
            return False
        self.model = PPO.load(str(self.model_dir / name))
        logger.info(f"PPO 모델 로드: {path}")
        return True
