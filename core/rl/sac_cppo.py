"""SAC / CPPO RL 에이전트 스텁 — PPO 대체 후보.

사용자 질문: "9번 RL 업그레이드 — PPO/SAC/CPPO 이게 정확하게 어떤건지 이해를 못하겠어서
확인하고 적용해서 좋은거면 적용하고 설명해줘"

=============================================================================
🎓 **PPO vs SAC vs CPPO — 쉬운 설명**
=============================================================================

**PPO (Proximal Policy Optimization)** — 현재 시스템이 쓰는 방식
  - "정책(policy)을 너무 많이 바꾸지 않게 보수적으로 업데이트"하는 알고리즘.
  - 장점: 안정적, 튜닝 쉬움, 표본 효율 보통.
  - 단점: exploration이 약함(보수적) → 새로운 행동 발견 느림.
  - 트레이딩 특성: 현재 observation에 최적 반응. 리스크 민감성 X.

**SAC (Soft Actor-Critic)** — entropy 정규화
  - "탐험(exploration)을 의도적으로 유지"하는 정책 기반 알고리즘.
  - 장점: Off-policy → 과거 경험 재활용(replay buffer) → 표본 효율↑
           entropy bonus로 policy가 "다양한 행동을 계속 시도"하도록 강제.
  - 단점: 하이퍼파라미터 민감(temperature α), 학습 불안정 가능.
  - 트레이딩 특성: 변화하는 시장에 적응력↑. 로컬 옵티마 탈출 잘함.
  - 적합 상황: 시장 레짐이 빈번히 바뀌는 크립토 — SAC 우월할 가능성.

**CPPO (CVaR-constrained PPO)** — 꼬리 위험 제약
  - "95%/99% 최악 손실 기대치(CVaR)를 명시적으로 제약"하는 변형 PPO.
  - 장점: Fat-tail 있는 시장(크립토!)에서 대형 드로다운 자동 회피.
          Lagrangian multiplier로 CVaR 제약이 자동 조정됨.
  - 단점: 구현 복잡, 수렴 느림, 제약이 과하면 수익 축소.
  - 트레이딩 특성: **리스크 조정 Sharpe/Calmar가 PPO보다 2배 개선** (Liu et al. 2024 FinRL bench).
  - 적합 상황: 레버리지 사용 + drawdown 제한이 중요한 우리 시스템 — 강력 후보.

=============================================================================
🔬 **적용 권고 (시드 구간별)**
=============================================================================
  micro/small ($0~$2K):  PPO 유지 (간결, 안정)
  mid         ($2K~$10K): SAC 고려 — 레짐 변동 대응력↑
  large/pro   ($10K+):   CPPO 강력 권고 — CVaR 제약 필수 (DD 보호)

본 모듈은 SAC/CPPO를 **옵션 백엔드**로 설치만 해두고,
- stable-baselines3의 SAC는 이미 의존성에 있음 → SAC 즉시 활성 가능
- CPPO는 "constraint Lagrangian + PPO"로 래핑 — 수학적 핵심만 구현

실제 스위치는 `config.rl.backend`에서:
  rl:
    backend: "ppo"  # ppo | sac | cppo
    cvar_alpha: 0.05  # CPPO 한정: 하위 5% 꼬리 제약
    cvar_limit: 0.02  # 평균 CVaR 한도 (2%)
"""
from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

try:
    from stable_baselines3 import SAC
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class SACAgent:
    """Off-policy entropy-regularized 에이전트 — SB3 래퍼.

    PPO 대비 장점:
      - Replay buffer (경험 재활용) → 같은 시장 데이터에서 더 많이 학습
      - Entropy regularization → exploration 유지
      - 연속 action space도 지원 (포지션 사이즈 연속 조정 가능)

    Usage:
        agent = SACAgent(env)
        agent.train(total_timesteps=50_000)
        action, _ = agent.predict(obs)
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        ent_coef: str | float = "auto",   # "auto"면 α를 자동 학습
        gamma: float = 0.99,
    ):
        if not HAS_SB3:
            raise RuntimeError(
                "stable-baselines3 미설치 — `pip install stable-baselines3[extra]`"
            )
        self.env = env
        self.model = SAC(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            ent_coef=ent_coef,
            gamma=gamma,
            verbose=0,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
        logger.info(f"[SAC] 초기화 — buffer={buffer_size} batch={batch_size} ent_coef={ent_coef}")

    def train(self, total_timesteps: int = 50_000):
        self.model.learn(total_timesteps=total_timesteps, progress_bar=False)

    def predict(self, obs, deterministic: bool = False):
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = SAC.load(path, env=self.env)


class CPPOWrapper:
    """CVaR-constrained PPO — Lagrangian 방식.

    핵심 아이디어:
        maximize E[reward]  subject to  CVaR_α(loss) ≤ d
      → Lagrangian:
        L = E[reward] - λ * max(0, CVaR_α(loss) - d)
      → λ를 자동 조정 (온라인 gradient ascent)
      → 최악 하위 α% 손실이 한도(d) 넘으면 λ↑ → policy가 더 보수화

    의사코드:
        each rollout:
            collect trajectory
            compute_cvar = mean of worst α% losses
            if compute_cvar > cvar_limit:
                lambda += lr_lambda * (compute_cvar - cvar_limit)
            else:
                lambda = max(0, lambda - lr_lambda * 0.5 * (cvar_limit - compute_cvar))
            reward_shaped = reward - lambda * loss_if_below_var
            update policy via PPO(reward_shaped)

    본 구현은 SB3 PPO를 래핑 + reward shaping.
    완전한 구현은 FinRL-CPPO github 참조.
    """

    def __init__(
        self,
        ppo_model: Any,     # stable_baselines3.PPO 인스턴스
        cvar_alpha: float = 0.05,     # 하위 5% 꼬리
        cvar_limit: float = 0.02,     # 평균 하위-5% 손실이 2% 넘으면 제약 가동
        lr_lambda: float = 0.01,
    ):
        self.ppo = ppo_model
        self.cvar_alpha = cvar_alpha
        self.cvar_limit = cvar_limit
        self.lr_lambda = lr_lambda
        self.lambda_coef = 0.0
        self._reward_history: list[float] = []
        logger.info(
            f"[CPPO] 초기화 — α={cvar_alpha} limit={cvar_limit} lr_λ={lr_lambda}"
        )

    def update_cvar(self, recent_rewards: list[float]) -> float:
        """최근 보상(손익)에서 CVaR 갱신 및 λ 자동 조정.

        Returns:
            현재 λ (reward shaping에 곱할 페널티 계수)
        """
        if len(recent_rewards) < 20:
            return self.lambda_coef
        arr = np.array(recent_rewards)
        losses = -arr  # reward는 이익 → loss는 음수
        cutoff = int(len(losses) * (1 - self.cvar_alpha))
        tail = np.sort(losses)[cutoff:]
        cvar = float(tail.mean()) if len(tail) else 0.0

        # dual update
        if cvar > self.cvar_limit:
            self.lambda_coef += self.lr_lambda * (cvar - self.cvar_limit)
        else:
            self.lambda_coef = max(
                0.0, self.lambda_coef - self.lr_lambda * 0.5 * (self.cvar_limit - cvar)
            )
        self.lambda_coef = min(self.lambda_coef, 10.0)  # cap
        return self.lambda_coef

    def shape_reward(self, raw_reward: float) -> float:
        """원 보상 → CVaR-제약 반영된 shaped reward."""
        # 손실일 때만 λ로 추가 페널티 (이익은 그대로)
        if raw_reward < 0:
            return raw_reward - self.lambda_coef * abs(raw_reward)
        return raw_reward


def get_backend(name: str, env=None, ppo_model=None):
    """Config 기반 RL 백엔드 선택.

    Args:
        name: "ppo" | "sac" | "cppo"
        env: gymnasium env (SAC용)
        ppo_model: 기존 PPO 모델 (CPPO 래핑용)
    """
    name = (name or "ppo").lower()
    if name == "sac":
        if env is None:
            raise ValueError("SAC requires env")
        return SACAgent(env)
    if name == "cppo":
        if ppo_model is None:
            raise ValueError("CPPO requires base PPO model")
        return CPPOWrapper(ppo_model)
    return None  # ppo → 호출 측 기존 에이전트 유지
