"""Bayesian Online Changepoint Detection (BOCPD) — 실시간 레짐 변화점 감지
================================================================================
HMMRegimeClassifier(3-state)와 보완:
  - HMM    : 사전 레짐 수 지정, 분류는 정확하나 변환 감지 지연(5-10봉)
  - BOCPD  : 변환점을 1-3봉 만에 포착, 변환 직후 진입 차단 게이트 역할

Adams & MacKay (2007) "Bayesian Online Changepoint Detection":
  매 시점마다 "마지막 변환점 이후 경과 봉 수(run length)"의 사후 확률을
  Student-t 켤레 prior로 온라인 업데이트. 변환점 확률 급증 시 레짐 전환 판단.

승률 개선:
  - 변환점 직후(changepoint_prob > 0.6) → 진입 차단 (잘못된 방향 진입 방지)
  - 변환점 근처(>0.3) → 포지션 사이즈 30%로 축소
  - 추세/평균회귀/고변동성 5-state 분류 → 전략별 진입 여부 판단

순수 numpy로 구현, 외부 의존성 없음 → M2 CPU에서 경량 실행.
StrategyManager.decide()의 stage-1 게이트로 통합 가능.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

# scipy.special.gammaln 사용 (np.lgamma는 존재하지 않음)
try:
    from scipy.special import gammaln as _gammaln
except Exception:  # pragma: no cover
    def _gammaln(x):
        return np.log(np.abs(np.vectorize(lambda v: float(np.math.gamma(v)))(x)))


@dataclass
class RegimeState:
    """현재 레짐 상태"""
    regime: str               # trending_up / trending_down / mean_reverting / high_volatility / transition / unknown
    confidence: float         # 레짐 확신도 [0, 1]
    run_length: int           # 현재 레짐 지속 봉 수 (MAP)
    changepoint_prob: float   # 변환점 확률 [0, 1]
    volatility_regime: str    # low / normal / high / extreme
    recommended_action: str   # aggressive / normal / defensive / exit_only


class StudentTPosterior:
    """Student-t 사후 분포 — 정규-역감마 켤레 prior

    수익률 분포의 평균/분산을 동시 추정. Student-t 의 두꺼운 꼬리는
    금융 수익률의 leptokurtic 특성에 적합.
    """

    def __init__(
        self, mu0: float = 0.0, kappa0: float = 1.0,
        alpha0: float = 1.0, beta0: float = 1.0,
    ):
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        # 충분통계량 — 길이가 점점 늘어남(run length 별)
        self.mu = np.array([mu0], dtype=np.float64)
        self.kappa = np.array([kappa0], dtype=np.float64)
        self.alpha = np.array([alpha0], dtype=np.float64)
        self.beta = np.array([beta0], dtype=np.float64)

    def pdf(self, x: float) -> np.ndarray:
        """Student-t 예측 분포 pdf (각 run length 별)"""
        df = 2.0 * self.alpha
        loc = self.mu
        scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        scale = np.maximum(scale, 1e-12)
        z = (x - loc) / scale
        # log_pdf = lgamma((df+1)/2) - lgamma(df/2) - 0.5*log(df*pi) - log(scale) - (df+1)/2 * log(1+z^2/df)
        log_pdf = (
            _gammaln((df + 1) / 2.0) - _gammaln(df / 2.0)
            - 0.5 * np.log(df * np.pi) - np.log(scale)
            - (df + 1) / 2.0 * np.log1p(z * z / df)
        )
        return np.exp(log_pdf)

    def update(self, x: float):
        """관측 x로 충분통계량 업데이트 + 새 run length=0 prior 추가"""
        new_kappa = self.kappa + 1.0
        new_mu = (self.kappa * self.mu + x) / new_kappa
        new_alpha = self.alpha + 0.5
        new_beta = self.beta + 0.5 * self.kappa * (x - self.mu) ** 2 / new_kappa

        self.mu = np.append([self.mu0], new_mu)
        self.kappa = np.append([self.kappa0], new_kappa)
        self.alpha = np.append([self.alpha0], new_alpha)
        self.beta = np.append([self.beta0], new_beta)

    def prune(self, keep_mask: np.ndarray):
        """확률 낮은 run length 제거 — 메모리 관리"""
        extended = np.append([True], keep_mask)
        self.mu = self.mu[extended]
        self.kappa = self.kappa[extended]
        self.alpha = self.alpha[extended]
        self.beta = self.beta[extended]


class BOCPDRegimeDetector:
    """Bayesian Online Changepoint Detection 레짐 감지기"""

    # 우리 시스템의 market_regime 문자열과 호환 매핑
    REGIME_TO_LEGACY = {
        "trending_up":     "strong_uptrend",
        "trending_down":   "strong_downtrend",
        "mean_reverting":  "ranging",
        "high_volatility": "extreme_volatility",
        "transition":      "extreme_volatility",
        "unknown":         "normal",
    }

    def __init__(
        self,
        hazard_rate: float = 1 / 250,     # 평균 250봉마다 레짐 전환 (5분봉 기준 ≈ 21시간)
        max_run_length: int = 500,
        volatility_window: int = 20,
        trend_window: int = 50,
        changepoint_threshold: float = 0.3,
        block_threshold: float = 0.6,     # 이 이상이면 진입 차단 추천
    ):
        self.hazard_rate = float(hazard_rate)
        self.max_run_length = int(max_run_length)
        self.volatility_window = int(volatility_window)
        self.trend_window = int(trend_window)
        self.changepoint_threshold = float(changepoint_threshold)
        self.block_threshold = float(block_threshold)

        self.posterior = StudentTPosterior()
        self.run_length_probs = np.array([1.0], dtype=np.float64)
        self.t = 0

        self._returns_history: list[float] = []
        self._prices_history: list[float] = []
        self._changepoint_history: list[float] = []

        self._current_regime = RegimeState(
            regime="unknown", confidence=0.0, run_length=0,
            changepoint_prob=0.0, volatility_regime="normal",
            recommended_action="defensive",
        )

    # ------------------------------------------------------------------
    def update(self, price: float, volume: float = 0.0) -> RegimeState:
        """새 가격 데이터 1봉으로 BOCPD 업데이트 → RegimeState 반환"""
        try:
            price = float(price)
            if price <= 0 or not np.isfinite(price):
                return self._current_regime
        except Exception:
            return self._current_regime

        self._prices_history.append(price)
        if len(self._prices_history) < 2:
            return self._current_regime

        prev_price = self._prices_history[-2]
        if prev_price <= 0:
            return self._current_regime
        ret = float(np.log(price / prev_price))
        if not np.isfinite(ret):
            return self._current_regime
        self._returns_history.append(ret)

        # 1) 예측 확률
        pred_probs = self.posterior.pdf(ret)
        # 2) 성장 (현 레짐 지속) vs 변환점 (새 레짐)
        growth_probs = self.run_length_probs * pred_probs * (1.0 - self.hazard_rate)
        changepoint_prob = float(np.sum(self.run_length_probs * pred_probs * self.hazard_rate))
        # 3) 다음 시점 사후분포
        new_probs = np.append(changepoint_prob, growth_probs)
        evidence = float(new_probs.sum())
        if evidence > 0 and np.isfinite(evidence):
            new_probs /= evidence
        else:
            new_probs = np.ones_like(new_probs) / len(new_probs)

        # 4) 사후분포 업데이트
        self.posterior.update(ret)

        # 5) 메모리 관리
        if len(new_probs) > self.max_run_length:
            keep_mask = new_probs[1:] > 1e-10
            if keep_mask.sum() < 10:
                top_indices = np.argsort(new_probs[1:])[-50:]
                keep_mask = np.zeros(len(new_probs) - 1, dtype=bool)
                keep_mask[top_indices] = True
            new_probs = np.append(new_probs[0], new_probs[1:][keep_mask])
            s = new_probs.sum()
            if s > 0:
                new_probs /= s
            self.posterior.prune(keep_mask)

        self.run_length_probs = new_probs
        self.t += 1
        self._changepoint_history.append(changepoint_prob)

        # 레짐 분류
        self._current_regime = self._classify_regime(changepoint_prob)
        return self._current_regime

    # ------------------------------------------------------------------
    def _classify_regime(self, changepoint_prob: float) -> RegimeState:
        returns = np.asarray(self._returns_history, dtype=np.float64)
        # MAP run length
        try:
            map_run_length = int(np.argmax(self.run_length_probs))
        except Exception:
            map_run_length = 0

        # 변동성 레짐
        vol_regime = "normal"
        vol_ratio = 1.0
        if len(returns) >= self.volatility_window:
            recent_vol = float(returns[-self.volatility_window:].std())
            ref_window = min(200, len(returns))
            long_vol = float(returns[-ref_window:].std()) if len(returns) >= 50 else recent_vol
            vol_ratio = recent_vol / (long_vol + 1e-10)
            if vol_ratio < 0.5:
                vol_regime = "low"
            elif vol_ratio < 1.5:
                vol_regime = "normal"
            elif vol_ratio < 2.5:
                vol_regime = "high"
            else:
                vol_regime = "extreme"

        # 추세
        regime = "mean_reverting"
        trend_confidence = 0.0

        if len(returns) >= self.trend_window:
            recent = returns[-self.trend_window:]
            cum_ret = float(recent.sum())
            std = float(recent.std()) + 1e-10
            trend_strength = abs(cum_ret) / (std * np.sqrt(self.trend_window))

            if changepoint_prob > self.changepoint_threshold:
                regime = "transition"
                trend_confidence = changepoint_prob
            elif trend_strength > 1.5 and cum_ret > 0:
                regime = "trending_up"
                trend_confidence = min(trend_strength / 3.0, 1.0)
            elif trend_strength > 1.5 and cum_ret < 0:
                regime = "trending_down"
                trend_confidence = min(trend_strength / 3.0, 1.0)
            elif vol_regime in ("high", "extreme"):
                regime = "high_volatility"
                trend_confidence = min(vol_ratio / 3.0, 1.0)
            else:
                regime = "mean_reverting"
                trend_confidence = 1.0 - min(trend_strength, 1.0)
        elif len(returns) >= 5:
            # 윈도우 부족 시 약한 평균회귀로 가정
            regime = "mean_reverting"
            trend_confidence = 0.3

        action_map = {
            "trending_up":     "aggressive",
            "trending_down":   "aggressive",
            "mean_reverting":  "normal",
            "high_volatility": "defensive",
            "transition":      "exit_only",
            "unknown":         "defensive",
        }
        recommended = action_map.get(regime, "normal")

        # 변환점 강하면 무조건 exit_only
        if changepoint_prob > self.block_threshold:
            recommended = "exit_only"

        return RegimeState(
            regime=regime,
            confidence=float(trend_confidence),
            run_length=int(map_run_length),
            changepoint_prob=float(changepoint_prob),
            volatility_regime=vol_regime,
            recommended_action=recommended,
        )

    # ------------------------------------------------------------------
    def warmup(self, prices: list[float]):
        """시작 시 과거 가격으로 BOCPD warmup — 전체 가격 시퀀스 적용"""
        for p in prices:
            self.update(float(p))

    def should_block_entry(self) -> tuple[bool, str]:
        """진입 차단 게이트 — True면 long/short 모두 차단 권고"""
        st = self._current_regime
        if st.changepoint_prob >= self.block_threshold:
            return True, (
                f"BOCPD changepoint_prob={st.changepoint_prob:.2f}≥{self.block_threshold:.2f} "
                f"(transition→{st.regime})"
            )
        return False, "ok"

    def get_position_size_multiplier(self) -> float:
        """레짐 기반 포지션 사이즈 배수 (0~1.5)"""
        st = self._current_regime
        if st.changepoint_prob > self.block_threshold:
            return 0.0
        if st.changepoint_prob > self.changepoint_threshold:
            return 0.3
        vol_mult = {"low": 1.3, "normal": 1.0, "high": 0.6, "extreme": 0.3}.get(
            st.volatility_regime, 1.0
        )
        regime_mult = {
            "trending_up": 1.2, "trending_down": 1.2,
            "mean_reverting": 0.8, "high_volatility": 0.5, "transition": 0.3,
            "unknown": 0.8,
        }.get(st.regime, 1.0)
        return float(min(vol_mult * regime_mult, 1.5))

    def get_regime_info(self) -> dict:
        """디버그/대시보드용 dict"""
        st = self._current_regime
        return {
            "regime": st.regime,
            "legacy_regime": self.REGIME_TO_LEGACY.get(st.regime, "normal"),
            "confidence": st.confidence,
            "run_length": st.run_length,
            "changepoint_prob": st.changepoint_prob,
            "volatility_regime": st.volatility_regime,
            "recommended_action": st.recommended_action,
            "is_transition": st.changepoint_prob > self.changepoint_threshold,
            "should_block": st.changepoint_prob > self.block_threshold,
        }

    def reset(self):
        self.posterior = StudentTPosterior()
        self.run_length_probs = np.array([1.0], dtype=np.float64)
        self.t = 0
        self._returns_history.clear()
        self._prices_history.clear()
        self._changepoint_history.clear()
        self._current_regime = RegimeState(
            regime="unknown", confidence=0.0, run_length=0,
            changepoint_prob=0.0, volatility_regime="normal",
            recommended_action="defensive",
        )
