"""전략 매니저 - ML/RL 시그널을 통합하여 최종 트레이딩 결정"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from loguru import logger


@dataclass
class TradeDecision:
    action: str  # "long", "short", "close", "hold"
    confidence: float
    size: float  # 0.0 ~ 1.0 (포지션 비율)
    reason: str
    timestamp: str = ""
    signals: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = str(datetime.utcnow())


class StrategyManager:
    """ML 앙상블 시그널 + RL 에이전트를 결합한 최종 의사결정"""

    def __init__(self, config: dict):
        self.config = config
        self.min_confidence = config.get("min_confidence", 0.55)
        self.signal_threshold = config.get("signal_threshold", 0.15)
        self.recent_decisions: list[TradeDecision] = []

    def decide(
        self,
        ml_signal: dict,
        rl_action: int,
        rl_confidence: float,
        current_position: float,
        market_regime: str = "normal",
    ) -> TradeDecision:
        """최종 트레이딩 결정"""
        action_map = {0: "hold", 1: "long", 2: "short", 3: "close"}
        rl_direction = action_map[rl_action]

        ml_direction = ml_signal.get("direction", "neutral")
        ml_confidence = ml_signal.get("confidence", 0)
        ml_agreement = ml_signal.get("agreement", 0)
        ml_signal_val = ml_signal.get("signal", 0)

        # 1. ML + RL 합의 확인
        directions_agree = (
            (rl_direction == "long" and ml_direction == "long") or
            (rl_direction == "short" and ml_direction == "short") or
            (rl_direction == "close" and ml_direction == "neutral")
        )

        # 2. 최종 결정 로직
        final_action = "hold"
        confidence = 0.0
        reason = ""

        if rl_direction == "close":
            # 청산은 한쪽만 동의해도 실행
            if rl_confidence > 0.5 or ml_direction == "neutral":
                final_action = "close"
                confidence = max(rl_confidence, ml_confidence)
                reason = "RL 청산 시그널"
        elif directions_agree:
            # 양쪽 합의 → 높은 확신으로 진입
            final_action = rl_direction
            confidence = (rl_confidence * 0.4 + ml_confidence * 0.4 + ml_agreement * 0.2)
            reason = f"ML+RL 합의 ({ml_direction})"
        elif rl_confidence > 0.7 and abs(ml_signal_val) < self.signal_threshold:
            # RL만 강한 확신 + ML 중립 → RL 따르기
            final_action = rl_direction
            confidence = rl_confidence * 0.7
            reason = f"RL 강한 시그널 ({rl_direction})"
        elif ml_confidence > 0.7 and ml_agreement > 0.8:
            # ML 강한 합의 → ML 따르기
            final_action = ml_direction if ml_direction != "neutral" else "hold"
            confidence = ml_confidence * 0.7
            reason = f"ML 강한 합의 ({ml_direction})"

        # 3. 최소 확신도 필터
        if confidence < self.min_confidence and final_action != "close":
            final_action = "hold"
            reason = f"확신도 부족 ({confidence:.2f} < {self.min_confidence})"

        # 4. 시장 레짐 필터
        if market_regime == "extreme_volatility" and final_action in ["long", "short"]:
            confidence *= 0.5
            if confidence < self.min_confidence:
                final_action = "hold"
                reason = "극심한 변동성 - 진입 보류"

        # 5. 포지션 크기 결정 (확신도 비례)
        size = min(confidence, 1.0) if final_action in ["long", "short"] else 0.0

        decision = TradeDecision(
            action=final_action,
            confidence=confidence,
            size=size,
            reason=reason,
            signals={"ml": ml_signal, "rl": {"action": rl_direction, "confidence": rl_confidence}},
        )
        self.recent_decisions.append(decision)

        # 최근 100개만 유지
        if len(self.recent_decisions) > 100:
            self.recent_decisions = self.recent_decisions[-100:]

        return decision
