"""전략 매니저 - ML/RL/외부요인/모멘텀 시그널을 통합하여 최종 트레이딩 결정

v3 - 적응형 + 모멘텀 fallback:
- 데드존 제거: ML/RL이 neutral/hold여도 모멘텀으로 거래 가능
- 동적 min_confidence: 장기 hold 시 자동 하향
- 가격 모멘텀 기반 fallback: 추세가 명확하면 ML/RL 무시하고 진입
- ML 확률 편향 활용: neutral이어도 long/short 확률 차이 사용
- 자기진단: stuck 상태 감지 및 자동 보정
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

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
    """ML + RL + 외부 요인 + 모멘텀 통합 의사결정 (v3 - 적응형 + 모멘텀)"""

    def __init__(self, config: dict):
        self.config = config
        self.base_min_confidence = config.get("min_confidence", 0.30)
        self.min_confidence = self.base_min_confidence
        self.signal_threshold = config.get("signal_threshold", 0.08)
        self.recent_decisions: list[TradeDecision] = []

        # 자기진단 상태
        self._consecutive_holds = 0
        self._last_trade_time: datetime | None = None
        self._confidence_decay_rate = 0.01
        self._min_floor = 0.15

    def _adaptive_min_confidence(self) -> float:
        """연속 hold 시 min_confidence를 점진적으로 낮춤"""
        if self._consecutive_holds > 30:
            decay = (self._consecutive_holds - 30) * self._confidence_decay_rate
            adjusted = max(self.base_min_confidence - decay, self._min_floor)
            return adjusted
        return self.base_min_confidence

    def decide(
        self,
        ml_signal: dict,
        rl_action: int,
        rl_confidence: float,
        current_position: float,
        market_regime: str = "normal",
        external_signal: dict | None = None,
        momentum: dict | None = None,
    ) -> TradeDecision:
        """최종 트레이딩 결정 (ML + RL + 외부 + 모멘텀)"""
        action_map = {0: "hold", 1: "long", 2: "short", 3: "close"}
        rl_direction = action_map.get(rl_action, "hold")

        ml_direction = ml_signal.get("direction", "neutral")
        ml_confidence = ml_signal.get("confidence", 0)
        ml_agreement = ml_signal.get("agreement", 0)
        ml_signal_val = ml_signal.get("signal", 0)

        # ML 확률 분포 활용 (neutral이어도 방향 편향 사용)
        ml_probs = {}
        for model_name, model_pred in ml_signal.get("models", {}).items():
            probs = model_pred.get("probabilities", {})
            if probs:
                ml_probs[model_name] = probs

        # 외부 신호 파싱
        ext = external_signal or {}
        ext_score = ext.get("score", 0)
        ext_direction = ext.get("direction", "neutral")
        ext_strength = ext.get("strength", "weak")
        ext_confidence = ext.get("confidence", 0)
        has_high_impact = ext.get("high_impact_events", False)

        # 모멘텀 파싱
        mom = momentum or {}
        mom_direction = mom.get("direction", "neutral")
        mom_strength = mom.get("strength", 0)
        mom_rsi = mom.get("rsi", 50)
        mom_trend = mom.get("trend_aligned", False)

        # 적응형 min_confidence
        self.min_confidence = self._adaptive_min_confidence()

        # 1. ML + RL 합의 확인
        directions_agree = (
            (rl_direction == "long" and ml_direction == "long") or
            (rl_direction == "short" and ml_direction == "short") or
            (rl_direction == "close" and ml_direction == "neutral")
        )

        # 2. 최종 결정 로직 (v3 - 다중 경로)
        final_action = "hold"
        confidence = 0.0
        reason = ""

        # PATH A: 청산 시그널
        if rl_direction == "close":
            if rl_confidence > 0.4 or ml_direction == "neutral":
                final_action = "close"
                confidence = max(rl_confidence, ml_confidence, 0.5)
                reason = "RL 청산"

        # PATH B: ML + RL 양쪽 합의
        elif directions_agree:
            final_action = rl_direction
            confidence = (
                rl_confidence * 0.20 +
                ml_confidence * 0.20 +
                ml_agreement * 0.10 +
                ext_confidence * 0.15 +
                0.15  # 합의 보너스
            )
            if mom_direction == final_action:
                confidence += 0.10
                reason = f"ML+RL+모멘텀 합의 ({ml_direction})"
            else:
                reason = f"ML+RL 합의 ({ml_direction})"

        # PATH C: RL 강한 시그널
        elif rl_direction in ("long", "short") and rl_confidence > 0.6:
            final_action = rl_direction
            confidence = rl_confidence * 0.4 + ext_confidence * 0.10 + 0.10
            reason = f"RL 시그널 ({rl_direction})"

        # PATH D: ML 방향이 있을 때
        elif ml_direction != "neutral" and ml_confidence > 0.4:
            final_action = ml_direction
            confidence = ml_confidence * 0.40 + ml_agreement * 0.10 + ext_confidence * 0.15 + 0.10
            reason = f"ML 시그널 ({ml_direction}, conf={ml_confidence:.2f})"

        # PATH E: ML signal 값 자체 방향
        elif abs(ml_signal_val) > self.signal_threshold:
            if ml_signal_val > self.signal_threshold:
                final_action = "long"
            else:
                final_action = "short"
            confidence = min(abs(ml_signal_val) * 2.5, 0.6) + ext_confidence * 0.15 + 0.10
            reason = f"ML값 ({ml_signal_val:.3f})"

        # PATH F: 외부 시그널만 (ML/RL 모두 neutral/hold)
        elif ext_confidence > 0.2 and ext_direction != "neutral":
            if ext_direction == "bullish":
                final_action = "long"
            elif ext_direction == "bearish":
                final_action = "short"
            confidence = ext_confidence * 0.5 + 0.10
            reason = f"외부시그널 ({ext_direction}, score={ext_score:.2f})"

        # PATH G: 모멘텀 fallback 또는 RSI 극값 반전
        elif (mom_direction in ("long", "short") and abs(mom_strength) > 0.3) or mom_rsi < 25 or mom_rsi > 75:
            # RSI 극값 반전 우선
            if mom_rsi < 25:
                final_action = "long"
                rsi_extreme = (25 - mom_rsi) / 25  # 0~1
                confidence = 0.20 + rsi_extreme * 0.30
                reason = f"RSI과매도 반전 (RSI={mom_rsi:.0f})"
            elif mom_rsi > 75:
                final_action = "short"
                rsi_extreme = (mom_rsi - 75) / 25  # 0~1
                confidence = 0.20 + rsi_extreme * 0.30
                reason = f"RSI과매수 반전 (RSI={mom_rsi:.0f})"
            else:
                final_action = mom_direction
                confidence = min(abs(mom_strength) * 0.6, 0.5) + 0.10
                reason = f"모멘텀 ({mom_direction}, str={mom_strength:.2f})"
            if mom_trend:
                confidence += 0.10
                reason += " +트렌드일치"

        # PATH H: ML 확률 편향 (argmax neutral이지만 long/short 확률 차이가 있을 때)
        elif ml_probs:
            avg_long_prob = np.mean([p.get("long", 0.33) for p in ml_probs.values()])
            avg_short_prob = np.mean([p.get("short", 0.33) for p in ml_probs.values()])
            prob_diff = avg_long_prob - avg_short_prob

            if abs(prob_diff) > 0.05:  # 5% 이상 차이
                if prob_diff > 0:
                    final_action = "long"
                else:
                    final_action = "short"
                confidence = min(abs(prob_diff) * 3, 0.4) + ext_confidence * 0.15 + 0.05
                reason = f"ML확률편향 (L:{avg_long_prob:.2f} S:{avg_short_prob:.2f})"

        # 2.5. 외부 신호 보정
        if final_action in ["long", "short"] and ext_direction != "neutral":
            ext_agrees = (
                (final_action == "long" and ext_direction == "bullish") or
                (final_action == "short" and ext_direction == "bearish")
            )
            if ext_agrees and ext_strength == "strong":
                confidence *= 1.25
                reason += " +외부강지지"
            elif not ext_agrees and ext_strength == "strong":
                confidence *= 0.6
                reason += f" !외부반대({ext_direction})"
            elif ext_agrees:
                confidence *= 1.15
                reason += " +외부지지"

        # 2.6. 고임팩트 이벤트 오버라이드
        if has_high_impact and abs(ext_score) > 0.4:
            if ext_direction == "bearish" and final_action == "long":
                final_action = "hold"
                reason = f"고임팩트-롱차단 (ext={ext_score:.2f})"
            elif ext_direction == "bullish" and final_action == "short":
                final_action = "hold"
                reason = f"고임팩트-숏차단 (ext={ext_score:.2f})"

        # 3. 최소 확신도 필터 (적응형)
        if confidence < self.min_confidence and final_action not in ("close", "hold"):
            reason = (
                f"확신도 부족 ({confidence:.3f} < {self.min_confidence:.3f}) "
                f"[ML:{ml_direction}/{ml_confidence:.2f} RL:{rl_direction}/{rl_confidence:.2f} "
                f"Ext:{ext_direction}/{ext_confidence:.2f} Mom:{mom_direction}/{mom_strength:.2f}]"
            )
            final_action = "hold"

        # 4. 시장 레짐 필터
        if market_regime == "extreme_volatility" and final_action in ["long", "short"]:
            confidence *= 0.7
            if confidence < self.min_confidence:
                final_action = "hold"
                reason = "극심한 변동성 - 진입 보류"

        # 5. 포지션 크기 결정
        confidence = min(confidence, 1.0)
        size = confidence if final_action in ["long", "short"] else 0.0

        # 자기진단: hold 카운터
        if final_action == "hold":
            self._consecutive_holds += 1
        else:
            self._consecutive_holds = 0
            if final_action in ("long", "short"):
                self._last_trade_time = datetime.utcnow()

        decision = TradeDecision(
            action=final_action,
            confidence=confidence,
            size=size,
            reason=reason,
            signals={
                "ml": ml_signal,
                "rl": {"action": rl_direction, "confidence": rl_confidence},
                "external": {"score": ext_score, "direction": ext_direction, "strength": ext_strength},
                "momentum": mom,
                "adaptive": {
                    "min_confidence": round(self.min_confidence, 3),
                    "consecutive_holds": self._consecutive_holds,
                    "last_trade": self._last_trade_time.isoformat() if self._last_trade_time else None,
                },
            },
        )
        self.recent_decisions.append(decision)

        if len(self.recent_decisions) > 100:
            self.recent_decisions = self.recent_decisions[-100:]

        # 주기적 자기진단 로그
        if self._consecutive_holds > 0 and self._consecutive_holds % 30 == 0:
            logger.warning(
                f"[Strategy] {self._consecutive_holds}회 연속 HOLD | "
                f"min_conf: {self.min_confidence:.3f} | "
                f"ML: {ml_direction}({ml_confidence:.3f}) sig={ml_signal_val:.3f} | "
                f"RL: {rl_direction}({rl_confidence:.3f}) | "
                f"Ext: {ext_direction}({ext_confidence:.3f}) | "
                f"Mom: {mom_direction}({mom_strength:.3f})"
            )

        return decision

    def get_diagnostics(self) -> dict:
        """자기진단 상태 반환"""
        return {
            "consecutive_holds": self._consecutive_holds,
            "current_min_confidence": round(self.min_confidence, 3),
            "base_min_confidence": self.base_min_confidence,
            "last_trade_time": self._last_trade_time.isoformat() if self._last_trade_time else None,
            "is_stuck": self._consecutive_holds > 100,
            "recent_actions": [d.action for d in self.recent_decisions[-10:]],
        }
