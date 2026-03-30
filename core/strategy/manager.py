"""전략 매니저 - ML/RL/외부요인/모멘텀 시그널을 통합하여 최종 트레이딩 결정

v4 - 다중확인(Multi-Confirmation) 시스템:
- 단일 시그널 진입 제거: 최소 2개 독립 시그널 합의 필요
- 투표 기반: ML, RL, 모멘텀, 외부요인이 각각 1표씩 투표
- 확신도 하한 강화: 0.40 기본, 최저 0.25 (0.15는 너무 느슨)
- 손실 후 엄격화: 연패 시 확신도 요구치 자동 상향
- 피드백 블랙리스트: 반복 실패 패턴 자동 차단
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
    confirming_sources: list = field(default_factory=list)  # 합의한 시그널 소스 목록
    signal_strength: str = "weak"  # "strong" (3+), "moderate" (2), "weak" (1)
    trade_type: str = "scalp"  # "scalp" or "swing"

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = str(datetime.utcnow())


class StrategyManager:
    """ML + RL + 외부 요인 + 모멘텀 통합 의사결정 (v4 - 다중확인 시스템)"""

    def __init__(self, config: dict, trade_profiles: dict | None = None):
        self.config = config
        self.trade_profiles = trade_profiles or {}
        self.base_min_confidence = config.get("min_confidence", 0.40)
        self.min_confidence = self.base_min_confidence
        self.signal_threshold = config.get("signal_threshold", 0.10)
        self.min_confirming = config.get("min_confirming_signals", 2)
        self.recent_decisions: list[TradeDecision] = []

        # 자기진단 상태
        self._consecutive_holds = 0
        self._last_trade_time: datetime | None = None
        self._confidence_decay_rate = 0.005  # 느린 감소 (0.01→0.005)
        self._min_floor = 0.25               # 최저 하한 (0.15→0.25)

        # 손실 후 엄격화
        self._recent_loss_count = 0          # 최근 연패 횟수
        self._loss_penalty_holds = 0         # 손실 패널티 남은 hold 수
        self._loss_confidence_boost = 0.0    # 손실 후 추가 확신도 요구

    def record_loss(self):
        """외부에서 손실 발생 알림 → 확신도 요구 임시 상향"""
        self._recent_loss_count += 1
        self._loss_penalty_holds = 24  # 12분(24루프×30초) 동안 엄격 모드
        # 연패할수록 더 엄격 (최대 +0.15)
        self._loss_confidence_boost = min(self._recent_loss_count * 0.05, 0.15)
        logger.info(
            f"[Strategy] 손실 기록 → {self._recent_loss_count}연패 | "
            f"확신도 +{self._loss_confidence_boost:.2f} 상향 ({self._loss_penalty_holds}루프)"
        )

    def record_win(self):
        """외부에서 수익 발생 알림 → 엄격 모드 해제"""
        self._recent_loss_count = 0
        self._loss_penalty_holds = 0
        self._loss_confidence_boost = 0.0

    def _adaptive_min_confidence(self) -> float:
        """연속 hold 시 min_confidence를 점진적으로 낮춤 (느리게, 높은 하한)"""
        base = self.base_min_confidence

        # 손실 패널티 적용
        if self._loss_penalty_holds > 0:
            base += self._loss_confidence_boost
            self._loss_penalty_holds -= 1

        # 연속 hold 60회(30분) 이후부터 점진 하향
        if self._consecutive_holds > 60:
            decay = (self._consecutive_holds - 60) * self._confidence_decay_rate
            adjusted = max(base - decay, self._min_floor)
            return adjusted
        return base

    def _classify_trade_type(self, confirming_sources: list[str]) -> str:
        """투표 소스 기반으로 scalp/swing 분류"""
        scalp_sources = set(
            self.trade_profiles.get("scalp", {}).get(
                "sources", ["ML", "ML_val", "MOM", "RSI_extreme"]
            )
        )
        swing_sources = set(
            self.trade_profiles.get("swing", {}).get(
                "sources", ["EXT", "EXT_boost", "RL"]
            )
        )
        confirming_set = set(confirming_sources)
        scalp_count = len(confirming_set & scalp_sources)
        swing_count = len(confirming_set & swing_sources)

        trade_type = "swing" if swing_count > scalp_count else "scalp"
        logger.info(
            f"[TradeType] {trade_type} | 소스: {confirming_sources} | "
            f"scalp={scalp_count} swing={swing_count}"
        )
        return trade_type

    def _count_signal_votes(
        self,
        ml_direction: str, ml_confidence: float, ml_signal_val: float,
        rl_direction: str, rl_confidence: float,
        mom_direction: str, mom_strength: float, mom_rsi: float,
        ext_direction: str, ext_confidence: float,
    ) -> dict:
        """각 독립 소스의 방향 투표 수집

        Returns: {"long": [(source, weight), ...], "short": [(source, weight), ...]}
        """
        votes = {"long": [], "short": []}

        # 1. ML 모델 투표 (confidence > 0.3 + 방향 명확)
        if ml_direction == "long" and ml_confidence > 0.3:
            votes["long"].append(("ML", ml_confidence))
        elif ml_direction == "short" and ml_confidence > 0.3:
            votes["short"].append(("ML", ml_confidence))
        # ML signal 값이 방향과 일치하면 추가 투표 (같은 소스이므로 0.5 가중)
        elif abs(ml_signal_val) > self.signal_threshold:
            if ml_signal_val > self.signal_threshold:
                votes["long"].append(("ML_val", min(abs(ml_signal_val) * 2, 0.6)))
            else:
                votes["short"].append(("ML_val", min(abs(ml_signal_val) * 2, 0.6)))

        # 2. RL 에이전트 투표 (confidence > 0.4)
        if rl_direction == "long" and rl_confidence > 0.4:
            votes["long"].append(("RL", rl_confidence))
        elif rl_direction == "short" and rl_confidence > 0.4:
            votes["short"].append(("RL", rl_confidence))

        # 3. 모멘텀 투표 (strength > 0.2 또는 RSI 극값)
        if mom_direction == "long" and abs(mom_strength) > 0.2:
            votes["long"].append(("MOM", abs(mom_strength)))
        elif mom_direction == "short" and abs(mom_strength) > 0.2:
            votes["short"].append(("MOM", abs(mom_strength)))
        # RSI 극값 (25 이하 = 과매도 반전, 75 이상 = 과매수 반전)
        if mom_rsi < 25:
            rsi_weight = (25 - mom_rsi) / 25 * 0.5
            votes["long"].append(("RSI_extreme", rsi_weight))
        elif mom_rsi > 75:
            rsi_weight = (mom_rsi - 75) / 25 * 0.5
            votes["short"].append(("RSI_extreme", rsi_weight))

        # 4. 외부 요인 투표 (confidence > 0.15)
        if ext_direction == "bullish" and ext_confidence > 0.15:
            votes["long"].append(("EXT", ext_confidence))
        elif ext_direction == "bearish" and ext_confidence > 0.15:
            votes["short"].append(("EXT", ext_confidence))

        return votes

    def decide(
        self,
        ml_signal: dict,
        rl_action: int,
        rl_confidence: float,
        current_position: float,
        market_regime: str = "normal",
        external_signal: dict | None = None,
        momentum: dict | None = None,
        feedback_blacklist: list | None = None,
        funding_rate: float = 0.0,
        fear_greed_index: float = 50.0,
    ) -> TradeDecision:
        """최종 트레이딩 결정 (v4 - 다중확인 시스템)"""
        action_map = {0: "hold", 1: "long", 2: "short", 3: "close"}
        rl_direction = action_map.get(rl_action, "hold")

        ml_direction = ml_signal.get("direction", "neutral")
        ml_confidence = ml_signal.get("confidence", 0)
        ml_agreement = ml_signal.get("agreement", 0)
        ml_signal_val = ml_signal.get("signal", 0)

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

        # === 다중확인 시스템: 투표 수집 ===
        votes = self._count_signal_votes(
            ml_direction, ml_confidence, ml_signal_val,
            rl_direction, rl_confidence,
            mom_direction, mom_strength, mom_rsi,
            ext_direction, ext_confidence,
        )

        long_votes = votes["long"]
        short_votes = votes["short"]
        long_count = len(long_votes)
        short_count = len(short_votes)

        # 2. 최종 결정 로직
        final_action = "hold"
        confidence = 0.0
        reason = ""
        confirming = []
        signal_str = "weak"

        # === PATH A: 청산 시그널 (RL이 close 요청) ===
        if rl_direction == "close":
            if rl_confidence > 0.4 or ml_direction == "neutral":
                final_action = "close"
                confidence = max(rl_confidence, ml_confidence, 0.5)
                reason = "RL 청산"
                confirming = ["RL_close"]

        # === PATH B: 다중확인 진입 (핵심 변경) ===
        # 더 많은 투표를 받은 방향으로, 최소 min_confirming 개 합의 필요
        elif long_count >= self.min_confirming and long_count > short_count:
            final_action = "long"
            confirming = [v[0] for v in long_votes]
            # 확신도 = 투표 가중치 평균 + 합의 보너스
            avg_weight = np.mean([v[1] for v in long_votes])
            consensus_bonus = min((long_count - 1) * 0.10, 0.20)  # 2개=+0.10, 3개=+0.20
            confidence = avg_weight * 0.60 + consensus_bonus + 0.15
            # ML 모델 합의도 반영
            if ml_agreement > 0.5:
                confidence += ml_agreement * 0.10
            signal_str = "strong" if long_count >= 3 else "moderate"
            reason = f"다중확인 LONG ({long_count}표: {','.join(confirming)})"

        elif short_count >= self.min_confirming and short_count > long_count:
            final_action = "short"
            confirming = [v[0] for v in short_votes]
            avg_weight = np.mean([v[1] for v in short_votes])
            consensus_bonus = min((short_count - 1) * 0.10, 0.20)
            confidence = avg_weight * 0.60 + consensus_bonus + 0.15
            if ml_agreement > 0.5:
                confidence += ml_agreement * 0.10
            signal_str = "strong" if short_count >= 3 else "moderate"
            reason = f"다중확인 SHORT ({short_count}표: {','.join(confirming)})"

        # === PATH C: ML+RL 강한 합의 (2표지만 모두 높은 확신) ===
        elif (ml_direction in ("long", "short") and rl_direction == ml_direction
              and ml_confidence > 0.5 and rl_confidence > 0.5):
            final_action = ml_direction
            confirming = ["ML_strong", "RL_strong"]
            confidence = (ml_confidence * 0.35 + rl_confidence * 0.35 +
                          ml_agreement * 0.10 + 0.15)
            signal_str = "moderate"
            reason = f"ML+RL 강합의 ({ml_direction}, ML:{ml_confidence:.2f} RL:{rl_confidence:.2f})"

        # 2.5. 외부 신호 보정 (진입 시그널이 있을 때만)
        if final_action in ["long", "short"] and ext_direction != "neutral":
            ext_agrees = (
                (final_action == "long" and ext_direction == "bullish") or
                (final_action == "short" and ext_direction == "bearish")
            )
            if ext_agrees and ext_strength == "strong":
                confidence *= 1.20
                reason += " +외부강지지"
                if "EXT" not in confirming:
                    confirming.append("EXT_boost")
            elif not ext_agrees and ext_strength == "strong":
                confidence *= 0.55  # 강한 외부 반대 = 큰 감액
                reason += f" !외부강반대({ext_direction})"
            elif ext_agrees:
                confidence *= 1.10
                reason += " +외부지지"
            elif not ext_agrees:
                confidence *= 0.75  # 약한 외부 반대도 감액
                reason += f" !외부반대({ext_direction})"

        # 2.6. 모멘텀 트렌드 일치 보너스
        if final_action in ["long", "short"] and mom_trend:
            if (final_action == "long" and mom_direction == "long") or \
               (final_action == "short" and mom_direction == "short"):
                confidence *= 1.10
                reason += " +트렌드일치"

        # 2.7. 고임팩트 이벤트 오버라이드
        if has_high_impact and abs(ext_score) > 0.4:
            if ext_direction == "bearish" and final_action == "long":
                final_action = "hold"
                reason = f"고임팩트-롱차단 (ext={ext_score:.2f})"
            elif ext_direction == "bullish" and final_action == "short":
                final_action = "hold"
                reason = f"고임팩트-숏차단 (ext={ext_score:.2f})"

        # 2.8. 레짐 방향 바이어스 (펀딩비 + 공포탐욕 + 가격추세 종합)
        if final_action in ["long", "short"]:
            direction_block = self._regime_direction_bias(
                final_action, funding_rate, fear_greed_index, mom_direction, mom_rsi,
            )
            if direction_block:
                old_action = final_action
                final_action = "hold"
                confidence = 0.0
                reason = direction_block
                confirming = []

        # 2.9. 피드백 블랙리스트 필터
        if final_action in ["long", "short"] and feedback_blacklist:
            combo_key = "+".join(sorted(confirming))
            if combo_key in feedback_blacklist:
                old_action = final_action
                final_action = "hold"
                confidence = 0.0
                reason = f"피드백 블랙리스트 차단 ({combo_key} → {old_action})"
                confirming = []

        # 3. 최소 확신도 필터 (강화)
        if confidence < self.min_confidence and final_action not in ("close", "hold"):
            max_long = max([v[1] for v in long_votes], default=0)
            max_short = max([v[1] for v in short_votes], default=0)
            reason = (
                f"확신도 부족 ({confidence:.3f} < {self.min_confidence:.3f}) "
                f"[L:{long_count}표(max={max_long:.2f}) S:{short_count}표(max={max_short:.2f}) "
                f"ML:{ml_direction}/{ml_confidence:.2f} RL:{rl_direction}/{rl_confidence:.2f}]"
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

        # 5.5 트레이드 타입 분류 (scalp vs swing)
        trade_type = "scalp"
        if final_action in ("long", "short"):
            trade_type = self._classify_trade_type(confirming)

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
            confirming_sources=confirming,
            signal_strength=signal_str,
            trade_type=trade_type,
            signals={
                "ml": ml_signal,
                "rl": {"action": rl_direction, "confidence": rl_confidence},
                "external": {"score": ext_score, "direction": ext_direction, "strength": ext_strength},
                "momentum": mom,
                "votes": {
                    "long": [(s, round(w, 3)) for s, w in long_votes],
                    "short": [(s, round(w, 3)) for s, w in short_votes],
                },
                "regime_bias": {
                    "funding_rate": round(funding_rate, 3),
                    "fear_greed": round(fear_greed_index, 1),
                },
                "adaptive": {
                    "min_confidence": round(self.min_confidence, 3),
                    "consecutive_holds": self._consecutive_holds,
                    "last_trade": self._last_trade_time.isoformat() if self._last_trade_time else None,
                    "loss_penalty": self._loss_confidence_boost,
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
                f"L투표: {long_count}({','.join(s for s,_ in long_votes)}) | "
                f"S투표: {short_count}({','.join(s for s,_ in short_votes)}) | "
                f"ML: {ml_direction}({ml_confidence:.3f}) RL: {rl_direction}({rl_confidence:.3f}) | "
                f"Mom: {mom_direction}({mom_strength:.3f}) Ext: {ext_direction}({ext_confidence:.3f})"
            )

        return decision

    def _regime_direction_bias(
        self, action: str, funding_rate: float, fear_greed: float,
        mom_direction: str, mom_rsi: float,
    ) -> str | None:
        """레짐 기반 방향 필터 — 시장 상태와 반대 방향 진입 차단

        Returns: 차단 사유 문자열 (None이면 통과)

        로직:
        - 극도의 공포(< 20) + 음펀비(< -0.3bp) + 반등 시그널 → 숏 차단 (롱 유리)
        - 극도의 탐욕(> 80) + 양펀비(> 1.5bp) + 과매수 → 롱 차단 (숏 유리)
        - 강한 음펀비(< -1bp)에서 숏 → 무조건 차단 (숏스퀴즈 리스크)
        - 강한 양펀비(> 3bp)에서 롱 → 무조건 차단 (롱스퀴즈 리스크)
        """
        # 1. 극단 펀딩비 → 무조건 차단
        if action == "short" and funding_rate < -1.0:
            return f"레짐차단: 강한음펀비({funding_rate:.1f}bp)→숏스퀴즈위험"
        if action == "long" and funding_rate > 3.0:
            return f"레짐차단: 강한양펀비({funding_rate:.1f}bp)→롱스퀴즈위험"

        # 2. 극도의 공포 + 음펀비 + 반등 조합 → 숏 차단
        if (action == "short" and fear_greed < 20 and funding_rate < -0.3
                and (mom_direction == "long" or mom_rsi < 30)):
            return (f"레짐차단: 극공포({fear_greed:.0f})+음펀비({funding_rate:.1f}bp)"
                    f"+반등시그널→숏차단")

        # 3. 극도의 탐욕 + 양펀비 + 과매수 → 롱 차단
        if (action == "long" and fear_greed > 80 and funding_rate > 1.5
                and (mom_direction == "short" or mom_rsi > 70)):
            return (f"레짐차단: 극탐욕({fear_greed:.0f})+양펀비({funding_rate:.1f}bp)"
                    f"+과매수→롱차단")

        return None

    def get_diagnostics(self) -> dict:
        """자기진단 상태 반환"""
        return {
            "consecutive_holds": self._consecutive_holds,
            "current_min_confidence": round(self.min_confidence, 3),
            "base_min_confidence": self.base_min_confidence,
            "last_trade_time": self._last_trade_time.isoformat() if self._last_trade_time else None,
            "is_stuck": self._consecutive_holds > 100,
            "recent_actions": [d.action for d in self.recent_decisions[-10:]],
            "loss_penalty": self._loss_confidence_boost,
            "min_confirming": self.min_confirming,
        }
