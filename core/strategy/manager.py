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
        # [2026-04-25] PAPER 전용 min_confirming — 데이터 수집 우선
        # decide(mode="paper")일 때 self.min_confirming 대신 사용.
        # config에 paper_min_confirming 없으면 self.min_confirming와 동일 (backward compat).
        self.paper_min_confirming = config.get(
            "paper_min_confirming", self.min_confirming
        )
        # 방향 필터 (mode-aware):
        # - live_long_only=True: LIVE 실행에서만 숏 차단 (PAPER는 학습 위해 양방향 허용)
        # - long_only=True     : 모든 모드에서 숏 차단 (강력 차단 — backward compat)
        # StrategyManager는 decide()에서 long_only만 반영한다.
        # live_long_only는 main.py의 LIVE 주문 게이트에서 처리.
        self.long_only = config.get("long_only", False)
        self.live_long_only = config.get("live_long_only", False)

        # === Smart Regime-Conditional SHORT Filter (2026-04-24) ===
        # 특정 레짐에서만 숏 차단 — 실측 WR < 30%인 레짐 자동 거부.
        # PAPER 학습 데이터에서 통계적으로 유의하게 실패한 레짐(예: strong_uptrend×SHORT n=23 WR 0%)만
        # 차단하고, 나머지는 학습 기회를 유지한다.
        self.smart_short_blocked_regimes = set(
            config.get("smart_short_blocked_regimes", []) or []
        )

        # === BOCPD 변환점 게이트 (2026-04-25) ===
        # Adams & MacKay (2007) Bayesian Online Changepoint Detection.
        # 레짐 전환 직후 잘못된 방향 진입 방지 — changepoint_prob >= block_threshold면 차단.
        # 외부에서 set_bocpd_detector(detector)로 주입 — 미설정 시 비활성.
        self.bocpd_detector = None
        bocpd_cfg = config.get("bocpd", {}) or {}
        self.bocpd_enabled = bool(bocpd_cfg.get("enabled", True))
        self.bocpd_block_threshold = float(bocpd_cfg.get("block_threshold", 0.6))
        if self.bocpd_enabled:
            try:
                from core.strategy.bocpd import BOCPDRegimeDetector
                self.bocpd_detector = BOCPDRegimeDetector(
                    hazard_rate=float(bocpd_cfg.get("hazard_rate", 1 / 250)),
                    volatility_window=int(bocpd_cfg.get("volatility_window", 20)),
                    trend_window=int(bocpd_cfg.get("trend_window", 50)),
                    changepoint_threshold=float(bocpd_cfg.get("changepoint_threshold", 0.3)),
                    block_threshold=self.bocpd_block_threshold,
                )
                logger.info(
                    f"[Strategy] BOCPD 변환점 게이트 활성 — block≥{self.bocpd_block_threshold:.2f}"
                )
            except Exception as e:
                logger.warning(f"[Strategy] BOCPD 초기화 실패({e}) — 게이트 비활성화")
                self.bocpd_detector = None

        # === BREAKOUT vote source config (2026-04-23 옵션 C) ===
        bv = config.get("breakout_vote", {}) or {}
        self.breakout_enabled = bool(bv.get("enabled", False))
        self.breakout_lookback = int(bv.get("lookback_bars", 20))
        self.breakout_vol_z_min = float(bv.get("volume_z_min", 1.5))
        self.breakout_atr_ratio_min = float(bv.get("atr_ratio_min", 0.005))
        self.breakout_regimes_long = set(bv.get("allowed_regimes_long", []) or [])
        self.breakout_regimes_short = set(bv.get("allowed_regimes_short", []) or [])
        self.breakout_vote_weight = float(bv.get("vote_weight", 0.55))
        if self.breakout_enabled:
            logger.info(
                f"[Strategy] BREAKOUT vote 활성 — N={self.breakout_lookback} "
                f"volZ≥{self.breakout_vol_z_min} ATR%≥{self.breakout_atr_ratio_min} "
                f"L레짐={sorted(self.breakout_regimes_long)} S레짐={sorted(self.breakout_regimes_short)}"
            )

        # === LIVE 공격 롱 모드 (2026-04-18) ===
        # 매크로/지정학 하방 필터 완화 + 롱 확신도 부스트 + 낮은 임계값
        self.live_aggressive_long = config.get("live_aggressive_long", False)
        self.live_min_confidence = config.get("live_min_confidence", self.base_min_confidence)
        self.live_long_conf_boost = config.get("live_long_conf_boost", 1.0)
        self.live_disable_macro_block = config.get("live_disable_macro_block", False)

        # === ⏰ Time Blacklist (Patch B+F, 2026-04-26) — LIVE 전용 ===
        # 7년치 백테스트(BTC/ETH/SOL/DOGE 5m, 2019-09 ~ 2026-04, ~2.5M 캔들)에서
        # pooled forward 1h EV < 0 인 시간대를 데이터 기반으로 자동 추출.
        # → 추천 차단 시간 (UTC): [1, 2, 8, 13, 16]
        #   - 13h UTC: ETH/SOL/DOGE 모두 최악 (변동성 hangover, NY pre-market 휩쏘)
        #   - 01-02h UTC: BTC/ETH 모두 음수 (Asia 야간 유동성 함정)
        #   - 08h UTC: London open 직전 휩쏘 (DOGE 최악)
        #   - 16h UTC: NY 점심 휩쏘 (SOL 음수)
        # PAPER는 차단 없음 (학습 데이터 수집 우선 — 손실도 정보).
        # config.live_blacklist_hours_utc 로 오버라이드 가능 (list[int]).
        self.live_blacklist_hours_utc = set(
            config.get("live_blacklist_hours_utc", [1, 2, 8, 13, 16]) or []
        )
        if self.live_blacklist_hours_utc:
            logger.info(
                f"[Strategy] LIVE Time Blacklist 활성 — "
                f"UTC {sorted(self.live_blacklist_hours_utc)}h 진입 차단 (PAPER는 무관)"
            )

        if self.long_only:
            logger.warning("[Strategy] LONG_ONLY (global) 모드 활성화 — 전 모드 숏 진입 차단")
        elif self.live_long_only:
            logger.warning("[Strategy] LIVE_LONG_ONLY 모드 활성화 — LIVE만 숏 차단, PAPER 양방향")
        if self.smart_short_blocked_regimes:
            logger.warning(
                f"[Strategy] SMART_SHORT_FILTER 활성 — {sorted(self.smart_short_blocked_regimes)} "
                f"레짐에서 LIVE 숏 차단 (실측 WR < 30%) — PAPER는 학습 위해 양방향 허용"
            )
        if self.live_aggressive_long:
            logger.warning(
                f"[Strategy] LIVE 공격 롱 모드 ON — min_conf={self.live_min_confidence:.2f} "
                f"boost=×{self.live_long_conf_boost:.2f} 매크로차단해제={self.live_disable_macro_block}"
            )
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

        # === [Phase J, 2026-04-25] 차단 사유 카운터 — 분석 마비 실증 진단 ===
        # decide()가 hold로 결정될 때마다 카테고리별 카운트를 증가.
        # log_block_stats() 호출 시 분포를 출력하고 카운터 리셋.
        # 카테고리:
        #   macro_block: 고임팩트 매크로 이벤트(베어/불) 차단
        #   regime_bias_block: 레짐 방향 바이어스 차단 (펀비/RSI/모멘텀)
        #   long_only_block: LONG_ONLY 정책 숏 차단
        #   smart_short_block: 레짐별 자동 숏 차단
        #   blacklist_block: 피드백 블랙리스트 차단
        #   bocpd_block: BOCPD 변환점 차단
        #   low_confidence: 최소 확신도 미달
        #   extreme_vol_block: 극심한 변동성 차단
        #   insufficient_votes: 투표 미달 (default fallthrough)
        #   close_signal: RL이 청산 요청 (참고용 — 실제 hold가 아님)
        self._hold_reason_counts: dict[str, int] = {}
        self._hold_total = 0
        self._decide_total = 0  # decide() 총 호출 수 (분모)
        self._block_log_last: datetime | None = None

    # ------------------------------------------------------------------
    # [Phase J, 2026-04-25] 차단 사유 카운터
    # ------------------------------------------------------------------
    def _record_hold(self, category: str) -> None:
        """decide()의 각 hold 지점에서 카테고리를 누적.

        카운터는 log_block_stats()가 출력 후 리셋한다.
        """
        try:
            self._hold_reason_counts[category] = self._hold_reason_counts.get(category, 0) + 1
            self._hold_total += 1
        except Exception:
            pass

    def log_block_stats(self, force: bool = False, min_interval_sec: int = 3600) -> dict | None:
        """1시간마다 차단 사유 분포 로깅 + 카운터 리셋.

        Args:
            force: True면 인터벌 무시
            min_interval_sec: 최소 호출 간격 (기본 1h)

        Returns:
            출력된 통계 dict (또는 인터벌 미달 시 None)
        """
        now = datetime.utcnow()
        if not force and self._block_log_last is not None:
            elapsed = (now - self._block_log_last).total_seconds()
            if elapsed < min_interval_sec:
                return None
        if self._decide_total == 0:
            self._block_log_last = now
            return None

        total = max(self._decide_total, 1)
        sorted_counts = sorted(
            self._hold_reason_counts.items(), key=lambda x: -x[1]
        )
        # Top 사유 + 비율 — 분석 마비 진단 핵심 지표
        breakdown = ", ".join(
            f"{cat}={cnt}({cnt/total*100:.1f}%)" for cat, cnt in sorted_counts
        )
        hold_rate = self._hold_total / total * 100
        logger.info(
            f"[BlockStats] decide={self._decide_total} hold={self._hold_total}({hold_rate:.1f}%) | {breakdown}"
        )
        snapshot = {
            "decide_total": self._decide_total,
            "hold_total": self._hold_total,
            "hold_rate_pct": round(hold_rate, 2),
            "by_category": dict(self._hold_reason_counts),
        }
        # 리셋
        self._hold_reason_counts.clear()
        self._hold_total = 0
        self._decide_total = 0
        self._block_log_last = now
        return snapshot

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
        """순수 스캘핑 모드 — 항상 scalp 반환"""
        return "scalp"

    def _detect_breakout(self, ohlcv_df, regime: str) -> dict:
        """Donchian N-bar 돌파 감지 (옵션 C, 2026-04-23)

        Args:
            ohlcv_df: DataFrame with columns [open, high, low, close, volume] (최신이 마지막)
            regime: 현재 adaptive regime
        Returns:
            {"direction": "long"|"short"|"none", "strength": float 0-1, "reason": str}
        """
        if not self.breakout_enabled or ohlcv_df is None:
            return {"direction": "none", "strength": 0.0, "reason": "disabled"}
        try:
            import numpy as _np
            N = self.breakout_lookback
            if len(ohlcv_df) < N + 14:  # ATR 14 + 롤링 윈도
                return {"direction": "none", "strength": 0.0, "reason": "표본부족"}

            close = float(ohlcv_df["close"].iloc[-1])
            # rolling high/low는 마지막 바 제외 (exclusive)
            prev_window = ohlcv_df.iloc[-(N + 1):-1]
            prev_high = float(prev_window["high"].max())
            prev_low = float(prev_window["low"].min())

            # volume z-score (마지막 바 vs 직전 N봉)
            vol_window = ohlcv_df["volume"].iloc[-(N + 1):-1]
            vol_mean = float(vol_window.mean())
            vol_std = float(vol_window.std()) + 1e-9
            last_vol = float(ohlcv_df["volume"].iloc[-1])
            vol_z = (last_vol - vol_mean) / vol_std

            # ATR(14) 간이 계산
            high = ohlcv_df["high"].iloc[-15:].values
            low = ohlcv_df["low"].iloc[-15:].values
            prev_close = ohlcv_df["close"].iloc[-16:-1].values
            tr = _np.maximum.reduce([
                high[-14:] - low[-14:],
                _np.abs(high[-14:] - prev_close[-14:]),
                _np.abs(low[-14:] - prev_close[-14:]),
            ])
            atr = float(_np.mean(tr))
            atr_ratio = atr / max(close, 1e-9)

            # 필터
            if atr_ratio < self.breakout_atr_ratio_min:
                return {"direction": "none", "strength": 0.0,
                        "reason": f"죽은시장 ATR%={atr_ratio:.4f}"}
            if vol_z < self.breakout_vol_z_min:
                return {"direction": "none", "strength": 0.0,
                        "reason": f"거래량부족 z={vol_z:.2f}"}

            # 방향 판정
            if close > prev_high and regime in self.breakout_regimes_long:
                # 돌파 강도 = max(1.0, (close-prev_high)/ATR) / 2 + volZ 가산
                excess = (close - prev_high) / max(atr, 1e-9)
                strength = min(self.breakout_vote_weight + min(excess, 1.0) * 0.15, 1.0)
                return {"direction": "long", "strength": strength,
                        "reason": f"상승돌파 close={close:.4f}>PH={prev_high:.4f} volZ={vol_z:.2f} ATR%={atr_ratio:.3f}"}
            if close < prev_low and regime in self.breakout_regimes_short:
                excess = (prev_low - close) / max(atr, 1e-9)
                strength = min(self.breakout_vote_weight + min(excess, 1.0) * 0.15, 1.0)
                return {"direction": "short", "strength": strength,
                        "reason": f"하락돌파 close={close:.4f}<PL={prev_low:.4f} volZ={vol_z:.2f} ATR%={atr_ratio:.3f}"}
            return {"direction": "none", "strength": 0.0,
                    "reason": f"돌파없음 close={close:.4f} PH={prev_high:.4f} PL={prev_low:.4f} regime={regime}"}
        except Exception as e:
            logger.warning(f"[BREAKOUT] 감지 실패: {e}")
            return {"direction": "none", "strength": 0.0, "reason": f"error:{e}"}

    def _count_signal_votes(
        self,
        ml_direction: str, ml_confidence: float, ml_signal_val: float,
        rl_direction: str, rl_confidence: float,
        mom_direction: str, mom_strength: float, mom_rsi: float,
        ext_direction: str, ext_confidence: float,
        regime: str = "normal",
    ) -> dict:
        """각 독립 소스의 방향 투표 수집

        레짐별 IC 기반 가중치 자동화 (2026-04-24 C):
        self.signal_weight_optimizer (optional) — (regime, source) → multiplier.
        미설정 시 1.0 고정 (기존 동작 유지).

        Returns: {"long": [(source, weight), ...], "short": [(source, weight), ...]}
        """
        votes = {"long": [], "short": []}

        # 레짐별 가중치 multiplier 제공자 (optional)
        opt = getattr(self, "signal_weight_optimizer", None)
        def wmult(source: str) -> float:
            if opt is None:
                return 1.0
            try:
                return float(opt.get(regime, source))
            except Exception:
                return 1.0

        # === [2026-04-25] 데이터 수집 극대화 — 투표 임계 일괄 완화 ===
        # 직전 1주일간 LONG 0건, 4/22 이후 SHORT 0건. 모든 vote source가
        # 임계 직전에서 컷오프됨. 임계를 한 단계 낮춰 학습 데이터 흐름 복구.
        # 위험: noisy entries↑. 완화: confidence 평균 가중 + min_confidence 게이트.

        # 1. ML 모델 투표 (confidence > 0.2 — was 0.3)
        if ml_direction == "long" and ml_confidence > 0.2:
            votes["long"].append(("ML", ml_confidence * wmult("ml")))
        elif ml_direction == "short" and ml_confidence > 0.2:
            votes["short"].append(("ML", ml_confidence * wmult("ml")))
        # ML signal 값이 방향과 일치하면 추가 투표 (같은 소스이므로 0.5 가중)
        elif abs(ml_signal_val) > self.signal_threshold:
            if ml_signal_val > self.signal_threshold:
                votes["long"].append(("ML_val", min(abs(ml_signal_val) * 2, 0.6) * wmult("ml")))
            else:
                votes["short"].append(("ML_val", min(abs(ml_signal_val) * 2, 0.6) * wmult("ml")))

        # 2. RL 에이전트 투표 (confidence > 0.3 — was 0.4)
        if rl_direction == "long" and rl_confidence > 0.3:
            votes["long"].append(("RL", rl_confidence * wmult("rl")))
        elif rl_direction == "short" and rl_confidence > 0.3:
            votes["short"].append(("RL", rl_confidence * wmult("rl")))

        # 3. 모멘텀 투표 (strength > 0.15 — was 0.2)
        if mom_direction == "long" and abs(mom_strength) > 0.15:
            votes["long"].append(("MOM", abs(mom_strength) * wmult("mom")))
        elif mom_direction == "short" and abs(mom_strength) > 0.15:
            votes["short"].append(("MOM", abs(mom_strength) * wmult("mom")))
        # RSI 극값 (30 이하 = 과매도 반전, 70 이상 = 과매수 반전 — was 25/75)
        if mom_rsi < 30:
            rsi_weight = (30 - mom_rsi) / 30 * 0.5
            votes["long"].append(("RSI_extreme", rsi_weight * wmult("rsi_extreme")))
        elif mom_rsi > 70:
            rsi_weight = (mom_rsi - 70) / 30 * 0.5
            votes["short"].append(("RSI_extreme", rsi_weight * wmult("rsi_extreme")))

        # 4. 외부 요인 투표 (confidence > 0.10 — was 0.15)
        if ext_direction == "bullish" and ext_confidence > 0.10:
            votes["long"].append(("EXT", ext_confidence * wmult("ext")))
        elif ext_direction == "bearish" and ext_confidence > 0.10:
            votes["short"].append(("EXT", ext_confidence * wmult("ext")))

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
        mode: str = "paper",
        variant_override: dict | None = None,
        ohlcv_df=None,
    ) -> TradeDecision:
        """최종 트레이딩 결정 (v5 - A/B variant 지원)

        mode: "paper" 또는 "live" — LIVE 공격 롱 모드 적용 여부 결정
        variant_override: A/B 테스트용 정책 오버라이드 (2026-04-21)
            - "disable_macro_block": bool — True면 2.7/2.8 매크로 차단 스킵
            - "min_confidence_override": float | None — 최소 신뢰도 대체
            ※ variant_override는 순수 함수 오버라이드 (self 상태 변경 금지)
        ohlcv_df: OHLCV 최근 N봉 (BREAKOUT vote source용, 2026-04-23 옵션 C)
        """
        vo = variant_override or {}
        # variant 오버라이드: 매크로 차단 정책
        # (self.live_disable_macro_block은 LIVE 공격 모드용 별개 플래그)
        variant_disable_macro = bool(vo.get("disable_macro_block", False))
        action_map = {0: "hold", 1: "long", 2: "short", 3: "close"}
        rl_direction = action_map.get(rl_action, "hold")

        # [Phase J] decide() 호출 카운트 (block_stats 분모)
        self._decide_total += 1
        # 카테고리 기록 시작점 — 끝에서 fallthrough hold를 잡기 위해
        _hold_total_snap = self._hold_total

        # LIVE 공격 롱 모드 활성 여부
        live_aggro = (mode == "live") and self.live_aggressive_long

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
            regime=market_regime,
        )

        # === BOCPD 변환점 게이트 (stage-1, 2026-04-25) ===
        # ohlcv_df의 마지막 close로 BOCPD 업데이트 → 변환점 직후 차단 권고면 hold 강제
        bocpd_info = {"enabled": False}
        if self.bocpd_detector is not None and ohlcv_df is not None and len(ohlcv_df) > 0:
            try:
                last_close = float(ohlcv_df["close"].iloc[-1])
                state = self.bocpd_detector.update(last_close)
                bocpd_info = self.bocpd_detector.get_regime_info()
                bocpd_info["enabled"] = True
            except Exception as e:
                logger.debug(f"[BOCPD] 업데이트 실패: {e}")

        # 5번째 vote source: BREAKOUT (2026-04-23 옵션 C)
        # Donchian N봉 돌파 + 거래량 확인 + 레짐 gate
        breakout_info = {"direction": "none", "strength": 0.0, "reason": "skip"}
        if self.breakout_enabled and ohlcv_df is not None:
            breakout_info = self._detect_breakout(ohlcv_df, market_regime)
            bd = breakout_info["direction"]
            bs = breakout_info["strength"]
            # 레짐별 가중치 적용 (C)
            _opt = getattr(self, "signal_weight_optimizer", None)
            bmult = 1.0
            if _opt is not None:
                try:
                    bmult = float(_opt.get(market_regime, "breakout"))
                except Exception:
                    bmult = 1.0
            if bd == "long":
                votes["long"].append(("BREAKOUT", bs * bmult))
                logger.info(f"[BREAKOUT] LONG vote 추가: {breakout_info['reason']}")
            elif bd == "short":
                votes["short"].append(("BREAKOUT", bs * bmult))
                logger.info(f"[BREAKOUT] SHORT vote 추가: {breakout_info['reason']}")

        long_votes = votes["long"]
        short_votes = votes["short"]
        long_count = len(long_votes)
        short_count = len(short_votes)

        # === [2026-04-25] mode-aware min_confirming ===
        # PAPER: 데이터 수집 우선 → 단일 소스도 진입 허용 (paper_min_confirming, 보통 1)
        # LIVE : 안전 우선 → 글로벌 min_confirming 유지 (보통 2)
        effective_min_confirming = (
            self.paper_min_confirming if mode == "paper" else self.min_confirming
        )

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
        # 더 많은 투표를 받은 방향으로, 최소 effective_min_confirming 개 합의 필요
        elif long_count >= effective_min_confirming and long_count > short_count:
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

        elif short_count >= effective_min_confirming and short_count > long_count:
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

        # 2.45. LIVE 공격 롱 부스트 — LIVE 모드에서 롱 확신도만 상향
        if live_aggro and final_action == "long" and self.live_long_conf_boost > 1.0:
            before = confidence
            confidence *= self.live_long_conf_boost
            reason += f" +LIVE공격롱(×{self.live_long_conf_boost:.2f})"
            logger.info(
                f"[LIVE-AGGRO] 롱 확신도 부스트: {before:.3f} → {confidence:.3f} "
                f"(×{self.live_long_conf_boost:.2f})"
            )

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

        # 2.7. 고임팩트 이벤트 오버라이드 [재활성화]
        # 강한 외부 시그널 + 고임팩트 뉴스 반대 방향 진입 → 차단
        # LIVE 공격 롱 모드 + macro_disable: 롱 차단만 해제 (숏 차단은 유지)
        # variant_override.disable_macro_block=True면 모든 방향 매크로 차단 스킵
        macro_disabled_long = live_aggro and self.live_disable_macro_block
        macro_fully_disabled = variant_disable_macro  # A/B variant용
        if has_high_impact and abs(ext_score) > 0.4 and not macro_fully_disabled:
            if ext_direction == "bearish" and final_action == "long":
                if macro_disabled_long:
                    logger.info(
                        f"[LIVE-AGGRO] 고임팩트 베어리시 이벤트 롱차단 무시 "
                        f"(ext_score={ext_score:.2f}) — 매크로 필터 해제"
                    )
                else:
                    logger.warning(f"[고임팩트차단] 롱 차단 (ext_score={ext_score:.2f}, 베어리시 이벤트)")
                    final_action = "hold"
                    reason = f"고임팩트 베어리시 이벤트 → 롱 차단 (ext={ext_score:.2f})"
                    confidence = 0.0
                    self._record_hold("macro_block")
            elif ext_direction == "bullish" and final_action == "short":
                logger.warning(f"[고임팩트차단] 숏 차단 (ext_score={ext_score:.2f}, 불리시 이벤트)")
                final_action = "hold"
                reason = f"고임팩트 불리시 이벤트 → 숏 차단 (ext={ext_score:.2f})"
                confidence = 0.0
                self._record_hold("macro_block")
        elif has_high_impact and abs(ext_score) > 0.4 and macro_fully_disabled:
            logger.debug(
                f"[A/B:MACRO_OFF] 고임팩트 이벤트 차단 무시 (ext_score={ext_score:.2f}) — variant 정책"
            )

        # 2.8. 레짐 방향 바이어스 [재활성화 v3] — funding_rate 극값 + 모멘텀/RSI만 사용
        # 공포탐욕 지수 완전 제거됨 (후행 지표, 예측력 없음)
        # LIVE 공격 롱 + macro_disable: 롱 차단만 무시 (숏 차단은 유지 — 펀비 스퀴즈 리스크 보호)
        # variant_override.disable_macro_block=True면 레짐 차단도 스킵
        if final_action in ["long", "short"]:
            direction_block = self._regime_direction_bias(
                final_action, funding_rate, mom_direction, mom_rsi,
            )
            if direction_block:
                if macro_fully_disabled:
                    logger.debug(f"[A/B:MACRO_OFF] 레짐 차단 무시: {direction_block}")
                elif macro_disabled_long and final_action == "long":
                    logger.info(
                        f"[LIVE-AGGRO] 레짐 롱차단 무시: {direction_block}"
                    )
                else:
                    logger.warning(f"[레짐차단] 진입 차단: {direction_block}")
                    final_action = "hold"
                    reason = direction_block
                    confidence = 0.0
                    self._record_hold("regime_bias_block")

        # 2.85. 방향 필터 (사용자 지시) — 숏 진입 차단
        # 주: live_long_only는 여기서 차단하지 않는다 (PAPER는 양방향 학습 필요).
        #     LIVE 전용 차단은 main.py의 주문 실행 게이트에서 수행.
        # (1) LONG_ONLY 전역 차단 (backward compat)
        if self.long_only and final_action == "short":
            logger.info(f"[LONG_ONLY] 숏 차단 → hold (원래 사유: {reason})")
            final_action = "hold"
            reason = f"LONG_ONLY 모드 — 숏 차단 (원신호: {reason})"
            confidence = 0.0
            self._record_hold("long_only_block")
        # (2) 스마트 레짐-조건부 숏필터 — 특정 레짐에서만 숏 거부 (2026-04-24)
        # [Patch B+, 2026-04-26] PAPER 학습 데이터 수집 우선 → LIVE에서만 차단.
        # PAPER에서는 이런 "통계상 나쁜 레짐"의 SHORT도 학습해야 향후 모델이 회피 가능.
        elif (mode == "live"
              and final_action == "short"
              and self.smart_short_blocked_regimes
              and market_regime in self.smart_short_blocked_regimes):
            logger.warning(
                f"[SMART_SHORT] LIVE 레짐={market_regime} 숏 자동차단 (실측 WR<30% 통계 거부) "
                f"→ hold (원 사유: {reason})"
            )
            final_action = "hold"
            reason = f"SMART_SHORT 차단: {market_regime} 레짐 숏 금지 LIVE (원신호: {reason})"
            confidence = 0.0
            self._record_hold("smart_short_block")

        # 2.9. 피드백 블랙리스트 필터 [재활성화] — 반복 실패 패턴 진입 차단
        if final_action in ["long", "short"] and feedback_blacklist:
            combo_key = "+".join(sorted(set(s.split("_")[0] for s in confirming)))
            if combo_key in feedback_blacklist:
                logger.warning(
                    f"[블랙리스트] 진입 차단: {combo_key} → {final_action} "
                    f"(피드백 학습 기반 — 승률 <25%)"
                )
                final_action = "hold"
                reason = f"블랙리스트 차단: {combo_key} (반복 실패 패턴)"
                confidence = 0.0
                self._record_hold("blacklist_block")

        # 2.93. ⏰ Time Blacklist — LIVE 전용 (Patch B, 2026-04-26)
        # 7년치 통계상 EV 음수 시간대 차단. PAPER는 학습 위해 통과.
        if (
            mode == "live"
            and final_action in ("long", "short")
            and self.live_blacklist_hours_utc
        ):
            try:
                cur_hour = datetime.utcnow().hour
                if cur_hour in self.live_blacklist_hours_utc:
                    logger.warning(
                        f"[TimeBlacklist] LIVE 차단: UTC {cur_hour}h 는 통계상 EV<0 시간대 "
                        f"→ {final_action} 차단 (원 사유: {reason})"
                    )
                    final_action = "hold"
                    reason = f"TimeBlacklist 차단: UTC {cur_hour}h (LIVE only)"
                    confidence = 0.0
                    self._record_hold("time_blacklist_block")
            except Exception as e:
                logger.debug(f"[TimeBlacklist] 시간 체크 실패: {e}")

        # 2.95. BOCPD 변환점 차단 (stage-1 final gate, 2026-04-25)
        # 레짐 전환점 직후에는 long/short 모두 차단 — 잘못된 방향 진입 방지.
        # variant_disable_macro 와 무관 — 통계적 가드 (Adams & MacKay 2007).
        if final_action in ["long", "short"] and bocpd_info.get("enabled"):
            cp_prob = float(bocpd_info.get("changepoint_prob", 0.0))
            if cp_prob >= self.bocpd_block_threshold:
                logger.warning(
                    f"[BOCPD] 변환점 차단: cp_prob={cp_prob:.2f}≥{self.bocpd_block_threshold:.2f} "
                    f"(regime={bocpd_info.get('regime')}, vol={bocpd_info.get('volatility_regime')}) "
                    f"→ {final_action} 차단"
                )
                final_action = "hold"
                reason = (
                    f"BOCPD 변환점 차단(cp={cp_prob:.2f}, "
                    f"regime={bocpd_info.get('regime')})"
                )
                confidence = 0.0
                self._record_hold("bocpd_block")
            else:
                # 변환점 근처면 포지션 사이즈 축소 (size에 곱해질 multiplier 저장)
                try:
                    size_mult = self.bocpd_detector.get_position_size_multiplier()
                    if size_mult < 1.0:
                        confidence *= max(size_mult, 0.3)
                        reason += f" *bocpd_size×{size_mult:.2f}"
                except Exception:
                    pass

        # 3. 최소 확신도 필터 (강화)
        # LIVE 공격 롱 모드에서 롱은 live_min_confidence 사용 (기본보다 낮음)
        effective_min_conf = self.min_confidence
        if live_aggro and final_action == "long":
            effective_min_conf = min(self.min_confidence, self.live_min_confidence)
        if confidence < effective_min_conf and final_action not in ("close", "hold"):
            max_long = max([v[1] for v in long_votes], default=0)
            max_short = max([v[1] for v in short_votes], default=0)
            reason = (
                f"확신도 부족 ({confidence:.3f} < {effective_min_conf:.3f}) "
                f"[L:{long_count}표(max={max_long:.2f}) S:{short_count}표(max={max_short:.2f}) "
                f"ML:{ml_direction}/{ml_confidence:.2f} RL:{rl_direction}/{rl_confidence:.2f}]"
            )
            final_action = "hold"
            self._record_hold("low_confidence")

        # 4. 시장 레짐 필터
        if market_regime == "extreme_volatility" and final_action in ["long", "short"]:
            confidence *= 0.7
            if confidence < effective_min_conf:
                final_action = "hold"
                reason = "극심한 변동성 - 진입 보류"
                self._record_hold("extreme_vol_block")

        # 5. 포지션 크기 결정
        confidence = min(confidence, 1.0)
        size = confidence if final_action in ["long", "short"] else 0.0

        # 5.5 트레이드 타입 분류 (scalp vs swing)
        trade_type = "scalp"
        if final_action in ("long", "short"):
            trade_type = self._classify_trade_type(confirming)

        # [Phase J] 카테고리 미기록 hold (투표 미달 fallthrough) 보충
        if final_action == "hold" and self._hold_total == _hold_total_snap:
            # 어떤 카테고리도 기록되지 않은 hold = 투표 부족 또는 RL=close 핸드오프
            if rl_direction == "close":
                self._record_hold("close_signal")
            else:
                self._record_hold("insufficient_votes")

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
                "breakout": {
                    "direction": breakout_info["direction"],
                    "strength": round(breakout_info["strength"], 3),
                    "reason": breakout_info["reason"],
                },
                "bocpd": bocpd_info,
                "regime_bias": {
                    "funding_rate": round(funding_rate, 3),
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
        self, action: str, funding_rate: float,
        mom_direction: str, mom_rsi: float,
    ) -> str | None:
        """레짐 기반 방향 필터 — funding_rate + 모멘텀/RSI 극값만 사용
        (공포탐욕 지수는 제거됨 — 후행 지표라 예측력 없음)

        Returns: 차단 사유 문자열 (None이면 통과)

        로직:
        - 강한 음펀비(< -1bp)에서 숏 → 무조건 차단 (숏스퀴즈 리스크)
        - 강한 양펀비(> 3bp)에서 롱 → 무조건 차단 (롱스퀴즈 리스크)
        - 중간 음펀비(< -0.5bp) + 과매도(RSI<30) + 반등 모멘텀 → 숏 차단
        - 중간 양펀비(> 1.5bp) + 과매수(RSI>70) + 하락 모멘텀 → 롱 차단
        """
        # 1. 극단 펀딩비 → 무조건 차단 (스퀴즈 리스크)
        if action == "short" and funding_rate < -1.0:
            return f"레짐차단: 강한음펀비({funding_rate:.1f}bp)→숏스퀴즈위험"
        if action == "long" and funding_rate > 3.0:
            return f"레짐차단: 강한양펀비({funding_rate:.1f}bp)→롱스퀴즈위험"

        # 2. 중간 음펀비 + 과매도 반등 → 숏 차단 (롱 유리)
        if (action == "short" and funding_rate < -0.5
                and (mom_direction == "long" or mom_rsi < 30)):
            return (f"레짐차단: 음펀비({funding_rate:.1f}bp)+반등시그널"
                    f"(RSI:{mom_rsi:.0f})→숏차단")

        # 3. 중간 양펀비 + 과매수 하락 → 롱 차단 (숏 유리)
        if (action == "long" and funding_rate > 1.5
                and (mom_direction == "short" or mom_rsi > 70)):
            return (f"레짐차단: 양펀비({funding_rate:.1f}bp)+과매수시그널"
                    f"(RSI:{mom_rsi:.0f})→롱차단")

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
