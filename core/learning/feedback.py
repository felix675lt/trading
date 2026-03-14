"""자기 피드백 학습 모듈 - 거래 결과를 분석하여 스스로 개선"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger


class TradeFeedbackAnalyzer:
    """
    자기 거래 결과를 분석하여 패턴을 학습하고 전략을 자동 개선

    분석 항목:
    1. 실패 패턴 감지 (반복적으로 지는 조건)
    2. 성공 패턴 강화 (잘 먹히는 조건 발견)
    3. 시간대별 성과 분석
    4. 시장 레짐별 성과 분석
    5. 연패/연승 감지 및 대응
    6. 이상 시장 경고
    """

    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_path = self.storage_dir / "feedback_history.json"
        self.feedback: dict = self._load()

        # 실시간 추적
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.recent_trades: list[dict] = []

    def _load(self) -> dict:
        if self.feedback_path.exists():
            with open(self.feedback_path) as f:
                return json.load(f)
        return {
            "pattern_scores": {},        # 조건별 성과 점수
            "hourly_performance": {},    # 시간대별 승률
            "regime_performance": {},    # 시장 레짐별 승률
            "symbol_performance": {},    # 종목별 승률
            "direction_performance": {}, # 롱/숏별 승률
            "signal_accuracy": {},       # 시그널 강도별 적중률
            "lessons": [],               # 학습된 교훈 (규칙)
            "adjustments": {},           # 자동 조정된 파라미터
            "total_analyzed": 0,
        }

    def _save(self):
        with open(self.feedback_path, "w") as f:
            json.dump(self.feedback, f, indent=2, default=str)

    def record_trade(self, trade: dict, market_context: dict):
        """
        거래 결과 기록 및 즉시 분석

        trade: {symbol, side, entry_price, exit_price, pnl, reason, ...}
        market_context: {regime, signal_strength, confidence, hour, volatility, ...}
        """
        enriched = {
            **trade,
            "regime": market_context.get("regime", "unknown"),
            "signal_strength": market_context.get("signal", 0),
            "confidence": market_context.get("confidence", 0),
            "hour": datetime.utcnow().hour,
            "volatility": market_context.get("volatility", 0),
            "analyzed_at": str(datetime.utcnow()),
        }
        self.recent_trades.append(enriched)
        if len(self.recent_trades) > 500:
            self.recent_trades = self.recent_trades[-500:]

        is_win = trade.get("pnl", 0) > 0

        # 연승/연패 추적
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # 각 차원별 성과 기록
        self._update_hourly(enriched)
        self._update_regime(enriched)
        self._update_symbol(enriched)
        self._update_direction(enriched)
        self._update_signal_accuracy(enriched)

        # 패턴 분석 및 교훈 도출
        self._detect_patterns()

        self.feedback["total_analyzed"] += 1
        self._save()

    def _update_hourly(self, trade: dict):
        """시간대별 성과 기록"""
        hour = str(trade["hour"])
        if hour not in self.feedback["hourly_performance"]:
            self.feedback["hourly_performance"][hour] = {"wins": 0, "losses": 0, "total_pnl": 0}
        h = self.feedback["hourly_performance"][hour]
        if trade["pnl"] > 0:
            h["wins"] += 1
        else:
            h["losses"] += 1
        h["total_pnl"] += trade["pnl"]

    def _update_regime(self, trade: dict):
        """시장 레짐별 성과 기록"""
        regime = trade["regime"]
        if regime not in self.feedback["regime_performance"]:
            self.feedback["regime_performance"][regime] = {"wins": 0, "losses": 0, "total_pnl": 0}
        r = self.feedback["regime_performance"][regime]
        if trade["pnl"] > 0:
            r["wins"] += 1
        else:
            r["losses"] += 1
        r["total_pnl"] += trade["pnl"]

    def _update_symbol(self, trade: dict):
        """종목별 성과 기록"""
        symbol = trade.get("symbol", "unknown")
        if symbol not in self.feedback["symbol_performance"]:
            self.feedback["symbol_performance"][symbol] = {"wins": 0, "losses": 0, "total_pnl": 0}
        s = self.feedback["symbol_performance"][symbol]
        if trade["pnl"] > 0:
            s["wins"] += 1
        else:
            s["losses"] += 1
        s["total_pnl"] += trade["pnl"]

    def _update_direction(self, trade: dict):
        """롱/숏별 성과 기록"""
        side = trade.get("side", "unknown")
        if side not in self.feedback["direction_performance"]:
            self.feedback["direction_performance"][side] = {"wins": 0, "losses": 0, "total_pnl": 0}
        d = self.feedback["direction_performance"][side]
        if trade["pnl"] > 0:
            d["wins"] += 1
        else:
            d["losses"] += 1
        d["total_pnl"] += trade["pnl"]

    def _update_signal_accuracy(self, trade: dict):
        """시그널 강도 구간별 적중률"""
        strength = abs(trade.get("signal_strength", 0))
        if strength < 0.2:
            bucket = "weak"
        elif strength < 0.4:
            bucket = "medium"
        else:
            bucket = "strong"

        if bucket not in self.feedback["signal_accuracy"]:
            self.feedback["signal_accuracy"][bucket] = {"wins": 0, "losses": 0}
        s = self.feedback["signal_accuracy"][bucket]
        if trade["pnl"] > 0:
            s["wins"] += 1
        else:
            s["losses"] += 1

    def _detect_patterns(self):
        """패턴 감지 및 자동 교훈 생성"""
        if len(self.recent_trades) < 10:
            return

        lessons = self.feedback["lessons"]
        adjustments = self.feedback["adjustments"]

        # 1. 연패 감지 → 포지션 축소
        if self.consecutive_losses >= 3:
            adj_key = "losing_streak_scale"
            scale = max(0.3, 1.0 - self.consecutive_losses * 0.15)
            adjustments[adj_key] = scale
            lesson = f"연패 {self.consecutive_losses}회 감지 → 포지션 크기 {scale:.0%}로 축소"
            if not lessons or lessons[-1] != lesson:
                lessons.append(lesson)
                logger.warning(f"[Feedback] {lesson}")

        elif self.consecutive_losses == 0 and adjustments.get("losing_streak_scale", 1.0) < 1.0:
            adjustments["losing_streak_scale"] = min(1.0, adjustments.get("losing_streak_scale", 0.5) + 0.1)

        # 2. 특정 시간대 불리 감지
        for hour, perf in self.feedback["hourly_performance"].items():
            total = perf["wins"] + perf["losses"]
            if total >= 5:
                win_rate = perf["wins"] / total
                if win_rate < 0.3 and perf["total_pnl"] < 0:
                    adj_key = f"avoid_hour_{hour}"
                    if adj_key not in adjustments:
                        adjustments[adj_key] = True
                        lesson = f"UTC {hour}시 승률 {win_rate:.0%} → 이 시간대 거래 회피"
                        lessons.append(lesson)
                        logger.info(f"[Feedback] {lesson}")

        # 3. 특정 레짐에서 반복 손실
        for regime, perf in self.feedback["regime_performance"].items():
            total = perf["wins"] + perf["losses"]
            if total >= 5:
                win_rate = perf["wins"] / total
                if win_rate < 0.3:
                    adj_key = f"regime_scale_{regime}"
                    adjustments[adj_key] = 0.3
                    lesson = f"'{regime}' 레짐 승률 {win_rate:.0%} → 이 레짐에서 포지션 70% 축소"
                    if lesson not in lessons:
                        lessons.append(lesson)
                        logger.info(f"[Feedback] {lesson}")
                elif win_rate > 0.65:
                    adj_key = f"regime_scale_{regime}"
                    adjustments[adj_key] = 1.3
                    lesson = f"'{regime}' 레짐 승률 {win_rate:.0%} → 이 레짐에서 포지션 30% 확대"
                    if lesson not in lessons:
                        lessons.append(lesson)
                        logger.info(f"[Feedback] {lesson}")

        # 4. 롱/숏 편향 감지
        for side, perf in self.feedback["direction_performance"].items():
            total = perf["wins"] + perf["losses"]
            if total >= 10:
                win_rate = perf["wins"] / total
                if win_rate < 0.35:
                    adj_key = f"direction_scale_{side}"
                    adjustments[adj_key] = 0.5
                    lesson = f"'{side}' 방향 승률 {win_rate:.0%} → 이 방향 포지션 50% 축소"
                    if lesson not in lessons:
                        lessons.append(lesson)
                        logger.info(f"[Feedback] {lesson}")

        # 5. 약한 시그널의 적중률 분석
        weak = self.feedback["signal_accuracy"].get("weak", {"wins": 0, "losses": 0})
        weak_total = weak["wins"] + weak["losses"]
        if weak_total >= 10:
            weak_wr = weak["wins"] / weak_total
            if weak_wr < 0.4:
                adjustments["min_signal_strength"] = 0.2
                lesson = f"약한 시그널 승률 {weak_wr:.0%} → 최소 시그널 강도 0.2로 상향"
                if lesson not in lessons:
                    lessons.append(lesson)
                    logger.info(f"[Feedback] {lesson}")

        # 최근 50개 교훈만 유지
        self.feedback["lessons"] = lessons[-50:]

    def get_adjustments(self) -> dict:
        """전략 매니저에 전달할 자동 조정값 반환"""
        return self.feedback.get("adjustments", {})

    def should_trade_now(self, hour: int, regime: str, side: str, signal_strength: float) -> tuple[bool, str]:
        """피드백 기반 거래 필터"""
        adj = self.feedback.get("adjustments", {})

        # 시간대 필터
        if adj.get(f"avoid_hour_{hour}"):
            return False, f"UTC {hour}시는 승률 낮은 시간대 (피드백 학습)"

        # 최소 시그널 강도 필터
        min_strength = adj.get("min_signal_strength", 0)
        if abs(signal_strength) < min_strength:
            return False, f"시그널 강도 부족: {abs(signal_strength):.2f} < {min_strength}"

        return True, "OK"

    def get_position_scale(self, regime: str, side: str) -> float:
        """피드백 기반 포지션 크기 조정 배율"""
        adj = self.feedback.get("adjustments", {})
        scale = 1.0

        # 연패 스케일
        scale *= adj.get("losing_streak_scale", 1.0)
        # 레짐 스케일
        scale *= adj.get(f"regime_scale_{regime}", 1.0)
        # 방향 스케일
        scale *= adj.get(f"direction_scale_{side}", 1.0)

        return max(0.1, min(2.0, scale))

    def get_report(self) -> dict:
        """피드백 분석 리포트"""
        report = {
            "total_trades_analyzed": self.feedback["total_analyzed"],
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "lessons_learned": len(self.feedback["lessons"]),
            "active_adjustments": self.feedback.get("adjustments", {}),
            "recent_lessons": self.feedback["lessons"][-5:],
        }

        # 시간대별 베스트/워스트
        best_hour = worst_hour = None
        best_wr = 0
        worst_wr = 1
        for hour, perf in self.feedback["hourly_performance"].items():
            total = perf["wins"] + perf["losses"]
            if total >= 3:
                wr = perf["wins"] / total
                if wr > best_wr:
                    best_wr = wr
                    best_hour = hour
                if wr < worst_wr:
                    worst_wr = wr
                    worst_hour = hour
        report["best_hour"] = f"UTC {best_hour}시 (승률 {best_wr:.0%})" if best_hour else "데이터 부족"
        report["worst_hour"] = f"UTC {worst_hour}시 (승률 {worst_wr:.0%})" if worst_hour else "데이터 부족"

        # 레짐별 성과
        regime_summary = {}
        for regime, perf in self.feedback["regime_performance"].items():
            total = perf["wins"] + perf["losses"]
            if total > 0:
                regime_summary[regime] = {
                    "win_rate": f"{perf['wins'] / total:.0%}",
                    "total_pnl": f"${perf['total_pnl']:.2f}",
                    "trades": total,
                }
        report["regime_performance"] = regime_summary

        return report


class AnomalyDetector:
    """시장 이상 감지 - 평소와 다른 움직임 경고"""

    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.price_history: list[float] = []
        self.volume_history: list[float] = []

    def update(self, price: float, volume: float) -> list[dict]:
        """가격/거래량 업데이트 및 이상 감지"""
        self.price_history.append(price)
        self.volume_history.append(volume)

        if len(self.price_history) > self.lookback * 2:
            self.price_history = self.price_history[-self.lookback * 2:]
            self.volume_history = self.volume_history[-self.lookback * 2:]

        if len(self.price_history) < self.lookback:
            return []

        alerts = []

        # 1. 가격 급변 감지
        returns = np.diff(self.price_history[-self.lookback:]) / np.array(self.price_history[-self.lookback:-1])
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        latest_ret = returns[-1] if len(returns) > 0 else 0

        if abs(latest_ret) > mean_ret + 3 * std_ret:
            direction = "급등" if latest_ret > 0 else "급락"
            alerts.append({
                "type": "price_spike",
                "severity": "high",
                "message": f"가격 {direction}: {latest_ret:.2%} (평소 {std_ret:.2%}의 {abs(latest_ret)/std_ret:.1f}배)",
                "recommendation": "신규 진입 보류, 기존 포지션 SL 타이트하게",
            })

        # 2. 거래량 이상 감지
        vol_arr = np.array(self.volume_history[-self.lookback:])
        vol_mean = np.mean(vol_arr)
        vol_std = np.std(vol_arr)
        latest_vol = vol_arr[-1]

        if latest_vol > vol_mean + 3 * vol_std:
            alerts.append({
                "type": "volume_spike",
                "severity": "medium",
                "message": f"거래량 급증: 평소의 {latest_vol/vol_mean:.1f}배",
                "recommendation": "큰 움직임 예상, 방향 확인 후 대응",
            })

        # 3. 변동성 급변 감지
        recent_vol = np.std(returns[-20:]) if len(returns) >= 20 else 0
        historical_vol = np.std(returns) if len(returns) > 0 else 1
        vol_ratio = recent_vol / (historical_vol + 1e-8)

        if vol_ratio > 2.5:
            alerts.append({
                "type": "volatility_spike",
                "severity": "high",
                "message": f"변동성 급등: 평소의 {vol_ratio:.1f}배",
                "recommendation": "포지션 크기 대폭 축소 또는 거래 중단",
            })
        elif vol_ratio < 0.3:
            alerts.append({
                "type": "volatility_compression",
                "severity": "low",
                "message": f"변동성 극도로 축소: 평소의 {vol_ratio:.1f}배",
                "recommendation": "큰 돌파 움직임 대비, 브레이크아웃 전략 준비",
            })

        for alert in alerts:
            logger.warning(f"[Anomaly] {alert['message']}")

        return alerts
