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
                data = json.load(f)
            # 신규 필드 마이그레이션
            if "entry_path_performance" not in data:
                data["entry_path_performance"] = {}
            if "exit_reason_analysis" not in data:
                data["exit_reason_analysis"] = {}
            if "entry_blacklist" not in data:
                data["entry_blacklist"] = []
            if "loss_patterns" not in data:
                data["loss_patterns"] = {}
            return data
        return {
            "pattern_scores": {},        # 조건별 성과 점수
            "hourly_performance": {},    # 시간대별 승률
            "regime_performance": {},    # 시장 레짐별 승률
            "symbol_performance": {},    # 종목별 승률
            "direction_performance": {}, # 롱/숏별 승률
            "signal_accuracy": {},       # 시그널 강도별 적중률
            "external_accuracy": {},     # 외부 신호별 적중률
            "entry_path_performance": {},  # 진입 시그널 조합별 성과 (v4)
            "exit_reason_analysis": {},    # 청산 사유별 분석 (v4)
            "entry_blacklist": [],         # 진입 차단 패턴 (v4)
            "loss_patterns": {},           # 손실 패턴 분석 (v4)
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
        market_context: {regime, signal_strength, confidence, hour, volatility,
                         entry_path, exit_reason, confirming_sources, atr_at_entry, ...}
        """
        enriched = {
            **trade,
            "regime": market_context.get("regime", "unknown"),
            "signal_strength": market_context.get("signal", 0),
            "confidence": market_context.get("confidence", 0),
            "hour": datetime.utcnow().hour,
            "volatility": market_context.get("volatility", 0),
            "external_score": market_context.get("external_score", 0),
            "external_direction": market_context.get("external_direction", "neutral"),
            "entry_path": market_context.get("entry_path", "unknown"),  # 진입 시그널 조합 (v4)
            "exit_reason": market_context.get("exit_reason", "unknown"),  # 청산 사유 (v4)
            "confirming_sources": market_context.get("confirming_sources", []),  # 합의 소스 (v4)
            "atr_at_entry": market_context.get("atr_at_entry", 0),  # 진입 시 ATR (v4)
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
        self._update_external_accuracy(enriched)
        self._update_entry_path(enriched)       # v4: 진입 경로별 성과
        self._update_exit_reason(enriched)       # v4: 청산 사유별 분석
        self._update_loss_patterns(enriched)     # v4: 손실 패턴 분석

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

    def _update_external_accuracy(self, trade: dict):
        """외부 신호 방향별 적중률 추적"""
        ext_dir = trade.get("external_direction", "neutral")
        ext_score = trade.get("external_score", 0)
        side = trade.get("side", "unknown")
        is_win = trade.get("pnl", 0) > 0

        if ext_dir == "neutral":
            return

        # 외부 신호가 거래 방향과 일치했는지
        ext_agreed = (
            (ext_dir == "bullish" and side == "long") or
            (ext_dir == "bearish" and side == "short")
        )
        agreement_key = "agreed" if ext_agreed else "disagreed"

        if "external_accuracy" not in self.feedback:
            self.feedback["external_accuracy"] = {}

        # 방향별 적중률
        if ext_dir not in self.feedback["external_accuracy"]:
            self.feedback["external_accuracy"][ext_dir] = {"wins": 0, "losses": 0, "total_pnl": 0}
        d = self.feedback["external_accuracy"][ext_dir]
        if is_win:
            d["wins"] += 1
        else:
            d["losses"] += 1
        d["total_pnl"] += trade.get("pnl", 0)

        # 합의/불일치별 적중률
        if agreement_key not in self.feedback["external_accuracy"]:
            self.feedback["external_accuracy"][agreement_key] = {"wins": 0, "losses": 0, "total_pnl": 0}
        a = self.feedback["external_accuracy"][agreement_key]
        if is_win:
            a["wins"] += 1
        else:
            a["losses"] += 1
        a["total_pnl"] += trade.get("pnl", 0)

        # 강도별 적중률
        strength_key = "strong" if abs(ext_score) > 0.3 else "moderate" if abs(ext_score) > 0.1 else "weak"
        ext_strength_key = f"ext_{strength_key}"
        if ext_strength_key not in self.feedback["external_accuracy"]:
            self.feedback["external_accuracy"][ext_strength_key] = {"wins": 0, "losses": 0, "total_pnl": 0}
        s = self.feedback["external_accuracy"][ext_strength_key]
        if is_win:
            s["wins"] += 1
        else:
            s["losses"] += 1
        s["total_pnl"] += trade.get("pnl", 0)

    def _update_entry_path(self, trade: dict):
        """v4: 진입 시그널 조합별 성과 추적"""
        sources = trade.get("confirming_sources", [])
        if not sources:
            return
        # 정렬된 조합 키 생성 (예: "EXT+ML+MOM")
        combo_key = "+".join(sorted(set(s.split("_")[0] for s in sources)))
        ep = self.feedback["entry_path_performance"]
        if combo_key not in ep:
            ep[combo_key] = {"wins": 0, "losses": 0, "total_pnl": 0, "trades": []}
        if trade["pnl"] > 0:
            ep[combo_key]["wins"] += 1
        else:
            ep[combo_key]["losses"] += 1
        ep[combo_key]["total_pnl"] += trade["pnl"]
        ep[combo_key]["trades"].append({
            "pnl": round(trade["pnl"], 4),
            "time": str(datetime.utcnow()),
        })
        # 최근 30건만 유지
        if len(ep[combo_key]["trades"]) > 30:
            ep[combo_key]["trades"] = ep[combo_key]["trades"][-30:]

    def _update_exit_reason(self, trade: dict):
        """v4: 청산 사유별 분석 (SL히트율 → SL 조정 근거)"""
        exit_reason = trade.get("exit_reason", "unknown")
        er = self.feedback["exit_reason_analysis"]
        if exit_reason not in er:
            er[exit_reason] = {"count": 0, "total_pnl": 0, "avg_pnl": 0}
        er[exit_reason]["count"] += 1
        er[exit_reason]["total_pnl"] += trade["pnl"]
        er[exit_reason]["avg_pnl"] = er[exit_reason]["total_pnl"] / er[exit_reason]["count"]

    def _update_loss_patterns(self, trade: dict):
        """v4: 손실 패턴 분석 — 어떤 조건에서 반복 손실이 발생하는지 추적"""
        if trade["pnl"] >= 0:
            return  # 수익 거래는 무시

        # 복합 키 생성: regime_side_시간대
        regime = trade.get("regime", "unknown")
        side = trade.get("side", "unknown")
        hour_bucket = "asia" if 0 <= trade["hour"] < 8 else "europe" if 8 <= trade["hour"] < 16 else "us"
        pattern_key = f"{regime}_{side}_{hour_bucket}"

        lp = self.feedback["loss_patterns"]
        if pattern_key not in lp:
            lp[pattern_key] = {"count": 0, "total_loss": 0}
        lp[pattern_key]["count"] += 1
        lp[pattern_key]["total_loss"] += trade["pnl"]

    def get_entry_blacklist(self) -> list[str]:
        """v4: 반복 실패하는 시그널 조합을 블랙리스트로 반환"""
        blacklist = []
        for combo, perf in self.feedback.get("entry_path_performance", {}).items():
            total = perf["wins"] + perf["losses"]
            if total >= 5:  # 최소 5건 이상
                win_rate = perf["wins"] / total
                if win_rate < 0.25 and perf["total_pnl"] < 0:
                    blacklist.append(combo)
                    if combo not in self.feedback.get("entry_blacklist", []):
                        self.feedback["entry_blacklist"].append(combo)
                        lesson = f"시그널조합 '{combo}' 승률 {win_rate:.0%} → 블랙리스트"
                        self.feedback["lessons"].append(lesson)
                        logger.warning(f"[Feedback] {lesson}")
        return blacklist

    def get_sl_adjustment_suggestion(self) -> dict:
        """v4: SL 히트율 분석 → SL 조정 제안"""
        er = self.feedback.get("exit_reason_analysis", {})
        sl_data = er.get("sl", er.get("sl_triggered", {}))
        tp_data = er.get("tp", er.get("tp_triggered", {}))

        sl_count = sl_data.get("count", 0) if isinstance(sl_data, dict) else 0
        tp_count = tp_data.get("count", 0) if isinstance(tp_data, dict) else 0
        total = sl_count + tp_count

        if total < 5:
            return {"suggestion": "데이터 부족", "sl_hit_rate": 0}

        sl_hit_rate = sl_count / total
        suggestion = {}
        if sl_hit_rate > 0.65:
            suggestion = {
                "suggestion": "SL 확대 필요 (히트율 과다)",
                "sl_hit_rate": round(sl_hit_rate, 2),
                "recommended_action": "atr_sl_multiplier +0.5",
            }
        elif sl_hit_rate < 0.30:
            suggestion = {
                "suggestion": "SL 적정 또는 축소 가능",
                "sl_hit_rate": round(sl_hit_rate, 2),
                "recommended_action": "현재 유지",
            }
        else:
            suggestion = {
                "suggestion": "SL 적정 범위",
                "sl_hit_rate": round(sl_hit_rate, 2),
                "recommended_action": "현재 유지",
            }
        return suggestion

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

        # 6. 외부 신호 적중률 분석 → 가중치 자동 조정
        ext_acc = self.feedback.get("external_accuracy", {})
        agreed = ext_acc.get("agreed", {"wins": 0, "losses": 0})
        disagreed = ext_acc.get("disagreed", {"wins": 0, "losses": 0})
        agreed_total = agreed["wins"] + agreed["losses"]
        disagreed_total = disagreed["wins"] + disagreed["losses"]

        if agreed_total >= 10:
            agreed_wr = agreed["wins"] / agreed_total
            if agreed_wr > 0.6:
                adjustments["external_weight_boost"] = 1.3
                lesson = f"외부요인 합의 시 승률 {agreed_wr:.0%} → 외부 가중치 30% 증가"
                if lesson not in lessons:
                    lessons.append(lesson)
                    logger.info(f"[Feedback] {lesson}")
            elif agreed_wr < 0.35:
                adjustments["external_weight_boost"] = 0.5
                lesson = f"외부요인 합의 시 승률 {agreed_wr:.0%} → 외부 가중치 50% 감소"
                if lesson not in lessons:
                    lessons.append(lesson)
                    logger.info(f"[Feedback] {lesson}")

        if disagreed_total >= 10:
            disagreed_wr = disagreed["wins"] / disagreed_total
            if disagreed_wr < 0.3:
                adjustments["external_disagree_block"] = True
                lesson = f"외부요인 반대 시 승률 {disagreed_wr:.0%} → 외부 반대 시 진입 차단"
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
        # 과다 레버리지 감지 스케일
        scale *= adj.get("overlev_scale", 1.0)

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

        # 외부 신호 정확도
        ext_summary = {}
        for key, perf in self.feedback.get("external_accuracy", {}).items():
            total = perf["wins"] + perf["losses"]
            if total > 0:
                ext_summary[key] = {
                    "win_rate": f"{perf['wins'] / total:.0%}",
                    "total_pnl": f"${perf['total_pnl']:.2f}",
                    "trades": total,
                }
        report["external_signal_accuracy"] = ext_summary

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
