"""적응형 파라미터 최적화 - 시장 상태에 따라 파라미터 자동 조정"""

import hashlib
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
from loguru import logger


class MarketRegimeDetector:
    """시장 레짐 감지 (추세/횡보/고변동성)"""

    def detect(self, prices: np.ndarray, volumes: np.ndarray) -> str:
        if len(prices) < 50:
            return "normal"

        # 변동성 측정
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-20:])
        avg_volatility = np.std(returns[-100:]) if len(returns) >= 100 else volatility

        # 추세 강도 (선형 회귀 R²)
        x = np.arange(min(50, len(prices)))
        y = prices[-len(x):]
        correlation = np.corrcoef(x, y)[0, 1]
        trend_strength = abs(correlation)

        # 거래량 이상 감지
        vol_ratio = np.mean(volumes[-5:]) / (np.mean(volumes[-50:]) + 1e-8)

        # 레짐 판별
        if volatility > avg_volatility * 2:
            return "extreme_volatility"
        elif trend_strength > 0.7:
            if correlation > 0:
                return "strong_uptrend"
            else:
                return "strong_downtrend"
        elif trend_strength < 0.3:
            return "ranging"
        elif vol_ratio > 2.0:
            return "high_volume_breakout"
        else:
            return "normal"


class AdaptiveOptimizer:
    """시장 레짐에 따라 전략 파라미터를 자동 조정"""

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.current_regime = "normal"
        self.regime_params = {
            "strong_uptrend": {
                "signal_threshold": 0.1,
                "min_confidence": 0.5,
                "position_scale": 1.2,
                "stop_loss_mult": 1.5,
                "prefer_direction": "long",
            },
            "strong_downtrend": {
                "signal_threshold": 0.1,
                "min_confidence": 0.5,
                "position_scale": 1.2,
                "stop_loss_mult": 1.5,
                "prefer_direction": "short",
            },
            "ranging": {
                "signal_threshold": 0.25,
                "min_confidence": 0.65,
                "position_scale": 0.6,
                "stop_loss_mult": 0.8,
                "prefer_direction": "neutral",
            },
            "extreme_volatility": {
                "signal_threshold": 0.3,
                "min_confidence": 0.75,
                "position_scale": 0.3,
                "stop_loss_mult": 2.0,
                "prefer_direction": "neutral",
            },
            "high_volume_breakout": {
                "signal_threshold": 0.12,
                "min_confidence": 0.55,
                "position_scale": 1.0,
                "stop_loss_mult": 1.2,
                "prefer_direction": "neutral",
            },
            "normal": {
                "signal_threshold": 0.15,
                "min_confidence": 0.55,
                "position_scale": 1.0,
                "stop_loss_mult": 1.0,
                "prefer_direction": "neutral",
            },
        }

    def update(self, prices: np.ndarray, volumes: np.ndarray) -> dict:
        """시장 상태 업데이트 및 최적 파라미터 반환"""
        new_regime = self.regime_detector.detect(prices, volumes)

        if new_regime != self.current_regime:
            logger.info(f"시장 레짐 전환: {self.current_regime} → {new_regime}")
            self.current_regime = new_regime

        return self.get_params()

    def get_params(self) -> dict:
        params = self.regime_params.get(self.current_regime, self.regime_params["normal"])
        params["regime"] = self.current_regime
        return params


class StrategyOptimizer:
    """자동 전략 최적화 시스템 — Paper/Live 독립 추적

    거래 결과를 config 해시별로 기록하고,
    최적 파라미터 조합을 자동으로 탐색합니다.
    """

    def __init__(self):
        self.current_config: dict = {}
        self.config_performance: dict[str, dict] = defaultdict(
            lambda: {"trades": [], "total_pnl": 0.0, "wins": 0, "losses": 0}
        )
        # performance_history: config_hash → [trade_list] (호환용 alias)
        self.performance_history: dict[str, list] = defaultdict(list)
        logger.info("[StrategyOptimizer] 자동 전략 최적화 시스템 초기화")

    def _config_to_hash(self, config: dict) -> str:
        """설정 dict를 고유 해시로 변환"""
        serialized = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()[:12]

    def record_trade(self, config_hash: str, trade: dict):
        """거래 결과를 config 해시에 연결하여 기록"""
        perf = self.config_performance[config_hash]
        pnl = trade.get("pnl", 0)
        trade_record = {
            "pnl": pnl,
            "timestamp": str(trade.get("timestamp", datetime.utcnow())),
            "symbol": trade.get("symbol", ""),
            "hour": trade.get("hour", 0),
        }
        perf["trades"].append(trade_record)
        perf["total_pnl"] += pnl
        if pnl > 0:
            perf["wins"] += 1
        else:
            perf["losses"] += 1

        # performance_history도 동기화 (호환용)
        self.performance_history[config_hash].append(trade_record)

        # 최근 100건만 유지
        if len(perf["trades"]) > 100:
            perf["trades"] = perf["trades"][-100:]
        if len(self.performance_history[config_hash]) > 100:
            self.performance_history[config_hash] = self.performance_history[config_hash][-100:]

    def get_best_config(self) -> tuple[str, dict]:
        """가장 성과 좋은 config 해시 반환"""
        best_hash = ""
        best_pnl = float("-inf")
        for h, perf in self.config_performance.items():
            if len(perf["trades"]) >= 3 and perf["total_pnl"] > best_pnl:
                best_pnl = perf["total_pnl"]
                best_hash = h
        return best_hash, self.config_performance.get(best_hash, {})

    def get_report(self) -> dict:
        """성과 리포트"""
        configs = {}
        for h, perf in self.config_performance.items():
            total = perf["wins"] + perf["losses"]
            configs[h] = {
                "trades": total,
                "win_rate": perf["wins"] / total if total > 0 else 0,
                "total_pnl": round(perf["total_pnl"], 2),
            }
        return {
            "total_configs": len(configs),
            "configs": configs,
        }

    def optimize_daily(self, all_trades: list[dict]):
        """일일 최적화 — 거래 결과 기반 파라미터 자동 조정"""
        if len(all_trades) < 10:
            return

        wins = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
        losses = len(all_trades) - wins
        total_pnl = sum(t.get("pnl", 0) for t in all_trades)
        win_rate = wins / len(all_trades) if all_trades else 0

        # 시간대별 성과 분석
        hour_pnl = defaultdict(list)
        for t in all_trades:
            hour_pnl[t.get("hour", 0)].append(t.get("pnl", 0))

        best_hours = sorted(hour_pnl.keys(), key=lambda h: sum(hour_pnl[h]), reverse=True)[:3]
        worst_hours = sorted(hour_pnl.keys(), key=lambda h: sum(hour_pnl[h]))[:3]

        logger.info(
            f"[StrategyOptimizer] 일일 최적화: {len(all_trades)}건 | "
            f"승률 {win_rate:.0%} | PnL: ${total_pnl:.2f} | "
            f"베스트시간: {best_hours} | 워스트시간: {worst_hours}"
        )
