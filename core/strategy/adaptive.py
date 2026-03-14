"""적응형 파라미터 최적화 - 시장 상태에 따라 파라미터 자동 조정"""

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
