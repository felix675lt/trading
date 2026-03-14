"""멀티타임프레임 합류 분석 - 여러 시간대에서 같은 방향일 때만 진입

프로 트레이더 규칙:
- 1h/4h에서 방향 확인 (큰 그림)
- 15m에서 진입 타이밍
- 5m에서 정확한 진입점

합류(Confluence) 점수:
- 모든 타임프레임 같은 방향: 강한 시그널 (1.0)
- 대부분 같은 방향: 중간 시그널 (0.5~0.8)
- 혼재: 약한 시그널 (0.0~0.3)
- 반대: 거래 금지 (-1.0)
"""

import numpy as np
import pandas as pd
import ta
from loguru import logger


class MultiTimeframeAnalyzer:
    """멀티타임프레임 합류 분석"""

    # 타임프레임별 가중치 (큰 프레임일수록 높은 가중치)
    TF_WEIGHTS = {
        "5m": 0.10,
        "15m": 0.20,
        "1h": 0.35,
        "4h": 0.35,
    }

    def __init__(self, timeframes: list[str] | None = None):
        self.timeframes = timeframes or ["5m", "15m", "1h", "4h"]
        self._signals: dict[str, dict] = {}  # tf → signal
        self._last_confluence: dict = {}

    def analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> dict:
        """단일 타임프레임 분석"""
        if len(df) < 50:
            return {"direction": "neutral", "strength": 0, "details": {}}

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        signals = {}

        # 1. 추세 (EMA 크로스)
        ema_fast = pd.Series(close).ewm(span=9).mean().values
        ema_slow = pd.Series(close).ewm(span=21).mean().values
        ema_trend = pd.Series(close).ewm(span=50).mean().values

        ema_cross = (ema_fast[-1] - ema_slow[-1]) / close[-1]
        above_ema50 = close[-1] > ema_trend[-1]

        signals["ema_cross"] = 1 if ema_cross > 0.001 else -1 if ema_cross < -0.001 else 0
        signals["ema_trend"] = 1 if above_ema50 else -1

        # 2. RSI
        rsi = ta.momentum.RSIIndicator(pd.Series(close), window=14).rsi().values
        current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
        signals["rsi"] = 1 if current_rsi < 30 else -1 if current_rsi > 70 else 0
        signals["rsi_value"] = current_rsi

        # 3. MACD
        macd_ind = ta.trend.MACD(pd.Series(close))
        macd_hist = macd_ind.macd_diff().values
        if len(macd_hist) > 0 and not np.isnan(macd_hist[-1]):
            signals["macd"] = 1 if macd_hist[-1] > 0 else -1
            # MACD 히스토그램 방향 (증가/감소)
            if len(macd_hist) > 1 and not np.isnan(macd_hist[-2]):
                signals["macd_momentum"] = 1 if macd_hist[-1] > macd_hist[-2] else -1
            else:
                signals["macd_momentum"] = 0
        else:
            signals["macd"] = 0
            signals["macd_momentum"] = 0

        # 4. 볼린저 밴드 위치
        bb = ta.volatility.BollingerBands(pd.Series(close))
        bb_upper = bb.bollinger_hband().values[-1]
        bb_lower = bb.bollinger_lband().values[-1]
        if not np.isnan(bb_upper) and not np.isnan(bb_lower) and bb_upper != bb_lower:
            bb_pct = (close[-1] - bb_lower) / (bb_upper - bb_lower)
            signals["bb_position"] = -1 if bb_pct > 0.95 else 1 if bb_pct < 0.05 else 0
        else:
            signals["bb_position"] = 0

        # 5. ADX (추세 강도)
        adx = ta.trend.ADXIndicator(pd.Series(high), pd.Series(low), pd.Series(close))
        adx_val = adx.adx().values[-1] if not np.isnan(adx.adx().values[-1]) else 20
        di_plus = adx.adx_pos().values[-1] if not np.isnan(adx.adx_pos().values[-1]) else 0
        di_minus = adx.adx_neg().values[-1] if not np.isnan(adx.adx_neg().values[-1]) else 0

        signals["adx_value"] = adx_val
        signals["trend_strength"] = adx_val / 100  # 0~1 정규화
        if adx_val > 25:  # 추세 존재
            signals["adx_direction"] = 1 if di_plus > di_minus else -1
        else:
            signals["adx_direction"] = 0

        # 6. 거래량 확인
        vol_sma = np.mean(volume[-20:])
        vol_ratio = volume[-1] / vol_sma if vol_sma > 0 else 1.0
        signals["volume_confirmation"] = 1 if vol_ratio > 1.5 else 0

        # 종합 방향 계산
        directional_signals = [
            signals["ema_cross"] * 2,
            signals["ema_trend"] * 2,
            signals["macd"],
            signals["macd_momentum"],
            signals["adx_direction"],
            signals["rsi"],  # 역추세 신호
            signals["bb_position"],  # 역추세 신호
        ]

        total_score = sum(directional_signals)
        max_possible = sum(abs(s) for s in [2, 2, 1, 1, 1, 1, 1])

        normalized = total_score / max_possible if max_possible > 0 else 0
        strength = abs(normalized)

        if normalized > 0.2:
            direction = "bullish"
        elif normalized < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"

        result = {
            "direction": direction,
            "strength": round(strength, 3),
            "score": round(normalized, 3),
            "rsi": round(current_rsi, 1),
            "adx": round(adx_val, 1),
            "volume_ratio": round(vol_ratio, 2),
            "details": signals,
        }

        self._signals[timeframe] = result
        return result

    def calculate_confluence(self) -> dict:
        """모든 타임프레임의 합류 점수 계산"""
        if len(self._signals) < 2:
            return {
                "score": 0, "direction": "neutral", "confidence": 0,
                "agreement": 0, "reason": "타임프레임 데이터 부족",
            }

        weighted_score = 0.0
        total_weight = 0.0
        directions = []

        for tf, signal in self._signals.items():
            weight = self.TF_WEIGHTS.get(tf, 0.1)
            weighted_score += signal["score"] * weight
            total_weight += weight
            directions.append(signal["direction"])

        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0

        # 합의율: 같은 방향인 타임프레임 비율
        bullish_count = sum(1 for d in directions if d == "bullish")
        bearish_count = sum(1 for d in directions if d == "bearish")
        total = len(directions)

        if bullish_count > bearish_count:
            agreement = bullish_count / total
            dominant = "bullish"
        elif bearish_count > bullish_count:
            agreement = bearish_count / total
            dominant = "bearish"
        else:
            agreement = 0
            dominant = "neutral"

        # 방향 결정
        if final_score > 0.15 and agreement > 0.5:
            direction = "bullish"
        elif final_score < -0.15 and agreement > 0.5:
            direction = "bearish"
        else:
            direction = "neutral"

        # 확신도
        confidence = min(abs(final_score) * agreement * 2, 1.0)

        # 이유 생성
        tf_summary = " | ".join(
            f"{tf}:{s['direction'][0].upper()}({s['score']:+.2f})"
            for tf, s in sorted(self._signals.items(), key=lambda x: self.TF_WEIGHTS.get(x[0], 0), reverse=True)
        )

        # 강한 반대 시그널 감지
        has_conflict = bullish_count > 0 and bearish_count > 0
        higher_tf_conflict = False
        if "4h" in self._signals and "1h" in self._signals:
            if self._signals["4h"]["direction"] != self._signals["1h"]["direction"]:
                if self._signals["4h"]["direction"] != "neutral" and self._signals["1h"]["direction"] != "neutral":
                    higher_tf_conflict = True

        result = {
            "score": round(final_score, 3),
            "direction": direction,
            "confidence": round(confidence, 3),
            "agreement": round(agreement, 3),
            "has_conflict": has_conflict,
            "higher_tf_conflict": higher_tf_conflict,
            "timeframes": {tf: {"direction": s["direction"], "score": s["score"]}
                          for tf, s in self._signals.items()},
            "reason": f"MTF 합류: {tf_summary} (합의 {agreement:.0%})",
        }

        self._last_confluence = result
        return result

    def get_features(self) -> dict:
        """ML 모델용 멀티타임프레임 피처"""
        features = {}
        for tf, signal in self._signals.items():
            prefix = f"mtf_{tf}"
            features[f"{prefix}_score"] = signal.get("score", 0)
            features[f"{prefix}_strength"] = signal.get("strength", 0)
            features[f"{prefix}_rsi"] = signal.get("rsi", 50) / 100
            features[f"{prefix}_adx"] = signal.get("adx", 20) / 100

        conf = self._last_confluence
        features["mtf_confluence_score"] = conf.get("score", 0)
        features["mtf_agreement"] = conf.get("agreement", 0)
        features["mtf_has_conflict"] = 1.0 if conf.get("has_conflict", False) else 0.0

        return features

    def get_signal_for_strategy(self) -> dict:
        """전략 매니저용 합류 시그널"""
        if not self._last_confluence:
            return {"score": 0, "direction": "neutral", "confidence": 0}
        c = self._last_confluence
        return {
            "score": c["score"],
            "direction": c["direction"],
            "confidence": c["confidence"],
            "agreement": c["agreement"],
            "has_conflict": c.get("has_conflict", False),
            "higher_tf_conflict": c.get("higher_tf_conflict", False),
        }

    def get_report(self) -> dict:
        """대시보드용"""
        return {
            "timeframes": {
                tf: {
                    "direction": s["direction"],
                    "score": s["score"],
                    "rsi": s.get("rsi", 50),
                    "adx": s.get("adx", 20),
                }
                for tf, s in self._signals.items()
            },
            "confluence": self._last_confluence,
        }
