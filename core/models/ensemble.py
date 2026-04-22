"""ML 모델 앙상블 - 다수 모델 시그널을 통합"""

import numpy as np
import pandas as pd
from loguru import logger

from core.models.lstm_model import LSTMPredictor
from core.models.xgboost_model import XGBoostPredictor


class EnsembleSignalGenerator:
    """XGBoost + LSTM 앙상블 시그널 생성기"""

    # === Regime-Conditional Signal Multiplier (2026-04-23 재조정) ===
    # 이전 "WR 13.8%(29건)"은 방향 미분해 집계였음 → 재분해 결과:
    # - strong_uptrend × LONG : n=3  WR 66.7% sum=+$13.90   (방향 긍정, 소표본)
    # - strong_uptrend × SHORT: n=23 WR  0.0% sum=-$634.62  (fade 참사 — 원인)
    # 즉, 이전 -0.5(fade) 가중치가 SHORT 진입을 유도 → 23건 전패.
    # 교정: 상승추세 원신호 그대로 존중(1.0), 숏 차단은 long_only=true로 분리 enforce.
    #
    # - strong_downtrend × SHORT: n=3 WR 66.7% → 유지 (1.0)
    # - high_volume_breakout × LONG: n=2 WR 50.0% +$21.91 → 돌파 edge 완전 반영 (1.0)
    # - unknown: 분류 실패 = 신뢰 불가, 계속 0.0
    # - extreme_volatility: Kelly f*→0, 계속 0.0
    # - ranging: 양방향 음수 기댓값 (LONG -$10, SHORT -$237) → 보수 0.6로 강화
    REGIME_SIGNAL_WEIGHT = {
        "strong_uptrend": 1.0,            # [2026-04-23] -0.5 → 1.0 (fade 제거, 추세순응)
        "strong_downtrend": 1.0,          # 정상 유지
        "unknown": 0.0,                   # 거래 중단
        "high_volume_breakout": 1.0,      # [2026-04-23] 0.5 → 1.0 (돌파 edge 완전 반영)
        "extreme_volatility": 0.0,        # 고변동성 → 거래 중단 (Kelly f*→0)
        "normal": 1.0,
        "ranging": 0.6,                   # [2026-04-23] 0.8 → 0.6 (양방향 음기댓값 확인)
    }

    def __init__(self, model_dir: str = "models_saved"):
        self.xgb = XGBoostPredictor(model_dir=model_dir)
        self.lstm = LSTMPredictor(model_dir=model_dir)
        self.weights = {"xgboost": 0.5, "lstm": 0.5}
        self._performance_history: list[dict] = []

    def train_all(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        walk_forward: bool = False,
        use_purged_kfold: bool = False,
        embargo_pct: float = 0.01,
    ):
        """모든 모델 학습

        Args:
            walk_forward: True이면 walk-forward CV (Capital Tier small+)
            use_purged_kfold: True이면 PurgedKFold + Embargo (tier=large+ 권장, Lopez de Prado)
            embargo_pct: PurgedKFold에서 test 직후 제거할 비율
        """
        if walk_forward:
            cv_name = "PurgedKFold+Embargo" if use_purged_kfold else "TimeSeriesSplit"
            logger.info(f"=== 앙상블 모델 Walk-Forward CV 학습 시작 ({cv_name}) ===")
            xgb_acc = self.xgb.train_walkforward(
                df, feature_cols,
                use_purged_kfold=use_purged_kfold, embargo_pct=embargo_pct,
            )
            lstm_acc = self.lstm.train_walkforward(df, feature_cols)
        else:
            logger.info("=== 앙상블 모델 학습 시작 ===")
            xgb_acc = self.xgb.train(df, feature_cols)
            lstm_acc = self.lstm.train(df, feature_cols)

        # 정확도 기반 가중치 자동 조정
        total = xgb_acc + lstm_acc
        if total > 0:
            self.weights["xgboost"] = xgb_acc / total
            self.weights["lstm"] = lstm_acc / total

        logger.info(f"앙상블 가중치 - XGBoost: {self.weights['xgboost']:.3f}, LSTM: {self.weights['lstm']:.3f}")

    def predict(self, df: pd.DataFrame, regime: str | None = None) -> dict:
        """앙상블 시그널 생성

        Args:
            df: OHLCV + 피처 포함된 최근 캔들
            regime: 현재 시장 레짐 (strong_uptrend, strong_downtrend, normal, ranging,
                    high_volume_breakout, extreme_volatility, unknown).
                    None이면 가중치 1.0 적용(기존 동작).
        """
        xgb_pred = self.xgb.predict(df)
        lstm_pred = self.lstm.predict(df)

        # 가중 평균 시그널
        w_xgb = self.weights["xgboost"]
        w_lstm = self.weights["lstm"]

        combined_signal = xgb_pred["signal"] * w_xgb + lstm_pred["signal"] * w_lstm
        combined_confidence = xgb_pred["confidence"] * w_xgb + lstm_pred["confidence"] * w_lstm

        # === Regime-Conditional 가중치 적용 (2026-04-20 추가) ===
        # 원시 시그널은 그대로 반환(디버깅용), 최종 signal은 레짐 가중치 반영
        raw_signal = combined_signal
        regime_mult = self.REGIME_SIGNAL_WEIGHT.get(regime, 1.0) if regime else 1.0
        combined_signal = combined_signal * regime_mult

        # 방향 결정 (레짐 가중치 적용 후)
        if combined_signal > 0.15:
            direction = "long"
        elif combined_signal < -0.15:
            direction = "short"
        else:
            direction = "neutral"

        # 모델 합의도 (agreement) — 원시 예측 기준 (레짐 가중치와 독립)
        agreement = 1.0 if xgb_pred["direction"] == lstm_pred["direction"] else 0.5

        return {
            "signal": float(combined_signal),
            "raw_signal": float(raw_signal),
            "regime_multiplier": float(regime_mult),
            "regime": regime,
            "confidence": float(combined_confidence * agreement),
            "direction": direction,
            "agreement": agreement,
            "models": {
                "xgboost": xgb_pred,
                "lstm": lstm_pred,
            },
        }

    def update_weights(self, model_name: str, recent_accuracy: float):
        """최근 성능 기반 가중치 동적 조정"""
        self._performance_history.append({"model": model_name, "accuracy": recent_accuracy})
        # 최근 10개 성능 기반 재조정
        recent = self._performance_history[-20:]
        xgb_scores = [p["accuracy"] for p in recent if p["model"] == "xgboost"]
        lstm_scores = [p["accuracy"] for p in recent if p["model"] == "lstm"]

        xgb_avg = np.mean(xgb_scores) if xgb_scores else 0.5
        lstm_avg = np.mean(lstm_scores) if lstm_scores else 0.5
        total = xgb_avg + lstm_avg
        if total > 0:
            self.weights["xgboost"] = xgb_avg / total
            self.weights["lstm"] = lstm_avg / total
            logger.info(f"앙상블 가중치 업데이트 - XGB: {self.weights['xgboost']:.3f}, LSTM: {self.weights['lstm']:.3f}")

    def save_all(self):
        self.xgb.save()
        self.lstm.save()

    def load_all(self) -> bool:
        xgb_ok = self.xgb.load()
        lstm_ok = self.lstm.load()
        return xgb_ok and lstm_ok
