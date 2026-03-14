"""ML 모델 앙상블 - 다수 모델 시그널을 통합"""

import numpy as np
import pandas as pd
from loguru import logger

from core.models.lstm_model import LSTMPredictor
from core.models.xgboost_model import XGBoostPredictor


class EnsembleSignalGenerator:
    """XGBoost + LSTM 앙상블 시그널 생성기"""

    def __init__(self, model_dir: str = "models_saved"):
        self.xgb = XGBoostPredictor(model_dir=model_dir)
        self.lstm = LSTMPredictor(model_dir=model_dir)
        self.weights = {"xgboost": 0.5, "lstm": 0.5}
        self._performance_history: list[dict] = []

    def train_all(self, df: pd.DataFrame, feature_cols: list[str]):
        """모든 모델 학습"""
        logger.info("=== 앙상블 모델 학습 시작 ===")
        xgb_acc = self.xgb.train(df, feature_cols)
        lstm_acc = self.lstm.train(df, feature_cols)

        # 정확도 기반 가중치 자동 조정
        total = xgb_acc + lstm_acc
        if total > 0:
            self.weights["xgboost"] = xgb_acc / total
            self.weights["lstm"] = lstm_acc / total

        logger.info(f"앙상블 가중치 - XGBoost: {self.weights['xgboost']:.3f}, LSTM: {self.weights['lstm']:.3f}")

    def predict(self, df: pd.DataFrame) -> dict:
        """앙상블 시그널 생성"""
        xgb_pred = self.xgb.predict(df)
        lstm_pred = self.lstm.predict(df)

        # 가중 평균 시그널
        w_xgb = self.weights["xgboost"]
        w_lstm = self.weights["lstm"]

        combined_signal = xgb_pred["signal"] * w_xgb + lstm_pred["signal"] * w_lstm
        combined_confidence = xgb_pred["confidence"] * w_xgb + lstm_pred["confidence"] * w_lstm

        # 방향 결정
        if combined_signal > 0.15:
            direction = "long"
        elif combined_signal < -0.15:
            direction = "short"
        else:
            direction = "neutral"

        # 모델 합의도 (agreement)
        agreement = 1.0 if xgb_pred["direction"] == lstm_pred["direction"] else 0.5

        return {
            "signal": float(combined_signal),
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
