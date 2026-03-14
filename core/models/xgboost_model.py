"""XGBoost 기반 방향 예측 모델"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


class XGBoostPredictor:
    """기술적 지표 → 시장 방향 예측 (하락/횡보/상승)"""

    def __init__(self, model_dir: str = "models_saved"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: XGBClassifier | None = None
        self.feature_columns: list[str] = []
        self.accuracy: float = 0.0

    def train(self, df: pd.DataFrame, feature_cols: list[str], label_col: str = "label"):
        """학습 데이터로 모델 훈련"""
        self.feature_columns = feature_cols
        X = df[feature_cols].values
        y = df[label_col].values

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"XGBoost 학습 완료 - 정확도: {self.accuracy:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['하락','횡보','상승'], zero_division=0)}")
        return self.accuracy

    def predict(self, df: pd.DataFrame) -> dict:
        """시그널 예측 반환"""
        if self.model is None:
            return {"signal": 0.0, "confidence": 0.0, "direction": "neutral"}

        X = df[self.feature_columns].values[-1:]
        proba = self.model.predict_proba(X)[0]
        pred = int(self.model.predict(X)[0])

        direction_map = {0: "short", 1: "neutral", 2: "long"}
        signal = proba[2] - proba[0]  # long확률 - short확률 → [-1, 1]

        return {
            "signal": float(signal),
            "confidence": float(max(proba)),
            "direction": direction_map[pred],
            "probabilities": {"short": float(proba[0]), "neutral": float(proba[1]), "long": float(proba[2])},
        }

    def get_feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self.feature_columns, importance.tolist()))

    def save(self, name: str = "xgboost"):
        path = self.model_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "features": self.feature_columns, "accuracy": self.accuracy}, f)
        logger.info(f"XGBoost 모델 저장: {path}")

    def load(self, name: str = "xgboost") -> bool:
        path = self.model_dir / f"{name}.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_columns = data["features"]
        self.accuracy = data["accuracy"]
        logger.info(f"XGBoost 모델 로드: 정확도 {self.accuracy:.4f}")
        return True
