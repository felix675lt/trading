"""Meta-Labeling (Lopez de Prado, Advances in Financial ML, Ch. 3).

1차 모델(앙상블)이 "어떤 방향으로 진입할지"를 결정 → "진입/스킵"의 2차 결정이 필요.
Meta-labeler는 1차 시그널 + 시장 피처를 입력으로 받아
"이 트레이드를 실제 취해야 하는가? (1) vs 스킵 (0)"을 출력.

효과:
- 정밀도(precision) 향상 — 약한 시그널 거르고 강한 것만 진입
- Kelly 사이징과 결합 시 자동으로 0(스킵)은 0 사이즈
- 앙상블 기존 구조 수정 없이 후단에 추가 가능

사용:
    ml = MetaLabeler()
    ml.train(df, primary_signal_col, outcome_col)  # outcome = 1(win) / 0(loss|skip)
    should_take = ml.predict(features, primary_signal)  # True/False
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class MetaLabeler:
    """2차 분류기 — 1차 시그널을 필터링.

    입력: [primary_signal, primary_confidence, 시장 피처들]
    출력: 진입(1) / 스킵(0) 이진 결정 + 확률

    tier=large+ 에서 활성화 권장.
    """

    def __init__(
        self,
        threshold: float = 0.55,
        model_dir: str | Path = "models_saved",
    ):
        self.threshold = threshold
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: RandomForestClassifier | None = None
        self.feature_columns: list[str] = []
        self.accuracy: float = 0.0
        self.precision: float = 0.0

    def _build_dataset(
        self,
        df: pd.DataFrame,
        primary_signal_col: str,
        primary_confidence_col: str,
        outcome_col: str,
        feature_cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """1차 시그널 + 피처 → X, y(=win 여부) 데이터셋 생성."""
        # 1차 시그널이 존재하는 샘플만 (방향 비neutral)
        mask = (df[primary_signal_col].abs() > 0.05) & df[outcome_col].notna()
        sub = df[mask].copy()

        X_base = sub[feature_cols].values
        sig = sub[primary_signal_col].values.reshape(-1, 1)
        conf = sub[primary_confidence_col].values.reshape(-1, 1) if primary_confidence_col in sub else np.ones_like(sig)
        X = np.hstack([X_base, sig, conf])

        y = sub[outcome_col].astype(int).values
        return X, y

    def train(
        self,
        df: pd.DataFrame,
        primary_signal_col: str = "primary_signal",
        primary_confidence_col: str = "primary_confidence",
        outcome_col: str = "tb_label",  # 0=loss, 2=win, 1=neutral
        feature_cols: list[str] | None = None,
    ) -> float:
        """Meta-labeler 학습.

        outcome_col 해석: 2(TP hit, win) → 1, 나머지(SL/time) → 0.
        """
        feature_cols = feature_cols or [
            c for c in df.columns
            if c not in {"open", "high", "low", "close", "volume", "label", "future_return",
                         "tb_label", "tb_ret", "tb_hit", "tb_t1",
                         primary_signal_col, primary_confidence_col, outcome_col}
        ]
        self.feature_columns = feature_cols + [primary_signal_col, primary_confidence_col]

        # outcome을 binary로 매핑 — TP(2) = 1, 나머지 = 0
        outcome_binary = (df[outcome_col] == 2).astype(int)
        df_mod = df.copy()
        df_mod["_outcome_bin"] = outcome_binary

        X, y = self._build_dataset(
            df_mod, primary_signal_col, primary_confidence_col, "_outcome_bin", feature_cols
        )
        if len(X) < 100:
            logger.warning(f"[Meta] 학습 샘플 부족 ({len(X)}) — 학습 생략")
            return 0.0

        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        if len(np.unique(y_tr)) < 2:
            logger.warning("[Meta] 학습 레이블 단일 — 학습 불가")
            return 0.0

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_tr, y_tr)

        y_pred = self.model.predict(X_te)
        y_proba = self.model.predict_proba(X_te)[:, 1] if self.model.classes_.shape[0] == 2 else np.full(len(y_pred), 0.5)

        self.accuracy = float(accuracy_score(y_te, y_pred))
        # precision: "진입" 결정했을 때 실제 win 비율
        take = y_proba >= self.threshold
        if take.sum() > 0:
            self.precision = float(y_te[take].sum() / take.sum())
        else:
            self.precision = 0.0
        logger.info(
            f"[Meta] 학습 완료 | acc={self.accuracy:.3f} precision={self.precision:.3f} "
            f"(n_train={len(X_tr)} n_test={len(X_te)} take_rate={take.mean():.2%})"
        )
        return self.accuracy

    def predict(
        self,
        feature_row: np.ndarray | pd.Series,
        primary_signal: float,
        primary_confidence: float,
    ) -> dict:
        """실시간 2차 결정.

        Returns:
            {"take": bool, "prob": float, "threshold": float}
        """
        if self.model is None:
            return {"take": True, "prob": 1.0, "threshold": self.threshold, "reason": "untrained"}
        try:
            if isinstance(feature_row, pd.Series):
                feature_row = feature_row.values
            base = np.asarray(feature_row, dtype=np.float64).ravel()
            X = np.concatenate([base, [primary_signal, primary_confidence]]).reshape(1, -1)
            proba = self.model.predict_proba(X)[0]
            # class 1 = 진입 권장
            if self.model.classes_[0] == 1:
                prob_take = float(proba[0])
            else:
                prob_take = float(proba[-1]) if len(proba) > 1 else 0.5
            take = prob_take >= self.threshold
            return {"take": bool(take), "prob": prob_take, "threshold": self.threshold}
        except Exception as e:
            logger.debug(f"[Meta] predict 실패: {e} → 통과")
            return {"take": True, "prob": 1.0, "threshold": self.threshold, "error": str(e)}

    def save(self, name: str = "meta_labeler"):
        if self.model is None:
            return
        path = self.model_dir / f"{name}.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "features": self.feature_columns,
                    "threshold": self.threshold,
                    "accuracy": self.accuracy,
                    "precision": self.precision,
                }, f)
            logger.info(f"[Meta] 저장: {path}")
        except Exception as e:
            logger.warning(f"[Meta] 저장 실패: {e}")

    def load(self, name: str = "meta_labeler") -> bool:
        path = self.model_dir / f"{name}.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.feature_columns = data["features"]
            self.threshold = data.get("threshold", 0.55)
            self.accuracy = data.get("accuracy", 0.0)
            self.precision = data.get("precision", 0.0)
            logger.info(f"[Meta] 로드: acc={self.accuracy:.3f} precision={self.precision:.3f}")
            return True
        except Exception as e:
            logger.warning(f"[Meta] 로드 실패: {e}")
            return False
