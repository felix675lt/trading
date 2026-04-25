"""LightGBM 기반 시장 방향 예측 모델 (XGBoost 보완 — Stawarz 2025)
==================================================================
M2 Apple Silicon 최적화 + 승률 개선 패키지:
  - 히스토그램 기반 학습 (XGBoost 대비 2-5배 빠름)
  - leaf-wise 성장 + max_bin=127 → M2 캐시 친화
  - Purged K-Fold CV + embargo (Lopez de Prado, Adv. Financial ML)
  - Time-decay × focal sample weighting (불균형 + 최근 패턴 강화)
  - Auto feature selection (importance 상위 K개) — 노이즈 차단
  - 클래스 불균형 보정 → strong_uptrend 시나리오 상승 recall ↑
  - 증분학습 + 피처 수 변경 자동 감지 + 5%p 하락 시 자동 롤백

XGBoostPredictor와 동일한 인터페이스를 제공한다(예: train, train_walkforward,
predict, save, load) → ensemble.py에서 단순 추가/교체 가능.

참고:
  - Stawarz (2025) "Crypto Trading via Gradient Boosting": LGB > XGB > RF
  - Lopez de Prado (2018) "Advances in Financial ML" Ch.7 Purged K-Fold
  - Lin et al. (2017) Focal Loss for Dense Object Detection
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight


class LightGBMPredictor:
    """LightGBM 시장 방향 예측기 — M2 Apple Silicon 최적화

    XGBoostPredictor와 동일한 공개 인터페이스:
      - train(df, feature_cols, label_col)
      - train_walkforward(df, feature_cols, label_col, n_splits, purge_gap, ...)
      - predict(df) → {"signal", "confidence", "direction", "probabilities"}
      - save(name) / load(name)
    """

    def __init__(self, model_dir: str = "models_saved"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.feature_columns: list[str] = []
        self.accuracy: float = 0.0
        self.f1: float = 0.0
        self.feature_importance: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 가중치 헬퍼
    # ------------------------------------------------------------------
    def _get_time_weights(self, n_samples: int, decay: float = 0.998) -> np.ndarray:
        """시간 가중치 — 최근 데이터에 더 높은 가중치 (지수 감쇠)"""
        w = np.array([decay ** (n_samples - i - 1) for i in range(n_samples)])
        m = w.mean()
        return w / m if m > 0 else np.ones_like(w)

    def _get_focal_sample_weight(self, y: np.ndarray, gamma: float = 2.0) -> np.ndarray:
        """Focal-style 샘플 가중 — 소수 클래스(횡보)에 강한 가중"""
        y_int = y.astype(int)
        class_counts = np.bincount(y_int, minlength=3).astype(float)
        total = max(len(y_int), 1)
        class_freq = class_counts / total

        w = np.ones(len(y_int), dtype=float)
        for cls in range(3):
            mask = y_int == cls
            if not mask.any():
                continue
            f = class_freq[cls]
            if f <= 0:
                continue
            focal = (1 - f) ** gamma
            w[mask] = focal / f
        m = w.mean()
        return w / m if m > 0 else w

    def _composite_weights(self, y: np.ndarray) -> np.ndarray:
        """time_decay × focal × balanced — 우리 시스템 통합 가중치"""
        n = len(y)
        if n == 0:
            return np.ones(0)
        time_w = self._get_time_weights(n)
        focal_w = self._get_focal_sample_weight(y)
        # 기존 XGB 호환: balanced도 곱해서 하락 클래스 편향 추가 보정
        try:
            bal_w = compute_sample_weight(class_weight="balanced", y=y)
        except Exception:
            bal_w = np.ones(n)
        composite = time_w * focal_w * bal_w
        m = composite.mean()
        return composite / m if m > 0 else composite

    # ------------------------------------------------------------------
    # CV 헬퍼
    # ------------------------------------------------------------------
    def _purged_kfold_split(
        self, n_samples: int, n_splits: int = 5, embargo_pct: float = 0.01
    ):
        """Purged K-Fold (Lopez de Prado) — 인덱스 기반 간이 구현"""
        embargo_size = max(1, int(n_samples * embargo_pct))
        fold_size = n_samples // n_splits

        for i in range(n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            train_end = max(0, test_start - embargo_size)
            train_start_after = min(n_samples, test_end + embargo_size)
            train_idx = np.concatenate([
                np.arange(0, train_end),
                np.arange(train_start_after, n_samples),
            ]).astype(int)
            test_idx = np.arange(test_start, test_end).astype(int)
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    # ------------------------------------------------------------------
    # 자동 피처 선택
    # ------------------------------------------------------------------
    def _auto_select_features(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str], top_k: int = 30
    ) -> list[int]:
        """1차 빠른 학습 → importance 상위 top_k 피처만 사용"""
        try:
            import lightgbm as lgb

            quick = lgb.LGBMClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                num_leaves=15, verbose=-1, n_jobs=-1,
            )
            quick.fit(X, y)
            imps = quick.feature_importances_
            top_indices = np.argsort(imps)[-top_k:]
            self.feature_importance = {
                feature_names[i]: float(imps[i]) for i in top_indices
            }
            logger.info(
                f"[LGB피처선택] {len(feature_names)}→{len(top_indices)} "
                f"(top: {feature_names[int(top_indices[-1])]}={imps[top_indices[-1]]:.0f})"
            )
            return sorted(int(i) for i in top_indices)
        except Exception as e:
            logger.warning(f"[LGB피처선택] 실패: {e} → 전체 사용")
            return list(range(len(feature_names)))

    # ------------------------------------------------------------------
    # 학습 (단일 80/20 + 옵션 PurgedCV best_iter)
    # ------------------------------------------------------------------
    def _build_params(self) -> dict:
        return {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 500,
            "max_depth": 7,
            "num_leaves": 63,           # 2^max_depth-1, leaf-wise 적합
            "learning_rate": 0.03,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_split_gain": 0.01,
            "verbose": -1,
            "n_jobs": -1,
            "force_col_wise": True,    # M2 캐시 친화
            "max_bin": 127,            # M2 메모리 최적화
        }

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "label",
        use_purged_kfold: bool = True,
        auto_select_features: bool = True,
    ):
        """80/20 학습 + 옵션 Purged K-Fold로 best_iter 결정"""
        import lightgbm as lgb

        self.feature_columns = feature_cols
        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(int)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 자동 피처 선택
        if auto_select_features and len(feature_cols) > 30:
            sel = self._auto_select_features(X, y, feature_cols)
            X = X[:, sel]
            self.feature_columns = [feature_cols[i] for i in sel]

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        sample_weight = self._composite_weights(y_train)

        params = self._build_params()
        prev_accuracy = self.accuracy

        # 피처 수 변경 시 모델 리셋
        if self.model is not None:
            try:
                prev_n_feat = self.model.n_features_in_
                if prev_n_feat != X_train.shape[1]:
                    logger.warning(
                        f"LightGBM 피처 수 불일치: {prev_n_feat} → {X_train.shape[1]} → 재학습"
                    )
                    self.model = None
                    prev_accuracy = 0.0
            except Exception:
                pass

        if self.model is not None:
            # 증분학습 (보수 학습률)
            logger.info(f"LightGBM 증분학습 (이전 acc: {prev_accuracy:.4f})")
            params["n_estimators"] = 200
            params["learning_rate"] = 0.01
            new_model = lgb.LGBMClassifier(**params)
            try:
                new_model.fit(
                    X_train, y_train,
                    sample_weight=sample_weight,
                    eval_set=[(X_test, y_test)],
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                    init_model=self.model.booster_,
                )
            except Exception as e:
                # init_model 실패 시 처음부터 다시 학습
                logger.warning(f"LightGBM 증분학습 실패({e}) → 최초학습 fallback")
                new_model = lgb.LGBMClassifier(**self._build_params())
                new_model.fit(
                    X_train, y_train, sample_weight=sample_weight,
                    eval_set=[(X_test, y_test)],
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
            self.model = new_model
        else:
            logger.info("LightGBM 최초학습 시작")
            if use_purged_kfold and len(X_train) >= 5 * 100:
                best_rounds = []
                for tr_idx, va_idx in self._purged_kfold_split(len(X_train), n_splits=5):
                    fold = lgb.LGBMClassifier(**{**params, "n_estimators": 1000})
                    fold.fit(
                        X_train[tr_idx], y_train[tr_idx],
                        sample_weight=sample_weight[tr_idx],
                        eval_set=[(X_train[va_idx], y_train[va_idx])],
                        callbacks=[lgb.early_stopping(20, verbose=False)],
                    )
                    if fold.best_iteration_:
                        best_rounds.append(fold.best_iteration_)
                if best_rounds:
                    optimal = int(np.median(best_rounds))
                    params["n_estimators"] = max(100, optimal)
                    logger.info(f"[LGB-PurgedCV] 최적 라운드: {optimal} (folds={best_rounds})")

            self.model = lgb.LGBMClassifier(**params)
            self.model.fit(
                X_train, y_train, sample_weight=sample_weight,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

        # 평가
        y_pred = self.model.predict(X_test)
        new_acc = float(accuracy_score(y_test, y_pred))
        new_f1 = float(f1_score(y_test, y_pred, average="weighted"))

        # 5%p 하락 → 롤백 (저장된 모델 다시 로드)
        if prev_accuracy > 0 and new_acc < prev_accuracy - 0.05:
            logger.warning(
                f"LightGBM 정확도 하락 {prev_accuracy:.4f}→{new_acc:.4f} → 롤백"
            )
            if self.load():
                return self.accuracy

        self.accuracy = new_acc
        self.f1 = new_f1

        if hasattr(self.model, "feature_importances_"):
            for i, imp in enumerate(self.model.feature_importances_):
                if i < len(self.feature_columns):
                    self.feature_importance[self.feature_columns[i]] = float(imp)

        improvement = f" ({new_acc - prev_accuracy:+.4f})" if prev_accuracy > 0 else ""
        logger.info(
            f"LightGBM 학습 완료 — Acc={new_acc:.4f}{improvement} F1={new_f1:.4f}\n"
            f"{classification_report(y_test, y_pred, target_names=['하락','횡보','상승'], zero_division=0)}"
        )
        return self.accuracy

    # ------------------------------------------------------------------
    # Walk-forward (XGB와 동일 시그니처)
    # ------------------------------------------------------------------
    def train_walkforward(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "label",
        n_splits: int = 5,
        purge_gap: int = 12,
        use_purged_kfold: bool = False,
        embargo_pct: float = 0.01,
    ) -> float:
        import lightgbm as lgb

        self.feature_columns = feature_cols
        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(int)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if len(X) < n_splits * 100:
            logger.warning(
                f"[WalkForward-LGB] 샘플 부족 ({len(X)} < {n_splits * 100}) → train() fallback"
            )
            return self.train(df, feature_cols, label_col)

        if use_purged_kfold:
            try:
                from core.ml.cv import PurgedKFold
                t1 = df["tb_t1"].values if "tb_t1" in df.columns else None
                cv = PurgedKFold(
                    n_splits=n_splits, embargo_pct=embargo_pct, purge_bars=purge_gap
                )
                splits = list(cv.split(X, y, t1=t1))
                logger.info(
                    f"[WalkForward-LGB] PurgedKFold (n={n_splits}, embargo={embargo_pct:.1%}, "
                    f"purge={purge_gap}, folds={len(splits)})"
                )
            except Exception as e:
                logger.warning(f"[WalkForward-LGB] PurgedKFold 로드 실패({e}) → TSS")
                tscv = TimeSeriesSplit(n_splits=n_splits)
                splits = list(tscv.split(X))
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = list(tscv.split(X))
            logger.info(f"[WalkForward-LGB] TSS (n={n_splits}, purge_gap={purge_gap})")

        fold_accs: list[float] = []
        last_model = None

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if not use_purged_kfold and len(train_idx) > purge_gap:
                train_idx = train_idx[:-purge_gap]

            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            sw_tr = self._composite_weights(y_tr)

            params = self._build_params()
            fold_model = lgb.LGBMClassifier(**params)
            fold_model.fit(
                X_tr, y_tr, sample_weight=sw_tr,
                eval_set=[(X_te, y_te)],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            y_pred = fold_model.predict(X_te)
            acc = float(accuracy_score(y_te, y_pred))
            fold_accs.append(acc)
            last_model = fold_model
            cv_name = "PurgedKFold" if use_purged_kfold else "TSS"
            logger.info(
                f"[WalkForward-LGB][{cv_name}] Fold {fold_idx+1}/{n_splits}: "
                f"train={len(train_idx)} test={len(test_idx)} acc={acc:.4f}"
            )

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        logger.info(
            f"[WalkForward-LGB] 평균 acc={mean_acc:.4f} ± {std_acc:.4f} "
            f"(min={min(fold_accs):.4f}, max={max(fold_accs):.4f})"
        )

        # 성능 가드
        if self.accuracy > 0 and mean_acc < self.accuracy - 0.05:
            logger.warning(
                f"[WalkForward-LGB] OOS {mean_acc:.4f} < 기존 {self.accuracy:.4f}-0.05 → 롤백"
            )
            if self.load():
                return self.accuracy

        self.model = last_model
        self.accuracy = mean_acc
        if hasattr(self.model, "feature_importances_"):
            for i, imp in enumerate(self.model.feature_importances_):
                if i < len(self.feature_columns):
                    self.feature_importance[self.feature_columns[i]] = float(imp)
        return mean_acc

    # ------------------------------------------------------------------
    # 추론
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> dict:
        """XGB와 동일한 dict 형식 반환"""
        if self.model is None:
            return {
                "signal": 0.0, "confidence": 0.0, "direction": "neutral",
                "probabilities": {"short": 0.0, "neutral": 0.0, "long": 0.0},
            }
        try:
            X = df[self.feature_columns].values.astype(np.float32)[-1:]
        except KeyError:
            # feature 누락 → fallback (모든 feature 0)
            X = np.zeros((1, len(self.feature_columns)), dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        proba = self.model.predict_proba(X)[0]
        pred = int(np.argmax(proba))
        direction_map = {0: "short", 1: "neutral", 2: "long"}
        signal = float(proba[2] - proba[0])  # [-1, 1]
        return {
            "signal": signal,
            "confidence": float(proba[pred]),
            "direction": direction_map[pred],
            "probabilities": {
                "short": float(proba[0]),
                "neutral": float(proba[1]),
                "long": float(proba[2]),
            },
        }

    def get_feature_importance(self) -> dict[str, float]:
        return dict(self.feature_importance)

    # ------------------------------------------------------------------
    # 영속화
    # ------------------------------------------------------------------
    def save(self, name: str = "lightgbm"):
        if self.model is None:
            return
        path = self.model_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "features": self.feature_columns,
                "accuracy": self.accuracy,
                "f1": self.f1,
                "feature_importance": self.feature_importance,
            }, f)
        logger.info(f"LightGBM 저장: {path} ({path.stat().st_size/1024:.0f}KB)")

    def load(self, name: str = "lightgbm") -> bool:
        path = self.model_dir / f"{name}.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.feature_columns = data.get("features") or data.get("feature_columns", [])
            self.accuracy = data.get("accuracy", 0.0)
            self.f1 = data.get("f1", 0.0)
            self.feature_importance = data.get("feature_importance", {})
            logger.info(f"LightGBM 로드: {path} (Acc: {self.accuracy:.4f})")
            return True
        except Exception as e:
            logger.error(f"LightGBM 로드 실패: {e}")
            return False
