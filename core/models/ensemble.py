"""ML 모델 앙상블 - 다수 모델 시그널을 통합

[2026-04-25] LightGBM 추가 — XGBoost와 병렬 운영(IC 비교) → 우수한 모델로 자동 가중.
- Stawarz (2025): 크립토 트레이딩에서 LGB > XGB > RF 보고
- M2 최적화: 히스토그램 학습 + max_bin=127 + force_col_wise
- 인터페이스는 XGBoostPredictor와 동일 → ensemble 내부에서만 통합
"""

import numpy as np
import pandas as pd
from loguru import logger

from core.models.lstm_model import LSTMPredictor
from core.models.xgboost_model import XGBoostPredictor

try:
    from core.models.lightgbm_model import LightGBMPredictor
    _LGB_AVAILABLE = True
except Exception as _lgb_err:  # pragma: no cover
    LightGBMPredictor = None  # type: ignore
    _LGB_AVAILABLE = False
    logger.warning(f"[Ensemble] LightGBM 비활성화 (import 실패: {_lgb_err})")

try:
    from core.models.cnn_attention_model import CNNAttentionPredictor
    _CNN_AVAILABLE = True
except Exception as _cnn_err:  # pragma: no cover
    CNNAttentionPredictor = None  # type: ignore
    _CNN_AVAILABLE = False
    logger.warning(f"[Ensemble] CNN-Attention 비활성화 (import 실패: {_cnn_err})")


class EnsembleSignalGenerator:
    """XGBoost + LightGBM + LSTM 앙상블 시그널 생성기"""

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
        # LightGBM은 옵션 — import 실패 시 무시
        self.lgb = LightGBMPredictor(model_dir=model_dir) if _LGB_AVAILABLE else None
        self.has_lgb = self.lgb is not None
        # CNN-Attention 6th vote source (옵션)
        self.cnn = CNNAttentionPredictor(model_dir=model_dir) if _CNN_AVAILABLE else None
        self.has_cnn = self.cnn is not None

        # 초기 가중치 — 활성 모델 균등 분배
        active = ["xgboost", "lstm"]
        if self.has_lgb:
            active.append("lightgbm")
        if self.has_cnn:
            active.append("cnn_attention")
        w = 1.0 / len(active)
        self.weights = {name: w for name in active}
        self._performance_history: list[dict] = []

    # ------------------------------------------------------------------
    def train_all(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        walk_forward: bool = False,
        use_purged_kfold: bool = False,
        embargo_pct: float = 0.01,
    ):
        """모든 모델 학습"""
        if walk_forward:
            cv_name = "PurgedKFold+Embargo" if use_purged_kfold else "TimeSeriesSplit"
            logger.info(f"=== 앙상블 모델 Walk-Forward CV 학습 시작 ({cv_name}) ===")
            xgb_acc = self.xgb.train_walkforward(
                df, feature_cols,
                use_purged_kfold=use_purged_kfold, embargo_pct=embargo_pct,
            )
            lstm_acc = self.lstm.train_walkforward(df, feature_cols)
            if self.has_lgb:
                try:
                    lgb_acc = self.lgb.train_walkforward(
                        df, feature_cols,
                        use_purged_kfold=use_purged_kfold, embargo_pct=embargo_pct,
                    )
                except Exception as e:
                    logger.warning(f"[Ensemble] LightGBM walkforward 실패({e}) → 단일 학습")
                    lgb_acc = self.lgb.train(df, feature_cols)
            else:
                lgb_acc = 0.0
        else:
            logger.info("=== 앙상블 모델 학습 시작 ===")
            xgb_acc = self.xgb.train(df, feature_cols)
            lstm_acc = self.lstm.train(df, feature_cols)
            if self.has_lgb:
                try:
                    lgb_acc = self.lgb.train(df, feature_cols)
                except Exception as e:
                    logger.warning(f"[Ensemble] LightGBM 학습 실패: {e}")
                    lgb_acc = 0.0
            else:
                lgb_acc = 0.0

        # CNN-Attention 학습 (시퀀스 기반 → walk_forward 미지원, 단일 학습만)
        cnn_acc = 0.0
        if self.has_cnn:
            try:
                cnn_acc = float(self.cnn.train(df, feature_cols) or 0.0)
            except Exception as e:
                logger.warning(f"[Ensemble] CNN-Attention 학습 실패: {e}")
                cnn_acc = 0.0

        # 정확도 기반 가중치 자동 조정 (활성 모델만)
        accs = {"xgboost": xgb_acc or 0.0, "lstm": lstm_acc or 0.0}
        if self.has_lgb and (lgb_acc or 0) > 0:
            accs["lightgbm"] = lgb_acc
        if self.has_cnn and cnn_acc > 0:
            accs["cnn_attention"] = cnn_acc
        total = sum(accs.values())
        if total > 0:
            for k, v in accs.items():
                self.weights[k] = v / total
        logger.info(
            "앙상블 가중치 — " + ", ".join(
                f"{k}:{self.weights.get(k,0):.3f}" for k in accs
            )
        )

    def predict(self, df: pd.DataFrame, regime: str | None = None) -> dict:
        """앙상블 시그널 생성 (XGB + LSTM + 옵션 LGB + 옵션 CNN-Attn 가중합)"""
        preds: dict[str, dict] = {
            "xgboost": self.xgb.predict(df),
            "lstm": self.lstm.predict(df),
        }
        if self.has_lgb and self.lgb is not None:
            try:
                preds["lightgbm"] = self.lgb.predict(df)
            except Exception as e:
                logger.debug(f"[Ensemble] LGB predict 실패: {e}")
        if self.has_cnn and self.cnn is not None:
            try:
                preds["cnn_attention"] = self.cnn.predict(df)
            except Exception as e:
                logger.debug(f"[Ensemble] CNN predict 실패: {e}")

        # 활성 모델만 가중치 정규화
        w = {k: float(self.weights.get(k, 0.0)) for k in preds}
        s = sum(w.values())
        if s > 0:
            w = {k: v / s for k, v in w.items()}
        else:
            # 모든 가중치 0이면 균등
            n = len(preds)
            w = {k: 1.0 / n for k in preds}

        combined_signal = sum(preds[k]["signal"] * w[k] for k in preds)
        combined_confidence = sum(preds[k]["confidence"] * w[k] for k in preds)

        # 레짐 가중치 (raw vs final)
        raw_signal = combined_signal
        regime_mult = self.REGIME_SIGNAL_WEIGHT.get(regime, 1.0) if regime else 1.0
        combined_signal = combined_signal * regime_mult

        # 방향 결정
        if combined_signal > 0.15:
            direction = "long"
        elif combined_signal < -0.15:
            direction = "short"
        else:
            direction = "neutral"

        # 합의도: 같은 방향 모델 비율
        dirs = [p["direction"] for p in preds.values()]
        non_neutral = [d for d in dirs if d != "neutral"]
        if not non_neutral:
            agreement = 0.5
        else:
            from collections import Counter
            cnt = Counter(non_neutral)
            top = cnt.most_common(1)[0][1]
            agreement = top / len(dirs)

        return {
            "signal": float(combined_signal),
            "raw_signal": float(raw_signal),
            "regime_multiplier": float(regime_mult),
            "regime": regime,
            "confidence": float(combined_confidence * agreement),
            "direction": direction,
            "agreement": float(agreement),
            "models": preds,
            "weights_used": {k: float(v) for k, v in w.items()},
        }

    def update_weights(self, model_name: str, recent_accuracy: float):
        """최근 성능 기반 가중치 동적 조정 — 활성 모델만 고려"""
        self._performance_history.append({"model": model_name, "accuracy": recent_accuracy})
        recent = self._performance_history[-30:]

        active = ["xgboost", "lstm"]
        if self.has_lgb:
            active.append("lightgbm")
        if self.has_cnn:
            active.append("cnn_attention")

        avgs = {}
        for name in active:
            scores = [p["accuracy"] for p in recent if p["model"] == name]
            avgs[name] = float(np.mean(scores)) if scores else 0.5

        total = sum(avgs.values())
        if total > 0:
            for k, v in avgs.items():
                self.weights[k] = v / total
            logger.info(
                "앙상블 가중치 업데이트 — " +
                ", ".join(f"{k}:{self.weights[k]:.3f}" for k in active)
            )

    def save_all(self):
        self.xgb.save()
        self.lstm.save()
        if self.has_lgb and self.lgb is not None:
            try:
                self.lgb.save()
            except Exception as e:
                logger.warning(f"[Ensemble] LGB 저장 실패: {e}")
        if self.has_cnn and self.cnn is not None:
            try:
                self.cnn.save()
            except Exception as e:
                logger.warning(f"[Ensemble] CNN 저장 실패: {e}")

    def load_all(self) -> bool:
        xgb_ok = self.xgb.load()
        lstm_ok = self.lstm.load()
        if self.has_lgb and self.lgb is not None:
            try:
                lgb_ok = self.lgb.load()
                if lgb_ok:
                    logger.info(f"[Ensemble] LGB 로드 성공 (acc={self.lgb.accuracy:.4f})")
            except Exception as e:
                logger.debug(f"[Ensemble] LGB 로드 실패(첫 가동일 수 있음): {e}")
        if self.has_cnn and self.cnn is not None:
            try:
                cnn_ok = self.cnn.load()
                if cnn_ok:
                    logger.info(f"[Ensemble] CNN 로드 성공 (acc={self.cnn.accuracy:.4f})")
            except Exception as e:
                logger.debug(f"[Ensemble] CNN 로드 실패(첫 가동일 수 있음): {e}")
        # 핵심 두 모델만 필수 — LGB/CNN은 옵션
        return xgb_ok and lstm_ok
