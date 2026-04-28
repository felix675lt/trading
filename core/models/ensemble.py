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

        # [Phase K, 2026-04-25] IC 기반 가중치 — predict() 시점의 per-model signal을
        # 보관하여, 거래 청산 시점에 (signal, realized_return) 쌍을 ICTracker에
        # source="model_xgb"/"model_lstm"/"model_lgb"/"model_cnn"으로 기록하기 위한 버퍼.
        # apply_ic_weights() 가 IC를 읽어 가중치를 정확도→IC로 교체.
        self.last_per_model_signals: dict[str, float] = {}
        # IC 가중 모드 ON/OFF — apply_ic_weights() 호출 시 True로 전환. False면 정확도 가중 유지.
        self._ic_weighting_active: bool = False

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
        # [Patch I, 2026-04-28] 모든 모델 predict를 try-except로 감싸 한 모델 실패 시
        # 전체 분석이 죽는 것을 방지 (특히 ext_llm_* 누락 컬럼 KeyError 케이스).
        preds: dict[str, dict] = {}
        try:
            preds["xgboost"] = self.xgb.predict(df)
        except Exception as e:
            logger.warning(f"[Ensemble] XGB predict 실패: {e} → 가중치 0으로 우회")
        try:
            preds["lstm"] = self.lstm.predict(df)
        except Exception as e:
            logger.warning(f"[Ensemble] LSTM predict 실패: {e} → 가중치 0으로 우회")
        if not preds:
            # 모든 핵심 모델 실패 시 neutral 신호 반환
            return {
                "signal": 0.0, "confidence": 0.0, "direction": "neutral",
                "raw_signal": 0.0, "regime_multiplier": 1.0,
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

        # [Patch I, 2026-04-28] NaN 출력 모델 자동 제외 — corrupt weights(LSTM 등)가
        # 합산 시 전체 신호를 NaN으로 만드는 문제 차단.
        import math
        bad = []
        for k, p in list(preds.items()):
            sig = p.get("signal", 0.0)
            conf = p.get("confidence", 0.0)
            if (sig is None or math.isnan(float(sig)) or math.isinf(float(sig))
                    or conf is None or math.isnan(float(conf))):
                bad.append(k)
                preds.pop(k)
        if bad and not getattr(self, "_nan_model_warned", False):
            logger.warning(
                f"[Ensemble] NaN 신호 모델 자동 제외 — {bad}. 해당 모델 재학습 필요 "
                f"(weights corrupt 가능성)."
            )
            self._nan_model_warned = True
        if not preds:
            return {
                "signal": 0.0, "confidence": 0.0, "direction": "neutral",
                "raw_signal": 0.0, "regime_multiplier": 1.0,
            }

        # 활성 모델만 가중치 정규화
        w = {k: float(self.weights.get(k, 0.0)) for k in preds}
        s = sum(w.values())
        if s > 0:
            w = {k: v / s for k, v in w.items()}
        else:
            # 모든 가중치 0이면 균등
            n = len(preds)
            w = {k: 1.0 / n for k in preds}

        # [Phase K] per-model signal 버퍼 갱신 — 청산 시점에 IC 기록 재료
        self.last_per_model_signals = {k: float(preds[k]["signal"]) for k in preds}

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

    def apply_ic_weights(
        self,
        ic_tracker,
        min_samples: int = 20,
        smoothing: float = 0.5,
    ) -> dict:
        """[Phase K, 2026-04-25] 모델별 IC 기반 가중치 갱신.

        Manus v3 (2026-04-25) 지적: 기존 self.weights는 *정확도* 비례.
        정확도와 IC는 다름 — 60% 정확도라도 맞을 때 5bp / 틀릴 때 50bp이면 IC<0.
        IC는 시그널 강도까지 반영하는 rank correlation이므로, 손익에 더 직결.

        ICTracker가 source="model_xgb"/"model_lstm"/"model_lgb"/"model_cnn"으로
        축적한 (signal, realized_return) 페어에서 IC를 읽어 가중치를 재계산한다.
        샘플 부족(min_samples<20) 모델은 정확도 가중치를 유지(혼합).

        Args:
            ic_tracker: ICTracker 인스턴스
            min_samples: IC 신뢰 임계 샘플 수
            smoothing: 신·구 가중치 지수평활 (0=교체 안함, 1=즉시 교체)

        Returns:
            {"before": {...}, "after": {...}, "ic": {model: ic_value}, "n": {model: n_samples}}
        """
        # 활성 모델 결정
        active = ["xgboost", "lstm"]
        if self.has_lgb:
            active.append("lightgbm")
        if self.has_cnn:
            active.append("cnn_attention")

        # 모델별 IC 조회
        source_map = {
            "xgboost": "model_xgb",
            "lstm": "model_lstm",
            "lightgbm": "model_lgb",
            "cnn_attention": "model_cnn",
        }
        ics: dict[str, float] = {}
        ns: dict[str, int] = {}
        for m in active:
            src = source_map.get(m, m)
            try:
                recs = ic_tracker._select(src, None)  # 내부 헬퍼 — 같은 모듈 호환
                n = len(recs)
                ns[m] = n
                if n >= min_samples:
                    ic_val = ic_tracker.ic(source=src)
                    ics[m] = float(ic_val) if np.isfinite(ic_val) else 0.0
                else:
                    ics[m] = 0.0
            except Exception as e:
                logger.debug(f"[Ensemble-IC] {m} IC 조회 실패: {e}")
                ics[m] = 0.0
                ns[m] = 0

        # 충분한 샘플이 있는 모델이 2개 미만이면 IC 가중 보류
        sufficient = [m for m in active if ns.get(m, 0) >= min_samples]
        if len(sufficient) < 2:
            logger.debug(
                f"[Ensemble-IC] 샘플 부족 — IC 가중 보류 (sufficient={len(sufficient)}/{len(active)})"
            )
            return {
                "before": dict(self.weights),
                "after": dict(self.weights),
                "ic": ics, "n": ns,
                "applied": False,
            }

        # IC → 가중치 매핑:
        #   IC > 0  : 양의 알파 (가중치 후보)
        #   IC <= 0 : 비유의/음의 알파 → 0 (그러나 모두 0 방지를 위해 floor=0.05)
        targets: dict[str, float] = {}
        for m in active:
            ic = ics.get(m, 0.0)
            n = ns.get(m, 0)
            if n < min_samples:
                # 샘플 부족 모델은 현재 가중치 유지(샘플 모일 때까지)
                targets[m] = float(self.weights.get(m, 1.0 / len(active)))
                continue
            # IC 양수만 살리고, 음/0은 최소 floor (완전 0이면 모델 죽음 → 회복 불가)
            score = max(ic, 0.0) + 0.05  # floor
            targets[m] = score

        total = sum(targets.values())
        if total <= 0:
            logger.warning("[Ensemble-IC] 모든 IC 0 이하 — 가중치 갱신 보류")
            return {
                "before": dict(self.weights),
                "after": dict(self.weights),
                "ic": ics, "n": ns,
                "applied": False,
            }

        target_weights = {m: targets[m] / total for m in active}

        # 지수평활 — 급격한 변경 방지
        s = max(0.0, min(1.0, smoothing))
        before = dict(self.weights)
        for m in active:
            prev = float(self.weights.get(m, 1.0 / len(active)))
            new = (1 - s) * prev + s * target_weights[m]
            self.weights[m] = new

        # 정규화 보정
        wt = sum(self.weights[m] for m in active)
        if wt > 0:
            for m in active:
                self.weights[m] = self.weights[m] / wt

        self._ic_weighting_active = True

        logger.info(
            "[Ensemble-IC] IC 기반 가중치 갱신 — "
            + ", ".join(
                f"{m}:{self.weights[m]:.3f}(IC={ics[m]:+.4f},n={ns[m]})" for m in active
            )
        )
        return {
            "before": before,
            "after": dict(self.weights),
            "ic": ics, "n": ns,
            "applied": True,
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
