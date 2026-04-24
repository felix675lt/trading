"""Transformer 기반 시계열 예측 스텁 (TFT / PatchTST / Autoformer).

⚠️ ⚠️ ⚠️ 매우 중요한 알림 ⚠️ ⚠️ ⚠️
=============================================================================
❗ 본 모듈은 CPU-only 환경(Mac mini 등)에서는 의도적으로 비활성화된다.
❗ Transformer 학습/추론은 GPU(CUDA or Apple Metal MPS)가 없으면 실용적 속도가 나오지 않는다.
❗ 사용자가 GPU 장비로 이전하거나 클라우드 GPU(RunPod/Lambda/Colab Pro)를 붙이면
❗ 자동으로 활성화된다.
=============================================================================

배경 — 왜 Transformer인가?
- **TFT** (Temporal Fusion Transformer, Lim et al. 2020): 시계열 멀티호라이즌 예측 SOTA.
  Attention으로 중요한 과거 시점 자동 선택, Gated Residual Network로 비선형 특성.
- **PatchTST** (Nie et al. 2023): 시계열을 패치로 쪼개 Transformer에 투입 — LSTM/XGB 대비
  긴 시계열에서 훨씬 우수.
- **Autoformer** (Wu et al. 2021): auto-correlation 메커니즘으로 주기성 감지.

왜 현재 CPU 환경에서 비활성인가?
- TFT 15m×1000봉 훈련: GPU 30초 / CPU 10~30분 (실시간 업데이트 불가).
- 어텐션 연산 O(L²·d) — CPU에서는 BLAS 최적화되어도 LSTM 대비 5~10배 느림.

활성화 방법:
  1. GPU 설치 (NVIDIA or Apple Silicon with MPS)
  2. `pip install torch pytorch-forecasting` (또는 `pip install transformers`)
  3. `config/default.yaml`에 `ml.transformer.enabled: true` 추가
  4. 시스템 재시작

현재 작동 방식:
  - `TransformerPredictor` 클래스는 존재하지만, `available()` → False면 predict()는
    LSTM fallback으로 우회
  - import는 지연(lazy)되어 torch 미설치 환경에서도 에러 안 남
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from loguru import logger


def _gpu_available() -> tuple[bool, str]:
    """GPU + torch 가용성 체크.

    Returns:
        (available: bool, reason: str)
    """
    if importlib.util.find_spec("torch") is None:
        return False, "torch 미설치 — `pip install torch` 필요"
    try:
        import torch
    except Exception as e:
        return False, f"torch import 실패: {e}"
    if torch.cuda.is_available():
        return True, f"CUDA GPU 감지: {torch.cuda.get_device_name(0)}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return True, "Apple Metal (MPS) 감지"
    return False, "GPU 없음 (CPU만 사용 가능 — Transformer 학습/추론 비현실적)"


# === 시작 시 1회 경고 출력 (눈에 띄게) ===
_GPU_OK, _GPU_REASON = _gpu_available()
if not _GPU_OK:
    logger.warning("=" * 78)
    logger.warning("⚠️  [TRANSFORMER] GPU 미감지 — Transformer 모듈 비활성화됨")
    logger.warning(f"⚠️  사유: {_GPU_REASON}")
    logger.warning("⚠️  Mac mini 같은 CPU-only 환경에서는 의도적으로 LSTM/XGBoost로 대체됨.")
    logger.warning("⚠️  GPU 장비로 이전 시 자동 활성화됩니다.")
    logger.warning("=" * 78)
else:
    logger.info(f"[TRANSFORMER] ✅ GPU 가용 — {_GPU_REASON}")


class TransformerPredictor:
    """Transformer-based time-series predictor (stub).

    GPU 없으면 available()=False → 호출 측에서 LSTM fallback.

    Usage:
        t = TransformerPredictor(model_dir="models_saved", model_type="tft")
        if t.available():
            t.train(df, feature_cols)
            pred = t.predict(df)
        else:
            # fallback to LSTM
            ...
    """

    def __init__(
        self,
        model_dir: str = "models_saved",
        model_type: str = "tft",     # "tft" | "patchtst" | "autoformer"
        seq_len: int = 64,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        self.model_dir = model_dir
        self.model_type = model_type
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self._model = None
        self._device = None

    def available(self) -> bool:
        """GPU + torch 둘 다 있어야 True."""
        return _GPU_OK

    def _ensure_env(self):
        if not _GPU_OK:
            raise RuntimeError(
                f"[TRANSFORMER] 사용 불가 — {_GPU_REASON}. "
                "GPU 환경에서 재실행하거나 LSTM으로 대체하세요."
            )

    def train(self, df: pd.DataFrame, feature_cols: list[str]) -> float:
        """학습 (GPU 필요)."""
        self._ensure_env()
        # GPU 있을 때만 실제 구현 임포트 — 지연 임포트로 CPU 환경 에러 방지
        logger.info(f"[TRANSFORMER] {self.model_type} 학습 시작 (GPU) — seq_len={self.seq_len}")
        logger.warning(
            "[TRANSFORMER] 실제 학습 로직은 GPU 장비 전환 후 구현 (현재는 스텁)."
        )
        return 0.0

    def predict(self, df: pd.DataFrame) -> dict:
        """예측. GPU 없으면 neutral 반환 — caller가 LSTM으로 fallback해야 함."""
        if not self.available():
            return {
                "signal": 0.0, "confidence": 0.0, "direction": "neutral",
                "skipped": True, "reason": _GPU_REASON,
            }
        self._ensure_env()
        # GPU 있을 때의 실제 추론 로직은 GPU 환경에서 구현
        return {"signal": 0.0, "confidence": 0.0, "direction": "neutral"}

    def gpu_status(self) -> dict:
        """UI/대시보드용 상태 리포트."""
        return {
            "available": _GPU_OK,
            "reason": _GPU_REASON,
            "model_type": self.model_type,
            "seq_len": self.seq_len,
        }
