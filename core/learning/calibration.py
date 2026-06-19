"""[Patch Y, 2026-06-19] 확률 캘리브레이션 — 신뢰도를 실제 승률에 정렬.

문제: 앙상블 conf 0.62가 실제 방향 적중률 ~34%인 과신(over-confidence).
      → Kelly 사이징(WR 입력)·진입 게이트가 잘못된 확률 위에서 작동.

해법: isotonic regression으로 raw_conf → P(적중) 단조 매핑을 과거 데이터로 적합.
      - 학습 라벨: 신호 시점 대비 1h 뒤 가격이 예측 방향으로 움직였는가(적중).
      - 추론 시: calibrated = transform(raw_conf).

안전장치:
  - out_of_bounds='clip' (외삽 금지).
  - 데이터 부족/적합 실패 시 identity 폴백(raw 그대로) — 절대 죽지 않음.
  - downward_only=True: 캘리브레이션이 conf를 올리는(과신 심화) 방향이면 무시,
    낮추는 방향만 반영 → LIVE 리스크는 단조 감소만.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


class ConfidenceCalibrator:
    def __init__(self, downward_only: bool = True):
        self.downward_only = downward_only
        self._x: Optional[np.ndarray] = None  # 정렬된 raw conf 그리드
        self._y: Optional[np.ndarray] = None  # 대응 calibrated prob
        self.n_samples: int = 0
        self.fitted: bool = False

    # ------------------------------------------------------------------
    def fit(self, conf: np.ndarray, correct: np.ndarray) -> bool:
        """conf: raw 신뢰도(0~1), correct: 0/1 적중 라벨."""
        try:
            from sklearn.isotonic import IsotonicRegression
        except Exception as e:
            logger.warning(f"[Calibration] sklearn 없음 — 적합 불가: {e}")
            return False

        conf = np.asarray(conf, dtype=float)
        correct = np.asarray(correct, dtype=float)
        mask = np.isfinite(conf) & np.isfinite(correct)
        conf, correct = conf[mask], correct[mask]
        if len(conf) < 200:
            logger.warning(f"[Calibration] 표본 부족 ({len(conf)}) — 적합 보류")
            return False

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(conf, correct)
        # 매핑을 그리드로 저장 (sklearn 의존 없이 transform 가능)
        grid = np.linspace(0.0, 1.0, 101)
        mapped = iso.predict(grid)
        self._x = grid
        self._y = np.clip(mapped, 0.0, 1.0)
        self.n_samples = int(len(conf))
        self.fitted = True
        logger.info(
            f"[Calibration] 적합 완료 — n={self.n_samples} | "
            f"0.5→{self.transform(0.5):.3f} 0.6→{self.transform(0.6):.3f} "
            f"0.7→{self.transform(0.7):.3f}"
        )
        return True

    # ------------------------------------------------------------------
    def transform(self, conf: float) -> float:
        if not self.fitted or self._x is None:
            return float(conf)
        cal = float(np.interp(conf, self._x, self._y))
        if self.downward_only:
            return min(float(conf), cal)
        return cal

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "x": self._x.tolist() if self._x is not None else None,
            "y": self._y.tolist() if self._y is not None else None,
            "n_samples": self.n_samples,
            "downward_only": self.downward_only,
        }))

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        try:
            d = json.loads(path.read_text())
            if d.get("x") is None:
                return False
            self._x = np.asarray(d["x"], dtype=float)
            self._y = np.asarray(d["y"], dtype=float)
            self.n_samples = int(d.get("n_samples", 0))
            self.downward_only = bool(d.get("downward_only", True))
            self.fitted = True
            logger.info(f"[Calibration] 로드 — n={self.n_samples} 0.6→{self.transform(0.6):.3f}")
            return True
        except Exception as e:
            logger.warning(f"[Calibration] 로드 실패: {e}")
            return False
