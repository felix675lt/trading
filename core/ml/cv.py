"""Purged K-Fold with Embargo — Lopez de Prado Ch. 7.

표준 KFold는 시계열에서 정보누수를 야기한다:
  - train 후반 샘플의 label이 test 초반과 겹침 (label horizon 동안)
  - 자기상관이 높은 금융 데이터 → OOS 성과 과대추정

PurgedKFold 해법:
  1. Purge: 각 test fold 경계에서 train 샘플 중 label이 test 기간과 overlap되는 것 제거
  2. Embargo: test fold 직후 일정 기간 train 샘플도 제거 (leak-through 차단)

사용:
    from core.ml.cv import PurgedKFold
    cv = PurgedKFold(n_splits=5, embargo_pct=0.01, purge_bars=12)
    for tr, te in cv.split(X, t1=df['tb_t1'].values):
        ...
"""

from __future__ import annotations

import numpy as np


class PurgedKFold:
    """정보누수 차단형 K-Fold CV.

    Args:
        n_splits: 폴드 수
        embargo_pct: 전체 샘플 수의 몇 %를 test 직후 embargo로 버릴지 (0.01 = 1%)
        purge_bars: label horizon — test 경계와 겹치는 train 샘플 제거 범위
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        purge_bars: int = 12,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_bars = purge_bars

    def split(
        self,
        X,
        y=None,
        t1: np.ndarray | None = None,
    ):
        """Yield (train_idx, test_idx) with purge + embargo.

        Args:
            X: feature matrix (len 기준)
            t1: 각 샘플의 label 만료 시점 (triple_barrier tb_t1). 없으면 purge_bars 사용.

        Yields:
            (train_idx, test_idx) 쌍 — numpy int 배열
        """
        n = len(X)
        indices = np.arange(n)
        embargo = int(n * self.embargo_pct)

        # 연속 구간 (contiguous) 테스트 폴드
        test_ranges = np.array_split(indices, self.n_splits)

        for test_idx in test_ranges:
            if len(test_idx) == 0:
                continue
            t_start, t_end = test_idx[0], test_idx[-1]

            # 1) Purge: train 중 t1이 test 기간과 overlap되는 샘플 제거
            if t1 is not None:
                # train 샘플 i는 t1[i]가 test 시작 이후면 오버랩 → 제거
                train_mask = np.ones(n, dtype=bool)
                train_mask[test_idx] = False

                for i in range(n):
                    if not train_mask[i]:
                        continue
                    if i < t_start:
                        # train 샘플이 test 시작 전 — t1[i]가 test 구간 내로 뻗으면 제거
                        t1_i = t1[i] if np.isfinite(t1[i]) else i + self.purge_bars
                        if t1_i >= t_start:
                            train_mask[i] = False
                    else:
                        # train 샘플이 test 끝난 이후 — embargo 구간이면 제거
                        if i <= t_end + embargo:
                            train_mask[i] = False
            else:
                # t1 없음 → 고정 bar 기반 purge
                train_mask = np.ones(n, dtype=bool)
                train_mask[test_idx] = False
                # test 시작 전 purge_bars개 + test 끝 후 embargo+purge_bars개 제거
                purge_start = max(0, t_start - self.purge_bars)
                purge_end_after = min(n, t_end + 1 + embargo + self.purge_bars)
                train_mask[purge_start:t_start] = False
                train_mask[t_end + 1:purge_end_after] = False

            train_idx = np.where(train_mask)[0]
            if len(train_idx) < 50:
                continue  # 너무 적으면 스킵
            yield train_idx, np.asarray(test_idx)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def purged_cv_score(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    t1: np.ndarray | None = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    purge_bars: int = 12,
    sample_weight: np.ndarray | None = None,
    scorer=None,
) -> dict:
    """PurgedKFold로 CV 점수 집계.

    Args:
        model_factory: () → 새 모델 인스턴스 (fit/predict 구현)
        scorer: (y_true, y_pred) → float. None이면 accuracy.

    Returns:
        {"mean": float, "std": float, "folds": [float, ...]}
    """
    from sklearn.metrics import accuracy_score

    if scorer is None:
        scorer = accuracy_score

    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct, purge_bars=purge_bars)
    scores = []
    for tr, te in cv.split(X, y, t1=t1):
        model = model_factory()
        if sample_weight is not None:
            try:
                model.fit(X[tr], y[tr], sample_weight=sample_weight[tr])
            except TypeError:
                model.fit(X[tr], y[tr])
        else:
            model.fit(X[tr], y[tr])
        y_pred = model.predict(X[te])
        scores.append(float(scorer(y[te], y_pred)))

    if not scores:
        return {"mean": 0.0, "std": 0.0, "folds": []}
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "folds": scores,
    }
