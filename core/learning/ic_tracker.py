"""Information Coefficient (IC) Tracker + Signal Half-life Analysis.

IC: 예측 시그널과 미래 실현수익률의 rank correlation (Spearman).
    - |IC| > 0.05: 유의미한 알파
    - |IC| > 0.10: 강한 알파
    - IC 음수 반전: 시그널 뒤집어야 함

Half-life: 알파가 시간 지남에 따라 얼마나 빨리 감쇠하는가?
    - AR(1) 적합 ρ → half-life = -ln(2)/ln(|ρ|)
    - 짧을수록 고빈도 교체 필요

사용:
    tracker = ICTracker()
    tracker.record(timestamp, signal=0.8, realized_return=0.012)
    stats = tracker.report()  # 최근 IC, half-life, decay
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import json
import numpy as np
from loguru import logger


class ICTracker:
    """시그널-수익률 rank correlation 및 알파 감쇠 추적.

    각 시그널 발생 시점 (signal, realized_return) 쌍을 저장하고
    롤링 Spearman IC, IC-IR(Information Ratio), half-life를 계산한다.
    """

    def __init__(
        self,
        max_samples: int = 500,
        lag_analysis: list[int] | None = None,
        persist_path: str | Path | None = "models_saved/ic_history.json",
    ):
        self.max_samples = max_samples
        self.lag_analysis = lag_analysis or [1, 3, 6, 12, 24]  # 지연 시각별 IC
        self.persist_path = Path(persist_path) if persist_path else None

        # (timestamp_iso, signal, realized_return, source) — 라운드트립당 1건
        self._records: deque = deque(maxlen=max_samples)
        # 소스별(xgb/lstm/ensemble/rl) 별도 집계
        self._by_source: dict[str, deque] = {}

        self._load()

    # ------------------------------------------------------------------
    # 기록
    # ------------------------------------------------------------------

    def record(
        self,
        signal: float,
        realized_return: float,
        source: str = "ensemble",
        timestamp: datetime | None = None,
    ):
        """시그널 관측 기록.

        Args:
            signal: 예측 시그널 값 ([-1, 1] 권장 또는 확률 차이)
            realized_return: 실제 청산 시점 수익률
            source: 시그널 출처 (ensemble/xgb/lstm/rl 등)
        """
        ts = (timestamp or datetime.utcnow()).isoformat()
        record = {
            "ts": ts,
            "signal": float(signal),
            "ret": float(realized_return),
            "source": source,
        }
        self._records.append(record)

        if source not in self._by_source:
            self._by_source[source] = deque(maxlen=self.max_samples)
        self._by_source[source].append(record)

        self._save()

    # ------------------------------------------------------------------
    # IC 계산
    # ------------------------------------------------------------------

    def ic(self, source: str | None = None, window: int | None = None) -> float:
        """Spearman rank correlation 반환.

        Args:
            source: None이면 전체 집계, 지정하면 해당 소스만
            window: 최근 N개만 사용 (None = 전체)
        """
        recs = self._select(source, window)
        if len(recs) < 10:
            return 0.0
        signals = np.array([r["signal"] for r in recs])
        rets = np.array([r["ret"] for r in recs])
        return _spearman(signals, rets)

    def ic_ir(self, source: str | None = None, window: int = 50) -> float:
        """Information Ratio = mean(IC) / std(IC) — 롤링 윈도우 평균/표준편차."""
        recs = self._select(source, None)
        if len(recs) < window * 2:
            return 0.0
        signals = np.array([r["signal"] for r in recs])
        rets = np.array([r["ret"] for r in recs])
        ics: list[float] = []
        for i in range(window, len(signals) + 1):
            ic = _spearman(signals[i - window:i], rets[i - window:i])
            if np.isfinite(ic):
                ics.append(ic)
        if len(ics) < 5:
            return 0.0
        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics))
        if std_ic < 1e-8:
            return 0.0
        return mean_ic / std_ic

    def hit_rate(self, source: str | None = None) -> float:
        """시그널 방향 일치율 (sign(signal) == sign(ret)) — 50%가 coin-flip."""
        recs = self._select(source, None)
        if not recs:
            return 0.0
        hits = sum(1 for r in recs if np.sign(r["signal"]) == np.sign(r["ret"]) and r["signal"] != 0)
        nonzero = sum(1 for r in recs if r["signal"] != 0)
        return hits / max(nonzero, 1)

    # ------------------------------------------------------------------
    # Signal Half-life (AR(1) 적합)
    # ------------------------------------------------------------------

    def half_life(self, source: str | None = None) -> dict:
        """시그널의 half-life 추정.

        approach: signal_t = ρ·signal_{t-1} + ε 적합 후
                  half-life = -ln(2)/ln(|ρ|)  (ρ>0일 때만 의미있음)

        Returns:
            {half_life_periods, rho, n_samples}
        """
        recs = self._select(source, None)
        if len(recs) < 30:
            return {"half_life_periods": None, "rho": None, "n_samples": len(recs)}

        signals = np.array([r["signal"] for r in recs])
        x = signals[:-1]
        y = signals[1:]
        if np.std(x) < 1e-8:
            return {"half_life_periods": None, "rho": 0.0, "n_samples": len(recs)}
        # OLS: y = ρ·x + intercept
        rho = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(rho) or abs(rho) < 1e-3:
            return {"half_life_periods": None, "rho": rho, "n_samples": len(recs)}

        # half-life only meaningful when 0 < rho < 1
        if rho <= 0 or rho >= 1:
            return {"half_life_periods": None, "rho": rho, "n_samples": len(recs)}

        hl = -np.log(2) / np.log(rho)
        return {
            "half_life_periods": round(float(hl), 2),
            "rho": round(rho, 4),
            "n_samples": len(recs),
        }

    def decay_curve(self, source: str | None = None) -> dict[int, float]:
        """lag별 IC — 시그널이 언제까지 유효한가.

        recs에 저장된 (signal, ret) 쌍에서 lag별 autocorrelation of returns가
        부정확하므로, 여기서는 signal의 autocorrelation을 lag별로 반환.
        """
        recs = self._select(source, None)
        if len(recs) < max(self.lag_analysis) + 10:
            return {lag: 0.0 for lag in self.lag_analysis}

        signals = np.array([r["signal"] for r in recs])
        curve = {}
        for lag in self.lag_analysis:
            if len(signals) > lag + 5:
                x = signals[:-lag]
                y = signals[lag:]
                if np.std(x) > 1e-8 and np.std(y) > 1e-8:
                    curve[lag] = float(np.corrcoef(x, y)[0, 1])
                else:
                    curve[lag] = 0.0
            else:
                curve[lag] = 0.0
        return curve

    # ------------------------------------------------------------------
    # 종합 리포트
    # ------------------------------------------------------------------

    def report(self) -> dict:
        """대시보드/로그용 종합 상태."""
        report = {
            "total_samples": len(self._records),
            "sources": list(self._by_source.keys()),
            "overall": {
                "ic": round(self.ic(), 4),
                "ic_ir": round(self.ic_ir(), 4),
                "hit_rate": round(self.hit_rate(), 4),
                "half_life": self.half_life(),
                "decay_curve": self.decay_curve(),
            },
            "by_source": {},
        }
        for src in self._by_source:
            report["by_source"][src] = {
                "n": len(self._by_source[src]),
                "ic": round(self.ic(src), 4),
                "hit_rate": round(self.hit_rate(src), 4),
                "half_life": self.half_life(src),
            }
        return report

    def log_summary(self):
        """최신 IC 요약 로그."""
        r = self.report()
        if r["total_samples"] < 10:
            return
        o = r["overall"]
        logger.info(
            f"[IC] n={r['total_samples']} IC={o['ic']:+.3f} "
            f"IC-IR={o['ic_ir']:+.3f} hit={o['hit_rate']*100:.0f}% "
            f"half-life={o['half_life'].get('half_life_periods')} bars"
        )
        for src, s in r["by_source"].items():
            if s["n"] >= 10:
                logger.info(
                    f"[IC:{src}] n={s['n']} IC={s['ic']:+.3f} hit={s['hit_rate']*100:.0f}%"
                )

    # ------------------------------------------------------------------
    # 영속화
    # ------------------------------------------------------------------

    def _select(self, source: str | None, window: int | None) -> list[dict]:
        recs = list(self._by_source[source]) if source else list(self._records)
        if window is not None and len(recs) > window:
            recs = recs[-window:]
        return recs

    def _save(self):
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump(list(self._records), f)
        except Exception as e:
            logger.debug(f"[IC] 저장 실패: {e}")

    def _load(self):
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            with open(self.persist_path) as f:
                data = json.load(f)
            # 최근 max_samples 건만 재로드
            for rec in data[-self.max_samples:]:
                self._records.append(rec)
                src = rec.get("source", "ensemble")
                if src not in self._by_source:
                    self._by_source[src] = deque(maxlen=self.max_samples)
                self._by_source[src].append(rec)
            logger.info(f"[IC] 이력 로드: {len(self._records)}건")
        except Exception as e:
            logger.debug(f"[IC] 로드 실패: {e}")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation — scipy 없이 numpy로."""
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    rx = _rankdata(x[mask])
    ry = _rankdata(y[mask])
    if np.std(rx) < 1e-8 or np.std(ry) < 1e-8:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Average-rank (동순위 평균 처리)."""
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64) + 1
    # 동순위 보정
    _, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.zeros_like(counts, dtype=np.float64)
        for i, r in enumerate(ranks):
            sums[inv[i]] += r
        avg = sums / counts
        ranks = avg[inv]
    return ranks
