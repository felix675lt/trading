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

        # (timestamp_iso, signal, realized_return, source, regime) — 라운드트립당 1건
        self._records: deque = deque(maxlen=max_samples)
        # 소스별(xgb/lstm/ensemble/rl) 별도 집계
        self._by_source: dict[str, deque] = {}
        # 레짐×소스별 집계 (2026-04-24 C): 레짐 가중치 자동화 근거
        self._by_regime_source: dict[tuple[str, str], deque] = {}

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
        regime: str | None = None,
    ):
        """시그널 관측 기록.

        Args:
            signal: 예측 시그널 값 ([-1, 1] 권장 또는 확률 차이)
            realized_return: 실제 청산 시점 수익률
            source: 시그널 출처 (ensemble/xgb/lstm/rl/mom/ext 등)
            regime: 진입 당시 시장 레짐 (trending/ranging/volatile 등).
                    None이면 "all" 버킷에만 집계됨.
        """
        ts = (timestamp or datetime.utcnow()).isoformat()
        record = {
            "ts": ts,
            "signal": float(signal),
            "ret": float(realized_return),
            "source": source,
            "regime": regime or "all",
        }
        self._records.append(record)

        if source not in self._by_source:
            self._by_source[source] = deque(maxlen=self.max_samples)
        self._by_source[source].append(record)

        # 레짐×소스 집계 (C: 레짐별 가중치 자동화용)
        if regime:
            key = (regime, source)
            if key not in self._by_regime_source:
                self._by_regime_source[key] = deque(maxlen=self.max_samples)
            self._by_regime_source[key].append(record)

        self._save()

    # ------------------------------------------------------------------
    # 레짐×소스 IC (C: 시그널 가중치 자동화용)
    # ------------------------------------------------------------------

    def ic_by_regime(self, regime: str, source: str, min_samples: int = 10) -> dict:
        """특정 레짐×소스의 IC + 샘플수 반환.

        Returns:
            {ic, n_samples, hit_rate}
            샘플 부족 시 ic=0.0 (가중치 optimizer가 "신뢰 안 함"으로 처리).
        """
        key = (regime, source)
        recs = list(self._by_regime_source.get(key, []))
        n = len(recs)
        if n < min_samples:
            return {"ic": 0.0, "n_samples": n, "hit_rate": 0.0, "sufficient": False}
        signals = np.array([r["signal"] for r in recs])
        rets = np.array([r["ret"] for r in recs])
        ic_val = _spearman(signals, rets)
        hits = sum(
            1 for r in recs
            if np.sign(r["signal"]) == np.sign(r["ret"]) and r["signal"] != 0
        )
        nonzero = sum(1 for r in recs if r["signal"] != 0)
        return {
            "ic": float(ic_val),
            "n_samples": n,
            "hit_rate": hits / max(nonzero, 1),
            "sufficient": True,
        }

    def regime_source_matrix(self, min_samples: int = 10) -> dict:
        """모든 (regime, source) 조합의 IC 매트릭스 — 대시보드/옵티마이저용."""
        out: dict = {}
        for (regime, source), recs in self._by_regime_source.items():
            if len(recs) < min_samples:
                continue
            if regime not in out:
                out[regime] = {}
            out[regime][source] = self.ic_by_regime(regime, source, min_samples)
        return out

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
                # regime×source 재구축 (C)
                reg = rec.get("regime")
                if reg and reg != "all":
                    key = (reg, src)
                    if key not in self._by_regime_source:
                        self._by_regime_source[key] = deque(maxlen=self.max_samples)
                    self._by_regime_source[key].append(rec)
            logger.info(f"[IC] 이력 로드: {len(self._records)}건")
        except Exception as e:
            logger.debug(f"[IC] 로드 실패: {e}")


class SignalWeightOptimizer:
    """레짐×소스 IC에 기반한 vote weight multiplier 공급자 (2026-04-24 C).

    목적: `strategy/manager._count_signal_votes()`가 생성한 각 소스(ML/RL/MOM/
    RSI_extreme/EXT/BREAKOUT)의 vote weight에, 해당 레짐에서 그 소스가
    실제로 얼마나 잘 맞혀왔는지(IC)를 곱해 가중치를 재조정한다.

    - IC ≤ -0.03: counter-productive → 0.3×  (더 가중치 깎기보단 방향 뒤집는
                                               건 위험하므로 줄이기만)
    - -0.03 < IC ≤ 0: 노이즈 → 0.5×
    - 0 < IC ≤ 0.03: 약신호 → 0.8×
    - 0.03 < IC ≤ 0.05: 보통 → 1.0× (중립)
    - 0.05 < IC ≤ 0.10: 강신호 → 1.3×
    - IC > 0.10: 최강 → 1.5× (상한)

    변경량 제한: 지수평활 smoothing=0.25 (4~5회 업데이트로 수렴).
    샘플 부족(min_samples<20): 1.0× 유지 (안전).
    """

    # (source, regime)가 매번 바뀌면 비용이 커서, 캐시는 { (regime, source): multiplier }
    def __init__(
        self,
        min_samples: int = 20,
        smoothing: float = 0.25,
        mult_min: float = 0.3,
        mult_max: float = 1.5,
    ):
        self.min_samples = int(min_samples)
        self.smoothing = float(smoothing)
        self.mult_min = float(mult_min)
        self.mult_max = float(mult_max)
        self._cache: dict[tuple[str, str], float] = {}
        self._last_update: datetime | None = None

    @staticmethod
    def _target_from_ic(ic: float) -> float:
        if ic <= -0.03:
            return 0.3
        if ic <= 0.0:
            return 0.5
        if ic <= 0.03:
            return 0.8
        if ic <= 0.05:
            return 1.0
        if ic <= 0.10:
            return 1.3
        return 1.5

    def update_from_tracker(self, tracker: "ICTracker") -> dict:
        """ICTracker의 regime×source 매트릭스에서 multiplier를 재계산.

        Returns:
            업데이트 요약 { (regime, source): {ic, n, mult_before, mult_after} }
        """
        matrix = tracker.regime_source_matrix(min_samples=self.min_samples)
        s = max(0.0, min(1.0, self.smoothing))
        summary: dict = {}
        for regime, srcs in matrix.items():
            for source, stats in srcs.items():
                ic = float(stats.get("ic", 0.0))
                n = int(stats.get("n_samples", 0))
                target = self._target_from_ic(ic)
                # 범위 클리핑
                target = max(self.mult_min, min(self.mult_max, target))
                key = (regime, source)
                prev = self._cache.get(key, 1.0)
                new = (1 - s) * prev + s * target
                # 최종 클리핑
                new = max(self.mult_min, min(self.mult_max, new))
                self._cache[key] = new
                if abs(new - prev) > 0.02:
                    summary[f"{regime}:{source}"] = {
                        "ic": round(ic, 4),
                        "n": n,
                        "mult_before": round(prev, 3),
                        "mult_after": round(new, 3),
                    }
        self._last_update = datetime.utcnow()
        if summary:
            logger.info(f"[SignalWeight] 갱신: {summary}")
        return summary

    def get(self, regime: str, source: str) -> float:
        """vote weight에 곱할 multiplier. 미등록이면 1.0 (영향 없음)."""
        return float(self._cache.get((regime, source), 1.0))

    def snapshot(self) -> dict:
        """대시보드용 현재 가중치 상태."""
        return {
            f"{r}:{s}": round(v, 3)
            for (r, s), v in sorted(self._cache.items())
        }


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
