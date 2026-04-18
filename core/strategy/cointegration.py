"""ADF + Cointegration tests — Stat Arb / Pairs Trading 전제조건.

- ADF (Augmented Dickey-Fuller): 단위근 검정 → 정상성(stationarity) 판별
- Engle-Granger 공적분: 두 비정상 시계열의 선형 결합이 정상이면 long-term 평형 관계
- Hedge ratio: OLS 기반 β — spread = Y − β·X
- Half-life of mean reversion: spread가 평균으로 얼마나 빨리 되돌아오는가

사용:
    coint = CointegrationTester()
    result = coint.test_pair(series_a, series_b)
    if result["is_cointegrated"]:
        spread = coint.compute_spread(series_a, series_b, result["hedge_ratio"])
        z = coint.zscore(spread, window=30)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller, coint
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class CointegrationTester:
    """두 가격 시리즈의 공적분 관계 검정 및 spread 생성."""

    def __init__(
        self,
        adf_pvalue_threshold: float = 0.05,
        coint_pvalue_threshold: float = 0.05,
        min_samples: int = 100,
    ):
        self.adf_p = adf_pvalue_threshold
        self.coint_p = coint_pvalue_threshold
        self.min_samples = min_samples

    def adf_test(self, series: np.ndarray | pd.Series) -> dict:
        """ADF 단위근 검정.

        p-value < threshold면 정상 시계열 (단위근 없음).
        """
        arr = np.asarray(series).astype(np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) < self.min_samples:
            return {"p_value": 1.0, "is_stationary": False, "reason": "sample too small"}

        if not HAS_STATSMODELS:
            return self._adf_fallback(arr)

        try:
            result = adfuller(arr, autolag="AIC")
            p = float(result[1])
            return {
                "adf_stat": float(result[0]),
                "p_value": p,
                "is_stationary": p < self.adf_p,
                "critical_values": {k: float(v) for k, v in result[4].items()},
            }
        except Exception as e:
            return {"p_value": 1.0, "is_stationary": False, "error": str(e)}

    def _adf_fallback(self, arr: np.ndarray) -> dict:
        """statsmodels 없을 때 간이 AR(1) 기반 단위근 검정."""
        if len(arr) < 50:
            return {"p_value": 1.0, "is_stationary": False}
        x = arr[:-1]
        y = np.diff(arr)
        if np.std(x) < 1e-8:
            return {"p_value": 1.0, "is_stationary": False}
        beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
        # 근사: 유의미하게 음수면 정상
        resid = y - beta * x
        se = np.std(resid) / (np.std(x) * np.sqrt(len(x)))
        t_stat = beta / max(se, 1e-8)
        # 매우 러프한 p-value (정규 근사, 실제는 Dickey 분포)
        p = 1.0 - (1.0 / (1.0 + np.exp(-t_stat)))
        return {
            "adf_stat": float(t_stat),
            "p_value": float(p),
            "is_stationary": t_stat < -2.86,  # 5% DF critical
            "fallback": True,
        }

    def test_pair(
        self,
        series_a: pd.Series | np.ndarray,
        series_b: pd.Series | np.ndarray,
    ) -> dict:
        """두 시리즈의 공적분 관계 + hedge ratio + mean reversion half-life."""
        a = np.asarray(series_a).astype(np.float64)
        b = np.asarray(series_b).astype(np.float64)
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if len(a) < self.min_samples:
            return {"is_cointegrated": False, "reason": "sample too small", "n": len(a)}

        # 1) hedge ratio β = Cov(A,B)/Var(B) (OLS A ~ β·B)
        beta = float(np.cov(a, b, ddof=1)[0, 1] / max(np.var(b, ddof=1), 1e-12))
        spread = a - beta * b

        # 2) 공적분 검정 — spread가 정상이면 공적분 성립
        if HAS_STATSMODELS:
            try:
                # Engle-Granger 원식
                _, p_coint, _ = coint(a, b)
            except Exception as e:
                p_coint = 1.0
        else:
            p_coint = self.adf_test(spread)["p_value"]

        is_coint = bool(p_coint < self.coint_p)

        # 3) Half-life of mean reversion (Ornstein-Uhlenbeck approximation)
        hl = self._half_life(spread)

        # 4) spread 통계 (z-score용 기준)
        return {
            "is_cointegrated": is_coint,
            "p_value": float(p_coint),
            "hedge_ratio": beta,
            "spread_mean": float(np.mean(spread)),
            "spread_std": float(np.std(spread)),
            "half_life": hl,
            "n_samples": len(a),
        }

    def _half_life(self, spread: np.ndarray) -> float | None:
        """OU: Δspread = λ(μ - spread) → half-life = -ln(2)/ln(1+λ_slope)."""
        if len(spread) < 30:
            return None
        s_lag = spread[:-1]
        s_diff = np.diff(spread)
        if np.std(s_lag) < 1e-8:
            return None
        # OLS: Δs = α + β·s_lag
        s_lag_mean = np.mean(s_lag)
        s_lag_cent = s_lag - s_lag_mean
        beta = float(np.sum(s_lag_cent * (s_diff - np.mean(s_diff))) / np.sum(s_lag_cent ** 2))
        if beta >= 0 or not np.isfinite(beta):
            return None  # mean-reverting 아님
        hl = -np.log(2) / np.log(1 + beta) if (1 + beta) > 0 else None
        return round(float(hl), 2) if hl and np.isfinite(hl) else None

    def compute_spread(
        self,
        series_a: pd.Series | np.ndarray,
        series_b: pd.Series | np.ndarray,
        hedge_ratio: float,
    ) -> np.ndarray:
        return np.asarray(series_a) - hedge_ratio * np.asarray(series_b)

    def zscore(self, spread: np.ndarray, window: int = 30) -> np.ndarray:
        """롤링 z-score — mean 회귀 진입/청산 시그널 생성용."""
        s = pd.Series(spread)
        mean = s.rolling(window).mean()
        std = s.rolling(window).std()
        return ((s - mean) / std.replace(0, np.nan)).fillna(0).values

    def find_pairs(
        self,
        price_dict: dict[str, pd.Series | np.ndarray],
        max_pairs: int = 10,
    ) -> list[dict]:
        """심볼별 가격 시리즈 딕셔너리에서 공적분 페어 자동 탐색.

        Returns:
            상위 N개 페어 (p-value 오름차순) — [{"a","b","hedge_ratio","p_value","half_life"}]
        """
        symbols = list(price_dict.keys())
        candidates = []
        for i, a in enumerate(symbols):
            for b in symbols[i + 1:]:
                res = self.test_pair(price_dict[a], price_dict[b])
                if res.get("is_cointegrated") and res.get("half_life"):
                    candidates.append({
                        "a": a, "b": b,
                        "hedge_ratio": round(res["hedge_ratio"], 4),
                        "p_value": round(res["p_value"], 4),
                        "half_life": res["half_life"],
                        "spread_std": round(res["spread_std"], 6),
                    })
        candidates.sort(key=lambda c: c["p_value"])
        return candidates[:max_pairs]
