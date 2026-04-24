"""Deflated Sharpe Ratio + Combinatorial Purged CV — 오버핏 방어 인프라.

퀀트 금융에서 가장 위험한 착각은 "백테스트 Sharpe 2.0이면 좋은 전략"이라는 생각이다.
수십 수백 번의 파라미터 조합을 시도해서 나온 "최고 Sharpe"는 selection bias로 부풀어져 있고,
실전 배포 시 평균적으로 0에 수렴한다(False Discovery).

본 모듈은 Lopez de Prado의 세 가지 기법을 제공한다:

1) **DSR (Deflated Sharpe Ratio)** — Bailey & Lopez de Prado (2014)
   N번 백테스트 시도 + skew/kurtosis 보정 후 "이 Sharpe가 우연이 아닐 확률"을 돌려준다.
   DSR > 0.95 → 통계적으로 유의.

2) **PSR (Probabilistic Sharpe Ratio)** — 위의 선행 단계 (n=1일 때의 DSR).

3) **CPCV (Combinatorial Purged Cross-Validation)** — Lopez de Prado (AFML Ch.12)
   시계열 데이터를 N개 그룹으로 나눠 C(N,k) 조합으로 train/test split을 여러 번 구성,
   각 조합에서 embargo로 누수 방지. 한 번의 split이 아닌 "경로의 분포(path distribution)"를
   얻어 전략 성과의 안정성을 평가.

사용:
    dsr = DeflatedSharpe.compute(returns=strategy_returns, n_trials=50)
    print(dsr)  # {"sharpe": 1.73, "dsr": 0.89, "psr": 0.97, "significant": False}

    cpcv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, embargo_pct=0.01)
    for train_idx, test_idx in cpcv.split(df):
        ...
    paths = cpcv.compute_paths(sharpes)  # 경로별 Sharpe 분포
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# DSR / PSR
# ---------------------------------------------------------------------------
@dataclass
class DSRResult:
    sharpe: float
    psr: float              # P(true_SR > 0 | observed_SR)
    dsr: float              # PSR deflated by n_trials
    n_trials: int
    n_samples: int
    skew: float
    kurt: float
    sr_benchmark: float     # 내부 계산된 "기대 최대 Sharpe" (random hypothesis)
    significant: bool       # dsr > 0.95 자동 판정

    def __str__(self):
        return (
            f"SR={self.sharpe:.3f} PSR={self.psr:.3f} DSR={self.dsr:.3f} "
            f"n_trials={self.n_trials} n_samples={self.n_samples} "
            f"skew={self.skew:.2f} kurt={self.kurt:.2f} "
            f"{'✅ 유의' if self.significant else '❌ 우연 가능'}"
        )


class DeflatedSharpe:
    """Bailey-Lopez de Prado (2014) Deflated Sharpe Ratio."""

    @staticmethod
    def compute(
        returns: np.ndarray | pd.Series,
        n_trials: int = 1,
        benchmark_sr: float = 0.0,
        annualization: int = 1,
    ) -> DSRResult:
        """Deflated Sharpe Ratio 계산.

        Args:
            returns: 전략의 (주기별) 수익률 시계열
            n_trials: 시도한 백테스트 수 (파라미터 조합 수). 기본 1.
            benchmark_sr: 귀무가설 Sharpe (보통 0)
            annualization: 연환산 계수 (일간=252, 시간=252*24 등). 기본 1(이미 연환산).
        """
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]
        n = len(r)
        if n < 30:
            logger.warning(f"[DSR] 샘플 부족 (n={n}) — 신뢰도 낮음")
            return DSRResult(0, 0, 0, n_trials, n, 0, 0, 0, False)

        mu = r.mean()
        sd = r.std(ddof=1)
        if sd == 0:
            return DSRResult(0, 0, 0, n_trials, n, 0, 0, 0, False)

        sr = mu / sd * math.sqrt(annualization)
        # Skew / excess kurtosis (Fisher)
        centered = (r - mu) / sd
        skew = float((centered ** 3).mean())
        kurt = float((centered ** 4).mean()) - 3.0  # excess

        # PSR: probability that true SR > benchmark_sr
        # SE(SR) = sqrt((1 - skew*SR + (kurt/4)*SR^2) / (n-1))
        sr_unann = mu / sd  # un-annualized SR for SE formula
        var_sr = (1.0 - skew * sr_unann + (kurt / 4.0) * sr_unann ** 2) / (n - 1)
        se_sr = math.sqrt(max(var_sr, 1e-12))
        # PSR (benchmark in un-annualized units)
        bench_un = benchmark_sr / math.sqrt(annualization)
        z = (sr_unann - bench_un) / se_sr
        psr = float(_norm_cdf(z))

        # Expected max SR under null (Bailey-Lopez de Prado) given n_trials
        # E[max SR] ≈ (1-γ)*Φ^-1(1 - 1/N) + γ*Φ^-1(1 - 1/(N*e))
        euler_mascheroni = 0.5772156649
        if n_trials > 1:
            n_t = float(n_trials)
            em = (1.0 - euler_mascheroni) * _norm_ppf(1 - 1.0 / n_t) + \
                 euler_mascheroni * _norm_ppf(1 - 1.0 / (n_t * math.e))
            sr_benchmark = em * se_sr
        else:
            sr_benchmark = bench_un
        # DSR: PSR deflated — P(true SR > expected_max_SR | observed)
        z2 = (sr_unann - sr_benchmark) / se_sr
        dsr = float(_norm_cdf(z2))

        return DSRResult(
            sharpe=sr, psr=psr, dsr=dsr, n_trials=n_trials, n_samples=n,
            skew=skew, kurt=kurt, sr_benchmark=sr_benchmark * math.sqrt(annualization),
            significant=dsr > 0.95,
        )


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    """표준 정규 분포 역함수 (Acklam approximation)."""
    if p <= 0: return -np.inf
    if p >= 1: return np.inf
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# ---------------------------------------------------------------------------
# CPCV (Combinatorial Purged Cross-Validation)
# ---------------------------------------------------------------------------
class CombinatorialPurgedCV:
    """시계열 CPCV with embargo — Lopez de Prado (AFML Ch.12).

    일반 K-Fold가 주는 한 개의 train/test split 대신 C(N, k) 개의 조합을 만든다.
    각 조합마다 k개 폴드를 test로 묶고 나머지 N-k를 train으로 쓴다. embargo로
    test 직후 시점의 train을 제거해 정보 누수 방지.

    장점:
    - 한 번의 일반 split보다 ~C(N,k)/k 배 많은 out-of-sample 경로 샘플 제공
    - 전략의 Sharpe 분포를 얻을 수 있어 "우연히 나온 값" vs "안정적 edge" 구분 가능

    Args:
        n_splits: 전체 폴드 수 N (기본 6)
        n_test_splits: 각 조합에서 test로 쓸 폴드 수 k (기본 2)
        embargo_pct: test 직후 제거할 비율 (기본 1%)
    """

    def __init__(self, n_splits: int = 6, n_test_splits: int = 2, embargo_pct: float = 0.01):
        if n_test_splits >= n_splits:
            raise ValueError(f"n_test_splits({n_test_splits}) must be < n_splits({n_splits})")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct
        logger.info(
            f"[CPCV] n_splits={n_splits} n_test_splits={n_test_splits} "
            f"embargo={embargo_pct*100:.1f}% → n_combos={self.n_combinations}"
        )

    @property
    def n_combinations(self) -> int:
        """총 조합 수 = C(N, k)."""
        return math.comb(self.n_splits, self.n_test_splits)

    def split(self, X):
        """(train_idx, test_idx) 튜플 generator — N번 yield."""
        n = len(X)
        fold_size = n // self.n_splits
        embargo = int(n * self.embargo_pct)
        fold_ranges = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n
            fold_ranges.append((start, end))

        for combo in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = []
            for f in combo:
                s, e = fold_ranges[f]
                test_idx.extend(range(s, e))
            test_set = set(test_idx)
            # train = 나머지 - embargo 영역
            embargo_set = set()
            for f in combo:
                _, e = fold_ranges[f]
                for j in range(e, min(e + embargo, n)):
                    embargo_set.add(j)
            train_idx = [i for i in range(n) if i not in test_set and i not in embargo_set]
            yield np.array(train_idx), np.array(test_idx)

    def compute_paths(self, fold_returns: dict[int, np.ndarray]) -> list[np.ndarray]:
        """각 조합에서 얻은 fold-별 OOS 수익률 → 경로(path) 재구성.

        각 경로는 조합별로 테스트 폴드들의 수익률을 시간순으로 이어붙인 배열.
        """
        paths = []
        for combo in combinations(range(self.n_splits), self.n_test_splits):
            pieces = [fold_returns[f] for f in combo if f in fold_returns]
            if pieces:
                paths.append(np.concatenate(pieces))
        return paths
