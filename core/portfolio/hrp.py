"""Hierarchical Risk Parity (HRP) — Lopez de Prado 2016.

Markowitz mean-variance의 단점:
  - Covariance matrix 역행렬 → 수치 불안정, 작은 변화에 큰 가중치 변동
  - 추정 오차 증폭 → OOS 성과 취약

HRP 해법:
  1. Hierarchical clustering (assets → dendrogram, corr 기반 거리)
  2. Quasi-diagonalization (비슷한 자산을 인접하게 재배치)
  3. Recursive bisection — 각 노드에서 인접 그룹에 역변동성 가중치로 분배

장점:
  - 역행렬 불필요 → 안정적
  - 자산 수 > 샘플 수일 때도 작동
  - OOS 샤프에서 MV 대비 개선

사용:
    hrp = HRPAllocator()
    weights = hrp.allocate(returns_df)  # dict: symbol → weight (∑=1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

try:
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class HRPAllocator:
    """Hierarchical Risk Parity 포트폴리오 배분."""

    def __init__(self, linkage_method: str = "single"):
        """
        Args:
            linkage_method: scipy.cluster.hierarchy.linkage 방식 ("single"=기본 HRP)
        """
        self.linkage_method = linkage_method

    def allocate(self, returns: pd.DataFrame) -> dict[str, float]:
        """자산별 수익률 DataFrame → HRP 가중치 dict.

        Args:
            returns: columns=symbols, rows=time, values=log_return 또는 pct_return

        Returns:
            {symbol: weight} — 가중치 합 1.0
        """
        if not HAS_SCIPY:
            logger.warning("[HRP] scipy 미설치 → equal-weight fallback")
            return self._equal_weight(returns.columns)

        # 결측치 제거
        rets = returns.dropna()
        if len(rets) < 20 or len(rets.columns) < 2:
            return self._equal_weight(returns.columns)

        cov = rets.cov().values
        corr = rets.corr().values

        # 1) distance matrix: d_ij = sqrt(0.5 * (1 - corr_ij))
        dist = np.sqrt(0.5 * np.clip(1 - corr, 0, 2))
        np.fill_diagonal(dist, 0.0)

        # scipy linkage는 condensed distance 필요
        try:
            condensed = squareform(dist, checks=False)
            link = linkage(condensed, method=self.linkage_method)
        except Exception as e:
            logger.warning(f"[HRP] linkage 실패 ({e}) → equal-weight")
            return self._equal_weight(returns.columns)

        # 2) quasi-diagonalization — dendrogram 리프 순서
        sort_idx = self._get_quasi_diag(link)

        # 3) recursive bisection
        weights = self._recursive_bisection(cov, sort_idx)

        symbols = list(returns.columns)
        w_dict = {symbols[sort_idx[i]]: float(weights[i]) for i in range(len(symbols))}
        # 정규화
        total = sum(w_dict.values())
        if total > 0:
            w_dict = {k: v / total for k, v in w_dict.items()}
        return w_dict

    def _get_quasi_diag(self, link: np.ndarray) -> list[int]:
        """Linkage → leaf order (재귀 분할용)."""
        link = link.astype(int)
        sort_idx = [link[-1, 0], link[-1, 1]]
        num_items = link[-1, 3]
        while max(sort_idx) >= num_items:
            new = []
            for i in sort_idx:
                if i < num_items:
                    new.append(i)
                else:
                    lnk = link[i - num_items]
                    new.append(lnk[0])
                    new.append(lnk[1])
            sort_idx = new
        return sort_idx

    def _recursive_bisection(self, cov: np.ndarray, sort_idx: list[int]) -> np.ndarray:
        """재귀 분할 — 각 클러스터 쌍에 역변동성 가중치 할당."""
        n = len(sort_idx)
        w = np.ones(n)
        clusters = [list(range(n))]
        while clusters:
            # 각 클러스터를 반으로 쪼개고 두 절반의 클러스터 분산 비율로 가중치 갱신
            next_clusters = []
            for c in clusters:
                if len(c) <= 1:
                    continue
                mid = len(c) // 2
                c1 = c[:mid]
                c2 = c[mid:]
                idx1 = [sort_idx[i] for i in c1]
                idx2 = [sort_idx[i] for i in c2]
                var1 = self._cluster_var(cov, idx1)
                var2 = self._cluster_var(cov, idx2)
                if var1 + var2 <= 0:
                    alpha = 0.5
                else:
                    alpha = 1 - var1 / (var1 + var2)
                for i in c1:
                    w[i] *= alpha
                for i in c2:
                    w[i] *= (1 - alpha)
                next_clusters.append(c1)
                next_clusters.append(c2)
            clusters = next_clusters
        return w

    def _cluster_var(self, cov: np.ndarray, idx: list[int]) -> float:
        """역변동성 가중치(inverse-variance) 기반 클러스터 분산."""
        sub = cov[np.ix_(idx, idx)]
        diag = np.diag(sub)
        if np.any(diag <= 0):
            return float(np.sum(diag))
        ivp = 1.0 / diag
        ivp = ivp / ivp.sum()
        return float(ivp @ sub @ ivp)

    def _equal_weight(self, symbols) -> dict[str, float]:
        symbols = list(symbols)
        if not symbols:
            return {}
        w = 1.0 / len(symbols)
        return {s: w for s in symbols}

    # ------------------------------------------------------------------
    # 포지션 사이즈 변환
    # ------------------------------------------------------------------

    def scale_positions(
        self,
        weights: dict[str, float],
        equity: float,
        max_per_asset: float = 0.4,
    ) -> dict[str, float]:
        """HRP 가중치 × equity → 심볼별 notional 상한.

        Args:
            max_per_asset: 단일 자산 최대 비중 (cap)
        """
        out = {}
        for s, w in weights.items():
            w_capped = min(max_per_asset, w)
            out[s] = round(equity * w_capped, 2)
        return out
