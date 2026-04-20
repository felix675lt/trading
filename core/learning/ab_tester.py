"""A/B 테스트 통계 모듈 — 수학적으로 엄밀한 두 variant 비교.

목적:
    같은 시장 조건에서 두 정책(예: macro_block ON vs OFF)을 병렬 실행하고,
    per-trade PnL 분포를 비교하여 어느 쪽이 통계적으로 우월한지 판정.

통계 절차:
    1) Welch's t-test (unequal variance) — 평균 PnL 차이 유의성
    2) Cohen's d — 효과 크기 (n 크면 p값은 쉽게 유의해지므로 effect size 필수)
    3) Mann-Whitney U — 분포 가정 없는 robust 검정 (scipy 있을 때)
    4) Bootstrap 95% CI — mean_diff / Sharpe diff 구간 추정
    5) Sharpe ratio 차이 (per-trade, annualized 아님)
    6) Risk metrics: MDD, downside deviation

정지규칙 (pre-registered — peeking 방지):
    - n_min ≥ 100 (각 variant)
    - Welch's t p < 0.05
    - |Cohen's d| ≥ 0.3 (small→medium effect)
    세 조건 동시 충족 시 승자 채택 권고.

다중검정:
    동시에 K개 A/B 테스트 운용 시 Bonferroni: α/K 사용 권고.
    현재는 1개 (macro ON/OFF) → α=0.05 그대로.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ============================================================================
# 저수준 통계 유틸 — scipy 없이도 동작하도록 순수 파이썬 fallback 포함
# ============================================================================


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _variance(xs: list[float], ddof: int = 1) -> float:
    """표본 분산 (ddof=1 = unbiased)"""
    n = len(xs)
    if n <= ddof:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - ddof)


def _std(xs: list[float], ddof: int = 1) -> float:
    return math.sqrt(_variance(xs, ddof))


def welch_t_test(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Welch's two-sample t-test (unequal variance).

    Returns (t_stat, df, p_value_two_sided)
    scipy 있으면 정확한 p값, 없으면 Student-t 근사 (대표본)
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 0.0, 1.0
    va = _variance(a)
    vb = _variance(b)
    if va == 0 and vb == 0:
        return 0.0, 0.0, 1.0
    diff = _mean(a) - _mean(b)
    se2 = va / na + vb / nb
    if se2 <= 0:
        return 0.0, 0.0, 1.0
    se = math.sqrt(se2)
    t = diff / se
    # Welch-Satterthwaite degrees of freedom
    df_num = (va / na + vb / nb) ** 2
    df_den = (va / na) ** 2 / max(na - 1, 1) + (vb / nb) ** 2 / max(nb - 1, 1)
    df = df_num / df_den if df_den > 0 else float(na + nb - 2)

    if _HAS_SCIPY:
        # two-sided p
        p = float(2 * (1 - _scipy_stats.t.cdf(abs(t), df)))
    else:
        # 정규 근사 (df 크면 충분히 정확)
        # Φ(|t|) 근사: error function
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    return t, df, p


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d — pooled SD 기반 효과 크기.

    |d|  0.2: small, 0.5: medium, 0.8: large
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va = _variance(a)
    vb = _variance(b)
    # pooled SD
    pooled_var = ((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1)
    if pooled_var <= 0:
        return 0.0
    pooled_sd = math.sqrt(pooled_var)
    return (_mean(a) - _mean(b)) / pooled_sd


def mann_whitney_u(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mann-Whitney U (non-parametric). scipy 없으면 (0, 1) 반환.

    Returns (u_stat, p_value_two_sided)
    """
    if not _HAS_SCIPY or len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    try:
        res = _scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def bootstrap_mean_ci(
    xs: list[float], n_boot: int = 2000, ci: float = 0.95, seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for mean.

    Returns (lo, hi) at confidence `ci`.
    """
    import random as _rnd
    if len(xs) < 5:
        m = _mean(xs)
        return (m, m)
    rng = _rnd.Random(seed)
    n = len(xs)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [xs[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_boot)]
    hi = means[int((1 - alpha) * n_boot) - 1]
    return lo, hi


def sharpe_per_trade(xs: list[float]) -> float:
    """Per-trade Sharpe = mean / std (annualization 없음, 상대 비교용)"""
    if len(xs) < 2:
        return 0.0
    sd = _std(xs)
    if sd == 0:
        return 0.0
    return _mean(xs) / sd


def max_drawdown(pnls: list[float]) -> float:
    """순차 PnL 누적의 MaxDD (금액 단위, 음수로 표시).

    Returns: MDD (negative number, e.g., -125.50)
    """
    if not pnls:
        return 0.0
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = cum - peak
        if dd < mdd:
            mdd = dd
    return mdd


def downside_deviation(xs: list[float], target: float = 0.0) -> float:
    """Sortino의 분모 — target 미만 손실 편차만 측정"""
    downs = [(x - target) ** 2 for x in xs if x < target]
    if not downs:
        return 0.0
    return math.sqrt(sum(downs) / len(downs))


# ============================================================================
# A/B 비교 상위 API
# ============================================================================


@dataclass
class VariantStats:
    variant: str
    n: int
    wr: float          # win rate
    mean_pnl: float
    median_pnl: float
    std_pnl: float
    total_pnl: float
    max_win: float
    max_loss: float
    sharpe_pt: float   # per-trade Sharpe
    mdd: float
    downside_dev: float
    mean_ci95: tuple[float, float]


@dataclass
class ABResult:
    variant_a: VariantStats
    variant_b: VariantStats
    # 검정 통계량 (A - B 기준)
    mean_diff: float
    t_stat: float
    t_df: float
    p_welch: float
    cohens_d: float
    u_stat: float
    p_mannwhitney: float
    mean_diff_ci95: tuple[float, float]
    sharpe_diff: float
    # 의사결정
    winner: Optional[str]            # "A", "B", or None
    significance: str                # "significant", "not_significant", "insufficient_data"
    reasoning: str                   # 사람이 읽는 요약
    stopping_rule_met: bool          # 정지규칙 전부 충족 여부
    # 상세 메트릭
    details: dict = field(default_factory=dict)

    def to_report_str(self) -> str:
        a, b = self.variant_a, self.variant_b
        lines = [
            f"━━━ A/B 비교: {a.variant} vs {b.variant} ━━━",
            f"N (A / B)        : {a.n} / {b.n}",
            f"WR (A / B)       : {a.wr:.2%} / {b.wr:.2%}",
            f"mean PnL (A/B)   : {a.mean_pnl:+.3f} / {b.mean_pnl:+.3f}  "
            f"(Δ={self.mean_diff:+.3f}, 95% CI [{self.mean_diff_ci95[0]:+.3f}, {self.mean_diff_ci95[1]:+.3f}])",
            f"std PnL (A/B)    : {a.std_pnl:.3f} / {b.std_pnl:.3f}",
            f"Sharpe/trade(A/B): {a.sharpe_pt:.3f} / {b.sharpe_pt:.3f}  (Δ={self.sharpe_diff:+.3f})",
            f"MDD (A / B)      : {a.mdd:.2f} / {b.mdd:.2f}",
            f"Welch's t        : t={self.t_stat:+.3f} (df={self.t_df:.1f})  p={self.p_welch:.4f}",
            f"Mann-Whitney U   : p={self.p_mannwhitney:.4f}",
            f"Cohen's d        : {self.cohens_d:+.3f}  ({_interpret_d(self.cohens_d)})",
            f"판정             : {self.significance}  →  winner={self.winner}",
            f"정지규칙 충족    : {self.stopping_rule_met}",
            f"해석             : {self.reasoning}",
        ]
        return "\n".join(lines)


def _interpret_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def compute_variant_stats(variant: str, pnls: list[float]) -> VariantStats:
    """단일 variant의 per-trade PnL 리스트에서 종합 통계 산출"""
    n = len(pnls)
    if n == 0:
        return VariantStats(
            variant=variant, n=0, wr=0.0, mean_pnl=0.0, median_pnl=0.0,
            std_pnl=0.0, total_pnl=0.0, max_win=0.0, max_loss=0.0,
            sharpe_pt=0.0, mdd=0.0, downside_dev=0.0, mean_ci95=(0.0, 0.0),
        )
    wins = [p for p in pnls if p > 0]
    sorted_pnls = sorted(pnls)
    median = sorted_pnls[n // 2] if n % 2 == 1 else (sorted_pnls[n // 2 - 1] + sorted_pnls[n // 2]) / 2
    ci_lo, ci_hi = bootstrap_mean_ci(pnls)
    return VariantStats(
        variant=variant,
        n=n,
        wr=len(wins) / n,
        mean_pnl=_mean(pnls),
        median_pnl=median,
        std_pnl=_std(pnls),
        total_pnl=sum(pnls),
        max_win=max(pnls),
        max_loss=min(pnls),
        sharpe_pt=sharpe_per_trade(pnls),
        mdd=max_drawdown(pnls),
        downside_dev=downside_deviation(pnls),
        mean_ci95=(ci_lo, ci_hi),
    )


def compare_variants(
    name_a: str, pnls_a: list[float],
    name_b: str, pnls_b: list[float],
    alpha: float = 0.05,
    n_min: int = 100,
    d_min: float = 0.3,
) -> ABResult:
    """두 variant의 per-trade PnL을 종합 비교.

    정지규칙 (pre-registered):
        n_a ≥ n_min AND n_b ≥ n_min
        AND p_welch < alpha
        AND |Cohen's d| ≥ d_min
    모두 충족 시 승자 채택 권고 (stopping_rule_met=True).

    Args:
        pnls_a, pnls_b: 각 variant의 per-trade PnL (USDT 기준)
        alpha: 유의수준 (default 0.05)
        n_min: 최소 표본 (default 100)
        d_min: 최소 효과 크기 (default 0.3)
    """
    stats_a = compute_variant_stats(name_a, pnls_a)
    stats_b = compute_variant_stats(name_b, pnls_b)

    t_stat, t_df, p_welch = welch_t_test(pnls_a, pnls_b)
    d = cohens_d(pnls_a, pnls_b)
    u_stat, p_mw = mann_whitney_u(pnls_a, pnls_b)
    mean_diff = stats_a.mean_pnl - stats_b.mean_pnl
    sharpe_diff = stats_a.sharpe_pt - stats_b.sharpe_pt

    # mean_diff bootstrap CI: 두 표본을 짝없는 resample
    diff_ci_lo, diff_ci_hi = _bootstrap_mean_diff_ci(pnls_a, pnls_b)

    # 정지규칙
    enough_samples = stats_a.n >= n_min and stats_b.n >= n_min
    stopping_met = enough_samples and p_welch < alpha and abs(d) >= d_min

    # 승자 판정
    winner = None
    if stopping_met:
        winner = name_a if mean_diff > 0 else name_b
        significance = "significant"
        reasoning = (
            f"통계적으로 유의 (p={p_welch:.4f}<{alpha}, |d|={abs(d):.2f}≥{d_min}). "
            f"{winner} 평균 PnL 우세 (Δ={mean_diff:+.3f})."
        )
    elif not enough_samples:
        significance = "insufficient_data"
        reasoning = (
            f"표본 부족 (A={stats_a.n}, B={stats_b.n}, 최소 n={n_min}). "
            f"현 추세: Δ={mean_diff:+.3f}, p={p_welch:.4f}, d={d:+.2f}"
        )
    else:
        significance = "not_significant"
        reasoning = (
            f"유의차 없음 (p={p_welch:.4f}, |d|={abs(d):.2f}). "
            f"두 정책 차이 불분명 → 보수적 기본값 유지 권고."
        )

    return ABResult(
        variant_a=stats_a,
        variant_b=stats_b,
        mean_diff=mean_diff,
        t_stat=t_stat,
        t_df=t_df,
        p_welch=p_welch,
        cohens_d=d,
        u_stat=u_stat,
        p_mannwhitney=p_mw,
        mean_diff_ci95=(diff_ci_lo, diff_ci_hi),
        sharpe_diff=sharpe_diff,
        winner=winner,
        significance=significance,
        reasoning=reasoning,
        stopping_rule_met=stopping_met,
        details={
            "alpha": alpha,
            "n_min": n_min,
            "d_min": d_min,
            "has_scipy": _HAS_SCIPY,
        },
    )


def _bootstrap_mean_diff_ci(
    a: list[float], b: list[float], n_boot: int = 2000, ci: float = 0.95, seed: int = 42,
) -> tuple[float, float]:
    """mean(A) - mean(B) 의 percentile bootstrap CI"""
    import random as _rnd
    if len(a) < 5 or len(b) < 5:
        md = _mean(a) - _mean(b)
        return (md, md)
    rng = _rnd.Random(seed)
    na, nb = len(a), len(b)
    diffs: list[float] = []
    for _ in range(n_boot):
        sa = [a[rng.randint(0, na - 1)] for _ in range(na)]
        sb = [b[rng.randint(0, nb - 1)] for _ in range(nb)]
        diffs.append(sum(sa) / na - sum(sb) / nb)
    diffs.sort()
    alpha = (1 - ci) / 2
    return diffs[int(alpha * n_boot)], diffs[int((1 - alpha) * n_boot) - 1]


# ============================================================================
# DB 조회 헬퍼 — storage에서 variant별 PnL 로드
# ============================================================================


def load_variant_pnls(storage, variant: str, limit: int = 5000) -> list[float]:
    """trades 테이블에서 특정 variant의 per-trade PnL 조회 (최신순 → 역순)"""
    try:
        cursor = storage.conn.execute(
            "SELECT pnl FROM trades WHERE variant = ? ORDER BY timestamp DESC LIMIT ?",
            (variant, int(limit)),
        )
        rows = cursor.fetchall()
        # 시간순 오름차순으로 뒤집기 (MDD 계산용)
        pnls = [float(r[0]) for r in rows if r and r[0] is not None]
        pnls.reverse()
        return pnls
    except Exception:
        return []
