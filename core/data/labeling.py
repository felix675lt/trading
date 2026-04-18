"""Triple-Barrier Labeling (Lopez de Prado, Advances in Financial ML, Ch. 3)

전통적 forward-return 라벨 대비 장점:
- TP(상단), SL(하단), 시간(우측) 배리어 중 먼저 도달한 것으로 분류
- 시장 노이즈에 강건 — 중간에 SL 맞고 돌아온 경로를 올바르게 -1로 라벨링
- 변동성에 따라 배리어 폭을 ATR 기반으로 동적 조정 → regime-agnostic

사용:
    from core.data.labeling import triple_barrier_labels
    df = triple_barrier_labels(df, pt_mult=2.0, sl_mult=1.0, max_hold=24)
    # df에 'tb_label' (0=SL, 1=시간만료, 2=TP), 'tb_ret' (실현수익률) 추가
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_hold: int = 24,
    atr_col: str = "atr_14",
    side_col: str | None = None,
    min_ret: float = 0.001,
) -> pd.DataFrame:
    """Triple-Barrier 라벨링 — Lopez de Prado 표준 구현.

    Args:
        df: OHLCV + ATR 포함 DataFrame (close, high, low 필수, atr_col 존재)
        pt_mult: Profit-Take 배리어 = ATR × pt_mult (상향 거리)
        sl_mult: Stop-Loss 배리어 = ATR × sl_mult (하향 거리)
        max_hold: 시간 배리어 — 이 바 이내에 TP/SL 미도달 시 시간 만료로 라벨
        atr_col: ATR 컬럼명 (없으면 close×0.01로 fallback)
        side_col: side 컬럼 (long=+1/short=-1). None이면 long 전용
        min_ret: 시간 만료 시 |return|이 이 값 미만이면 중립(1), 아니면 방향 부호

    Returns:
        df (원본 + 컬럼 추가):
          - tb_label: 0=SL(하락), 1=시간만료(중립), 2=TP(상승) — XGB 호환 3-class
          - tb_ret: 배리어 히트 시점의 실현 수익률
          - tb_hit: "pt"/"sl"/"time" — 어느 배리어에 먼저 닿았는지
          - tb_t1: 만료 바 인덱스
    """
    out = df.copy()
    n = len(out)
    if n < max_hold + 1:
        out["tb_label"] = 1
        out["tb_ret"] = 0.0
        out["tb_hit"] = "time"
        out["tb_t1"] = np.nan
        return out

    close = out["close"].values.astype(np.float64)
    high = out["high"].values.astype(np.float64)
    low = out["low"].values.astype(np.float64)

    if atr_col in out.columns:
        atr = out[atr_col].fillna(method="ffill").fillna(close * 0.01).values.astype(np.float64)
    else:
        atr = close * 0.01

    side = np.ones(n, dtype=np.int64)
    if side_col and side_col in out.columns:
        side = out[side_col].fillna(1).astype(int).values

    labels = np.full(n, 1, dtype=np.int64)  # 기본: 시간만료(중립)
    rets = np.zeros(n, dtype=np.float64)
    hits = np.array(["time"] * n, dtype=object)
    t1s = np.full(n, np.nan, dtype=np.float64)

    for i in range(n - 1):
        entry = close[i]
        a = atr[i]
        if not np.isfinite(entry) or not np.isfinite(a) or a <= 0:
            continue

        s = side[i]  # +1 long, -1 short
        # 롱: 위쪽 TP / 아래쪽 SL, 숏: 반대
        pt_price = entry + s * pt_mult * a
        sl_price = entry - s * sl_mult * a

        t_end = min(i + max_hold, n - 1)
        hit_idx = t_end
        hit_kind = "time"

        for j in range(i + 1, t_end + 1):
            if s == 1:  # long
                # SL 먼저 체크 (보수적)
                if low[j] <= sl_price:
                    hit_idx = j
                    hit_kind = "sl"
                    break
                if high[j] >= pt_price:
                    hit_idx = j
                    hit_kind = "pt"
                    break
            else:  # short
                if high[j] >= sl_price:
                    hit_idx = j
                    hit_kind = "sl"
                    break
                if low[j] <= pt_price:
                    hit_idx = j
                    hit_kind = "pt"
                    break

        exit_price = close[hit_idx]
        if hit_kind == "pt":
            exit_price = pt_price
        elif hit_kind == "sl":
            exit_price = sl_price

        ret = s * (exit_price - entry) / entry
        rets[i] = ret
        hits[i] = hit_kind
        t1s[i] = hit_idx

        if hit_kind == "pt":
            labels[i] = 2
        elif hit_kind == "sl":
            labels[i] = 0
        else:  # time barrier
            if ret > min_ret:
                labels[i] = 2
            elif ret < -min_ret:
                labels[i] = 0
            else:
                labels[i] = 1

    out["tb_label"] = labels
    out["tb_ret"] = rets
    out["tb_hit"] = hits
    out["tb_t1"] = t1s
    return out


def get_sample_weights(df: pd.DataFrame, label_col: str = "tb_label") -> np.ndarray:
    """라벨 불균형 보정용 샘플 가중치.

    tb_label 분포에 따라 1/freq로 가중 — XGB/sklearn sample_weight에 투입.
    """
    labels = df[label_col].values
    counts = pd.Series(labels).value_counts()
    freqs = counts / counts.sum()
    weights = np.array([1.0 / freqs.get(lbl, 1.0) for lbl in labels])
    return weights / weights.mean()
