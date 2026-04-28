"""[Patch M, 2026-04-28] Pattern Memory Bank — Retrieval-Augmented Trading

사용자 통찰 (2026-04-28):
> "데이터를 백업해서 저장한다면 비대가 될 일이 없다.
>  패턴이나 데이터가 필요할 때 백업된 자료들에서 확인하면 됨.
>  퀀트트레이딩에 데이터가 필요한거니깐."

이 통찰을 시스템 차원에서 구현. ML 모델이 691k 캔들의 패턴을 weights로
"압축" 저장하는 대신, 데이터를 그대로 두고 필요할 때 직접 retrieve.

설계 철학:
  - Raw 데이터 = 정직한 사실. 절대 손실/압축하지 않음.
  - 모델 = 도구. 작게 유지.
  - 결정 = retrieval 기반 reasoning.

Phase 1 (이 파일):
  - Compact 16-dim 패턴 시그니처 (key indicators 위주)
  - Numpy brute-force cosine similarity (외부 의존성 없음)
  - 691k 패턴 검색 ~5ms (벡터화)
  - Forward return 통계 집계 (1h, 4h horizon)
  - 디스크 persist (npz 압축)

Phase 2+ (추후):
  - Learned encoder (LSTM hidden state)
  - FAISS HNSW (대규모 가속)
  - Decision Fusion + Conformal Prediction

사용:
    bank = PatternMemoryBank()
    bank.build_from_dataframe(df)  # 학습 데이터로 인덱스 구축
    bank.save("data/pattern_bank/btc_5m.npz")

    bank = PatternMemoryBank.load("data/pattern_bank/btc_5m.npz")
    pred = bank.predict(current_features)  # → forward return distribution
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# Pattern signature 구성 — 시계열 패턴 식별을 위한 핵심 피처들.
# 각 캔들의 "현재 상태"를 16-dim 벡터로 압축.
# 이 벡터들의 cosine similarity = 패턴 유사도.
# 주의: features.py 실제 출력 컬럼만 사용 (btc_*는 학습 시에만 주입되므로 제외).
SIGNATURE_FEATURES = [
    # 단기/중기 모멘텀 (3 dim)
    "returns_1", "returns_5", "returns_20",
    # 변동성 레짐 (3 dim)
    "atr_pct", "bb_pct", "bb_width",
    # 모멘텀 지표 (3 dim)
    "rsi_14", "rsi_7", "rsi_21",
    # 추세 (1 dim)
    "macd_hist",
    # 스토캐스틱 (2 dim)
    "stoch_k", "stoch_d",
    # 거래량 레짐 (2 dim)
    "vol_ratio", "vol_std",
    # 시간 사이클 (2 dim) — Patch A에서 추가
    "hour_sin", "hour_cos",
]
# Total: 16 dim — features.py 실제 출력에 모두 존재 확인 (2026-04-28)


@dataclass
class PatternStats:
    """패턴 검색 결과 통계."""
    n_neighbors: int
    fwd_1h_mean: float
    fwd_1h_median: float
    fwd_1h_std: float
    fwd_1h_winrate: float
    fwd_4h_mean: float
    fwd_4h_winrate: float
    confidence: float          # 0~1, 표준편차 기반 (낮을수록 일관됨)
    similarity_top1: float     # 가장 유사한 패턴의 cosine similarity
    similarity_meanK: float    # top-K 평균 similarity

    def to_signal(self) -> dict:
        """ML 신호 형식으로 변환 — strategy_manager가 ml_signal과 동일하게 처리 가능."""
        # forward 1h return 부호로 방향, magnitude로 confidence
        direction = "long" if self.fwd_1h_mean > 0.001 else (
            "short" if self.fwd_1h_mean < -0.001 else "neutral"
        )
        # signal: -1~1 범위로 정규화 (1h fwd return을 1% 기준)
        signal = float(np.clip(self.fwd_1h_mean / 0.01, -1.0, 1.0))
        return {
            "signal": signal,
            "confidence": float(self.confidence),
            "direction": direction,
            "n_neighbors": self.n_neighbors,
            "winrate_1h": self.fwd_1h_winrate,
            "winrate_4h": self.fwd_4h_winrate,
            "ev_1h_pct": self.fwd_1h_mean * 100,
            "ev_4h_pct": self.fwd_4h_mean * 100,
            "similarity": self.similarity_meanK,
            "source": "pattern_memory_bank",
        }


class PatternMemoryBank:
    """패턴 메모리 뱅크 — 과거 캔들 패턴을 retrieval 가능한 형태로 인덱싱.

    Phase 1: Numpy brute-force cosine similarity.
    - 691k 패턴 × 16 dim = 44MB float32
    - Search time: ~5ms (벡터화 dot product)
    - 적용 가능 한도: ~5M 패턴 (그 이상은 Phase 3 FAISS로)

    Forward return horizons:
    - 1h = 12 bars (5분봉 기준)
    - 4h = 48 bars
    """

    FORWARD_BARS_1H = 12
    FORWARD_BARS_4H = 48

    def __init__(
        self,
        signature_dim: int = len(SIGNATURE_FEATURES),
        k_neighbors: int = 100,
        min_neighbors_for_signal: int = 30,
    ):
        self.signature_dim = signature_dim
        self.k_neighbors = k_neighbors
        self.min_neighbors_for_signal = min_neighbors_for_signal

        # Index 구조
        self.embeddings: np.ndarray = np.zeros((0, signature_dim), dtype=np.float32)
        self.timestamps: np.ndarray = np.zeros(0, dtype=np.int64)  # unix sec
        self.fwd_1h_returns: np.ndarray = np.zeros(0, dtype=np.float32)
        self.fwd_4h_returns: np.ndarray = np.zeros(0, dtype=np.float32)
        self.symbols: list[str] = []

        # Normalize 통계 (전체 인덱스 평균/std로 z-score)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

        # 메타데이터
        self.built_at: Optional[datetime] = None
        self.symbol_filter: Optional[str] = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _extract_signature(self, df: pd.DataFrame) -> np.ndarray:
        """DataFrame에서 signature 피처 추출 + NaN 처리."""
        cols = [c for c in SIGNATURE_FEATURES if c in df.columns]
        missing = [c for c in SIGNATURE_FEATURES if c not in df.columns]
        if missing:
            logger.debug(f"[PatternBank] 누락 피처 {len(missing)}개 → 0.0 fallback: {missing[:5]}")
        # 누락 컬럼은 0으로 채워 합치기
        sig_data = []
        for c in SIGNATURE_FEATURES:
            if c in df.columns:
                sig_data.append(df[c].values)
            else:
                sig_data.append(np.zeros(len(df), dtype=np.float32))
        sig = np.column_stack(sig_data).astype(np.float32)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        return sig

    def _compute_forward_returns(
        self, df: pd.DataFrame, n_bars: int
    ) -> np.ndarray:
        """forward N-bar pct return 계산."""
        if "close" not in df.columns:
            return np.zeros(len(df), dtype=np.float32)
        close = df["close"].values.astype(np.float64)
        fwd = np.zeros(len(close), dtype=np.float32)
        if len(close) > n_bars:
            fwd[:-n_bars] = ((close[n_bars:] - close[:-n_bars]) / close[:-n_bars]).astype(np.float32)
        # 마지막 n_bars개는 forward 계산 불가 → 0 (제외 마스크에서 처리)
        return fwd

    def build_from_dataframe(self, df: pd.DataFrame, symbol: str = "?"):
        """주어진 캔들 DF로 인덱스 구축. 마지막 48봉(4h)은 forward 계산 불가라 제외."""
        if len(df) < self.FORWARD_BARS_4H + 100:
            logger.warning(f"[PatternBank] 데이터 부족 ({len(df)}봉) — 인덱스 구축 skip")
            return

        sig = self._extract_signature(df)
        fwd_1h = self._compute_forward_returns(df, self.FORWARD_BARS_1H)
        fwd_4h = self._compute_forward_returns(df, self.FORWARD_BARS_4H)

        # forward 계산 가능한 부분만 채택 (마지막 4h 제외)
        n_valid = len(df) - self.FORWARD_BARS_4H
        sig = sig[:n_valid]
        fwd_1h = fwd_1h[:n_valid]
        fwd_4h = fwd_4h[:n_valid]

        # Normalize: z-score (전체 데이터 통계)
        self._mean = sig.mean(axis=0)
        self._std = sig.std(axis=0) + 1e-8
        sig_normalized = (sig - self._mean) / self._std

        # L2 정규화 → cosine similarity = dot product
        norms = np.linalg.norm(sig_normalized, axis=1, keepdims=True) + 1e-8
        sig_unit = sig_normalized / norms

        # timestamp (index가 datetime이면 unix로)
        try:
            ts = df.index[:n_valid].astype(np.int64) // 10**9
        except Exception:
            ts = np.arange(n_valid, dtype=np.int64)

        # 인덱스에 누적 (incremental 가능하도록)
        self.embeddings = sig_unit
        self.timestamps = ts.astype(np.int64)
        self.fwd_1h_returns = fwd_1h
        self.fwd_4h_returns = fwd_4h
        self.symbols = [symbol] * n_valid
        self.symbol_filter = symbol
        self.built_at = datetime.utcnow()

        logger.info(
            f"[PatternBank] {symbol} 인덱스 구축 완료 — "
            f"{n_valid:,}개 패턴 | dim={sig.shape[1]} | "
            f"메모리 {sig_unit.nbytes / 1e6:.1f}MB"
        )

    # ------------------------------------------------------------------
    # Search & Predict
    # ------------------------------------------------------------------
    def predict(self, current_row: pd.Series | pd.DataFrame, k: Optional[int] = None) -> Optional[PatternStats]:
        """현재 상태에 대한 retrieval 예측.

        Args:
            current_row: 단일 시점의 피처 (Series 또는 1-row DataFrame)
            k: top-K (기본 self.k_neighbors)

        Returns:
            PatternStats 또는 None (인덱스 비어있거나 데이터 부족 시)
        """
        if len(self.embeddings) < self.min_neighbors_for_signal:
            return None
        if self._mean is None or self._std is None:
            return None

        k = k or self.k_neighbors

        # 단일 row → DataFrame 정규화
        if isinstance(current_row, pd.Series):
            df_one = current_row.to_frame().T
        else:
            df_one = current_row.iloc[[-1]] if isinstance(current_row, pd.DataFrame) else current_row

        sig = self._extract_signature(df_one)  # (1, signature_dim)
        sig_normalized = (sig - self._mean) / self._std
        norm = np.linalg.norm(sig_normalized, axis=1, keepdims=True) + 1e-8
        query = sig_normalized / norm  # (1, dim)

        # Cosine similarity (이미 unit vectors → dot product)
        sims = (self.embeddings @ query.T).flatten()  # (N,)

        # Top-K
        if len(sims) <= k:
            top_idx = np.argsort(sims)[::-1]
        else:
            top_idx = np.argpartition(sims, -k)[-k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        sel_fwd_1h = self.fwd_1h_returns[top_idx]
        sel_fwd_4h = self.fwd_4h_returns[top_idx]
        sel_sims = sims[top_idx]

        # 통계
        fwd_1h_mean = float(np.mean(sel_fwd_1h))
        fwd_1h_median = float(np.median(sel_fwd_1h))
        fwd_1h_std = float(np.std(sel_fwd_1h))
        fwd_1h_winrate = float((sel_fwd_1h > 0).mean())
        fwd_4h_mean = float(np.mean(sel_fwd_4h))
        fwd_4h_winrate = float((sel_fwd_4h > 0).mean())

        # confidence: 표준편차가 작을수록(일관될수록) 높음 + similarity가 높을수록 높음
        # 정규화: |mean| / (std + eps) = signal-to-noise ratio
        snr = abs(fwd_1h_mean) / (fwd_1h_std + 1e-4)
        sim_factor = float(sel_sims.mean())
        confidence = float(np.clip(snr * sim_factor, 0.0, 1.0))

        return PatternStats(
            n_neighbors=int(len(top_idx)),
            fwd_1h_mean=fwd_1h_mean,
            fwd_1h_median=fwd_1h_median,
            fwd_1h_std=fwd_1h_std,
            fwd_1h_winrate=fwd_1h_winrate,
            fwd_4h_mean=fwd_4h_mean,
            fwd_4h_winrate=fwd_4h_winrate,
            confidence=confidence,
            similarity_top1=float(sel_sims[0]) if len(sel_sims) > 0 else 0.0,
            similarity_meanK=sim_factor,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        """디스크에 npz 압축 저장."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            embeddings=self.embeddings,
            timestamps=self.timestamps,
            fwd_1h=self.fwd_1h_returns,
            fwd_4h=self.fwd_4h_returns,
            mean=self._mean if self._mean is not None else np.zeros(self.signature_dim),
            std=self._std if self._std is not None else np.ones(self.signature_dim),
            meta=np.array([
                self.symbol_filter or "?",
                self.built_at.isoformat() if self.built_at else "?",
                str(self.signature_dim),
                str(self.k_neighbors),
            ], dtype=object),
        )
        tmp.rename(p)
        logger.info(
            f"[PatternBank] 저장 완료: {p} "
            f"({p.stat().st_size / 1e6:.1f}MB, {len(self.embeddings):,}개 패턴)"
        )

    @classmethod
    def load(cls, path: str | Path) -> "PatternMemoryBank":
        """디스크에서 인덱스 로드."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        data = np.load(p, allow_pickle=True)
        meta = data["meta"]
        bank = cls(
            signature_dim=int(meta[2]),
            k_neighbors=int(meta[3]),
        )
        bank.embeddings = data["embeddings"]
        bank.timestamps = data["timestamps"]
        bank.fwd_1h_returns = data["fwd_1h"]
        bank.fwd_4h_returns = data["fwd_4h"]
        bank._mean = data["mean"]
        bank._std = data["std"]
        bank.symbol_filter = str(meta[0])
        try:
            bank.built_at = datetime.fromisoformat(str(meta[1]))
        except Exception:
            bank.built_at = None
        logger.info(
            f"[PatternBank] 로드 완료: {p.name} "
            f"({len(bank.embeddings):,}개 패턴, symbol={bank.symbol_filter})"
        )
        return bank
