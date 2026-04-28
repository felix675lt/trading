#!/usr/bin/env python3
"""[Patch M, 2026-04-28] Pattern Memory Bank 사전 빌드

용도: 학습 사이클 기다리지 않고 즉시 Pattern Bank 빌드.
4개 심볼(BTC/ETH/SOL/DOGE)의 전체 캔들 데이터를 인덱싱.

실행: ./venv/bin/python3 scripts/build_pattern_banks.py
출력: data/pattern_bank/<symbol>_5m.npz
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

# Project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.features import FeatureEngineer
from core.patterns.memory_bank import PatternMemoryBank

SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT"]
TIMEFRAME = "5m"
DB_PATH = "data/autotrader.db"
OUT_DIR = Path("data/pattern_bank")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fe = FeatureEngineer()

    print("=" * 70)
    print("🏗️  Pattern Memory Bank 사전 빌드")
    print("=" * 70)

    total_start = time.time()
    total_patterns = 0

    for sym in SYMBOLS:
        print(f"\n📊 {sym}")
        sym_safe = sym.replace("/", "_").replace(":", "_")
        out_path = OUT_DIR / f"{sym_safe}_{TIMEFRAME}.npz"

        # 1) 캔들 로드
        t0 = time.time()
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT timestamp, open, high, low, close, volume FROM candles "
            "WHERE symbol=? AND timeframe=? ORDER BY timestamp ASC",
            conn,
            params=(sym, TIMEFRAME),
        )
        conn.close()
        if df.empty:
            print(f"  ⚠️  데이터 없음 — skip")
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        print(f"  📁 캔들 {len(df):,}개 로드 ({time.time()-t0:.1f}s)")

        # 2) 피처 생성
        t0 = time.time()
        df = fe.generate(df)
        print(f"  🔧 피처 생성 ({time.time()-t0:.1f}s) — {df.shape[1]}개 컬럼")

        # 3) Pattern Bank 빌드
        t0 = time.time()
        bank = PatternMemoryBank()
        bank.build_from_dataframe(df, symbol=sym)
        print(f"  🧠 인덱스 빌드 ({time.time()-t0:.1f}s) — {len(bank.embeddings):,}개 패턴")

        # 4) 저장
        t0 = time.time()
        bank.save(out_path)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  💾 저장 ({time.time()-t0:.1f}s) — {size_mb:.1f}MB")

        # 5) 추론 테스트
        t0 = time.time()
        result = bank.predict(df.iloc[[-1]])
        elapsed_ms = (time.time() - t0) * 1000
        if result:
            print(
                f"  🔍 추론 테스트 {elapsed_ms:.1f}ms — "
                f"fwd_1h={result.fwd_1h_mean*100:+.3f}% "
                f"WR={result.fwd_1h_winrate*100:.0f}% "
                f"sim={result.similarity_meanK:.3f}"
            )
        total_patterns += len(bank.embeddings)

    print("\n" + "=" * 70)
    print(f"✅ 완료 — {total_patterns:,}개 패턴 인덱싱 ({time.time()-total_start:.1f}s)")
    print(f"   디스크: {sum(p.stat().st_size for p in OUT_DIR.glob('*.npz')) / 1e6:.1f}MB")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
