#!/usr/bin/env python3
"""[Patch Y, 2026-06-19] 신뢰도 캘리브레이터 적합.

라벨 생성: 각 방향성 신호(long/short)에 대해 신호 시점 종가 대비 1h(12바) 뒤
종가의 방향이 예측과 일치하면 적중(1), 아니면 0.
→ isotonic으로 raw_conf → P(적중) 매핑 적합 후 models_saved/confidence_calibrator.json 저장.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.learning.calibration import ConfidenceCalibrator

DB = "data/autotrader.db"
FWD_BARS = 12  # 1h (5m × 12)
OUT = "models_saved/confidence_calibrator.json"


def main() -> int:
    conn = sqlite3.connect(DB)
    sig = pd.read_sql_query(
        "SELECT timestamp, symbol, confidence, "
        "json_extract(metadata,'$.action') AS action "
        "FROM signals WHERE confidence > 0",
        conn,
    )
    sig = sig[sig["action"].isin(["long", "short"])].copy()
    sig["ts"] = pd.to_datetime(sig["timestamp"], utc=True, errors="coerce")
    sig = sig.dropna(subset=["ts"])
    print(f"방향성 신호 {len(sig):,}건")

    rows = []
    for symbol, grp in sig.groupby("symbol"):
        cd = pd.read_sql_query(
            "SELECT timestamp, close FROM candles WHERE symbol=? AND timeframe='5m' ORDER BY timestamp ASC",
            conn, params=(symbol,),
        )
        if cd.empty:
            continue
        cd["ts"] = pd.to_datetime(cd["timestamp"], utc=True, errors="coerce")
        cd = cd.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        cd["fwd_close"] = cd["close"].shift(-FWD_BARS)
        # 신호 시각을 가장 가까운(과거) 캔들에 매칭
        g = grp.sort_values("ts")
        merged = pd.merge_asof(g, cd[["ts", "close", "fwd_close"]], on="ts", direction="backward")
        merged = merged.dropna(subset=["close", "fwd_close"])
        if merged.empty:
            continue
        fwd_ret = (merged["fwd_close"] - merged["close"]) / merged["close"]
        correct = np.where(
            merged["action"] == "long", (fwd_ret > 0).astype(int),
            (fwd_ret < 0).astype(int),
        )
        for c, ok in zip(merged["confidence"].values, correct):
            rows.append((float(c), int(ok)))

    conn.close()
    if not rows:
        print("❌ 라벨 생성 실패")
        return 1
    conf = np.array([r[0] for r in rows])
    correct = np.array([r[1] for r in rows])
    print(f"라벨 {len(rows):,}건 | 전체 적중률 {correct.mean()*100:.1f}%")

    # 신뢰도 구간별 실제 적중률 (리포트용)
    print("\nraw_conf → 실제 적중률 (캘리브레이션 전):")
    for lo in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = (conf >= lo) & (conf < lo + 0.1)
        if m.sum() > 0:
            print(f"  {lo:.1f}~{lo+0.1:.1f}: n={m.sum():>6,} 적중 {correct[m].mean()*100:5.1f}%")

    cal = ConfidenceCalibrator(downward_only=True)
    if not cal.fit(conf, correct):
        print("❌ 적합 실패")
        return 1
    cal.save(OUT)
    print(f"\n✅ 저장: {OUT}")
    print("매핑 (raw → calibrated, downward_only):")
    for c in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"  {c:.1f} → {cal.transform(c):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
