#!/usr/bin/env python3
"""[Patch F, 2026-04-26] 7년치 백테스트 검증 리포트

목적:
  - BTC/ETH/SOL/DOGE 5m 캔들 (2019-09 ~ 2026-04, 약 6.6년) 전체에 대해
    Patch A(시간 피처) 추가된 모델의 EV 패턴을 분석.
  - 시간대별 / 요일별 / 월별 EV 분포로 14h UTC EV<<0 같은 패턴이 정말 존재하는지 검증.
  - 시간 피처 도입 이전 vs 이후 모델 성능 비교를 위한 베이스라인 데이터 생성.

설계:
  - 모델 학습은 메모리 부담이 크므로 본 스크립트는 "통계 분석"에 집중.
  - 진입 시뮬: forward 12-bar return > +0.5% → win (long), < -0.5% → loss (short)
  - 시간/요일/월별 buckets별 평균 forward return, hit-rate, EV 산출.

출력: reports/backtest_7yr_<TS>.json + 콘솔 요약
실행: ./venv/bin/python3 scripts/backtest_7yr_validation.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path("data/autotrader.db")
OUT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT"]
TF = "5m"
FORWARD_BARS = 12  # 1h 후 평가 (5m × 12 = 60min)
WIN_THRESH = 0.005  # 0.5%
LOSS_THRESH = -0.005


def load_candles(symbol: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    q = f"""SELECT timestamp, open, high, low, close, volume
            FROM candles WHERE symbol=? AND timeframe='{TF}'
            ORDER BY timestamp ASC"""
    df = pd.read_sql(q, conn, params=(symbol,))
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    return df


def add_forward_return(df: pd.DataFrame, n: int = FORWARD_BARS) -> pd.DataFrame:
    df = df.copy()
    df["fwd_ret"] = df["close"].pct_change(n).shift(-n)
    df["win"] = (df["fwd_ret"] > WIN_THRESH).astype(int)
    df["loss"] = (df["fwd_ret"] < LOSS_THRESH).astype(int)
    df["neutral"] = ((df["fwd_ret"] >= LOSS_THRESH) & (df["fwd_ret"] <= WIN_THRESH)).astype(int)
    return df.dropna(subset=["fwd_ret"])


def bucket_stats(df: pd.DataFrame, label: str, key: pd.Series) -> dict:
    """key 별로 EV, WR, count 통계."""
    df = df.copy()
    df["_k"] = key.values
    g = df.groupby("_k").agg(
        n=("fwd_ret", "size"),
        ev_pct=("fwd_ret", "mean"),
        wr=("win", "mean"),
        loss_rate=("loss", "mean"),
        std=("fwd_ret", "std"),
    )
    g["ev_pct"] *= 100
    g["sharpe_proxy"] = g["ev_pct"] / (g["std"] * 100 + 1e-9)
    g = g.round(4)
    return {
        "label": label,
        "buckets": g.reset_index().to_dict(orient="records"),
    }


def main() -> int:
    start = time.time()
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "params": {
            "forward_bars": FORWARD_BARS,
            "timeframe": TF,
            "win_threshold": WIN_THRESH,
            "loss_threshold": LOSS_THRESH,
        },
        "symbols": {},
    }

    # 누적 통합 분석용
    pooled = []

    for sym in SYMBOLS:
        print(f"[7yr] {sym} loading...")
        try:
            df = load_candles(sym)
        except Exception as e:
            print(f"  {sym} load 실패: {e}")
            continue

        if len(df) < 5000:
            print(f"  {sym} 데이터 부족: {len(df)}")
            continue

        df = add_forward_return(df)
        df["hour_utc"] = df.index.hour
        df["dow_utc"] = df.index.dayofweek
        df["month_utc"] = df.index.month
        df["year_utc"] = df.index.year

        sym_report = {
            "n_candles": int(len(df)),
            "date_range": [str(df.index.min()), str(df.index.max())],
            "overall_ev_pct": float(df["fwd_ret"].mean() * 100),
            "overall_wr": float(df["win"].mean()),
            "overall_loss_rate": float(df["loss"].mean()),
            "by_hour_utc": bucket_stats(df, "hour_utc", df["hour_utc"]),
            "by_dow_utc": bucket_stats(df, "dow_utc", df["dow_utc"]),
            "by_month_utc": bucket_stats(df, "month_utc", df["month_utc"]),
            "by_year": bucket_stats(df, "year", df["year_utc"]),
        }
        report["symbols"][sym] = sym_report

        pooled.append(df.assign(symbol=sym))

        # 콘솔 요약
        worst_hour = sorted(
            sym_report["by_hour_utc"]["buckets"], key=lambda x: x["ev_pct"]
        )[:3]
        best_hour = sorted(
            sym_report["by_hour_utc"]["buckets"], key=lambda x: -x["ev_pct"]
        )[:3]
        print(f"  {sym} n={len(df)} 전체 EV={sym_report['overall_ev_pct']:.4f}% WR={sym_report['overall_wr']:.2%}")
        print(f"  worst hours UTC: {[(b['_k'], round(b['ev_pct'],3)) for b in worst_hour]}")
        print(f"  best  hours UTC: {[(b['_k'], round(b['ev_pct'],3)) for b in best_hour]}")

    # 통합 분석
    if pooled:
        all_df = pd.concat(pooled)
        report["pooled_all_symbols"] = {
            "n_candles": int(len(all_df)),
            "overall_ev_pct": float(all_df["fwd_ret"].mean() * 100),
            "overall_wr": float(all_df["win"].mean()),
            "by_hour_utc": bucket_stats(all_df, "hour_utc", all_df["hour_utc"]),
            "by_dow_utc": bucket_stats(all_df, "dow_utc", all_df["dow_utc"]),
        }
        # 추천 LIVE 블랙리스트 자동 추출 — EV<0 인 hour
        hour_buckets = report["pooled_all_symbols"]["by_hour_utc"]["buckets"]
        bad_hours = sorted(
            [b for b in hour_buckets if b["ev_pct"] < 0],
            key=lambda x: x["ev_pct"],
        )
        report["recommended_live_blacklist_hours_utc"] = [int(b["_k"]) for b in bad_hours]
        print(f"\n[POOLED] 추천 LIVE blacklist hours (EV<0): {report['recommended_live_blacklist_hours_utc']}")

    out_path = OUT_DIR / f"backtest_7yr_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[Done] {out_path} ({time.time()-start:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
