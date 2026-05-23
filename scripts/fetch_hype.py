#!/usr/bin/env python3
"""[Patch P, 2026-05-22] HYPE/USDT:USDT 5분봉 캔들 수집 + DB 적재.

HYPE는 신규 추가 종목이라 DB에 데이터 없음. 직접 ccxt로 fetch.
Binance Futures HYPE/USDT perpetual.
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import ccxt

SYMBOL = "HYPE/USDT:USDT"
TIMEFRAME = "5m"
DB_PATH = "data/autotrader.db"
LIMIT_PER_REQ = 1000


def main() -> int:
    print(f"=== HYPE 캔들 수집 시작 ===")
    ex = ccxt.binance({"options": {"defaultType": "future"}})
    ex.load_markets()
    if SYMBOL not in ex.markets:
        print(f"❌ {SYMBOL} not in Binance futures")
        return 1

    # 가능한 가장 옛날부터 — Binance는 ~5-6개월 정도 보존
    # HYPE 토큰 발행 ~2024-11 → 약 6개월 데이터
    since = ex.parse8601("2024-11-01T00:00:00Z")
    all_candles = []
    while True:
        try:
            ohlcv = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=LIMIT_PER_REQ)
            if not ohlcv:
                break
            all_candles.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if last_ts <= since:
                break
            since = last_ts + 1
            print(f"  수집 중… {len(all_candles):,}개 (마지막 {ex.iso8601(last_ts)})")
            time.sleep(0.1)
            if len(ohlcv) < LIMIT_PER_REQ:
                break
        except Exception as e:
            print(f"  오류: {e} → 재시도 5초 후")
            time.sleep(5)
            continue

    if not all_candles:
        print("❌ 데이터 수집 실패")
        return 1

    # 중복 제거
    seen = set()
    uniq = []
    for c in all_candles:
        if c[0] not in seen:
            seen.add(c[0])
            uniq.append(c)
    uniq.sort(key=lambda x: x[0])
    print(f"\n총 {len(uniq):,}개 캔들 수집 완료 ({ex.iso8601(uniq[0][0])} ~ {ex.iso8601(uniq[-1][0])})")

    # DB 적재 — 기존 포맷 맞춤 (exchange='binance', timestamp='YYYY-MM-DD HH:MM:SS')
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = []
    from datetime import datetime, timezone
    for ts_ms, o, h, l, c, v in uniq:
        ts_str = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(("binance", SYMBOL, TIMEFRAME, ts_str, o, h, l, c, v))
    cur.executemany(
        "INSERT OR IGNORE INTO candles (exchange, symbol, timeframe, timestamp, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    inserted = cur.rowcount
    cur.execute("SELECT COUNT(*) FROM candles WHERE symbol=? AND timeframe=?", (SYMBOL, TIMEFRAME))
    total = cur.fetchone()[0]
    conn.close()
    print(f"✅ DB 적재 완료 — 새로 추가 {inserted}, 총 {total:,}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
