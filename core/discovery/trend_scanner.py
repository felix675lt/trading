"""[Patch AE, 2026-06-30] 트렌드/네러티브 자동 스크리너.

사용자 전략: "거래량 + 추세(네러티브)를 타는 종목을 따라가야 한다."
바이낸스 선물 전 USDT-perp(639종목)을 fetch_tickers 1콜로 받아
거래량 + 모멘텀 + 신규상장을 스코어링 → 상위 후보를 발굴.

반자동 운영: 상위 후보를 PAPER에 자동 편입(데이터 수집) + 텔레그램 통지.
LIVE는 live_symbol_whitelist로 보호되어 자동 편입 대상이 아님(실자본 안전).
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Optional

import ccxt
from loguru import logger

SNAPSHOT_PATH = Path("data/known_symbols.json")

# 스테이블/래핑/이상 심볼 제외 (네러티브 무관)
_EXCLUDE_BASES = {
    "USDC", "USDT", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "USDD",
    "WBTC", "WBETH", "BTCDOM", "USTC",
}


class TrendScanner:
    def __init__(self, config: dict):
        cfg = (config.get("trading", {}) or {}).get("trend_scanner", {}) or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.min_volume_usd = float(cfg.get("min_volume_usd", 300_000_000))  # 유동성 바닥 $300M
        self.max_auto_add = int(cfg.get("max_auto_add", 3))      # 1회 스캔당 자동편입 최대
        self.top_n_report = int(cfg.get("top_n_report", 12))     # 텔레그램 보고 상위 N
        self.min_momentum_pct = float(cfg.get("min_momentum_pct", 3.0))  # |24h 변동| 최소

    # ------------------------------------------------------------------
    def _load_snapshot(self) -> set[str]:
        try:
            if SNAPSHOT_PATH.exists():
                return set(json.loads(SNAPSHOT_PATH.read_text()))
        except Exception:
            pass
        return set()

    def _save_snapshot(self, symbols: set[str]) -> None:
        try:
            SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = SNAPSHOT_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(sorted(symbols)))
            tmp.replace(SNAPSHOT_PATH)
        except Exception as e:
            logger.debug(f"[TrendScanner] 스냅샷 저장 실패: {e}")

    # ------------------------------------------------------------------
    def _scan_sync(self, tracked: set[str]) -> dict:
        ex = ccxt.binance({"options": {"defaultType": "future"}})
        ex.load_markets()
        tickers = ex.fetch_tickers()

        rows = []
        all_syms = set()
        tracked_info: dict[str, dict] = {}  # [Patch AF] 퇴출 판정용 — 추적심볼은 필터 무관 정보 제공
        for sym, t in tickers.items():
            if not sym.endswith("USDT:USDT"):
                continue
            all_syms.add(sym)
            qv = float(t.get("quoteVolume") or 0)
            pc = float(t.get("percentage") or 0)
            if sym in tracked:
                tracked_info[sym] = {
                    "volume_usd": qv, "pct": pc,
                    "last": float(t.get("last") or 0),
                }
            base = sym.split("/")[0]
            if base in _EXCLUDE_BASES:
                continue
            if qv < self.min_volume_usd:
                continue
            rows.append({"symbol": sym, "base": base, "volume_usd": qv, "pct": pc})

        # 신규 상장 감지 (직전 스냅샷에 없던 심볼)
        prev = self._load_snapshot()
        new_syms = (all_syms - prev) if prev else set()
        self._save_snapshot(all_syms)

        if not rows:
            return {"candidates": [], "new_listings": [], "report": [],
                    "tracked_info": tracked_info, "scanned": len(all_syms)}

        # 스코어링: z(log 거래량) + z(|모멘텀|) — 유동성·추세 동시 보상
        vols = [math.log10(r["volume_usd"]) for r in rows]
        moms = [abs(r["pct"]) for r in rows]
        vmean, vstd = _mean_std(vols)
        mmean, mstd = _mean_std(moms)
        for r in rows:
            zv = (math.log10(r["volume_usd"]) - vmean) / vstd if vstd > 0 else 0.0
            zm = (abs(r["pct"]) - mmean) / mstd if mstd > 0 else 0.0
            r["score"] = round(zv + zm, 3)
            r["is_new"] = r["symbol"] in new_syms

        rows.sort(key=lambda r: r["score"], reverse=True)

        # 자동편입 후보: 미추적 + 모멘텀 기준 충족 (신규상장은 모멘텀 무관 우대)
        candidates = [
            r for r in rows
            if r["symbol"] not in tracked
            and (abs(r["pct"]) >= self.min_momentum_pct or r["is_new"])
        ]
        new_listings = [r for r in rows if r["is_new"] and r["symbol"] not in tracked]

        return {
            "candidates": candidates,
            "new_listings": new_listings,
            "report": rows[: self.top_n_report],
            "tracked_info": tracked_info,
            "scanned": len(all_syms),
        }

    async def scan(self, tracked: set[str]) -> dict:
        """이벤트 루프 차단 없이 스캔 (fetch_tickers ~1.6초)."""
        try:
            return await asyncio.to_thread(self._scan_sync, tracked)
        except Exception as e:
            logger.warning(f"[TrendScanner] 스캔 실패: {e}")
            return {"candidates": [], "new_listings": [], "report": [],
                    "tracked_info": {}, "scanned": 0}


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    return m, math.sqrt(var)
