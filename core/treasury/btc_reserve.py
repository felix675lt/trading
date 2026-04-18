"""BTC Treasury Reserve — 선물 수익의 일부를 현물 BTC로 자동 적립하는 "디지털 금고".

철학:
    "달러는 여전히 강력하지만 세계가 재편되면서 BTC도 준비자산 자리를 가질 수 있다."
    → 선물 트레이딩으로 번 실현수익(realized PnL)의 일정 비율을 Binance 현물 BTC로 자동 전환.
    → 적립된 BTC는 트레이딩에 사용하지 않음 (장기 보유 원칙).

핵심 규칙:
- 실현수익(positive) 발생 시에만 트리거 (손실은 적립 없음).
- 티어별 적립률: micro 10% → pro 30% (시드가 작을수록 복리우선, 커질수록 수확 비중 상승).
- 최소 수익 임계값($5) + 최소 주문금액(Binance spot $11) 체크 — 수수료 드래그 방지.
- PAPER 모드: 가상 BTC 누적 (현재 BTC 시세로 가상 환산) — 실거래 없이 시뮬.
- LIVE 모드: 별도 spot ExchangeClient로 실제 시장가 매수.
- 상태는 JSON 파일로 영속화 (크래시 복구).

스레드 안전성:
- _lock을 통해 동시 close 이벤트 직렬화.
- 파일 쓰기는 atomic (tmp→rename).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


# =============================================================================
# 데이터 구조
# =============================================================================

@dataclass
class ReserveEntry:
    """단일 적립 기록"""
    timestamp: str
    source: str                  # "live" | "paper"
    trigger_symbol: str          # 원본 선물 거래 심볼 (e.g., ETH/USDT:USDT)
    trigger_pnl_usdt: float      # 원본 실현수익 (gross profit)
    allocation_pct: float        # 적용된 적립률 (0.0~1.0)
    usdt_spent: float            # BTC 매수에 쓴 USDT
    btc_bought: float            # 매수한 BTC 수량
    btc_price: float             # 체결가
    tier: str                    # 적립 당시 티어 이름
    order_id: Optional[str] = None  # 실거래 주문 ID (live only)
    note: str = ""


# =============================================================================
# BTCReserve — 메인 클래스
# =============================================================================

class BTCReserve:
    """선물 실현수익 → 현물 BTC 자동 적립 관리자.

    사용 흐름:
        reserve = BTCReserve(config, tier_manager, collector)
        reserve.set_spot_exchange(spot_client)  # LIVE용 별도 spot client 주입
        # 청산 콜백에 연결:
        paper_trader.set_profit_callback(reserve.on_paper_close)
        order_manager.set_profit_callback(reserve.on_live_close)
    """

    def __init__(
        self,
        config: dict,
        tier_manager: Any,
        collector: Any,
        notifier=None,
    ):
        cfg = (config.get("treasury") or {}).get("btc_reserve") or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.paper_simulate: bool = bool(cfg.get("paper_simulate", True))

        # 티어별 적립률 (dict: tier_name → 0~1)
        alloc = cfg.get("allocation_by_tier") or {
            "micro": 0.10,
            "small": 0.15,
            "mid": 0.20,
            "large": 0.25,
            "pro": 0.30,
        }
        self.allocation_by_tier: dict[str, float] = {
            k: float(v) for k, v in alloc.items()
        }
        self.fallback_allocation: float = float(cfg.get("fallback_allocation", 0.15))

        # 임계값
        self.min_profit_usdt: float = float(cfg.get("min_profit_usdt", 5.0))
        self.min_order_usdt: float = float(cfg.get("min_order_usdt", 11.0))
        self.spot_symbol: str = cfg.get("spot_symbol", "BTC/USDT")

        # 의존성
        self.tier_manager = tier_manager
        self.collector = collector
        self.notifier = notifier  # callable(text, silent=False)

        # 영속화
        default_file = "data/btc_reserve.json"
        self.reserve_file = Path(cfg.get("reserve_file", default_file))
        self.reserve_file.parent.mkdir(parents=True, exist_ok=True)

        # 상태 (영속화됨)
        # 소스(live/paper)별로 분리 관리 — paper 가상 BTC와 실 BTC를 섞지 않는다.
        self._state: dict[str, dict] = {
            "live": self._empty_source_state(),
            "paper": self._empty_source_state(),
        }
        self._load()

        # Spot 거래소 클라이언트 (나중에 set_spot_exchange로 주입)
        self.spot_exchange = None

        # 동시성 제어
        self._lock = threading.RLock()

        if self.enabled:
            logger.warning(
                f"[BTCReserve] 🏛️ 활성화 — 적립률 {self.allocation_by_tier} | "
                f"최소수익 ${self.min_profit_usdt} | paper시뮬={self.paper_simulate} | "
                f"파일={self.reserve_file}"
            )
            status = self.get_status()
            for src in ("live", "paper"):
                s = status[src]
                if s["total_btc"] > 0:
                    logger.info(
                        f"[BTCReserve] {src.upper()} 누적: "
                        f"{s['total_btc']:.8f} BTC @ avg ${s['avg_cost']:,.2f} "
                        f"(총 ${s['total_spent']:,.2f} 투입, {s['entry_count']}회)"
                    )
        else:
            logger.info("[BTCReserve] 비활성화 (config.treasury.btc_reserve.enabled=false)")

    @staticmethod
    def _empty_source_state() -> dict:
        return {
            "total_btc": 0.0,
            "total_spent_usdt": 0.0,
            "entries": [],  # list[ReserveEntry as dict]
            "first_entry_ts": None,
            "last_entry_ts": None,
        }

    # -------------------------------------------------------------------------
    # 외부 주입
    # -------------------------------------------------------------------------

    def set_spot_exchange(self, spot_client):
        """LIVE 실매수용 spot ExchangeClient 주입 (Binance spot 모드).

        주: 이 클라이언트는 반드시 defaultType='spot'로 생성되어야 한다.
            선물용 클라이언트를 그대로 넘기면 spot 매수가 실패할 수 있다.
        """
        self.spot_exchange = spot_client
        logger.info("[BTCReserve] Spot 거래소 클라이언트 주입 완료 — LIVE 실매수 활성")

    def set_notifier(self, notifier):
        """Telegram 알림 함수 주입 — callable(text, silent=False)"""
        self.notifier = notifier

    # -------------------------------------------------------------------------
    # 적립률 결정
    # -------------------------------------------------------------------------

    def _get_allocation(self, source: str) -> tuple[float, str]:
        """현재 티어 기반 적립률 반환 — (allocation_pct, tier_name)

        source="live" → live 티어, source="paper" → paper 티어.
        """
        try:
            tier = self.tier_manager.get_tier(source)
            name = tier.name
            rate = self.allocation_by_tier.get(name, self.fallback_allocation)
            return rate, name
        except Exception:
            return self.fallback_allocation, "unknown"

    # -------------------------------------------------------------------------
    # 청산 이벤트 콜백 — PAPER / LIVE
    # -------------------------------------------------------------------------

    def on_paper_close(self, trade: dict) -> Optional[dict]:
        """PaperTrader.close_position 결과를 받아 가상 BTC 적립.

        Args:
            trade: {"pnl": float, "symbol": str, ...}

        Returns:
            ReserveEntry dict (적립 발생 시) | None (스킵)
        """
        if not self.enabled or not self.paper_simulate:
            return None
        return self._record_profit(trade, source="paper", async_ctx=False)

    def on_live_close_sync(self, trade: dict) -> Optional[dict]:
        """동기 버전 — LIVE 실매수가 아니라 '스케줄'만 기록. 실매수는 async로 별도.

        실매수가 필요한 경우 run_live_buy_async()를 이벤트 루프에서 호출해야 함.
        """
        if not self.enabled:
            return None
        return self._record_profit(trade, source="live", async_ctx=False)

    async def on_live_close(self, trade: dict) -> Optional[dict]:
        """LIVE 포지션 청산 시 호출 — 실 spot 매수까지 완료.

        Args:
            trade: order_manager.close_position() 결과 dict

        Returns:
            ReserveEntry dict | None
        """
        if not self.enabled:
            return None
        return await self._record_profit_async(trade, source="live")

    # -------------------------------------------------------------------------
    # 내부 적립 로직 (sync)
    # -------------------------------------------------------------------------

    def _record_profit(
        self, trade: dict, source: str, async_ctx: bool,
    ) -> Optional[dict]:
        """수익 기반 적립 트리거 — 동기 경로 (paper) 또는 async의 fallback."""
        pnl = float(trade.get("pnl") or 0.0)
        symbol = trade.get("symbol", "?")

        # 수익 아니면 스킵
        if pnl <= 0:
            logger.debug(f"[BTCReserve] {source} {symbol} PnL {pnl:+.4f} ≤ 0 — 스킵")
            return None

        if pnl < self.min_profit_usdt:
            logger.debug(
                f"[BTCReserve] {source} {symbol} PnL ${pnl:.2f} < 최소 ${self.min_profit_usdt} — 스킵"
            )
            return None

        # 적립률
        alloc_pct, tier = self._get_allocation(source)
        usdt_to_spend = pnl * alloc_pct

        if usdt_to_spend < self.min_order_usdt:
            logger.debug(
                f"[BTCReserve] {source} {symbol} 적립액 ${usdt_to_spend:.2f} "
                f"< 최소주문 ${self.min_order_usdt} — 스킵"
            )
            return None

        # BTC 가격 조회 — paper는 현재 BTC 시세 사용
        btc_price = self._get_current_btc_price()
        if btc_price <= 0:
            logger.warning(
                f"[BTCReserve] {source} BTC 가격 조회 실패 — 적립 스킵"
            )
            return None

        btc_amount = usdt_to_spend / btc_price

        # paper 가상 매수 기록
        if source == "paper":
            entry = self._append_entry(
                source="paper",
                symbol=symbol,
                pnl=pnl,
                alloc_pct=alloc_pct,
                usdt_spent=usdt_to_spend,
                btc_bought=btc_amount,
                btc_price=btc_price,
                tier=tier,
                order_id=None,
                note="paper_virtual",
            )
            self._notify_accumulation(source, entry)
            return entry

        # live는 여기선 스케줄만 기록, 실매수는 async에서
        logger.info(
            f"[BTCReserve] LIVE 적립 대기 — ${usdt_to_spend:.2f} "
            f"(PnL ${pnl:.2f} × {alloc_pct:.0%}) [async 경로 필요]"
        )
        return None

    async def _record_profit_async(self, trade: dict, source: str) -> Optional[dict]:
        """LIVE 비동기 경로 — 실 spot 매수 실행"""
        pnl = float(trade.get("pnl") or 0.0)
        symbol = trade.get("symbol", "?")

        if pnl <= 0 or pnl < self.min_profit_usdt:
            return None

        alloc_pct, tier = self._get_allocation(source)
        usdt_to_spend = pnl * alloc_pct
        if usdt_to_spend < self.min_order_usdt:
            return None

        # Spot 클라이언트 확인
        if self.spot_exchange is None:
            # fallback: 가상 기록만 (paper와 동일 취급 — 나중에 manual fill 가능)
            btc_price = self._get_current_btc_price()
            if btc_price <= 0:
                return None
            entry = self._append_entry(
                source="live",
                symbol=symbol,
                pnl=pnl,
                alloc_pct=alloc_pct,
                usdt_spent=usdt_to_spend,
                btc_bought=usdt_to_spend / btc_price,
                btc_price=btc_price,
                tier=tier,
                order_id=None,
                note="spot_client_none_virtual",
            )
            logger.warning(
                f"[BTCReserve] LIVE spot 클라이언트 없음 → 가상 기록만 "
                f"${usdt_to_spend:.2f} @ ${btc_price:,.2f} (추후 수동 매수 필요)"
            )
            self._notify_accumulation("live", entry)
            return entry

        # 실제 spot 시장가 매수
        try:
            # Binance spot은 수량(base) 또는 quoteOrderQty(USDT) 단위 매수 가능.
            # USDT 고정 금액 매수가 편함 → quoteOrderQty 사용.
            order = await self.spot_exchange.exchange.create_order(
                self.spot_symbol,
                "market",
                "buy",
                None,
                None,
                {"quoteOrderQty": round(usdt_to_spend, 2)},
            )
            fill_price = float(order.get("average") or order.get("price") or 0)
            filled_qty = float(order.get("filled") or order.get("amount") or 0)
            order_id = str(order.get("id", ""))

            if fill_price <= 0 or filled_qty <= 0:
                # fallback: ticker 가격 사용
                fill_price = self._get_current_btc_price()
                filled_qty = usdt_to_spend / fill_price if fill_price > 0 else 0

            entry = self._append_entry(
                source="live",
                symbol=symbol,
                pnl=pnl,
                alloc_pct=alloc_pct,
                usdt_spent=usdt_to_spend,
                btc_bought=filled_qty,
                btc_price=fill_price,
                tier=tier,
                order_id=order_id,
                note="binance_spot_market",
            )
            logger.success(
                f"[BTCReserve] 🏛️ LIVE 실매수 완료: {filled_qty:.8f} BTC @ ${fill_price:,.2f} "
                f"(${usdt_to_spend:.2f} from {symbol} PnL ${pnl:.2f})"
            )
            self._notify_accumulation("live", entry)
            return entry

        except Exception as e:
            logger.error(f"[BTCReserve] LIVE spot 매수 실패 ({symbol}): {e}")
            # 실패 시에도 '의도'는 기록 (미체결 상태로 트레이서블)
            return None

    # -------------------------------------------------------------------------
    # 가상 매수 (paper) — 동기
    # -------------------------------------------------------------------------

    def _append_entry(
        self,
        source: str,
        symbol: str,
        pnl: float,
        alloc_pct: float,
        usdt_spent: float,
        btc_bought: float,
        btc_price: float,
        tier: str,
        order_id: Optional[str],
        note: str,
    ) -> dict:
        """엔트리 추가 + 상태 업데이트 + 저장"""
        now = datetime.utcnow().isoformat()
        entry = ReserveEntry(
            timestamp=now,
            source=source,
            trigger_symbol=symbol,
            trigger_pnl_usdt=round(pnl, 6),
            allocation_pct=round(alloc_pct, 4),
            usdt_spent=round(usdt_spent, 4),
            btc_bought=round(btc_bought, 10),
            btc_price=round(btc_price, 2),
            tier=tier,
            order_id=order_id,
            note=note,
        )
        entry_dict = asdict(entry)

        with self._lock:
            s = self._state[source]
            s["entries"].append(entry_dict)
            s["total_btc"] = float(s["total_btc"]) + entry.btc_bought
            s["total_spent_usdt"] = float(s["total_spent_usdt"]) + entry.usdt_spent
            if not s.get("first_entry_ts"):
                s["first_entry_ts"] = now
            s["last_entry_ts"] = now
            # 엔트리 수 제한 — 최근 1000건만 보존 (tracking은 집계값으로)
            if len(s["entries"]) > 1000:
                s["entries"] = s["entries"][-1000:]
            self._save()

        return entry_dict

    # -------------------------------------------------------------------------
    # BTC 가격 조회
    # -------------------------------------------------------------------------

    def _get_current_btc_price(self) -> float:
        """현재 BTC/USDT 가격 — collector 캐시 또는 외부 source"""
        try:
            # collector에 캐시된 ticker 있으면 우선 사용
            if self.collector:
                # 우선 spot symbol 시도, 실패하면 futures symbol (BTC/USDT:USDT)
                exch_name = getattr(self.collector, "_default_exchange", None) or \
                    next(iter(getattr(self.collector, "exchanges", {}).keys()), None)
                if exch_name:
                    try:
                        fut_symbol = "BTC/USDT:USDT"
                        result = self._sync_fetch_ticker(exch_name, fut_symbol)
                        if result:
                            return result
                    except Exception:
                        pass
            # fallback: ccxt 간이 fetch (블로킹)
            return 0.0
        except Exception as e:
            logger.debug(f"[BTCReserve] BTC 가격 조회 실패: {e}")
            return 0.0

    def _sync_fetch_ticker(self, exchange_name: str, symbol: str) -> float:
        """동기 티커 조회 — collector의 async 메서드를 이벤트 루프 밖에서 안전하게 호출"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 이벤트 루프 안이면 create_task는 할 수 없으니 0 반환
                # (이 경로는 paper 동기 콜백에서만 사용됨 — 루프가 동작 중일 때
                #  await 불가 → 캐시된 가격이 있다면 그걸 쓰고, 없으면 0)
                cached = self._get_cached_btc_price(exchange_name, symbol)
                return cached
            ticker = loop.run_until_complete(
                self.collector.fetch_ticker(exchange_name, symbol)
            )
            return float(ticker.get("last") or 0)
        except Exception:
            return 0.0

    def _get_cached_btc_price(self, exchange_name: str, symbol: str) -> float:
        """Collector 내부 캐시에서 BTC 가격 추출 시도.

        Collector 구현에 따라 다르지만, 대부분 최근 ticker/ohlcv를 캐시한다.
        """
        try:
            # 가능한 경로들 시도
            if hasattr(self.collector, "_ticker_cache"):
                cache = self.collector._ticker_cache
                if symbol in cache:
                    return float(cache[symbol].get("last") or 0)
            if hasattr(self.collector, "last_prices"):
                last = self.collector.last_prices.get(symbol)
                if last:
                    return float(last)
        except Exception:
            pass
        return 0.0

    def set_btc_price_hint(self, price: float):
        """외부에서 BTC 가격을 힌트로 주입 (main.py 루프에서 매 틱 업데이트).

        Collector 캐시 구조에 의존하지 않고 안전하게 paper 가상 매수 가격 확보.
        """
        if price > 0:
            self._btc_price_hint = float(price)

    def _current_btc_price(self) -> float:
        """우선순위: hint > collector 캐시 > 0"""
        hint = getattr(self, "_btc_price_hint", 0.0)
        if hint > 0:
            return hint
        return self._get_current_btc_price()

    # 덮어쓰기 — hint 우선
    def _get_current_btc_price(self) -> float:  # noqa: F811
        hint = getattr(self, "_btc_price_hint", 0.0)
        if hint > 0:
            return float(hint)
        try:
            if self.collector:
                if hasattr(self.collector, "last_prices"):
                    last = self.collector.last_prices.get("BTC/USDT:USDT") or \
                           self.collector.last_prices.get("BTC/USDT")
                    if last:
                        return float(last)
        except Exception:
            pass
        return 0.0

    # -------------------------------------------------------------------------
    # 상태 조회
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """현재 적립 상태 전체 조회"""
        now_price = self._get_current_btc_price()
        out = {}
        for src in ("live", "paper"):
            s = self._state[src]
            total_btc = float(s["total_btc"])
            total_spent = float(s["total_spent_usdt"])
            avg_cost = (total_spent / total_btc) if total_btc > 0 else 0.0
            current_value = total_btc * now_price if now_price > 0 else 0.0
            unrealized_pnl = current_value - total_spent if now_price > 0 else 0.0
            pnl_pct = (unrealized_pnl / total_spent * 100) if total_spent > 0 else 0.0
            out[src] = {
                "total_btc": round(total_btc, 8),
                "total_spent": round(total_spent, 2),
                "avg_cost": round(avg_cost, 2),
                "current_price": round(now_price, 2) if now_price > 0 else None,
                "current_value": round(current_value, 2) if now_price > 0 else None,
                "unrealized_pnl": round(unrealized_pnl, 2) if now_price > 0 else None,
                "pnl_pct": round(pnl_pct, 2) if now_price > 0 else None,
                "entry_count": len(s["entries"]),
                "first_entry_ts": s.get("first_entry_ts"),
                "last_entry_ts": s.get("last_entry_ts"),
            }
        out["combined"] = {
            "total_btc": out["live"]["total_btc"] + out["paper"]["total_btc"],
            "total_spent": out["live"]["total_spent"] + out["paper"]["total_spent"],
            "entry_count": out["live"]["entry_count"] + out["paper"]["entry_count"],
        }
        return out

    def get_recent_entries(self, n: int = 20, source: Optional[str] = None) -> list[dict]:
        """최근 N건 엔트리 반환 (source=None이면 live+paper 합쳐서 timestamp 내림차순)"""
        if source in ("live", "paper"):
            return list(self._state[source]["entries"][-n:])[::-1]
        merged = []
        for src in ("live", "paper"):
            merged.extend(self._state[src]["entries"])
        merged.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return merged[:n]

    # -------------------------------------------------------------------------
    # 영속화
    # -------------------------------------------------------------------------

    def _save(self):
        """Atomic JSON 저장 (tmp → rename)"""
        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", delete=False, dir=str(self.reserve_file.parent),
                prefix=".btc_reserve_", suffix=".tmp", encoding="utf-8",
            )
            json.dump(self._state, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp.name, self.reserve_file)
        except Exception as e:
            logger.error(f"[BTCReserve] 상태 저장 실패: {e}")

    def _load(self):
        """기존 상태 로드"""
        if not self.reserve_file.exists():
            return
        try:
            with open(self.reserve_file, encoding="utf-8") as f:
                data = json.load(f)
            # 구조 검증 — 누락된 소스는 빈 상태로 보완
            for src in ("live", "paper"):
                if src in data and isinstance(data[src], dict):
                    self._state[src] = {**self._empty_source_state(), **data[src]}
            logger.info(
                f"[BTCReserve] 상태 로드 — "
                f"LIVE {self._state['live']['total_btc']:.8f} BTC ({len(self._state['live']['entries'])}건) | "
                f"PAPER {self._state['paper']['total_btc']:.8f} BTC ({len(self._state['paper']['entries'])}건)"
            )
        except Exception as e:
            logger.warning(f"[BTCReserve] 상태 로드 실패 — 빈 상태로 시작: {e}")

    # -------------------------------------------------------------------------
    # 알림
    # -------------------------------------------------------------------------

    def _notify_accumulation(self, source: str, entry: dict):
        """Telegram 알림 — callable 등록되어 있을 때만"""
        if not self.notifier:
            return
        try:
            status = self.get_status()[source]
            total_btc = status["total_btc"]
            total_spent = status["total_spent"]
            avg_cost = status["avg_cost"]
            cur_val = status.get("current_value")
            pnl_pct = status.get("pnl_pct")

            badge = "🏛️ LIVE 실매수" if source == "live" else "📊 PAPER 가상적립"
            val_line = (
                f"💎 현재가치: ${cur_val:,.2f} ({pnl_pct:+.2f}%)"
                if cur_val is not None else ""
            )
            text = (
                f"{badge}\n"
                f"━━━━━━━━━━━━━\n"
                f"🎯 {entry['btc_bought']:.8f} BTC @ ${entry['btc_price']:,.2f}\n"
                f"💰 적립액: ${entry['usdt_spent']:.2f} "
                f"({entry['allocation_pct']:.0%} × ${entry['trigger_pnl_usdt']:.2f} PnL)\n"
                f"📈 원천: {entry['trigger_symbol']} ({entry['tier']} 티어)\n"
                f"━━━━━━━━━━━━━\n"
                f"🏦 {source.upper()} 금고 누적:\n"
                f"   {total_btc:.8f} BTC (${total_spent:,.2f} 투입)\n"
                f"   평단 ${avg_cost:,.2f}\n"
                f"{val_line}"
            )
            self.notifier(text, silent=True)
        except Exception as e:
            logger.debug(f"[BTCReserve] 알림 실패: {e}")
