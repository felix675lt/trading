"""한국 거래소 상장 시그널 감지 모듈

바이낸스 선물 vs 업비트/빗썸 교차 비교:
- 1순위: 바이낸스 선물 O + 한국 전혀 X → 신규 상장 후보
- 2순위: 한국 BTC/USDT 마켓만 O + KRW X → KRW 상장 후보

주의: 업비트/빗썸 상장폐지 이력이 있는 코인은 재상장 가능성이
불확실하므로 점수 할인 적용 (delisted_discount).
과거 상장폐지된 코인 목록을 관리하여 참고.

모니터링 지표:
- 거래량 급증 (24h volume spike)
- 가격 급등 (price momentum)
- OI 급증 (open interest surge)
- 소셜 트렌딩 (trending on CoinGecko)
"""

import json
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger


class ListingDetector:
    """한국 거래소 상장 시그널 감지"""

    # 업비트/빗썸 마켓 캐시 갱신 주기
    MARKET_REFRESH_HOURS = 1
    # 시그널 스캔 주기
    SCAN_INTERVAL_MINUTES = 15
    # 상장 시그널 임계값
    SIGNAL_THRESHOLD = 0.65

    # 업비트/빗썸 과거 상장폐지 이력 (확인된 것들)
    # 재상장 가능성은 있지만 불확실 → 점수 50% 할인
    KNOWN_DELISTED = {
        # 업비트 상장폐지 이력
        "SXP", "LINA", "FLM", "KEY", "MDT", "DAR", "AMB",
        "LEVER", "COMBO", "HIFI", "BAKE", "XEM", "PERP",
        "LUNA", "USTC", "ANC", "MIR",  # Terra 관련
        "FTT", "SRM",  # FTX 관련
        "TON",  # 구 TON (현재 Toncoin과 다름)
        "BSV",  # 빗코인SV
        "NPXS", "MOC", "PLY", "MBL",  # 빗썸 폐지
    }

    def __init__(self, exchange_client=None):
        self.exchange = exchange_client  # ccxt binanceusdm
        self._upbit_krw: set[str] = set()
        self._upbit_btc: set[str] = set()
        self._upbit_usdt: set[str] = set()
        self._bithumb_krw: set[str] = set()
        self._binance_futures: dict[str, str] = {}  # coin -> symbol
        self._last_market_refresh: datetime | None = None
        self._last_scan: datetime | None = None

        # 모니터링 결과
        self._tier1_candidates: list[dict] = []  # 완전 신규
        self._tier2_candidates: list[dict] = []  # KRW 미상장
        self._active_signals: dict[str, dict] = {}  # coin -> signal
        self._history: list[dict] = []  # 과거 시그널 기록 (학습용)

        # 볼륨/가격 기준선 (이전 스캔 대비)
        self._prev_volumes: dict[str, float] = {}
        self._prev_prices: dict[str, float] = {}

        # 상장폐지 파일에서 추가 로드
        self._delisted = set(self.KNOWN_DELISTED)
        self._load_delisted_file()

    def _load_delisted_file(self):
        """data/delisted_coins.txt에서 추가 상장폐지 코인 로드"""
        try:
            path = Path(__file__).parent.parent.parent / "data" / "delisted_coins.txt"
            if path.exists():
                with open(path) as f:
                    for line in f:
                        coin = line.strip().upper()
                        if coin and not coin.startswith("#"):
                            self._delisted.add(coin)
                logger.info(f"[ListingDetector] 상장폐지 목록: {len(self._delisted)}종목")
        except Exception:
            pass

    def _fetch_upbit_markets(self):
        """업비트 마켓 목록 조회"""
        try:
            url = "https://api.upbit.com/v1/market/all"
            data = json.loads(urllib.request.urlopen(url, timeout=10).read())
            self._upbit_krw.clear()
            self._upbit_btc.clear()
            self._upbit_usdt.clear()
            for m in data:
                coin = m["market"].split("-")[1]
                if m["market"].startswith("KRW-"):
                    self._upbit_krw.add(coin)
                elif m["market"].startswith("BTC-"):
                    self._upbit_btc.add(coin)
                elif m["market"].startswith("USDT-"):
                    self._upbit_usdt.add(coin)
            logger.info(f"[ListingDetector] 업비트 마켓 갱신: KRW={len(self._upbit_krw)} BTC={len(self._upbit_btc)} USDT={len(self._upbit_usdt)}")
        except Exception as e:
            logger.warning(f"[ListingDetector] 업비트 마켓 조회 실패: {e}")

    def _fetch_bithumb_markets(self):
        """빗썸 KRW 마켓 목록 조회"""
        try:
            url = "https://api.bithumb.com/public/ticker/ALL_KRW"
            data = json.loads(urllib.request.urlopen(url, timeout=10).read())
            self._bithumb_krw = set(data.get("data", {}).keys()) - {"date"}
            logger.info(f"[ListingDetector] 빗썸 마켓 갱신: KRW={len(self._bithumb_krw)}")
        except Exception as e:
            logger.warning(f"[ListingDetector] 빗썸 마켓 조회 실패: {e}")

    async def _fetch_binance_futures(self):
        """바이낸스 선물 심볼 매핑"""
        if not self.exchange:
            return
        try:
            if not self.exchange.exchange.markets:
                await self.exchange.exchange.load_markets()

            self._binance_futures.clear()
            for sym, market in self.exchange.exchange.markets.items():
                if "USDT:USDT" in sym:
                    coin = sym.split("/")[0]
                    # 1000PEPE -> PEPE 정규화
                    normalized = coin.replace("1000", "") if coin.startswith("1000") else coin
                    self._binance_futures[normalized] = sym
            logger.info(f"[ListingDetector] 바이낸스 선물: {len(self._binance_futures)}종목")
        except Exception as e:
            logger.warning(f"[ListingDetector] 바이낸스 선물 조회 실패: {e}")

    async def refresh_markets(self):
        """마켓 목록 갱신 (캐시 주기 적용)"""
        now = datetime.utcnow()
        if (
            self._last_market_refresh
            and (now - self._last_market_refresh).total_seconds() < self.MARKET_REFRESH_HOURS * 3600
        ):
            return

        self._fetch_upbit_markets()
        self._fetch_bithumb_markets()
        await self._fetch_binance_futures()
        self._last_market_refresh = now

    def _classify_coins(self) -> tuple[dict[str, str], dict[str, str]]:
        """코인 분류: 1순위 (완전 신규) / 2순위 (KRW 미상장)"""
        kr_all = self._upbit_krw | self._upbit_btc | self._upbit_usdt | self._bithumb_krw
        kr_krw = self._upbit_krw | self._bithumb_krw

        # 1순위: 바이낸스 선물 O + 한국 전혀 X
        tier1 = {coin: sym for coin, sym in self._binance_futures.items() if coin not in kr_all}

        # 2순위: 한국 BTC/USDT에만 O + KRW X
        tier2 = {coin: sym for coin, sym in self._binance_futures.items() if coin in kr_all and coin not in kr_krw}

        return tier1, tier2

    async def scan_signals(self) -> list[dict]:
        """상장 시그널 스캔 — 비정상 움직임 감지"""
        now = datetime.utcnow()
        if (
            self._last_scan
            and (now - self._last_scan).total_seconds() < self.SCAN_INTERVAL_MINUTES * 60
        ):
            return list(self._active_signals.values())

        await self.refresh_markets()
        tier1, tier2 = self._classify_coins()

        if not tier1 and not tier2:
            return []

        signals = []

        # 각 티어에서 시그널 스캔
        for tier_name, candidates, weight in [("tier1", tier1, 1.0), ("tier2", tier2, 0.8)]:
            syms = list(candidates.values())
            if not syms:
                continue

            try:
                # 배치로 티커 조회 (최대 100개씩)
                all_tickers = {}
                for i in range(0, len(syms), 80):
                    batch = syms[i : i + 80]
                    tickers = await self.exchange.exchange.fetch_tickers(batch)
                    all_tickers.update(tickers)
                    if i + 80 < len(syms):
                        import asyncio
                        await asyncio.sleep(0.5)

                for coin, sym in candidates.items():
                    ticker = all_tickers.get(sym)
                    if not ticker:
                        continue

                    signal = self._score_coin(coin, sym, ticker, tier_name, weight)
                    if signal and signal["score"] >= self.SIGNAL_THRESHOLD:
                        signals.append(signal)

            except Exception as e:
                logger.warning(f"[ListingDetector] {tier_name} 스캔 실패: {e}")

        # 점수 기준 정렬
        signals.sort(key=lambda x: x["score"], reverse=True)

        # 상위 10개만 유지
        self._active_signals = {s["coin"]: s for s in signals[:10]}
        if tier_name == "tier1":
            self._tier1_candidates = [s for s in signals if s["tier"] == "tier1"]
        self._tier2_candidates = [s for s in signals if s["tier"] == "tier2"]

        self._last_scan = now

        if signals:
            top3 = signals[:3]
            top_str = ", ".join(f"{s['coin']}({s['score']:.2f})" for s in top3)
            logger.info(f"[ListingDetector] 상장 시그널 {len(signals)}건 감지 | Top: {top_str}")

        return signals

    def _score_coin(
        self, coin: str, symbol: str, ticker: dict, tier: str, tier_weight: float
    ) -> Optional[dict]:
        """개별 코인 상장 시그널 점수 계산"""
        price = ticker.get("last", 0) or 0
        pct_24h = ticker.get("percentage", 0) or 0
        volume = (ticker.get("quoteVolume", 0) or 0) / 1e6  # $M
        high = ticker.get("high", 0) or 0
        low = ticker.get("low", 0) or 0

        if price == 0 or volume < 1:  # $1M 미만 거래량 무시
            return None

        # 24h 변동 범위
        range_pct = (high - low) / low * 100 if low > 0 else 0

        # === 점수 계산 ===
        score = 0.0
        reasons = []

        # 1. 가격 급등 (양방향, 하락도 상장 전 조정일 수 있음)
        if abs(pct_24h) > 20:
            score += 0.30
            reasons.append(f"급변동 {pct_24h:+.1f}%")
        elif abs(pct_24h) > 10:
            score += 0.20
            reasons.append(f"변동 {pct_24h:+.1f}%")
        elif abs(pct_24h) > 5:
            score += 0.10
            reasons.append(f"소폭변동 {pct_24h:+.1f}%")

        # 2. 거래량 급증 (이전 대비)
        prev_vol = self._prev_volumes.get(coin, 0)
        if prev_vol > 0:
            vol_ratio = volume / prev_vol
            if vol_ratio > 3:
                score += 0.25
                reasons.append(f"볼륨 {vol_ratio:.1f}배 급증")
            elif vol_ratio > 2:
                score += 0.15
                reasons.append(f"볼륨 {vol_ratio:.1f}배")
        elif volume > 50:  # 첫 스캔이지만 절대 볼륨 높음
            score += 0.15
            reasons.append(f"고볼륨 ${volume:.0f}M")
        self._prev_volumes[coin] = volume

        # 3. 변동성 (일중 범위)
        if range_pct > 30:
            score += 0.20
            reasons.append(f"고변동성 {range_pct:.0f}%")
        elif range_pct > 15:
            score += 0.10
            reasons.append(f"변동성 {range_pct:.0f}%")

        # 4. 가격 상승 추세 (이전 대비)
        prev_price = self._prev_prices.get(coin, 0)
        if prev_price > 0:
            price_change = (price - prev_price) / prev_price
            if price_change > 0.10:
                score += 0.15
                reasons.append(f"가격상승 {price_change:+.1%}")
            elif price_change > 0.05:
                score += 0.08
                reasons.append(f"소폭상승 {price_change:+.1%}")
        self._prev_prices[coin] = price

        # 5. 티어 가중치
        score *= tier_weight

        # 6. 상장폐지 이력 할인 (재상장 불확실)
        is_delisted = coin in self._delisted
        if is_delisted:
            score *= 0.5
            reasons.append("⚠상폐이력")

        # 최소 점수 필터
        if score < 0.1:
            return None

        return {
            "coin": coin,
            "symbol": symbol,
            "tier": tier,
            "score": round(min(score, 1.0), 3),
            "price": price,
            "pct_24h": round(pct_24h, 2),
            "volume_m": round(volume, 1),
            "range_pct": round(range_pct, 1),
            "reasons": reasons,
            "timestamp": datetime.utcnow().isoformat(),
            "action": "long" if pct_24h > 0 else "short",
        }

    def get_tradeable_signals(self, min_score: float = 0.65) -> list[dict]:
        """거래 가능한 시그널 반환 (심볼 + 방향 + 확신도)"""
        return [
            {
                "symbol": s["symbol"],
                "coin": s["coin"],
                "action": s["action"],
                "confidence": s["score"],
                "tier": s["tier"],
                "reasons": s["reasons"],
            }
            for s in self._active_signals.values()
            if s["score"] >= min_score
        ]

    def get_report(self) -> dict:
        """대시보드/텔레그램 보고용"""
        return {
            "tier1_count": len(self._tier1_candidates),
            "tier2_count": len(self._tier2_candidates),
            "active_signals": len(self._active_signals),
            "top_signals": [
                {
                    "coin": s["coin"],
                    "tier": s["tier"],
                    "score": s["score"],
                    "pct_24h": s["pct_24h"],
                    "volume_m": s["volume_m"],
                    "reasons": ", ".join(s["reasons"]),
                }
                for s in sorted(self._active_signals.values(), key=lambda x: x["score"], reverse=True)[:5]
            ],
            "market_coverage": {
                "binance_futures": len(self._binance_futures),
                "upbit_krw": len(self._upbit_krw),
                "bithumb_krw": len(self._bithumb_krw),
                "tier1_pool": len(self._tier1_candidates) if self._tier1_candidates else "N/A",
                "tier2_pool": len(self._tier2_candidates) if self._tier2_candidates else "N/A",
            },
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
        }
