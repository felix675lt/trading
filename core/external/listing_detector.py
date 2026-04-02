"""한국 거래소 상장 시그널 감지 + 선매집 패턴 추적

2단계 감지:
1. 선매집 패턴 — 바이낸스 선물에서 tier1 코인의 볼륨/가격 이상 움직임
2. 상장 공지 — Google News RSS에서 "업비트 상장" / "빗썸 상장" 키워드

3단계 안전 필터:
- 24h +100% 펌프 금지
- 유통비율 30% 미만 금지
- 토큰 언락 스케줄 직접 확인 (DefiLlama emissions-adapters)
- 백커 등급 판단: S-Tier 백커 있으면 언락 있어도 허용 (VC가 펌프할 가능성)
  백커 없이 언락만 있으면 차단 (순수 덤프 위험)

트레이딩 전략:
- 전체 시드의 10%, 고배율(10x), 빠른 진입/청산
- SL 5%, TP 20~50%+ (트레일링)
"""

import asyncio
import json
import re as _re
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger


class ListingDetector:
    """한국 거래소 상장 시그널 감지 + 선매집 스나이핑"""

    MARKET_REFRESH_HOURS = 1
    SCAN_INTERVAL_MINUTES = 5  # 15분 → 5분으로 단축

    # 선매집 감지 임계값
    VOLUME_SPIKE_RATIO = 3.0     # 24h 평균 대비 3배
    PRICE_SURGE_PCT = 15.0       # 1시간 내 15%+ 상승
    MIN_VOLUME_M = 5.0           # 최소 거래량 $5M (노이즈 필터)

    # 알림/거래 임계값
    ALERT_THRESHOLD = 0.50       # 텔레그램 알림
    TRADE_THRESHOLD = 0.65       # 자동 거래 진입

    # 상장폐지 이력
    KNOWN_DELISTED = {
        "SXP", "LINA", "FLM", "KEY", "MDT", "DAR", "AMB",
        "LEVER", "COMBO", "HIFI", "BAKE", "XEM", "PERP",
        "LUNA", "USTC", "ANC", "MIR", "FTT", "SRM",
        "TON", "BSV", "NPXS", "MOC", "PLY", "MBL",
    }

    # 상장 뉴스 키워드
    LISTING_KEYWORDS_KR = ["상장", "마켓 추가", "거래 지원", "원화 마켓"]
    LISTING_KEYWORDS_EN = ["listing", "listed", "new market", "trading support"]
    EXCHANGE_KEYWORDS = ["업비트", "빗썸", "upbit", "bithumb"]

    # 안전 필터
    MAX_24H_PUMP_PCT = 100.0     # 24h 이미 +100% 이상이면 진입 금지 (늦은 진입)
    MIN_CIRCULATING_RATIO = 0.30  # 유통비율 30% 미만 = 언락 덤프 위험

    # ── 토큰 언락 스케줄 설정 ──
    UNLOCK_DANGER_DAYS = 14       # 향후 N일 이내 대규모 언락 감지
    UNLOCK_DANGER_PCT = 1.5       # 총 공급량의 N% 이상 언락 = 위험 (투자자+팀 월 1.4%도 위험)
    DEFILLAMA_EMISSIONS_BASE = "https://raw.githubusercontent.com/DefiLlama/emissions-adapters/master/protocols"

    # ── 코인 심볼 → DefiLlama 슬러그 매핑 ──
    # (DefiLlama은 프로젝트명을 슬러그로 사용, 코인 심볼과 다름)
    COIN_TO_DEFILLAMA = {
        "sto": "stakestone", "arb": "arbitrum", "op": "optimism",
        "sui": "sui", "apt": "aptos", "sei": "sei", "tia": "celestia",
        "zro": "layerzero", "eigen": "eigenlayer", "strk": "starknet",
        "jup": "jupiter", "w": "wormhole", "pyth": "pyth-network",
        "pendle": "pendle", "ethfi": "ether-fi", "ena": "ethena",
        "zk": "zksync", "blast": "blast", "manta": "manta",
        "alt": "altlayer", "dym": "dymension", "pixel": "pixels",
        "portal": "portal", "aevo": "aevo", "ondo": "ondo-finance",
        "rez": "renzo", "io": "io-net", "not": "notcoin",
        "bb": "bouncebit", "lista": "lista-dao",
    }

    # ── 백커(VC) 등급 — S/A Tier가 있으면 언락 있어도 허용 ──
    # (VC가 언락 전에 펌프를 만드는 패턴이 매우 흔함)
    S_TIER_VCS = {
        "binance labs", "yzi labs", "a16z", "andreessen horowitz",
        "paradigm", "polychain", "polychain capital",
        "sequoia", "coinbase ventures", "jump crypto", "jump trading",
        "dragonfly", "dragonfly capital", "multicoin", "multicoin capital",
        "pantera", "pantera capital", "lightspeed",
    }
    A_TIER_VCS = {
        "okx ventures", "hashkey", "hashkey capital", "animoca",
        "animoca ventures", "galaxy digital", "delphi digital",
        "spartan", "framework ventures", "electric capital",
        "wintermute", "alameda",  # historical
        "blockchain capital", "digital currency group", "dcg",
        "hashed", "amber group", "bain capital crypto",
    }

    def __init__(self, exchange_client=None):
        self.exchange = exchange_client
        self._upbit_krw: set[str] = set()
        self._upbit_btc: set[str] = set()
        self._upbit_usdt: set[str] = set()
        self._bithumb_krw: set[str] = set()
        self._binance_futures: dict[str, str] = {}
        self._last_market_refresh: datetime | None = None
        self._last_scan: datetime | None = None

        # 시그널 결과
        self._active_signals: dict[str, dict] = {}
        self._tier1_candidates: list[dict] = []
        self._tier2_candidates: list[dict] = []

        # 24시간 볼륨 히스토리 (24h 평균 비교용)
        self._volume_history: dict[str, deque] = {}  # coin -> deque of (timestamp, volume)
        self._price_history: dict[str, deque] = {}   # coin -> deque of (timestamp, price)

        # 상장 뉴스 감지
        self._alerted_listings: set[str] = set()  # 이미 알린 코인
        self._listing_news: list[dict] = []  # 최근 상장 뉴스

        # 상장폐지 목록
        self._delisted = set(self.KNOWN_DELISTED)
        self._load_delisted_file()

        # 토큰 언락 스케줄 캐시 (coin_id -> parsed unlock data)
        self._unlock_cache: dict[str, dict] = {}
        self._unlock_cache_time: dict[str, float] = {}
        self._unlock_cache_ttl = 3600 * 6  # 6시간 캐시

        # 백커 정보 캐시
        self._backer_cache: dict[str, dict] = {}
        self._backer_cache_time: dict[str, float] = {}

    def _load_delisted_file(self):
        try:
            path = Path(__file__).parent.parent.parent / "data" / "delisted_coins.txt"
            if path.exists():
                with open(path) as f:
                    for line in f:
                        coin = line.strip().upper()
                        if coin and not coin.startswith("#"):
                            self._delisted.add(coin)
        except Exception:
            pass

    def _fetch_upbit_markets(self):
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
            logger.info(f"[ListingDetector] 업비트: KRW={len(self._upbit_krw)} BTC={len(self._upbit_btc)} USDT={len(self._upbit_usdt)}")
        except Exception as e:
            logger.warning(f"[ListingDetector] 업비트 조회 실패: {e}")

    def _fetch_bithumb_markets(self):
        try:
            url = "https://api.bithumb.com/public/ticker/ALL_KRW"
            data = json.loads(urllib.request.urlopen(url, timeout=10).read())
            self._bithumb_krw = set(data.get("data", {}).keys()) - {"date"}
            logger.info(f"[ListingDetector] 빗썸: KRW={len(self._bithumb_krw)}")
        except Exception as e:
            logger.warning(f"[ListingDetector] 빗썸 조회 실패: {e}")

    async def _fetch_binance_futures(self):
        if not self.exchange:
            return
        try:
            if not self.exchange.exchange.markets:
                await self.exchange.exchange.load_markets()

            self._binance_futures.clear()
            for sym, market in self.exchange.exchange.markets.items():
                if "USDT:USDT" in sym:
                    coin = sym.split("/")[0]
                    normalized = coin.replace("1000", "") if coin.startswith("1000") else coin
                    self._binance_futures[normalized] = sym
            logger.info(f"[ListingDetector] 바이낸스 선물: {len(self._binance_futures)}종목")
        except Exception as e:
            logger.warning(f"[ListingDetector] 바이낸스 선물 조회 실패: {e}")

    async def refresh_markets(self):
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
        kr_all = self._upbit_krw | self._upbit_btc | self._upbit_usdt | self._bithumb_krw
        kr_krw = self._upbit_krw | self._bithumb_krw

        tier1 = {coin: sym for coin, sym in self._binance_futures.items() if coin not in kr_all}
        tier2 = {coin: sym for coin, sym in self._binance_futures.items() if coin in kr_all and coin not in kr_krw}

        return tier1, tier2

    def _get_volume_avg_24h(self, coin: str) -> float:
        """24시간 평균 볼륨 반환"""
        history = self._volume_history.get(coin, deque())
        if len(history) < 2:
            return 0.0
        cutoff = time.time() - 86400
        recent = [v for t, v in history if t > cutoff]
        return sum(recent) / len(recent) if recent else 0.0

    def _get_price_change_1h(self, coin: str, current_price: float) -> float:
        """1시간 전 대비 가격 변동률"""
        history = self._price_history.get(coin, deque())
        if not history:
            return 0.0
        cutoff = time.time() - 3600
        old_prices = [p for t, p in history if t < cutoff]
        if not old_prices:
            oldest = history[0][1]
            return (current_price - oldest) / oldest if oldest > 0 else 0.0
        ref_price = old_prices[-1]
        return (current_price - ref_price) / ref_price if ref_price > 0 else 0.0

    def _update_history(self, coin: str, price: float, volume: float):
        """볼륨/가격 히스토리 업데이트 (24시간 유지)"""
        now = time.time()
        cutoff = now - 86400

        if coin not in self._volume_history:
            self._volume_history[coin] = deque(maxlen=300)
        if coin not in self._price_history:
            self._price_history[coin] = deque(maxlen=300)

        self._volume_history[coin].append((now, volume))
        self._price_history[coin].append((now, price))

        # 24시간 이전 데이터 정리
        while self._volume_history[coin] and self._volume_history[coin][0][0] < cutoff:
            self._volume_history[coin].popleft()
        while self._price_history[coin] and self._price_history[coin][0][0] < cutoff:
            self._price_history[coin].popleft()

    async def scan_signals(self) -> list[dict]:
        """상장 시그널 스캔 — 선매집 패턴 + 상장 뉴스"""
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

        for tier_name, candidates, weight in [("tier1", tier1, 1.0), ("tier2", tier2, 0.7)]:
            syms = list(candidates.values())
            if not syms:
                continue

            try:
                all_tickers = {}
                for i in range(0, len(syms), 80):
                    batch = syms[i: i + 80]
                    tickers = await self.exchange.exchange.fetch_tickers(batch)
                    all_tickers.update(tickers)
                    if i + 80 < len(syms):
                        await asyncio.sleep(0.5)

                for coin, sym in candidates.items():
                    ticker = all_tickers.get(sym)
                    if not ticker:
                        continue

                    signal = self._score_coin(coin, sym, ticker, tier_name, weight)
                    if signal and signal["score"] >= 0.30:
                        signals.append(signal)

            except Exception as e:
                logger.warning(f"[ListingDetector] {tier_name} 스캔 실패: {e}")

        # 상장 뉴스 스캔
        news_signals = self._scan_listing_news()
        signals.extend(news_signals)

        signals.sort(key=lambda x: x["score"], reverse=True)

        self._active_signals = {s["coin"]: s for s in signals[:20]}
        self._tier1_candidates = [s for s in signals if s.get("tier") == "tier1"]
        self._tier2_candidates = [s for s in signals if s.get("tier") == "tier2"]

        self._last_scan = now

        tradeable = [s for s in signals if s["score"] >= self.TRADE_THRESHOLD]
        if tradeable:
            top_str = ", ".join(f"{s['coin']}({s['score']:.2f})" for s in tradeable[:5])
            logger.info(f"[ListingDetector] 🎯 거래가능 시그널 {len(tradeable)}건 | Top: {top_str}")
        elif signals:
            top_str = ", ".join(f"{s['coin']}({s['score']:.2f})" for s in signals[:3])
            logger.info(f"[ListingDetector] 시그널 {len(signals)}건 (거래 미달) | Top: {top_str}")

        return signals

    def _score_coin(
        self, coin: str, symbol: str, ticker: dict, tier: str, tier_weight: float
    ) -> Optional[dict]:
        """선매집 패턴 점수 계산 — 24시간 평균 볼륨 대비"""
        price = ticker.get("last", 0) or 0
        pct_24h = ticker.get("percentage", 0) or 0
        volume_m = (ticker.get("quoteVolume", 0) or 0) / 1e6
        high = ticker.get("high", 0) or 0
        low = ticker.get("low", 0) or 0

        if price == 0 or volume_m < self.MIN_VOLUME_M:
            self._update_history(coin, price, volume_m)
            return None

        range_pct = (high - low) / low * 100 if low > 0 else 0

        # 히스토리 업데이트
        self._update_history(coin, price, volume_m)

        # === 안전 필터: 이미 너무 올랐으면 진입 금지 ===
        if pct_24h > self.MAX_24H_PUMP_PCT:
            logger.debug(
                f"[ListingDetector] {coin} 진입금지: 24h +{pct_24h:.0f}% "
                f"(한도 +{self.MAX_24H_PUMP_PCT:.0f}%) — 늦은 진입/펌프앤덤프 위험"
            )
            return None

        score = 0.0
        reasons = []

        # === 1. 24시간 평균 대비 볼륨 급증 (핵심 지표) ===
        avg_vol = self._get_volume_avg_24h(coin)
        if avg_vol > 0:
            vol_ratio = volume_m / avg_vol
            if vol_ratio >= 5.0:
                score += 0.35
                reasons.append(f"볼륨 {vol_ratio:.1f}배 폭증 (${volume_m:.0f}M)")
            elif vol_ratio >= self.VOLUME_SPIKE_RATIO:
                score += 0.25
                reasons.append(f"볼륨 {vol_ratio:.1f}배 급증 (${volume_m:.0f}M)")
            elif vol_ratio >= 2.0:
                score += 0.15
                reasons.append(f"볼륨 {vol_ratio:.1f}배 (${volume_m:.0f}M)")
        elif volume_m > 50:
            score += 0.15
            reasons.append(f"고볼륨 ${volume_m:.0f}M")

        # === 2. 가격 급등 ===
        if abs(pct_24h) > 30:
            score += 0.30
            reasons.append(f"급등 {pct_24h:+.1f}%")
        elif abs(pct_24h) > 15:
            score += 0.20
            reasons.append(f"강세 {pct_24h:+.1f}%")
        elif abs(pct_24h) > 8:
            score += 0.10
            reasons.append(f"상승 {pct_24h:+.1f}%")

        # === 3. 1시간 가격 변동 (단기 모멘텀) ===
        price_1h = self._get_price_change_1h(coin, price)
        if price_1h > 0.15:
            score += 0.20
            reasons.append(f"1h +{price_1h:.0%} 급등")
        elif price_1h > 0.08:
            score += 0.10
            reasons.append(f"1h +{price_1h:.0%}")

        # === 4. 변동성 (일중 범위) ===
        if range_pct > 40:
            score += 0.15
            reasons.append(f"고변동성 {range_pct:.0f}%")
        elif range_pct > 20:
            score += 0.08
            reasons.append(f"변동성 {range_pct:.0f}%")

        # === 5. 선매집 복합 패턴 (볼륨+가격 동시) ===
        if avg_vol > 0 and volume_m / avg_vol >= 2.0 and pct_24h > 10:
            score += 0.15
            reasons.append("선매집 패턴")

        # 티어 가중치
        score *= tier_weight

        # 상장폐지 할인
        if coin in self._delisted:
            score *= 0.5
            reasons.append("⚠상폐이력")

        if score < 0.20:
            return None

        return {
            "coin": coin,
            "symbol": symbol,
            "tier": tier,
            "score": round(min(score, 1.0), 3),
            "price": price,
            "pct_24h": round(pct_24h, 2),
            "volume_m": round(volume_m, 1),
            "vol_ratio": round(volume_m / avg_vol, 1) if avg_vol > 0 else 0,
            "range_pct": round(range_pct, 1),
            "reasons": reasons,
            "timestamp": datetime.utcnow().isoformat(),
            "action": "long",
            "source": "accumulation",
        }

    def _scan_listing_news(self) -> list[dict]:
        """Google News RSS에서 업비트/빗썸 상장 뉴스 감지"""
        signals = []
        rss_urls = [
            ("upbit_kr", "https://news.google.com/rss/search?q=%EC%97%85%EB%B9%84%ED%8A%B8+%EC%83%81%EC%9E%A5&hl=ko&gl=KR&ceid=KR:ko"),
            ("bithumb_kr", "https://news.google.com/rss/search?q=%EB%B9%97%EC%8D%B8+%EC%83%81%EC%9E%A5&hl=ko&gl=KR&ceid=KR:ko"),
            ("upbit_en", "https://news.google.com/rss/search?q=upbit+listing+crypto&hl=en-US&gl=US&ceid=US:en"),
        ]

        for source, url in rss_urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                resp = urllib.request.urlopen(req, timeout=10)
                root = ET.fromstring(resp.read())

                for item in root.findall(".//item")[:10]:
                    title_el = item.find("title")
                    pub_el = item.find("pubDate")
                    if title_el is None:
                        continue

                    title = title_el.text or ""
                    title_lower = title.lower()

                    # 24시간 이내 뉴스만
                    if pub_el is not None and pub_el.text:
                        try:
                            from email.utils import parsedate_to_datetime
                            pub_dt = parsedate_to_datetime(pub_el.text)
                            if (datetime.utcnow() - pub_dt.replace(tzinfo=None)).total_seconds() > 86400:
                                continue
                        except Exception:
                            pass

                    # 상장 키워드 매칭
                    has_listing = any(kw in title for kw in self.LISTING_KEYWORDS_KR) or \
                                  any(kw in title_lower for kw in self.LISTING_KEYWORDS_EN)
                    has_exchange = any(kw in title for kw in self.EXCHANGE_KEYWORDS)

                    if has_listing and has_exchange:
                        # 코인 이름 추출 시도
                        coins = self._extract_coin_from_title(title)
                        for coin in coins:
                            if coin in self._alerted_listings:
                                continue

                            # 바이낸스 선물에 있는지 확인
                            binance_sym = self._binance_futures.get(coin)
                            if not binance_sym:
                                continue

                            self._alerted_listings.add(coin)
                            signals.append({
                                "coin": coin,
                                "symbol": binance_sym,
                                "tier": "listing_news",
                                "score": 0.90,  # 뉴스 확인 = 높은 점수
                                "price": 0,
                                "pct_24h": 0,
                                "volume_m": 0,
                                "vol_ratio": 0,
                                "range_pct": 0,
                                "reasons": [f"📰 상장뉴스: {title[:60]}"],
                                "timestamp": datetime.utcnow().isoformat(),
                                "action": "long",
                                "source": "news",
                            })
                            logger.info(f"[ListingDetector] 📰 상장 뉴스 감지: {coin} | {title[:80]}")

            except Exception as e:
                logger.debug(f"[ListingDetector] 뉴스 스캔 실패 ({source}): {e}")

        return signals

    def _extract_coin_from_title(self, title: str) -> list[str]:
        """뉴스 제목에서 코인 심볼 추출"""
        import re
        coins = []

        # 괄호 안의 영문 대문자 (예: "스카이프로토콜(SKY)")
        matches = re.findall(r'\(([A-Z]{2,10})\)', title)
        coins.extend(matches)

        # 영문 대문자 단어 (3글자 이상)
        words = re.findall(r'\b([A-Z]{3,10})\b', title)
        for w in words:
            if w in self._binance_futures and w not in ["THE", "FOR", "AND", "USD", "BTC", "ETH", "KRW"]:
                coins.append(w)

        return list(set(coins))

    def check_token_safety(self, coin: str) -> tuple[bool, str]:
        """토큰 안전성 체크 — 3단계 필터

        1. CoinGecko 유통비율 (<30% 위험)
        2. DefiLlama 언락 스케줄 직접 확인 (향후 14일 내 대규모 언락)
        3. 백커 등급 판단: S/A-Tier 백커 있으면 언락 있어도 허용
           (VC가 언락 전 펌프하는 패턴 — STO/YZi Labs 사례)
           백커 없이 언락만 있으면 차단 (순수 덤프 위험)

        Returns: (안전 여부, 사유)
        """
        coin_id = coin.lower()
        # 심볼 → DefiLlama 슬러그 매핑 (STO → stakestone)
        defillama_slug = self.COIN_TO_DEFILLAMA.get(coin_id, coin_id)

        # ── 1. CoinGecko 유통비율 + 백커 정보 ──
        circulating_ratio = None
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&community_data=false&developer_data=false"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=8)
            data = json.loads(resp.read())

            mc = data.get("market_data", {})
            circulating = mc.get("circulating_supply") or 0
            total = mc.get("total_supply") or mc.get("max_supply") or 0

            if total > 0 and circulating > 0:
                circulating_ratio = circulating / total

        except Exception:
            pass

        # ── 2. 언락 스케줄 직접 확인 (DefiLlama GitHub) ──
        unlock_info = self._check_unlock_schedule(defillama_slug)
        has_upcoming_unlock = unlock_info.get("has_danger", False)
        unlock_pct = unlock_info.get("unlock_pct_next_14d", 0.0)
        unlock_detail = unlock_info.get("detail", "")

        # ── 3. 백커 등급 확인 ──
        backer_info = self._check_backers(defillama_slug)
        has_strong_backer = backer_info.get("tier") in ("S", "A")
        backer_names = backer_info.get("names", [])
        backer_tier = backer_info.get("tier", "none")

        # ── 판단 로직 ──

        # Case 1: 유통비율 낮음 + 언락 있음 + 백커 없음 → 차단
        if circulating_ratio is not None and circulating_ratio < self.MIN_CIRCULATING_RATIO:
            if has_upcoming_unlock and not has_strong_backer:
                return False, (
                    f"⛔ 유통비율 {circulating_ratio:.0%} + 14일내 언락 {unlock_pct:.1f}% "
                    f"+ 백커 미확인 → 덤프 위험 | {unlock_detail}"
                )
            elif has_upcoming_unlock and has_strong_backer:
                # 백커 있으면 경고만 하고 허용 (VC 펌프 가능성)
                logger.warning(
                    f"[ListingDetector] ⚠️ {coin} 유통비율 {circulating_ratio:.0%} "
                    f"+ 언락 {unlock_pct:.1f}% BUT {backer_tier}-Tier 백커: "
                    f"{', '.join(backer_names[:3])} → 허용 (VC펌프 가능성)"
                )
            elif not has_upcoming_unlock:
                # 언락 없으면 유통비율만으로는 차단하지 않음 (이전에는 차단했음)
                if circulating_ratio < 0.15:
                    return False, f"유통비율 극저 {circulating_ratio:.0%} — 위험"

        # Case 2: 유통비율은 OK지만 대규모 언락 + 백커 없음 → 차단
        if has_upcoming_unlock and unlock_pct >= self.UNLOCK_DANGER_PCT and not has_strong_backer:
            return False, (
                f"⛔ 14일내 대규모 언락 {unlock_pct:.1f}% + 백커 미확인 → 덤프 위험 | {unlock_detail}"
            )

        # Case 3: 대규모 언락 + 백커 있음 → 허용 + 로그
        if has_upcoming_unlock and has_strong_backer:
            logger.info(
                f"[ListingDetector] ✅ {coin} 언락 {unlock_pct:.1f}% 있지만 "
                f"{backer_tier}-Tier 백커: {', '.join(backer_names[:3])} → 허용"
            )

        # 결과 로그
        safety_msg = "OK"
        if backer_names:
            safety_msg = f"OK | 백커: {', '.join(backer_names[:3])} ({backer_tier}-Tier)"
        if has_upcoming_unlock:
            safety_msg += f" | 언락: {unlock_pct:.1f}% in 14d"

        return True, safety_msg

    # ═══════════════════════════════════════════════════════════
    #  DefiLlama Emissions Adapters — 토큰 언락 스케줄 파싱
    # ═══════════════════════════════════════════════════════════

    def _check_unlock_schedule(self, coin_id: str) -> dict:
        """DefiLlama emissions-adapters GitHub에서 토큰 언락 스케줄 확인

        TypeScript 파일을 파싱하여 향후 14일 이내 언락 규모를 계산.
        manualCliff(timestamp, amount) → 특정 시점 일괄 언락
        manualStep(start, interval, steps, amountPerStep) → 주기적 선형 언락
        """
        now = time.time()

        # 캐시 확인
        if coin_id in self._unlock_cache:
            if now - self._unlock_cache_time.get(coin_id, 0) < self._unlock_cache_ttl:
                return self._unlock_cache[coin_id]

        result = {"has_danger": False, "unlock_pct_next_14d": 0.0, "detail": "", "events": []}

        try:
            # DefiLlama GitHub에서 TypeScript 파일 다운로드
            url = f"{self.DEFILLAMA_EMISSIONS_BASE}/{coin_id}.ts"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=10)
            ts_content = resp.read().decode("utf-8")

            result = self._parse_emissions_ts(ts_content, coin_id)
            logger.info(
                f"[ListingDetector] 📅 {coin_id} 언락 스케줄: "
                f"14일내 {result['unlock_pct_next_14d']:.1f}% "
                f"({len(result['events'])}건 이벤트)"
            )

        except urllib.request.HTTPError as e:
            if e.code == 404:
                logger.debug(f"[ListingDetector] {coin_id} DefiLlama 언락 데이터 없음")
            else:
                logger.debug(f"[ListingDetector] {coin_id} 언락 조회 HTTP {e.code}")
        except Exception as e:
            logger.debug(f"[ListingDetector] {coin_id} 언락 스케줄 조회 실패: {e}")

        self._unlock_cache[coin_id] = result
        self._unlock_cache_time[coin_id] = now
        return result

    def _parse_emissions_ts(self, ts_content: str, coin_id: str) -> dict:
        """TypeScript emissions adapter 파싱 → 향후 14일 언락 계산

        포맷:
        - manualCliff(timestamp, amount) → 일괄 언락
        - manualStep(start, interval, steps, amountPerStep) → 주기적 언락
        - periodToSeconds.year = 31536000
        - periodToSeconds.month = 2628000 (평균)
        - periodToSeconds.months(N) = N * 2628000
        """
        now = time.time()
        window_start = now
        window_end = now + self.UNLOCK_DANGER_DAYS * 86400

        # 상수 추출
        total_supply = self._extract_number(ts_content, r'totalSupply\s*=\s*([\d_]+)')
        start_ts = self._extract_number(ts_content, r'(?:const\s+)?start\s*=\s*(\d{10})')

        if not total_supply or total_supply == 0:
            total_supply = 1_000_000_000  # fallback

        # periodToSeconds 상수
        YEAR = 31_536_000
        MONTH = 2_628_000  # ~30.4일

        events = []
        total_unlock_in_window = 0.0

        # 전처리: 줄바꿈/탭 제거 → 한 줄로 (파싱 간소화)
        flat = _re.sub(r'\s+', ' ', ts_content)

        # ── manualCliff(timestamp, amount) 파싱 ──
        # 중첩 괄호 지원: months(6) 같은 내부 괄호 허용
        cliff_pattern = r'manualCliff\s*\(\s*([^,]+?)\s*,\s*(.+?)\s*\)(?:\s*[,\]\)])'
        for match in _re.finditer(cliff_pattern, flat):
            ts_expr = match.group(1).strip().rstrip(",").strip()
            amount_expr = match.group(2).strip().rstrip(",").strip()

            ts_val = self._eval_ts_expr(ts_expr, start_ts, YEAR, MONTH)
            amount_val = self._eval_amount_expr(amount_expr, ts_content, total_supply)

            if ts_val and amount_val:
                if window_start <= ts_val <= window_end:
                    pct = (amount_val / total_supply) * 100
                    total_unlock_in_window += amount_val
                    events.append({
                        "type": "cliff",
                        "timestamp": ts_val,
                        "date": datetime.utcfromtimestamp(ts_val).strftime("%Y-%m-%d"),
                        "amount": amount_val,
                        "pct": round(pct, 2),
                    })

        # ── manualStep(start, interval, steps, amountPerStep) 파싱 ──
        # 중첩 괄호 지원: (foundation * 0.889) / 48 같은 표현식 허용
        step_pattern = r'manualStep\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*(.+?)\s*,?\s*\)(?:\s*[,\]\)])'
        for match in _re.finditer(step_pattern, flat):
            start_expr = match.group(1).strip().rstrip(",")
            interval_expr = match.group(2).strip().rstrip(",")
            steps_expr = match.group(3).strip().rstrip(",")
            amount_expr = match.group(4).strip().rstrip(",")

            step_start = self._eval_ts_expr(start_expr, start_ts, YEAR, MONTH)
            interval = self._eval_ts_expr(interval_expr, start_ts, YEAR, MONTH)
            n_steps = self._eval_simple(steps_expr)
            per_step = self._eval_amount_expr(amount_expr, ts_content, total_supply)

            if step_start and interval and n_steps and per_step:
                for i in range(int(n_steps)):
                    unlock_ts = step_start + interval * i
                    if unlock_ts > window_end:
                        break
                    if window_start <= unlock_ts <= window_end:
                        pct = (per_step / total_supply) * 100
                        total_unlock_in_window += per_step
                        events.append({
                            "type": "step",
                            "timestamp": unlock_ts,
                            "date": datetime.utcfromtimestamp(unlock_ts).strftime("%Y-%m-%d"),
                            "amount": per_step,
                            "pct": round(pct, 2),
                        })

        total_pct = (total_unlock_in_window / total_supply) * 100 if total_supply > 0 else 0
        has_danger = total_pct >= self.UNLOCK_DANGER_PCT

        detail_parts = []
        for e in sorted(events, key=lambda x: x["timestamp"]):
            detail_parts.append(f"{e['date']}: {e['type']} {e['pct']:.1f}%")

        return {
            "has_danger": has_danger,
            "unlock_pct_next_14d": round(total_pct, 2),
            "detail": " | ".join(detail_parts[:5]),
            "events": events,
            "total_supply": total_supply,
        }

    def _extract_number(self, text: str, pattern: str) -> float:
        """정규식으로 숫자 추출"""
        m = _re.search(pattern, text)
        if m:
            return float(m.group(1).replace("_", ""))
        return 0.0

    def _eval_ts_expr(self, expr: str, start_ts: float, year: int, month: int) -> float:
        """타임스탬프 표현식 평가

        예: start + periodToSeconds.year → start_ts + 31536000
            start + periodToSeconds.months(6) → start_ts + 6*2628000
            periodToSeconds.month → 2628000
        """
        expr = expr.strip()

        # 순수 숫자
        try:
            val = float(expr.replace("_", ""))
            if val > 1_000_000_000:  # 타임스탬프
                return val
            return val
        except ValueError:
            pass

        # periodToSeconds 치환
        result = expr
        result = result.replace("periodToSeconds.year", str(year))
        result = result.replace("periodToSeconds.month", str(month))

        # periodToSeconds.months(N)
        months_match = _re.search(r'periodToSeconds\.months\((\d+)\)', result)
        if months_match:
            n = int(months_match.group(1))
            result = result.replace(months_match.group(0), str(n * month))

        # start 치환
        if start_ts:
            result = result.replace("start", str(int(start_ts)))

        # 간단한 산술 평가
        try:
            # 안전한 평가: 숫자, +, -, *, / 만 허용
            if _re.match(r'^[\d\s+\-*/().]+$', result):
                return float(eval(result))  # noqa: S307
        except Exception:
            pass

        return 0.0

    def _eval_amount_expr(self, expr: str, ts_content: str, total_supply: float) -> float:
        """금액 표현식 평가

        예: investor / 24 → (totalSupply * 0.215) / 24
            foundation * 0.111 → (totalSupply * 0.1865) * 0.111
        """
        expr = expr.strip()

        # 순수 숫자
        try:
            return float(expr.replace("_", ""))
        except ValueError:
            pass

        # 변수 참조 해결 (investor, team, foundation 등)
        resolved = expr
        var_pattern = r'const\s+(\w+)\s*=\s*totalSupply\s*\*\s*([\d.]+)'
        variables = {}
        for m in _re.finditer(var_pattern, ts_content):
            var_name = m.group(1)
            var_pct = float(m.group(2))
            variables[var_name] = total_supply * var_pct

        # 변수 치환
        for var_name, var_val in sorted(variables.items(), key=lambda x: -len(x[0])):
            resolved = resolved.replace(var_name, str(var_val))

        # 산술 평가
        try:
            if _re.match(r'^[\d\s+\-*/().eE]+$', resolved):
                return float(eval(resolved))  # noqa: S307
        except Exception:
            pass

        return 0.0

    def _eval_simple(self, expr: str) -> float:
        """간단한 숫자/산술 평가"""
        try:
            return float(expr.replace("_", "").strip())
        except ValueError:
            pass
        try:
            cleaned = expr.strip()
            if _re.match(r'^[\d\s+\-*/().]+$', cleaned):
                return float(eval(cleaned))  # noqa: S307
        except Exception:
            pass
        return 0.0

    # ═══════════════════════════════════════════════════════════
    #  백커(VC) 등급 확인
    # ═══════════════════════════════════════════════════════════

    def _check_backers(self, coin_id: str) -> dict:
        """CoinGecko + DefiLlama에서 백커/투자자 정보 확인

        Returns: {"tier": "S"|"A"|"B"|"none", "names": [...]}
        """
        now = time.time()

        # 캐시 확인
        if coin_id in self._backer_cache:
            if now - self._backer_cache_time.get(coin_id, 0) < self._unlock_cache_ttl:
                return self._backer_cache[coin_id]

        result = {"tier": "none", "names": [], "raw": []}

        # 방법 1: CoinGecko API (일부 코인은 investors 필드 있음)
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&community_data=false&developer_data=false"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=8)
            data = json.loads(resp.read())

            # description에서 백커 추출
            desc = (data.get("description", {}).get("en", "") or "").lower()
            found_vcs = []

            for vc in self.S_TIER_VCS:
                if vc in desc:
                    found_vcs.append(("S", vc.title()))
            for vc in self.A_TIER_VCS:
                if vc in desc:
                    found_vcs.append(("A", vc.title()))

            if found_vcs:
                result["raw"] = found_vcs
                result["names"] = [name for _, name in found_vcs]
                best_tier = "S" if any(t == "S" for t, _ in found_vcs) else "A"
                result["tier"] = best_tier

        except Exception as e:
            logger.debug(f"[ListingDetector] {coin_id} CoinGecko 백커 조회 실패: {e}")

        # 방법 2: DefiLlama 프로토콜 정보 (추가 검증)
        if result["tier"] == "none":
            try:
                url = f"https://api.llama.fi/protocol/{coin_id}"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                resp = urllib.request.urlopen(req, timeout=8)
                data = json.loads(resp.read())

                # raises 필드에서 투자자 정보
                raises = data.get("raises", [])
                if raises:
                    all_investors = []
                    for r in raises:
                        investors = r.get("leadInvestors", []) + r.get("otherInvestors", [])
                        all_investors.extend(investors)

                    found_vcs = []
                    for inv in all_investors:
                        inv_lower = inv.lower()
                        for vc in self.S_TIER_VCS:
                            if vc in inv_lower:
                                found_vcs.append(("S", inv))
                                break
                        for vc in self.A_TIER_VCS:
                            if vc in inv_lower:
                                found_vcs.append(("A", inv))
                                break

                    if found_vcs:
                        result["raw"] = found_vcs
                        result["names"] = list(set(name for _, name in found_vcs))
                        best_tier = "S" if any(t == "S" for t, _ in found_vcs) else "A"
                        result["tier"] = best_tier
                    elif all_investors:
                        # VC는 찾았지만 S/A tier 아님
                        result["tier"] = "B"
                        result["names"] = all_investors[:5]

            except Exception as e:
                logger.debug(f"[ListingDetector] {coin_id} DefiLlama 백커 조회 실패: {e}")

        # 방법 3: 하드코딩된 주요 프로젝트 (확인된 정보)
        KNOWN_BACKERS = {
            "stakestone": {"tier": "S", "names": ["Polychain Capital", "YZi Labs (Binance Labs)", "OKX Ventures", "Animoca Ventures"]},
            "arbitrum": {"tier": "S", "names": ["a16z", "Pantera Capital", "Polychain Capital"]},
            "optimism": {"tier": "S", "names": ["a16z", "Paradigm"]},
            "sui": {"tier": "S", "names": ["a16z", "Jump Crypto", "Coinbase Ventures"]},
            "aptos": {"tier": "S", "names": ["a16z", "Jump Crypto", "Multicoin Capital"]},
            "sei": {"tier": "S", "names": ["Jump Crypto", "Multicoin Capital"]},
            "celestia": {"tier": "S", "names": ["Polychain Capital", "Bain Capital Crypto"]},
            "layerzero": {"tier": "S", "names": ["a16z", "Sequoia Capital"]},
            "eigenlayer": {"tier": "S", "names": ["a16z", "Polychain Capital"]},
        }

        if result["tier"] == "none" and coin_id in KNOWN_BACKERS:
            known = KNOWN_BACKERS[coin_id]
            result["tier"] = known["tier"]
            result["names"] = known["names"]

        if result["names"]:
            logger.info(f"[ListingDetector] 🏢 {coin_id} 백커: {result['tier']}-Tier | {', '.join(result['names'][:3])}")

        self._backer_cache[coin_id] = result
        self._backer_cache_time[coin_id] = now
        return result

    def get_tradeable_signals(self, min_score: float = None) -> list[dict]:
        """거래 가능한 시그널 반환"""
        threshold = min_score or self.TRADE_THRESHOLD
        return [
            {
                "symbol": s["symbol"],
                "coin": s["coin"],
                "action": s["action"],
                "confidence": s["score"],
                "tier": s["tier"],
                "reasons": s["reasons"],
                "source": s.get("source", "accumulation"),
            }
            for s in sorted(self._active_signals.values(), key=lambda x: x["score"], reverse=True)
            if s["score"] >= threshold
        ]

    def get_report(self) -> dict:
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
                    "vol_ratio": s.get("vol_ratio", 0),
                    "reasons": ", ".join(s["reasons"]),
                }
                for s in sorted(self._active_signals.values(), key=lambda x: x["score"], reverse=True)[:5]
            ],
            "market_coverage": {
                "binance_futures": len(self._binance_futures),
                "upbit_krw": len(self._upbit_krw),
                "bithumb_krw": len(self._bithumb_krw),
            },
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
        }
