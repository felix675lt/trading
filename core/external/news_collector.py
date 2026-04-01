"""크립토 뉴스 수집 - 다중 RSS + CoinGecko 트렌딩 + 지정학 뉴스"""

import asyncio
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import aiohttp
from loguru import logger

# lxml이 있으면 더 관대한 파서 사용
try:
    from lxml import etree as lxml_ET
    _LXML_AVAILABLE = True
except ImportError:
    _LXML_AVAILABLE = False


class NewsCollector:
    """
    무료 소스에서 크립토 + 매크로 뉴스/트렌딩 수집

    소스:
    - CoinDesk RSS (크립토 전문)
    - CoinTelegraph RSS (크립토 전문)
    - Google News RSS (크립토 + 지정학)
    - CoinGecko 트렌딩 (키 불필요)
    """

    # RSS 피드 목록 — 크립토 + 지정학 + 매크로 속보 종합
    RSS_FEEDS = [
        # 크립토 전문
        ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("cointelegraph", "https://cointelegraph.com/rss"),

        # 지정학 속보 (중동/전쟁/제재/유가)
        ("aljazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("bbc_world", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("bbc_mideast", "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml"),
        ("investing_world", "https://www.investing.com/rss/news_287.rss"),
        ("investing_commodities", "https://www.investing.com/rss/news_11.rss"),

        # 매크로 경제 (금리/인플레/연준/달러)
        ("bbc_business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
        ("investing_economy", "https://www.investing.com/rss/news_14.rss"),
        ("investing_indicators", "https://www.investing.com/rss/news_95.rss"),
        ("cnbc_economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"),
        ("cnbc_world", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),

        # Google News (보충)
        ("google_crypto", "https://news.google.com/rss/search?q=cryptocurrency+bitcoin&hl=en-US&gl=US&ceid=US:en"),
        ("google_geopolitics", "https://news.google.com/rss/search?q=iran+hormuz+oil+war+sanctions+ceasefire&hl=en-US&gl=US&ceid=US:en"),
        ("google_macro", "https://news.google.com/rss/search?q=fed+rate+cut+inflation+cpi+fomc&hl=en-US&gl=US&ceid=US:en"),
    ]

    COINGECKO_TRENDING = "https://api.coingecko.com/api/v3/search/trending"

    def __init__(self):
        self.news: list[dict] = []
        self.trending: list[dict] = []
        self.last_fetch: datetime | None = None
        self.fetch_interval = timedelta(minutes=3)  # 3분마다 뉴스 갱신 (속보 빠른 감지)

    async def fetch(self) -> dict:
        """뉴스 및 트렌딩 수집"""
        now = datetime.utcnow()
        if self.last_fetch and (now - self.last_fetch) < self.fetch_interval:
            return {"news": self.news, "trending": self.trending}

        # 모든 RSS + 트렌딩 병렬 수집
        tasks = [self._fetch_rss(name, url) for name, url in self.RSS_FEEDS]
        tasks.append(self._fetch_trending())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # RSS 결과 통합
        all_news = []
        for r in results[:-1]:  # 마지막은 trending
            if isinstance(r, list):
                all_news.extend(r)

        # 중복 제거 (제목 기준)
        seen_titles = set()
        unique_news = []
        for n in all_news:
            title_key = n["title"][:50].lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(n)

        self.news = unique_news[:100]  # 최대 100개 (소스 다양화)
        self.last_fetch = now

        logger.info(f"[News] 총 {len(self.news)}개 뉴스 수집 (소스: {len([r for r in results[:-1] if isinstance(r, list) and r])}개)")

        return {"news": self.news, "trending": self.trending}

    async def _fetch_rss(self, source_name: str, url: str) -> list[dict]:
        """RSS 피드에서 뉴스 수집 (lxml fallback)"""
        news_items = []
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; CryptoTrader/1.0)",
                "Accept": "application/rss+xml, application/xml, text/xml",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        logger.debug(f"[News] {source_name} HTTP {resp.status}")
                        return []

                    text = await resp.text()

                    # HTML 응답 감지 (RSS가 아닌 경우)
                    if text.strip().startswith("<!DOCTYPE") or text.strip().startswith("<html"):
                        logger.debug(f"[News] {source_name} HTML 응답 (RSS 아님)")
                        return []

                    items = self._parse_rss(text, source_name)
                    if items:
                        logger.debug(f"[News] {source_name}: {len(items)}개")
                    return items

        except Exception as e:
            logger.debug(f"[News] {source_name} 수집 실패: {e}")
            return []

    def _parse_rss(self, text: str, source: str) -> list[dict]:
        """RSS XML 파싱 (다중 방법)"""
        items = []

        # 1차: 표준 XML 파서
        root = None
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            pass

        # 2차: 정리 후 재시도
        if root is None:
            try:
                cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)
                cleaned = re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;)', '&amp;', cleaned)
                root = ET.fromstring(cleaned)
            except ET.ParseError:
                pass

        # 3차: lxml 관대 파서
        if root is None and _LXML_AVAILABLE:
            try:
                parser = lxml_ET.XMLParser(recover=True)
                root = lxml_ET.fromstring(text.encode('utf-8'), parser)
            except Exception:
                return []

        if root is None:
            return []

        # RSS 2.0: .//item, Atom: .//entry
        rss_items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
        if not rss_items:
            rss_items = root.findall(".//item")

        for item in rss_items[:20]:
            title = ""
            pub_date = ""
            link = ""

            # RSS 2.0
            title = item.findtext("title", "")
            pub_date = item.findtext("pubDate", "") or item.findtext("published", "")
            link = item.findtext("link", "")

            # Atom 포맷
            if not title:
                title = item.findtext("{http://www.w3.org/2005/Atom}title", "")
            if not pub_date:
                pub_date = item.findtext("{http://www.w3.org/2005/Atom}published", "") or \
                           item.findtext("{http://www.w3.org/2005/Atom}updated", "")
            if not link:
                link_elem = item.find("{http://www.w3.org/2005/Atom}link")
                if link_elem is not None:
                    link = link_elem.get("href", "")

            if title:
                items.append({
                    "title": html.unescape(title.strip()),
                    "published": pub_date,
                    "source": source,
                    "url": link,
                })

        return items

    async def _fetch_trending(self):
        """CoinGecko 트렌딩 코인"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.COINGECKO_TRENDING, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        coins = data.get("coins", [])
                        self.trending = [
                            {
                                "name": c["item"]["name"],
                                "symbol": c["item"]["symbol"],
                                "market_cap_rank": c["item"].get("market_cap_rank", 0),
                                "score": c["item"].get("score", 0),
                            }
                            for c in coins[:10]
                        ]
                        logger.info(f"[News] 트렌딩: {[t['symbol'] for t in self.trending[:5]]}")
        except Exception as e:
            logger.warning(f"[News] 트렌딩 수집 실패: {e}")

    def get_features(self, symbol: str = "") -> dict:
        """뉴스 기반 ML 피처"""
        features = {
            "news_count_1h": 0.0,
            "news_count_24h": 0.0,
            "news_velocity": 0.0,
            "is_trending": 0.0,
            "trending_rank": 0.0,
            "news_geo_count": 0.0,       # 지정학 뉴스 수
            "news_macro_count": 0.0,     # 매크로 뉴스 수
        }

        # 심볼 관련 뉴스 카운트
        coin = symbol.split("/")[0] if "/" in symbol else symbol
        coin_lower = coin.lower()

        count_1h = 0
        count_24h = 0
        geo_count = 0
        macro_count = 0

        for i, n in enumerate(self.news):
            title_lower = n["title"].lower()

            # 코인 관련 뉴스
            if coin_lower in title_lower or coin.upper() in n["title"]:
                count_24h += 1
                if i < 5:
                    count_1h += 1

            # 지정학 뉴스 (유가/전쟁/제재 관련)
            if n.get("source") == "google_geopolitics":
                geo_count += 1
            elif any(kw in title_lower for kw in ["war", "sanctions", "geopolit", "iran", "oil price", "opec", "tariff"]):
                geo_count += 1

            # 매크로 뉴스 (금리/인플레 관련)
            if n.get("source") == "google_macro":
                macro_count += 1
            elif any(kw in title_lower for kw in ["fed", "rate cut", "rate hike", "inflation", "cpi", "fomc", "dxy", "dollar"]):
                macro_count += 1

        features["news_count_1h"] = float(count_1h)
        features["news_count_24h"] = float(count_24h)
        features["news_velocity"] = count_1h / max(count_24h, 1)
        features["news_geo_count"] = float(geo_count)
        features["news_macro_count"] = float(macro_count)

        # 트렌딩 여부
        for t in self.trending:
            if t["symbol"].upper() == coin.upper():
                features["is_trending"] = 1.0
                features["trending_rank"] = 1.0 - (t.get("score", 5) / 10)
                break

        return features
