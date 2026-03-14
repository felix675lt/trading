"""크립토 뉴스 수집 - CryptoPanic RSS + CoinGecko 트렌딩"""

import asyncio
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import aiohttp
from loguru import logger


class NewsCollector:
    """
    무료 소스에서 크립토 뉴스/트렌딩 수집
    - CryptoPanic RSS (키 불필요)
    - CoinGecko 트렌딩 (키 불필요)
    """

    CRYPTOPANIC_RSS = "https://cryptopanic.com/news/rss/"
    COINGECKO_TRENDING = "https://api.coingecko.com/api/v3/search/trending"

    def __init__(self):
        self.news: list[dict] = []
        self.trending: list[dict] = []
        self.last_fetch: datetime | None = None
        self.fetch_interval = timedelta(minutes=15)

    async def fetch(self) -> dict:
        """뉴스 및 트렌딩 수집"""
        now = datetime.utcnow()
        if self.last_fetch and (now - self.last_fetch) < self.fetch_interval:
            return {"news": self.news, "trending": self.trending}

        await asyncio.gather(
            self._fetch_cryptopanic(),
            self._fetch_trending(),
            return_exceptions=True,
        )
        self.last_fetch = now
        return {"news": self.news, "trending": self.trending}

    async def _fetch_cryptopanic(self):
        """CryptoPanic RSS 피드에서 최신 뉴스"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.CRYPTOPANIC_RSS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        root = ET.fromstring(text)
                        items = root.findall(".//item")
                        self.news = []
                        for item in items[:30]:
                            title = item.findtext("title", "")
                            pub_date = item.findtext("pubDate", "")
                            link = item.findtext("link", "")
                            self.news.append({
                                "title": title,
                                "published": pub_date,
                                "source": "cryptopanic",
                                "url": link,
                            })
                        logger.info(f"[News] CryptoPanic: {len(self.news)}개 뉴스 수집")
        except Exception as e:
            logger.warning(f"[News] CryptoPanic 수집 실패: {e}")

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
            "news_velocity": 0.0,  # 뉴스 증가율
            "is_trending": 0.0,
            "trending_rank": 0.0,
        }

        # 심볼 관련 뉴스 카운트
        coin = symbol.split("/")[0] if "/" in symbol else symbol
        coin_lower = coin.lower()

        now = datetime.utcnow()
        count_1h = 0
        count_24h = 0
        for n in self.news:
            title_lower = n["title"].lower()
            if coin_lower in title_lower or coin.upper() in n["title"]:
                count_24h += 1
                # 대략적 시간 체크 (최근 뉴스만)
                if self.news.index(n) < 5:
                    count_1h += 1

        features["news_count_1h"] = float(count_1h)
        features["news_count_24h"] = float(count_24h)
        features["news_velocity"] = count_1h / max(count_24h, 1)

        # 트렌딩 여부
        for t in self.trending:
            if t["symbol"].upper() == coin.upper():
                features["is_trending"] = 1.0
                features["trending_rank"] = 1.0 - (t.get("score", 5) / 10)
                break

        return features
