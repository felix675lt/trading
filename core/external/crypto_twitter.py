"""크립토 트위터/소셜 실시간 수집 - Stocktwits + Free Crypto News API"""

import time
from datetime import datetime, timedelta

import aiohttp
from loguru import logger


class CryptoTwitterCollector:
    """
    무료 크립토 소셜 데이터 수집기

    소스:
    1. Stocktwits - 실시간 불/베어 센티먼트 (API 키 불필요)
    2. Free Crypto News - AI 센티먼트 분석 (API 키 불필요)
    3. CoinGlass 공개 데이터 - 청산/거래량 급등 감지
    """

    STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"
    CRYPTO_NEWS_BASE = "https://cryptocurrency.cv/api"

    # 심볼 매핑 (Stocktwits용)
    SYMBOL_MAP = {
        "BTC": "BTC.X",
        "ETH": "ETH.X",
        "SOL": "SOL.X",
        "XRP": "XRP.X",
        "DOGE": "DOGE.X",
    }

    # 영향력 있는 크립토 인물 키워드 (트위터/소셜에서 자주 언급)
    INFLUENCER_KEYWORDS = {
        "elon": 3.0, "musk": 3.0,
        "saylor": 2.5, "michael saylor": 2.5,
        "cathie": 2.0, "ark invest": 2.0,
        "gary gensler": 2.5, "sec": 2.5,
        "powell": 3.0, "fed": 2.5, "fomc": 3.0,
        "trump": 2.5, "biden": 2.0,
        "blackrock": 2.5, "etf": 3.0,
        "binance": 2.0, "cz": 2.0,
        "vitalik": 2.0, "buterin": 2.0,
    }

    # 긴급 이벤트 키워드
    URGENT_KEYWORDS = {
        "hack": -4.0, "hacked": -4.0, "exploit": -3.5,
        "ban": -3.0, "banned": -3.0, "lawsuit": -2.5,
        "crash": -3.0, "dump": -2.5, "plunge": -3.0,
        "moon": 2.0, "pump": 2.0, "rally": 2.5,
        "approval": 3.0, "approved": 3.0,
        "adoption": 2.5, "partnership": 2.0,
        "halving": 2.0, "upgrade": 1.5,
        "liquidation": -2.5, "liquidated": -2.5,
        "whale": 1.5, "accumulation": 2.0,
        "war": -3.0, "sanctions": -2.5,
        "rate cut": 2.5, "rate hike": -2.5,
    }

    def __init__(self):
        self._cache = {}
        self._cache_ttl = 120  # 2분 캐시
        self._last_fetch = {}
        self._history = []  # 과거 센티먼트 기록 (트렌드 분석용)

    async def fetch(self, symbol: str = "BTC") -> dict:
        """모든 소스에서 크립토 소셜 데이터 수집"""
        cache_key = f"twitter_{symbol}"
        now = time.time()

        if cache_key in self._cache and (now - self._cache.get(f"{cache_key}_time", 0)) < self._cache_ttl:
            return self._cache[cache_key]

        results = {
            "stocktwits": {},
            "crypto_news_sentiment": {},
            "urgent_events": [],
            "influencer_mentions": [],
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                # 병렬 수집
                import asyncio
                tasks = [
                    self._fetch_stocktwits(session, symbol),
                    self._fetch_crypto_news_sentiment(session, symbol),
                    self._fetch_crypto_news_latest(session, symbol),
                ]
                stocktwits, ai_sentiment, latest_news = await asyncio.gather(
                    *tasks, return_exceptions=True
                )

                if not isinstance(stocktwits, Exception):
                    results["stocktwits"] = stocktwits
                if not isinstance(ai_sentiment, Exception):
                    results["crypto_news_sentiment"] = ai_sentiment
                if not isinstance(latest_news, Exception):
                    results["urgent_events"] = latest_news.get("urgent_events", [])
                    results["influencer_mentions"] = latest_news.get("influencer_mentions", [])

        except Exception as e:
            logger.warning(f"[CryptoTwitter] 수집 실패: {e}")

        features = self._calculate_features(results, symbol)
        self._cache[cache_key] = features
        self._cache[f"{cache_key}_time"] = now

        # 히스토리 기록
        self._history.append({
            "time": datetime.utcnow(),
            "symbol": symbol,
            "score": features.get("twitter_composite", 0),
        })
        # 최근 100개만 유지
        if len(self._history) > 100:
            self._history = self._history[-100:]

        return features

    async def _fetch_stocktwits(self, session: aiohttp.ClientSession, symbol: str) -> dict:
        """Stocktwits 실시간 센티먼트 수집"""
        st_symbol = self.SYMBOL_MAP.get(symbol.upper(), f"{symbol.upper()}.X")
        url = f"{self.STOCKTWITS_BASE}/streams/symbol/{st_symbol}.json"

        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()

                messages = data.get("messages", [])
                if not messages:
                    return {}

                bullish = 0
                bearish = 0
                total = len(messages)
                recent_scores = []

                for msg in messages:
                    sentiment = msg.get("entities", {}).get("sentiment", {})
                    if sentiment:
                        basic = sentiment.get("basic", "")
                        if basic == "Bullish":
                            bullish += 1
                            recent_scores.append(1.0)
                        elif basic == "Bearish":
                            bearish += 1
                            recent_scores.append(-1.0)
                        else:
                            recent_scores.append(0.0)
                    else:
                        recent_scores.append(0.0)

                bull_ratio = bullish / max(total, 1)
                bear_ratio = bearish / max(total, 1)
                avg_score = sum(recent_scores) / max(len(recent_scores), 1)

                logger.debug(f"[Stocktwits] {symbol}: {bullish}불/{bearish}베어 (총{total})")

                return {
                    "bullish_count": bullish,
                    "bearish_count": bearish,
                    "total_messages": total,
                    "bull_ratio": bull_ratio,
                    "bear_ratio": bear_ratio,
                    "avg_sentiment": avg_score,
                    "message_velocity": total,  # 메시지 수 자체가 활동도
                }
        except Exception as e:
            logger.debug(f"[Stocktwits] {symbol} 수집 실패: {e}")
            return {}

    async def _fetch_crypto_news_sentiment(self, session: aiohttp.ClientSession, symbol: str) -> dict:
        """Free Crypto News API의 AI 센티먼트 분석"""
        url = f"{self.CRYPTO_NEWS_BASE}/ai/sentiment"
        params = {"asset": symbol.lower()}

        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()

                return {
                    "ai_sentiment": data.get("sentiment", 0),
                    "ai_confidence": data.get("confidence", 0),
                    "ai_summary": data.get("summary", ""),
                }
        except Exception as e:
            logger.debug(f"[CryptoNews AI] {symbol} 센티먼트 실패: {e}")
            return {}

    async def _fetch_crypto_news_latest(self, session: aiohttp.ClientSession, symbol: str) -> dict:
        """최신 뉴스에서 긴급 이벤트 + 영향력 인물 감지"""
        url = f"{self.CRYPTO_NEWS_BASE}/news"

        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return {"urgent_events": [], "influencer_mentions": []}
                data = await resp.json()

                articles = data if isinstance(data, list) else data.get("articles", data.get("data", []))
                if not isinstance(articles, list):
                    return {"urgent_events": [], "influencer_mentions": []}

                urgent_events = []
                influencer_mentions = []

                for article in articles[:30]:  # 최근 30개만 분석
                    title = article.get("title", "").lower() if isinstance(article, dict) else ""
                    content = article.get("description", article.get("summary", "")).lower() if isinstance(article, dict) else ""
                    text = f"{title} {content}"

                    # 심볼 관련성 체크
                    if symbol.lower() not in text and "bitcoin" not in text and "crypto" not in text:
                        continue

                    # 긴급 이벤트 감지
                    for keyword, impact in self.URGENT_KEYWORDS.items():
                        if keyword in text:
                            urgent_events.append({
                                "keyword": keyword,
                                "impact": impact,
                                "title": title[:100],
                                "time": article.get("published", "") if isinstance(article, dict) else "",
                            })

                    # 영향력 인물 감지
                    for keyword, weight in self.INFLUENCER_KEYWORDS.items():
                        if keyword in text:
                            influencer_mentions.append({
                                "keyword": keyword,
                                "weight": weight,
                                "title": title[:100],
                            })

                return {
                    "urgent_events": urgent_events[:10],
                    "influencer_mentions": influencer_mentions[:10],
                }

        except Exception as e:
            logger.debug(f"[CryptoNews Latest] 수집 실패: {e}")
            return {"urgent_events": [], "influencer_mentions": []}

    def _calculate_features(self, results: dict, symbol: str) -> dict:
        """수집 데이터 → ML 피처 변환"""
        st = results.get("stocktwits", {})
        ai = results.get("crypto_news_sentiment", {})
        events = results.get("urgent_events", [])
        influencers = results.get("influencer_mentions", [])

        # Stocktwits 센티먼트 (가장 실시간)
        st_sentiment = st.get("avg_sentiment", 0)
        st_bull_ratio = st.get("bull_ratio", 0.5)
        st_velocity = min(st.get("message_velocity", 0) / 30, 1.0)  # 30개 기준 정규화

        # AI 센티먼트
        ai_sentiment = ai.get("ai_sentiment", 0)
        ai_confidence = ai.get("ai_confidence", 0)

        # 긴급 이벤트 점수
        event_score = 0.0
        high_impact_events = []
        for event in events:
            impact = event.get("impact", 0)
            event_score += impact
            if abs(impact) >= 3.0:
                high_impact_events.append(event)

        event_score = max(-1.0, min(1.0, event_score / 5.0))  # 정규화

        # 영향력 인물 점수
        influencer_score = 0.0
        for mention in influencers:
            influencer_score += mention.get("weight", 0) * 0.1
        influencer_score = max(-1.0, min(1.0, influencer_score))

        # 종합 소셜 점수 (가중 합산)
        composite = (
            st_sentiment * 0.35 +       # Stocktwits 실시간 (가장 중요)
            ai_sentiment * 0.25 +        # AI 분석
            event_score * 0.25 +         # 긴급 이벤트
            influencer_score * 0.15      # 인플루언서
        )
        composite = max(-1.0, min(1.0, composite))

        # 트렌드 변화율 (이전 데이터 대비)
        trend = 0.0
        if len(self._history) >= 3:
            recent = [h["score"] for h in self._history[-3:]]
            older = [h["score"] for h in self._history[-6:-3]] if len(self._history) >= 6 else recent
            trend = sum(recent) / len(recent) - sum(older) / len(older)

        features = {
            "twitter_composite": composite,
            "twitter_stocktwits_sentiment": st_sentiment,
            "twitter_bull_ratio": st_bull_ratio,
            "twitter_message_velocity": st_velocity,
            "twitter_ai_sentiment": ai_sentiment,
            "twitter_ai_confidence": ai_confidence,
            "twitter_event_score": event_score,
            "twitter_influencer_score": influencer_score,
            "twitter_high_impact_count": len(high_impact_events),
            "twitter_trend": trend,
        }

        if st_sentiment != 0 or ai_sentiment != 0 or events:
            direction = "bullish" if composite > 0.1 else ("bearish" if composite < -0.1 else "neutral")
            logger.info(
                f"[CryptoTwitter] {symbol}: {composite:.2f}({direction}) | "
                f"ST:{st_sentiment:.2f} AI:{ai_sentiment:.2f} "
                f"이벤트:{len(events)} 인플:{len(influencers)}"
            )

        return features

    def get_signal(self) -> dict:
        """전략 매니저용 신호 반환"""
        if not self._cache:
            return {"score": 0, "direction": "neutral", "strength": "weak", "events": []}

        # 가장 최근 캐시에서 가져오기
        for key, val in self._cache.items():
            if isinstance(val, dict) and "twitter_composite" in val:
                score = val["twitter_composite"]
                direction = "bullish" if score > 0.1 else ("bearish" if score < -0.1 else "neutral")
                strength = "strong" if abs(score) > 0.4 else ("moderate" if abs(score) > 0.2 else "weak")
                return {
                    "score": score,
                    "direction": direction,
                    "strength": strength,
                    "high_impact": val.get("twitter_high_impact_count", 0) > 0,
                    "events": [],
                }

        return {"score": 0, "direction": "neutral", "strength": "weak", "events": []}
