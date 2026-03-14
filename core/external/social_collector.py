"""소셜 미디어 센티먼트 수집 - Reddit 공개 API"""

import asyncio
from datetime import datetime, timedelta

import aiohttp
from loguru import logger


class SocialCollector:
    """
    소셜 미디어 크립토 센티먼트 수집
    - Reddit: r/CryptoCurrency, r/Bitcoin (공개 JSON API)
    - 활동량, 감성, 화제성 추적
    """

    SUBREDDITS = ["CryptoCurrency", "Bitcoin", "ethtrader"]
    REDDIT_BASE = "https://www.reddit.com/r"

    def __init__(self):
        self.posts: list[dict] = []
        self.metrics: dict = {}
        self.last_fetch: datetime | None = None
        self.fetch_interval = timedelta(minutes=20)

    async def fetch(self) -> dict:
        """Reddit 크립토 커뮤니티 데이터 수집"""
        now = datetime.utcnow()
        if self.last_fetch and (now - self.last_fetch) < self.fetch_interval:
            return {"posts": self.posts, "metrics": self.metrics}

        tasks = [self._fetch_subreddit(sub) for sub in self.SUBREDDITS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_posts = []
        for result in results:
            if isinstance(result, list):
                all_posts.extend(result)

        self.posts = sorted(all_posts, key=lambda x: x.get("score", 0), reverse=True)[:50]
        self._compute_metrics()
        self.last_fetch = now

        return {"posts": self.posts, "metrics": self.metrics}

    async def _fetch_subreddit(self, subreddit: str) -> list[dict]:
        """서브레딧 핫 포스트 수집"""
        posts = []
        try:
            headers = {"User-Agent": "AutoTrader/1.0"}
            async with aiohttp.ClientSession(headers=headers) as session:
                url = f"{self.REDDIT_BASE}/{subreddit}/hot.json?limit=25"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for child in data.get("data", {}).get("children", []):
                            post = child.get("data", {})
                            posts.append({
                                "title": post.get("title", ""),
                                "score": post.get("score", 0),
                                "num_comments": post.get("num_comments", 0),
                                "upvote_ratio": post.get("upvote_ratio", 0.5),
                                "subreddit": subreddit,
                                "created_utc": post.get("created_utc", 0),
                                "is_self": post.get("is_self", True),
                            })
                        logger.debug(f"[Social] r/{subreddit}: {len(posts)}개 포스트 수집")
        except Exception as e:
            logger.warning(f"[Social] r/{subreddit} 수집 실패: {e}")

        return posts

    def _compute_metrics(self):
        """수집된 포스트에서 메트릭 계산"""
        if not self.posts:
            self.metrics = {}
            return

        scores = [p["score"] for p in self.posts]
        comments = [p["num_comments"] for p in self.posts]
        upvotes = [p["upvote_ratio"] for p in self.posts]

        import numpy as np

        self.metrics = {
            "total_posts": len(self.posts),
            "avg_score": float(np.mean(scores)),
            "max_score": max(scores),
            "avg_comments": float(np.mean(comments)),
            "total_engagement": sum(scores) + sum(comments),
            "avg_upvote_ratio": float(np.mean(upvotes)),
            "engagement_std": float(np.std(scores)),
        }

    def get_features(self, symbol: str = "") -> dict:
        """ML 피처로 변환"""
        if not self.metrics:
            return {
                "social_engagement": 0.0,
                "social_sentiment_ratio": 0.5,
                "social_buzz_score": 0.0,
                "social_symbol_mentions": 0.0,
                "social_controversy": 0.0,
            }

        coin = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()

        # 심볼 관련 포스트 필터
        mentions = 0
        mention_scores = []
        for p in self.posts:
            title = p["title"].upper()
            if coin in title or (coin == "BTC" and "BITCOIN" in title) or (coin == "ETH" and "ETHEREUM" in title):
                mentions += 1
                mention_scores.append(p["score"])

        import numpy as np

        features = {
            "social_engagement": min(self.metrics.get("total_engagement", 0) / 10000, 1.0),
            "social_sentiment_ratio": self.metrics.get("avg_upvote_ratio", 0.5),
            "social_buzz_score": min(self.metrics.get("avg_score", 0) / 1000, 1.0),
            "social_symbol_mentions": float(mentions),
            "social_controversy": 1.0 - self.metrics.get("avg_upvote_ratio", 0.5),  # 낮은 upvote = 논란
        }

        # 심볼 관련 감성
        if mention_scores:
            features["social_symbol_avg_score"] = float(np.mean(mention_scores))
        else:
            features["social_symbol_avg_score"] = 0.0

        return features

    def get_titles(self) -> list[str]:
        """센티먼트 분석용 제목 목록 반환"""
        return [p["title"] for p in self.posts]
