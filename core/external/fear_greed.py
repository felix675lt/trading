"""Fear & Greed Index 수집 - Alternative.me API (무료, 키 불필요)"""

import asyncio
from datetime import datetime, timedelta

import aiohttp
from loguru import logger


class FearGreedCollector:
    """
    크립토 공포/탐욕 지수 수집
    - 0~25: 극도의 공포 (매수 기회)
    - 25~45: 공포
    - 45~55: 중립
    - 55~75: 탐욕
    - 75~100: 극도의 탐욕 (매도 시그널)
    """

    API_URL = "https://api.alternative.me/fng/"

    def __init__(self):
        self.current: dict = {}
        self.history: list[dict] = []
        self.last_fetch: datetime | None = None
        self.fetch_interval = timedelta(minutes=30)

    async def fetch(self) -> dict:
        """현재 Fear & Greed Index 조회"""
        now = datetime.utcnow()
        if self.last_fetch and (now - self.last_fetch) < self.fetch_interval:
            return self.current

        try:
            async with aiohttp.ClientSession() as session:
                # 최근 30일 데이터
                async with session.get(f"{self.API_URL}?limit=30&format=json", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        entries = data.get("data", [])

                        if entries:
                            latest = entries[0]
                            self.current = {
                                "value": int(latest["value"]),
                                "classification": latest["value_classification"],
                                "timestamp": datetime.fromtimestamp(int(latest["timestamp"])).isoformat(),
                            }

                            self.history = [
                                {
                                    "value": int(e["value"]),
                                    "classification": e["value_classification"],
                                    "timestamp": datetime.fromtimestamp(int(e["timestamp"])).isoformat(),
                                }
                                for e in entries
                            ]

                            self.last_fetch = now
                            logger.info(f"[FearGreed] 현재: {self.current['value']} ({self.current['classification']})")
        except Exception as e:
            logger.warning(f"[FearGreed] 수집 실패: {e}")

        return self.current

    def get_features(self) -> dict:
        """ML 피처로 변환"""
        if not self.current:
            return {
                "fg_value": 50.0,
                "fg_normalized": 0.0,
                "fg_extreme_fear": 0.0,
                "fg_extreme_greed": 0.0,
                "fg_trend_7d": 0.0,
                "fg_volatility": 0.0,
            }

        value = self.current["value"]
        features = {
            "fg_value": float(value),
            "fg_normalized": (value - 50) / 50,  # -1 ~ 1
            "fg_extreme_fear": 1.0 if value <= 20 else 0.0,
            "fg_extreme_greed": 1.0 if value >= 80 else 0.0,
        }

        # 7일 트렌드
        if len(self.history) >= 7:
            recent = [h["value"] for h in self.history[:7]]
            features["fg_trend_7d"] = (recent[0] - recent[-1]) / max(recent[-1], 1)
            features["fg_volatility"] = float(__import__("numpy").std(recent) / max(__import__("numpy").mean(recent), 1))
        else:
            features["fg_trend_7d"] = 0.0
            features["fg_volatility"] = 0.0

        return features
