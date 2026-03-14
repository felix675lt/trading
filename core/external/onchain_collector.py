"""온체인 데이터 수집 - 공개 API (키 불필요)"""

import asyncio
from datetime import datetime, timedelta

import aiohttp
from loguru import logger


class OnchainCollector:
    """
    온체인 지표 수집 (무료 공개 API 사용)
    - 거래소 BTC 잔고 (유출입 추적)
    - 해시레이트
    - 활성 주소 수
    - 멤풀 크기
    """

    # Blockchain.com 공개 API
    BLOCKCHAIN_API = "https://api.blockchain.info"
    # Mempool.space 공개 API
    MEMPOOL_API = "https://mempool.space/api"

    def __init__(self):
        self.data: dict = {}
        self.last_fetch: datetime | None = None
        self.fetch_interval = timedelta(minutes=30)
        self.history: list[dict] = []

    async def fetch(self) -> dict:
        """온체인 데이터 수집"""
        now = datetime.utcnow()
        if self.last_fetch and (now - self.last_fetch) < self.fetch_interval:
            return self.data

        results = await asyncio.gather(
            self._fetch_exchange_balance(),
            self._fetch_hashrate(),
            self._fetch_mempool(),
            self._fetch_difficulty(),
            return_exceptions=True,
        )

        self.last_fetch = now
        self.history.append({**self.data, "timestamp": now.isoformat()})
        if len(self.history) > 200:
            self.history = self.history[-200:]

        return self.data

    async def _fetch_exchange_balance(self):
        """거래소 BTC 잔고 추적 (Blockchain.com)"""
        try:
            async with aiohttp.ClientSession() as session:
                # 거래소 유출입 대용: 대량 거래 수
                url = f"{self.BLOCKCHAIN_API}/charts/n-transactions?timespan=7days&format=json"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        values = data.get("values", [])
                        if values:
                            latest = values[-1]["y"]
                            avg = sum(v["y"] for v in values) / len(values)
                            self.data["tx_count"] = latest
                            self.data["tx_count_ratio"] = latest / avg if avg > 0 else 1.0
                            logger.debug(f"[Onchain] 일일 트랜잭션: {latest:,.0f} (평균 대비 {latest/avg:.2f}x)")
        except Exception as e:
            logger.warning(f"[Onchain] 트랜잭션 수집 실패: {e}")

    async def _fetch_hashrate(self):
        """BTC 해시레이트"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.BLOCKCHAIN_API}/charts/hash-rate?timespan=30days&format=json"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        values = data.get("values", [])
                        if values:
                            latest = values[-1]["y"]
                            prev_week = values[-8]["y"] if len(values) >= 8 else latest
                            self.data["hashrate"] = latest
                            self.data["hashrate_change_7d"] = (latest - prev_week) / prev_week if prev_week > 0 else 0
        except Exception as e:
            logger.warning(f"[Onchain] 해시레이트 수집 실패: {e}")

    async def _fetch_mempool(self):
        """멤풀 상태 (Mempool.space)"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.MEMPOOL_API}/mempool"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.data["mempool_count"] = data.get("count", 0)
                        self.data["mempool_vsize"] = data.get("vsize", 0)
                        # 멤풀이 크면 네트워크 혼잡 = 수요 증가 신호
                        logger.debug(f"[Onchain] 멤풀: {data.get('count', 0):,} txs")
        except Exception as e:
            logger.warning(f"[Onchain] 멤풀 수집 실패: {e}")

    async def _fetch_difficulty(self):
        """BTC 채굴 난이도"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.MEMPOOL_API}/v1/mining/difficulty-adjustments"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and len(data) >= 2:
                            latest_diff = data[0]
                            prev_diff = data[1]
                            # 난이도 변화율 (양수 = 채굴자 증가 = 강세 신호)
                            if isinstance(latest_diff, list) and isinstance(prev_diff, list):
                                latest_val = latest_diff[2] if len(latest_diff) > 2 else 0
                                prev_val = prev_diff[2] if len(prev_diff) > 2 else 1
                                self.data["difficulty_change"] = (latest_val - prev_val) / prev_val if prev_val > 0 else 0
        except Exception as e:
            logger.warning(f"[Onchain] 난이도 수집 실패: {e}")

    def get_features(self) -> dict:
        """ML 피처로 변환"""
        features = {
            "onchain_tx_ratio": self.data.get("tx_count_ratio", 1.0),
            "onchain_hashrate_change": self.data.get("hashrate_change_7d", 0.0),
            "onchain_mempool_congestion": min(self.data.get("mempool_count", 0) / 100000, 1.0),
            "onchain_difficulty_change": self.data.get("difficulty_change", 0.0),
        }

        # 종합 온체인 점수 (-1 약세 ~ 1 강세)
        score = 0.0
        if features["onchain_tx_ratio"] > 1.2:
            score += 0.3  # 트랜잭션 증가 = 네트워크 활성
        elif features["onchain_tx_ratio"] < 0.8:
            score -= 0.3

        if features["onchain_hashrate_change"] > 0.05:
            score += 0.3  # 해시레이트 증가 = 채굴자 강세
        elif features["onchain_hashrate_change"] < -0.05:
            score -= 0.3

        if features["onchain_mempool_congestion"] > 0.5:
            score += 0.2  # 멤풀 혼잡 = 수요 증가
        if features["onchain_difficulty_change"] > 0:
            score += 0.2  # 난이도 증가 = 강세

        features["onchain_composite_score"] = max(-1, min(1, score))

        return features
