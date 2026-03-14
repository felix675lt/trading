"""매크로 경제 지표 수집 - 공개 API"""

import asyncio
from datetime import datetime, timedelta

import aiohttp
from loguru import logger


class MacroCollector:
    """
    매크로 경제 지표 수집 (무료 API)
    - DXY (달러 인덱스) 대용 지표
    - 금 가격 (안전자산 선호도)
    - 미국 국채 금리
    - 주요 지수 방향성
    """

    def __init__(self):
        self.data: dict = {}
        self.last_fetch: datetime | None = None
        self.fetch_interval = timedelta(hours=1)
        self.history: list[dict] = []

    async def fetch(self) -> dict:
        """매크로 데이터 수집"""
        now = datetime.utcnow()
        if self.last_fetch and (now - self.last_fetch) < self.fetch_interval:
            return self.data

        await asyncio.gather(
            self._fetch_dxy_proxy(),
            self._fetch_global_market(),
            return_exceptions=True,
        )

        self.last_fetch = now
        self.history.append({**self.data, "timestamp": now.isoformat()})
        if len(self.history) > 200:
            self.history = self.history[-200:]

        return self.data

    async def _fetch_dxy_proxy(self):
        """달러 강세 대용 지표 - CoinGecko BTC/USD 변동으로 간접 추정"""
        try:
            async with aiohttp.ClientSession() as session:
                # BTC 시장 데이터에서 USD 기준 변동 추출
                url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,tether&vs_currencies=usd,eur,gbp&include_24hr_change=true"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        btc = data.get("bitcoin", {})

                        # USD vs EUR 환율 변동으로 달러 강세 추정
                        usd_price = btc.get("usd", 0)
                        eur_price = btc.get("eur", 0)
                        if eur_price > 0 and usd_price > 0:
                            self.data["usd_eur_ratio"] = usd_price / eur_price
                        self.data["btc_24h_change"] = btc.get("usd_24h_change", 0) / 100

                        # Tether premium (USDT != $1이면 시장 스트레스)
                        usdt = data.get("tether", {})
                        usdt_price = usdt.get("usd", 1.0)
                        self.data["usdt_peg_deviation"] = abs(usdt_price - 1.0)

                        logger.debug(f"[Macro] BTC 24h: {self.data.get('btc_24h_change', 0):.2%}, USDT 디페그: {self.data.get('usdt_peg_deviation', 0):.4f}")
        except Exception as e:
            logger.warning(f"[Macro] DXY 프록시 수집 실패: {e}")

    async def _fetch_global_market(self):
        """글로벌 크립토 시장 데이터"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.coingecko.com/api/v3/global"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        market = data.get("data", {})

                        self.data["total_market_cap_change_24h"] = market.get("market_cap_change_percentage_24h_usd", 0) / 100
                        self.data["btc_dominance"] = market.get("market_cap_percentage", {}).get("btc", 0) / 100
                        self.data["eth_dominance"] = market.get("market_cap_percentage", {}).get("eth", 0) / 100
                        self.data["active_cryptocurrencies"] = market.get("active_cryptocurrencies", 0)
                        self.data["total_volume_24h"] = market.get("total_volume", {}).get("usd", 0)

                        logger.debug(f"[Macro] 시장 24h: {self.data.get('total_market_cap_change_24h', 0):.2%}, BTC 도미넌스: {self.data.get('btc_dominance', 0):.1%}")
        except Exception as e:
            logger.warning(f"[Macro] 글로벌 시장 수집 실패: {e}")

    def get_features(self) -> dict:
        """ML 피처로 변환"""
        features = {
            "macro_market_change_24h": self.data.get("total_market_cap_change_24h", 0.0),
            "macro_btc_dominance": self.data.get("btc_dominance", 0.45),
            "macro_eth_dominance": self.data.get("eth_dominance", 0.18),
            "macro_usdt_depeg": self.data.get("usdt_peg_deviation", 0.0),
            "macro_btc_24h_change": self.data.get("btc_24h_change", 0.0),
        }

        # 매크로 종합 점수
        score = 0.0

        # 시장 전체 방향
        market_change = features["macro_market_change_24h"]
        if market_change > 0.03:
            score += 0.4
        elif market_change > 0.01:
            score += 0.2
        elif market_change < -0.03:
            score -= 0.4
        elif market_change < -0.01:
            score -= 0.2

        # USDT 디페그 (시장 스트레스 시그널)
        if features["macro_usdt_depeg"] > 0.005:
            score -= 0.3  # 스트레스 신호

        # BTC 도미넌스 변화 (상승 = 리스크 오프)
        btc_dom = features["macro_btc_dominance"]
        if btc_dom > 0.55:
            score -= 0.1  # 알트 약세
        elif btc_dom < 0.40:
            score += 0.1  # 알트 강세 = 위험선호

        features["macro_composite_score"] = max(-1, min(1, score))

        return features
