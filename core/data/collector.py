"""거래소 데이터 수집 모듈 - ccxt 기반 실시간/과거 데이터 수집"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger


class DataCollector:
    """거래소에서 OHLCV, 펀딩비, 오더북 데이터를 수집"""

    def __init__(self, exchange_configs: dict):
        self.exchanges: dict = {}
        self.exchange_configs = exchange_configs

    async def initialize(self):
        for name, cfg in self.exchange_configs.items():
            exchange_class = getattr(ccxt, name)
            self.exchanges[name] = exchange_class({
                "apiKey": cfg.get("api_key", ""),
                "secret": cfg.get("secret", ""),
                "options": cfg.get("options", {}),
                "enableRateLimit": True,
            })
            if cfg.get("testnet"):
                self.exchanges[name].set_sandbox_mode(True)
            logger.info(f"거래소 초기화: {name}")

    async def close(self):
        for exchange in self.exchanges.values():
            await exchange.close()

    async def fetch_ohlcv(
        self,
        exchange_name: str,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """OHLCV 캔들 데이터 조회"""
        exchange = self.exchanges[exchange_name]
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    async def fetch_all_ohlcv(
        self,
        exchange_name: str,
        symbol: str,
        timeframe: str,
        days: int = 90,
    ) -> pd.DataFrame:
        """지정 기간 전체 OHLCV 데이터를 페이지네이션으로 수집"""
        exchange = self.exchanges[exchange_name]
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        all_data = []

        while True:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
            await asyncio.sleep(exchange.rateLimit / 1000)

        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        logger.info(f"{symbol} {timeframe} 데이터 수집 완료: {len(df)}개 캔들")
        return df

    async def fetch_funding_rate(self, exchange_name: str, symbol: str) -> float:
        """현재 펀딩비 조회"""
        exchange = self.exchanges[exchange_name]
        try:
            funding = await exchange.fetch_funding_rate(symbol)
            return funding.get("fundingRate", 0.0)
        except Exception as e:
            logger.warning(f"펀딩비 조회 실패 ({symbol}): {e}")
            return 0.0

    async def fetch_orderbook(self, exchange_name: str, symbol: str, limit: int = 20) -> dict:
        """오더북 스냅샷 조회"""
        exchange = self.exchanges[exchange_name]
        ob = await exchange.fetch_order_book(symbol, limit=limit)
        bid_vol = sum(b[1] for b in ob["bids"][:limit])
        ask_vol = sum(a[1] for a in ob["asks"][:limit])
        spread = ob["asks"][0][0] - ob["bids"][0][0] if ob["asks"] and ob["bids"] else 0
        return {
            "bid_volume": bid_vol,
            "ask_volume": ask_vol,
            "spread": spread,
            "imbalance": (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0,
        }

    async def fetch_ticker(self, exchange_name: str, symbol: str) -> dict:
        """현재 티커 정보"""
        exchange = self.exchanges[exchange_name]
        return await exchange.fetch_ticker(symbol)
