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
        target_ts = int(datetime.utcnow().timestamp() * 1000)
        all_data = []
        batch_count = 0
        # [Patch AG, 2026-07-29] 무한 재시도 차단 — 재시도 상한 + 지수 백오프.
        # 사고: 7/28 02:30부터 동일 요청이 16,925회(약 40시간) 반복되며 트레이딩 전면 정지.
        # 원인: except 블록이 상한 없이 continue만 수행 → 영구 루프(다음 단계 진입 불가).
        # 수정: MAX_RETRIES 초과 시 포기하고 그때까지 모은 부분 데이터로 진행(루프 생존 우선).
        MAX_RETRIES = 5
        retries = 0

        while True:
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                retries = 0  # 성공 시 카운터 리셋
            except Exception as e:
                retries += 1
                if retries > MAX_RETRIES:
                    logger.error(
                        f"[DataCollect] {symbol} {timeframe} 수집 {MAX_RETRIES}회 연속 실패 → 중단 "
                        f"(부분 데이터 {len(all_data):,}개로 진행): {type(e).__name__}: {e}"
                    )
                    break
                backoff = min(5 * (2 ** (retries - 1)), 60)  # 5→10→20→40→60초
                logger.warning(
                    f"[DataCollect] {symbol} 수집 에러 (재시도 {retries}/{MAX_RETRIES}, "
                    f"{backoff}s 대기): {type(e).__name__}: {e}"
                )
                await asyncio.sleep(backoff)
                continue

            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            batch_count += 1

            # 대량 수집 시 진행률 로깅 (50배치=50,000캔들마다)
            if batch_count % 50 == 0:
                progress = min(100, (since - (target_ts - days * 86400000)) / (days * 86400000) * 100)
                logger.info(
                    f"[DataCollect] {symbol} {timeframe} 수집 중... "
                    f"{len(all_data):,}개 캔들 ({progress:.0f}%)"
                )

            if len(ohlcv) < 1000:
                break
            # rate limit 준수 (대량 수집 시 여유있게)
            await asyncio.sleep(max(exchange.rateLimit / 1000, 0.2))

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
