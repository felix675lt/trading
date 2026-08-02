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

    _TF_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
                   "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440}

    async def fetch_all_ohlcv(
        self,
        exchange_name: str,
        symbol: str,
        timeframe: str,
        days: int = 90,
        storage=None,
    ) -> pd.DataFrame:
        """지정 기간 전체 OHLCV 데이터를 수집.

        [Patch AJ, 2026-07-30] DB 캐시 우선 — storage를 넘기면 이미 저장된 캔들을
        읽고 '부족한 최신 구간만' 거래소에서 받는다(증분 수집).
        기존: 학습 사이클마다 lookback_days(2400일=6.6년, 종목당 약 69만 캔들)를
              통째로 재수집 → 종목당 수십 분, 12종목에서 메인 루프 수 시간 블로킹.
              DB에 저장은 했지만 다시 읽지 않는 write-only 구조였음.
        """
        exchange = self.exchanges[exchange_name]
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        window_start = since
        target_ts = int(datetime.utcnow().timestamp() * 1000)
        all_data = []
        batch_count = 0

        # === DB 캐시 로드 → 갭만 수집 ===
        cached_df = None
        if storage is not None:
            try:
                tf_min = self._TF_MINUTES.get(timeframe, 5)
                need = int(days * 1440 / tf_min) + 100
                cached_df = storage.load_candles(exchange_name, symbol, timeframe, limit=need)
                if cached_df is not None and not cached_df.empty:
                    cached_df = cached_df[~cached_df.index.duplicated(keep="last")].sort_index()
                    last_ts = int(cached_df.index[-1].timestamp() * 1000)
                    if last_ts > since:
                        since = last_ts + 1  # 캐시 이후 구간만 요청
                    gap_min = max(0, (target_ts - last_ts) / 60000)
                    logger.info(
                        f"[DataCollect] {symbol} {timeframe} DB 캐시 {len(cached_df):,}개 활용 "
                        f"→ 최근 {gap_min:.0f}분 갭만 수집"
                    )
                    if since >= target_ts:  # 이미 최신
                        return cached_df[cached_df.index >= pd.to_datetime(window_start, unit="ms")]
                else:
                    cached_df = None
            except Exception as e:
                logger.warning(f"[DataCollect] {symbol} 캐시 로드 실패 → 전체 수집: {e}")
                cached_df = None
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

        # [Patch AJ] 신규분만 DB에 저장 후 캐시와 병합 (증분 수집)
        if storage is not None and not df.empty:
            try:
                storage.save_candles(exchange_name, symbol, timeframe, df)
            except Exception as e:
                logger.warning(f"[DataCollect] {symbol} 증분 저장 실패(무시): {e}")
        if cached_df is not None and not cached_df.empty:
            new_cnt = len(df)
            df = pd.concat([cached_df, df]) if not df.empty else cached_df
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df = df[df.index >= pd.to_datetime(window_start, unit="ms")]
            logger.info(
                f"{symbol} {timeframe} 수집 완료: 캐시 {len(cached_df):,} + 신규 {new_cnt:,} "
                f"→ 총 {len(df):,}개 캔들"
            )
            return df

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
