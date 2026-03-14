"""자기학습 트레이너 - 주기적 재학습 및 모델 관리"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from core.data.collector import DataCollector
from core.data.features import FeatureEngineer
from core.data.storage import Storage
from core.models.ensemble import EnsembleSignalGenerator
from core.rl.agent import RLAgent
from core.rl.environment import TradingEnvironment


class SelfLearningTrainer:
    """자기학습 루프 - 데이터 수집 → 학습 → 평가 → 모델 교체"""

    def __init__(
        self,
        collector: DataCollector,
        storage: Storage,
        ensemble: EnsembleSignalGenerator,
        rl_agent: RLAgent,
        config: dict,
    ):
        self.collector = collector
        self.storage = storage
        self.ensemble = ensemble
        self.rl_agent = rl_agent
        self.config = config
        self.feature_engineer = FeatureEngineer(config.get("ml", {}).get("features"))
        self.last_train_time = datetime.utcnow() - timedelta(hours=999)

    def should_retrain(self) -> bool:
        interval = self.config.get("ml", {}).get("retrain_interval_hours", 24)
        return (datetime.utcnow() - self.last_train_time) > timedelta(hours=interval)

    async def collect_training_data(self, exchange_name: str, symbol: str, timeframe: str) -> pd.DataFrame:
        """학습 데이터 수집 및 피처 생성"""
        lookback = self.config.get("ml", {}).get("lookback_days", 90)
        df = await self.collector.fetch_all_ohlcv(exchange_name, symbol, timeframe, days=lookback)
        self.storage.save_candles(exchange_name, symbol, timeframe, df)
        df = self.feature_engineer.generate(df)
        return df

    async def train_cycle(self, exchange_name: str, symbol: str, timeframe: str = "1h"):
        """전체 학습 사이클 실행"""
        logger.info(f"=== 자기학습 사이클 시작: {symbol} {timeframe} ===")

        # 1. 데이터 수집
        df = await self.collect_training_data(exchange_name, symbol, timeframe)
        if len(df) < 200:
            logger.warning(f"학습 데이터 부족: {len(df)}개")
            return

        feature_cols = self.feature_engineer.get_feature_columns(df)

        # 2. ML 앙상블 학습
        self.ensemble.train_all(df, feature_cols)

        # 3. RL 에이전트 학습
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        data = np.hstack([df[feature_cols].values, df[ohlcv_cols].values]).astype(np.float32)
        # NaN 처리
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        env = self.rl_agent.create_env(
            data=data,
            feature_dim=len(feature_cols),
            initial_capital=self.config.get("backtest", {}).get("initial_capital", 10000),
            commission=self.config.get("backtest", {}).get("commission_pct", 0.0004),
            leverage=self.config.get("trading", {}).get("leverage", 5),
        )
        rl_metrics = self.rl_agent.train(env)

        # 4. 모델 저장
        self.ensemble.save_all()
        self.rl_agent.save()

        # 5. 성능 기록
        self.storage.save_model_performance(
            "ensemble", self.ensemble.xgb.accuracy, rl_metrics.get("sharpe_ratio", 0),
            rl_metrics.get("win_rate", 0),
        )

        self.last_train_time = datetime.utcnow()
        logger.info(f"=== 자기학습 사이클 완료 ===")

    async def run_continuous(self, exchange_name: str, symbols: list[str], timeframe: str = "1h"):
        """연속 학습 루프"""
        while True:
            if self.should_retrain():
                for symbol in symbols:
                    try:
                        await self.train_cycle(exchange_name, symbol, timeframe)
                    except Exception as e:
                        logger.error(f"학습 실패 ({symbol}): {e}")
            await asyncio.sleep(300)  # 5분 대기
