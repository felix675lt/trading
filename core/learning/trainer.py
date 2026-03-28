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
        self._train_time_file = Path("models_saved/.last_train_time")
        self.last_train_time = self._load_last_train_time()

    def _load_last_train_time(self) -> datetime:
        try:
            if self._train_time_file.exists():
                ts = self._train_time_file.read_text().strip()
                return datetime.fromisoformat(ts)
        except Exception:
            pass
        return datetime.utcnow() - timedelta(hours=999)

    def _save_last_train_time(self):
        try:
            self._train_time_file.parent.mkdir(parents=True, exist_ok=True)
            self._train_time_file.write_text(self.last_train_time.isoformat())
        except Exception:
            pass

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
        """전체 학습 사이클 실행 (기존 모델이 있으면 이어서 학습)"""
        logger.info(f"=== 자기학습 사이클 시작: {symbol} {timeframe} ===")

        # 1. 데이터 수집
        df = await self.collect_training_data(exchange_name, symbol, timeframe)
        if len(df) < 200:
            logger.warning(f"학습 데이터 부족: {len(df)}개")
            return

        # ML은 외부 피처 포함 전체 사용, RL은 기본 피처만 사용 (차원 고정)
        all_feature_cols = self.feature_engineer.get_feature_columns(df)
        base_feature_cols = self.feature_engineer.get_base_feature_columns(df)

        # 1.5. 기존 모델 로드 (이어서 학습하기 위해)
        #      이미 메모리에 있으면 그대로, 없으면 파일에서 로드
        if self.ensemble.xgb.model is None:
            self.ensemble.load_all()
            if self.ensemble.xgb.model:
                logger.info("기존 ML 모델 로드 → 증분학습 모드")
        if self.rl_agent.model is None:
            self.rl_agent.load()
            if self.rl_agent.model:
                logger.info("기존 RL 모델 로드 → 이어서 학습 모드")

        prev_xgb_acc = self.ensemble.xgb.accuracy
        prev_lstm_acc = self.ensemble.lstm.accuracy

        # 2. ML 앙상블 학습 (기존 모델 이어서 / 외부 피처 포함)
        self.ensemble.train_all(df, all_feature_cols)

        # 3. RL 에이전트 학습 (기존 모델 이어서 / 기본 피처만)
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        rl_data = np.hstack([df[base_feature_cols].values, df[ohlcv_cols].values]).astype(np.float32)
        rl_data = np.nan_to_num(rl_data, nan=0.0, posinf=0.0, neginf=0.0)

        is_incremental = self.rl_agent.model is not None
        logger.info(
            f"{'증분' if is_incremental else '최초'}학습 | "
            f"RL 피처: {len(base_feature_cols)}개 / ML 피처: {len(all_feature_cols)}개"
        )

        env = self.rl_agent.create_env(
            data=rl_data,
            feature_dim=len(base_feature_cols),
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

        # 학습 결과 비교 로그
        xgb_diff = self.ensemble.xgb.accuracy - prev_xgb_acc if prev_xgb_acc > 0 else 0
        lstm_diff = self.ensemble.lstm.accuracy - prev_lstm_acc if prev_lstm_acc > 0 else 0
        logger.info(
            f"XGB: {self.ensemble.xgb.accuracy:.4f} ({xgb_diff:+.4f}) | "
            f"LSTM: {self.ensemble.lstm.accuracy:.4f} ({lstm_diff:+.4f}) | "
            f"RL Sharpe: {rl_metrics.get('sharpe_ratio', 0):.2f}"
        )

        self.last_train_time = datetime.utcnow()
        self._save_last_train_time()
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
