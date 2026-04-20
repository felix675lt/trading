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
        tier_manager=None,  # CapitalTierManager (optional, walk-forward CV gating용)
        meta_labeler=None,  # MetaLabeler (optional, tier=large+)
        hmm_regime=None,    # HMMRegimeClassifier (optional, tier=large+)
    ):
        self.collector = collector
        self.storage = storage
        self.ensemble = ensemble
        self.rl_agent = rl_agent
        self.config = config
        self.tier_manager = tier_manager
        self.meta_labeler = meta_labeler
        self.hmm_regime = hmm_regime
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
        """학습 데이터 수집 및 피처 생성

        통찰 #3 (2026-04-20): 파생 스냅샷(funding/OI/LS ratio) 역사 데이터를
        DB에서 로드하여 학습 DF에 시간 조인 — 기존 상수값 ext_deriv_* 컬럼을
        실제 시계열로 덮어써서 XGBoost가 의미있는 split을 생성하도록.
        """
        lookback = self.config.get("ml", {}).get("lookback_days", 90)
        df = await self.collector.fetch_all_ohlcv(exchange_name, symbol, timeframe, days=lookback)
        self.storage.save_candles(exchange_name, symbol, timeframe, df)
        df = self.feature_engineer.generate(df)

        # === 통찰 #3: 역사적 파생 스냅샷 조인 ===
        try:
            deriv_df = self.storage.load_derivatives_snapshots(symbol, days=lookback)
            if deriv_df is not None and not deriv_df.empty:
                # 컬럼 prefix를 `ext_deriv_` 로 맞춤 (기존 상수값 컬럼명과 동일)
                deriv_df = deriv_df.add_prefix("ext_deriv_")
                # df.index(타임스탬프)에 맞춰 ffill — 미래 정보 유입 차단
                # (스냅샷이 캔들 시각 직전에 찍힌 마지막 값만 사용)
                deriv_aligned = deriv_df.reindex(df.index, method="ffill")
                # 기존에 external_features 로부터 주입된 상수 ext_deriv_* 컬럼은
                # 시계열로 덮어씀 (존재하면). 없으면 새로 추가.
                overlap = [c for c in deriv_aligned.columns if c in df.columns]
                new_cols = [c for c in deriv_aligned.columns if c not in df.columns]
                if overlap:
                    df[overlap] = deriv_aligned[overlap]
                if new_cols:
                    df = pd.concat([df, deriv_aligned[new_cols]], axis=1)
                logger.info(
                    f"[Derivatives] 역사 스냅샷 {len(deriv_df)}개 조인 완료 — "
                    f"overwrite={len(overlap)}, new={len(new_cols)}"
                )
            else:
                logger.debug(
                    f"[Derivatives] {symbol} 역사 스냅샷 없음 — 상수값 ext_deriv_* 유지 "
                    f"(시간 누적 후 재학습 시 활성화)"
                )
        except Exception as e:
            logger.warning(f"[Derivatives] 역사 스냅샷 조인 실패: {e} — 기존 상수값 유지")

        return df

    async def _prepare_btc_reference(self, exchange_name: str, timeframe: str):
        """BTC 데이터를 먼저 준비하여 cross-asset 피처용 reference 세팅

        통찰 #2: alt 심볼 학습 시 BTC 5m 선행 피처 주입 — set_btc_reference() 호출.
        BTC 자신의 학습에는 btc_reference=None 으로 복구 (자기참조 방지).
        """
        btc_symbol = "BTC/USDT:USDT"
        try:
            # BTC raw OHLCV (피처 없이, 순수 OHLCV + 기본 returns/rsi만 계산)
            lookback = self.config.get("ml", {}).get("lookback_days", 90)
            btc_raw = await self.collector.fetch_all_ohlcv(
                exchange_name, btc_symbol, timeframe, days=lookback
            )
            # 최소한의 선행 피처만 계산 (전체 feature 생성은 과중)
            import ta
            btc_raw["returns_1"] = btc_raw["close"].pct_change(1)
            btc_raw["returns_5"] = btc_raw["close"].pct_change(5)
            btc_raw["returns_20"] = btc_raw["close"].pct_change(20)
            btc_raw["rsi_14"] = ta.momentum.RSIIndicator(btc_raw["close"], window=14).rsi()
            btc_raw["volatility_20"] = btc_raw["returns_1"].rolling(20).std()
            self.feature_engineer.set_btc_reference(btc_raw)
            logger.info(f"[CrossAsset] BTC reference 준비 완료 — {len(btc_raw)}개 캔들")
        except Exception as e:
            logger.warning(f"[CrossAsset] BTC reference 준비 실패: {e} — 독립 학습으로 fallback")
            self.feature_engineer.set_btc_reference(None)

    async def train_cycle(self, exchange_name: str, symbol: str, timeframe: str = "1h"):
        """전체 학습 사이클 실행 (기존 모델이 있으면 이어서 학습)"""
        logger.info(f"=== 자기학습 사이클 시작: {symbol} {timeframe} ===")

        # 0. Cross-Asset BTC Reference 주입 (통찰 #2) —
        #    [2026-04-20 수정] 피처 수 일관성을 위해 BTC 학습 시에도 자기 자신을
        #    reference로 사용. btc_returns_1 == returns_1 (동어반복) 이 되지만
        #    단일 XGBoost/LSTM 모델이 모든 심볼에서 같은 피처 차원(39개)으로 동작.
        #    중복 피처는 XGBoost가 자동으로 split importance=0에 근접하게 처리.
        await self._prepare_btc_reference(exchange_name, timeframe)

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
        # ⚠️ 비대 가드 (2026-04-20 추가): xgboost.pkl이 200MB 초과하면
        #    증분학습 누적으로 비정상 부풀었다는 뜻 → 백업 후 처음부터 재학습
        xgb_path = Path("models_saved/xgboost.pkl")
        if xgb_path.exists():
            size_mb = xgb_path.stat().st_size / (1024 * 1024)
            if size_mb > 200:
                from datetime import datetime as _dt
                backup_dir = Path("models_saved/auto_reset")
                backup_dir.mkdir(parents=True, exist_ok=True)
                ts = _dt.utcnow().strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"xgboost_{size_mb:.0f}MB_{ts}.pkl"
                logger.warning(
                    f"[모델가드] xgboost.pkl 비정상 비대 감지 ({size_mb:.0f}MB > 200MB) "
                    f"→ {backup_path} 로 백업 후 초기화"
                )
                xgb_path.rename(backup_path)
                # 메모리의 모델 참조도 리셋
                self.ensemble.xgb.model = None

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
        #    티어 기반 Walk-Forward CV: small+ 에서 활성화 (PAPER 가상 시드 포함)
        use_walk_forward = False
        if self.tier_manager is not None:
            try:
                use_walk_forward = (
                    self.tier_manager.feature_enabled("walk_forward_cv", mode="paper")
                    or self.tier_manager.feature_enabled("walk_forward_cv", mode="live")
                )
            except Exception as e:
                logger.debug(f"[WalkForward] 티어 조회 실패: {e}")

        if use_walk_forward:
            logger.info(
                f"[WalkForward] 활성화 — tier(paper={self.tier_manager.get_tier('paper').name}"
                f", live={self.tier_manager.get_tier('live').name})"
            )

        # tier=large+ 에서 PurgedKFold + Embargo (Lopez de Prado) 활성화 — meta_labeling 플래그와 동일 gating
        use_purged = False
        if self.tier_manager is not None:
            try:
                use_purged = (
                    self.tier_manager.feature_enabled("meta_labeling", mode="paper")
                    or self.tier_manager.feature_enabled("meta_labeling", mode="live")
                )
            except Exception:
                pass

        self.ensemble.train_all(
            df, all_feature_cols,
            walk_forward=use_walk_forward,
            use_purged_kfold=use_purged,
        )

        # === Meta-Labeler 학습 (tier=large+) ===
        if self.meta_labeler is not None and self.tier_manager is not None:
            try:
                meta_on = (
                    self.tier_manager.feature_enabled("meta_labeling", mode="paper")
                    or self.tier_manager.feature_enabled("meta_labeling", mode="live")
                )
                if meta_on and "tb_label" in df.columns:
                    # 1차 시그널 주입: XGB 예측값을 DF에 컬럼으로 추가
                    df_meta = df.copy()
                    try:
                        X_all = df_meta[all_feature_cols].values
                        if self.ensemble.xgb.model is not None:
                            proba = self.ensemble.xgb.model.predict_proba(X_all)
                            # signal = P(up) - P(down), confidence = max proba
                            df_meta["primary_signal"] = proba[:, 2] - proba[:, 0]
                            df_meta["primary_confidence"] = proba.max(axis=1)
                            feats_for_meta = [c for c in all_feature_cols
                                              if c not in ("primary_signal", "primary_confidence")]
                            self.meta_labeler.train(
                                df_meta,
                                primary_signal_col="primary_signal",
                                primary_confidence_col="primary_confidence",
                                outcome_col="tb_label",
                                feature_cols=feats_for_meta,
                            )
                            self.meta_labeler.save()
                            logger.info(
                                f"[Meta] 학습 완료 → precision={self.meta_labeler.precision:.3f} "
                                f"acc={self.meta_labeler.accuracy:.3f}"
                            )
                    except Exception as e:
                        logger.warning(f"[Meta] 학습 실패: {e}")
            except Exception as e:
                logger.debug(f"[Meta] gating 실패: {e}")

        # === HMM Regime Classifier 학습 (tier=large+) ===
        if self.hmm_regime is not None and self.tier_manager is not None:
            try:
                hmm_on = (
                    self.tier_manager.feature_enabled("hmm_regime", mode="paper")
                    or self.tier_manager.feature_enabled("hmm_regime", mode="live")
                )
                if hmm_on and "close" in df.columns and len(df) >= 200:
                    self.hmm_regime.fit(df["close"])
                    self.hmm_regime.save()
                    logger.info("[HMM] 레짐 분류기 학습 완료")
            except Exception as e:
                logger.warning(f"[HMM] 학습 실패: {e}")

        # === ML 모델 먼저 저장 (2026-04-20 수정) ===
        # RL 학습에서 예외 발생해도 XGBoost/LSTM은 이미 잘 학습된 상태이므로 선저장.
        # 기존 로직은 RL 실패 시 전체 train_cycle이 터져서 ML 저장도 누락됨.
        try:
            self.ensemble.save_all()
            logger.info("[모델저장] ML 앙상블(XGB/LSTM) 저장 완료 — RL 학습 시작 전")
        except Exception as e:
            logger.error(f"[모델저장] ML 앙상블 저장 실패: {e}")

        # 3. RL 에이전트 학습 (기존 모델 이어서 / 기본 피처만)
        rl_metrics = {"sharpe_ratio": 0, "win_rate": 0}
        try:
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
            self.rl_agent.save()
        except Exception as e:
            logger.exception(f"[RL학습] 실패 — ML 모델은 이미 저장됨, RL만 건너뜀: {e}")

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
