"""자기학습 트레이너 - 주기적 재학습 및 모델 관리"""

import asyncio
import json
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

try:
    from core.learning.smart_scheduler import SmartTrainingScheduler
    _SCHED_AVAILABLE = True
except Exception as _sched_err:  # pragma: no cover
    SmartTrainingScheduler = None  # type: ignore
    _SCHED_AVAILABLE = False


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

        # === SmartTrainingScheduler (2026-04-25) ===
        # 24h 무조건 재학습 → 조건부 (성능 하락/레짐 전환/최대 간격 초과 시).
        # 기존 should_retrain()는 24h 게이트 그대로 유지(backward compat),
        # 신규 should_retrain_smart()로 보완 게이트 OR 결합.
        ml_cfg = config.get("ml", {})
        scheduler_cfg = ml_cfg.get("smart_scheduler", {})
        self.smart_scheduler = None
        if _SCHED_AVAILABLE and scheduler_cfg.get("enabled", True):
            try:
                self.smart_scheduler = SmartTrainingScheduler(
                    memory_limit_gb=float(scheduler_cfg.get("memory_limit_gb", 6.0)),
                    cpu_cooldown_minutes=float(scheduler_cfg.get("cpu_cooldown_minutes", 5.0)),
                    models=scheduler_cfg.get("models", ["ensemble", "rl_agent"]),
                )
                # 마지막 학습 시각을 동기화 (트레이너 last_train_time → ensemble 스케줄)
                if "ensemble" in self.smart_scheduler.schedules:
                    self.smart_scheduler.schedules["ensemble"].last_train_time = self.last_train_time
                logger.info(
                    f"[SmartSched] 활성 — models={list(self.smart_scheduler.schedules)}"
                )
            except Exception as e:
                logger.warning(f"[SmartSched] 초기화 실패: {e} — 기존 24h 게이트만 사용")
                self.smart_scheduler = None

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
        """기존 24h 게이트 — backward compat 유지"""
        interval = self.config.get("ml", {}).get("retrain_interval_hours", 24)
        return (datetime.utcnow() - self.last_train_time) > timedelta(hours=interval)

    def should_retrain_smart(
        self,
        current_accuracy: float = 0.0,
        regime_changed: bool = False,
    ) -> tuple[bool, str]:
        """SmartScheduler 통합 게이트 — 24h 게이트와 OR 결합.

        반환: (재학습필요?, 사유)
        - 기존 24h 게이트가 True면 즉시 True
        - SmartScheduler가 perf_decline / regime_changed / max_interval 등에서 True면 True
        - 둘 다 False면 False
        """
        # 1) 기존 24h 게이트
        if self.should_retrain():
            return True, "interval(legacy 24h)"

        # 2) SmartScheduler
        if self.smart_scheduler is not None:
            try:
                ok, reason = self.smart_scheduler.should_retrain_any(
                    accuracies={"ensemble": float(current_accuracy)},
                    regime_changed=bool(regime_changed),
                )
                if ok:
                    return True, f"smart:{reason}"
            except Exception as e:
                logger.debug(f"[SmartSched] 판단 실패: {e}")
        return False, "no_trigger"

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

        # === LLM Signal 스냅샷 조인 (2026-04-24, Claude-Native) ===
        # ExternalDataManager가 매 update마다 누적한 LLM 분석 결과를 학습 DF에 시간 조인.
        # 충분한 누적(최소 50행) 있을 때만 활성 — 모델이 실 신호로 인식하게.
        try:
            if hasattr(self.storage, "load_llm_snapshots"):
                llm_df = self.storage.load_llm_snapshots(symbol, days=lookback)
                if llm_df is not None and not llm_df.empty and len(llm_df) >= 50:
                    # prefix로 피처 컬럼 네이밍 통일 — XGBoost가 자동 인식
                    llm_df = llm_df.add_prefix("ext_llm_")
                    # 누락 시각은 ffill (last-known LLM 분석이 다음 캔들까지 유효)
                    llm_aligned = llm_df.reindex(df.index, method="ffill")
                    overlap_llm = [c for c in llm_aligned.columns if c in df.columns]
                    new_llm = [c for c in llm_aligned.columns if c not in df.columns]
                    if overlap_llm:
                        df[overlap_llm] = llm_aligned[overlap_llm]
                    if new_llm:
                        df = pd.concat([df, llm_aligned[new_llm]], axis=1)
                    logger.info(
                        f"[LLM] Claude 스냅샷 {len(llm_df)}개 조인 완료 — "
                        f"overwrite={len(overlap_llm)}, new={len(new_llm)} "
                        f"→ XGBoost가 LLM reasoning을 피처로 학습"
                    )
                else:
                    snap_count = len(llm_df) if llm_df is not None else 0
                    logger.debug(
                        f"[LLM] {symbol} 스냅샷 누적 부족({snap_count}개 < 50) — "
                        f"상수값 유지. LIVE 돌면서 자동 누적 중."
                    )
        except Exception as e:
            logger.warning(f"[LLM] 스냅샷 조인 실패: {e} — 기존 상수값 유지")

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

        # === [Patch L, 2026-04-28] 진정한 심화학습 (Continual Learning) ===
        # 사용자 직관 적용: "재학습 아니라 심화학습. 데이터를 계속 쌓아가면서"
        #
        # 변경 전: tier=large+ 면 walk_forward=True → 매 사이클 from-scratch 5-fold
        #   → 691k 캔들을 매번 처음부터 학습 → 시간 지나도 누적 효과 0
        # 변경 후: 매 7번째 사이클만 walk_forward CV (평가 목적), 나머지 6번은 incremental
        #   → fine-tune (XGB 트리 +100 / LSTM epochs 추가 / LGB init_model)
        #   → 시간 갈수록 모델이 더 똑똑해짐 (진정한 누적 학습)
        #
        # 효과:
        #   - 학습 시간 12h → ~2h (incremental fast-path) → 더 자주 학습 가능
        #   - 시장 변화 적응 빨라짐
        #   - Walk-Forward OOS acc 평가는 매 7번째 사이클에 보존 (모델 품질 보장)
        cycle_state_path = Path("data/learning_cycle_state.json")
        cycle_count = 0
        try:
            if cycle_state_path.exists():
                cycle_count = int(json.loads(cycle_state_path.read_text()).get("cycle", 0))
        except Exception:
            cycle_count = 0

        # 매 7번째만 평가 (eval cycle)
        is_eval_cycle = (cycle_count > 0 and cycle_count % 7 == 0)
        use_walk_forward = is_eval_cycle  # 평가 사이클만 from-scratch CV
        use_purged = is_eval_cycle

        if is_eval_cycle:
            logger.info(
                f"[Patch L] 사이클 #{cycle_count} = 평가 사이클 (매 7회 1번) → "
                f"Walk-Forward CV from-scratch (모델 품질 OOS 검증)"
            )
        else:
            logger.info(
                f"[Patch L] 사이클 #{cycle_count} = 심화학습 사이클 → "
                f"incremental fine-tune (XGB 트리+100 / LSTM 추가 epochs / LGB init_model)"
            )

        # 사이클 카운터 갱신 + 디스크 persist
        try:
            cycle_state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cycle_state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "cycle": cycle_count + 1,
                "last_mode": "eval" if is_eval_cycle else "incremental",
                "updated_at": datetime.utcnow().isoformat(),
            }, indent=2))
            tmp.rename(cycle_state_path)
        except Exception as e:
            logger.debug(f"[Patch L] cycle state 저장 실패: {e}")

        self.ensemble.train_all(
            df, all_feature_cols,
            walk_forward=use_walk_forward,
            use_purged_kfold=use_purged,
        )

        # === Meta-Labeler 학습 (Patch G, 2026-04-26: 티어 무관 강제 활성) ===
        # 기존: tier=large+에서만 — 자본 적은 시점엔 메타라벨 학습 자체가 안 돼
        # 누적 시그널이 부족했음. 이제 meta_labeler 객체가 존재하면 무조건 학습.
        # (precision/recall 가드는 inference 시점에 self.meta_labeler.precision으로 자동 처리)
        if self.meta_labeler is not None:
            try:
                meta_on = True  # 강제 ON
                if meta_on and "tb_label" in df.columns:
                    # 1차 시그널 주입: XGB 예측값을 DF에 컬럼으로 추가
                    df_meta = df.copy()
                    try:
                        # [Patch H, 2026-04-27] XGB가 롤백되었거나 다른 차원으로 학습됐을 수 있으므로
                        # XGB 모델이 실제 학습된 feature_columns를 우선 사용 (shape mismatch 방지)
                        xgb_feats = getattr(self.ensemble.xgb, "feature_columns", None) or all_feature_cols
                        # df_meta에 없는 컬럼은 제외하여 안전하게 정합화
                        xgb_feats = [c for c in xgb_feats if c in df_meta.columns]
                        if not xgb_feats:
                            raise RuntimeError("xgb feature_columns 비어있음")
                        X_all = df_meta[xgb_feats].values
                        if self.ensemble.xgb.model is not None:
                            # 추가 안전장치: 모델이 기대하는 차원과 다르면 우회
                            expected = getattr(self.ensemble.xgb.model, "n_features_in_", X_all.shape[1])
                            if expected != X_all.shape[1]:
                                raise RuntimeError(
                                    f"XGB 차원 불일치 expected={expected} got={X_all.shape[1]} → 메타 학습 스킵"
                                )
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

        # === SmartScheduler 동기화 (2026-04-25) ===
        # 학습 완료 시 ensemble 정확도를 스케줄러에 기록 → 다음 사이클 perf_decline 판단.
        if self.smart_scheduler is not None:
            try:
                ens_acc = float(self.ensemble.xgb.accuracy)
                self.smart_scheduler.mark_training_complete("ensemble", ens_acc)
                if "rl_agent" in self.smart_scheduler.schedules:
                    self.smart_scheduler.mark_training_complete(
                        "rl_agent", float(rl_metrics.get("sharpe_ratio", 0.0))
                    )
            except Exception as e:
                logger.debug(f"[SmartSched] 완료 기록 실패: {e}")

        # === [Patch M, 2026-04-28] Pattern Memory Bank 인덱스 갱신 ===
        # 학습 사이클이 fetch한 모든 데이터를 그대로 retrieval 인덱스로 빌드.
        # raw 데이터 직접 활용 — ML 모델 압축 손실 우회 (사용자 통찰 적용).
        try:
            from core.patterns.memory_bank import PatternMemoryBank
            bank = PatternMemoryBank()
            bank.build_from_dataframe(df, symbol=symbol)
            sym_safe = symbol.replace("/", "_").replace(":", "_")
            bank_path = Path(f"data/pattern_bank/{sym_safe}_{timeframe}.npz")
            bank.save(bank_path)
        except Exception as e:
            logger.warning(f"[PatternBank] {symbol} 인덱스 빌드 실패 (학습은 정상): {e}")

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
