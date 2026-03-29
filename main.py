"""AutoTrader AI - 자기학습 선물 트레이딩 시스템 메인 실행"""

import asyncio
import os
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import uvicorn
import yaml
from dotenv import load_dotenv
from loguru import logger

from backtest.engine import BacktestEngine
from core.data.collector import DataCollector
from core.data.features import FeatureEngineer
from core.data.storage import Storage
from core.execution.exchange import ExchangeClient
from core.execution.order_manager import OrderManager
from core.execution.paper_trader import PaperTrader
from core.external.external_manager import ExternalDataManager
from core.learning.trainer import SelfLearningTrainer
from core.models.ensemble import EnsembleSignalGenerator
from core.rl.agent import RLAgent
from core.risk.manager import RiskManager
from core.learning.feedback import AnomalyDetector, TradeFeedbackAnalyzer
from core.strategy.adaptive import AdaptiveOptimizer, StrategyOptimizer
from core.strategy.manager import StrategyManager
from core.external.listing_detector import ListingDetector
from dashboard.app import app as dashboard_app, set_state, add_live_log

# Telegram 알림 (실패해도 트레이딩에 영향 없음)
try:
    from scripts.telegram_bot import send_message as tg_send, format_trade_open, format_trade_close, format_system_alert
    _tg_ok = True
except Exception:
    _tg_ok = False
    # fallback 함수 정의 (telegram_bot.py 없을 때)
    def format_trade_open(mode, symbol, action, price, notional, lev, reason):
        return f"{mode} {action.upper()} {symbol} @ {price} ({notional:.0f}$ x{lev}) | {reason}"
    def format_trade_close(mode, symbol, pnl, reason, duration=0):
        return f"{mode} 청산 {symbol} PnL: ${pnl:+.2f} | {reason}"
    def format_system_alert(msg):
        return f"⚠️ {msg}"

def tg_notify(text, silent=False):
    """비동기 안전 텔레그램 알림 (실패 무시)"""
    if not _tg_ok:
        return
    try:
        threading.Thread(target=tg_send, args=(text, "HTML", silent), daemon=True).start()
    except:
        pass


class AutoTrader:
    """자기학습 선물 트레이딩 시스템 메인 클래스"""

    def __init__(self, config_path: str = "config/default.yaml"):
        load_dotenv()
        self.config = self._load_config(config_path)
        self.mode = self.config["trading"]["mode"]  # paper / live / dual
        self.dual_mode = (self.mode == "dual")
        self.start_time = datetime.utcnow()

        # 로깅 설정
        log_cfg = self.config.get("logging", {})
        log_file = log_cfg.get("file", "logs/autotrader.log")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, rotation="10 MB", level=log_cfg.get("level", "INFO"))

        # 컴포넌트 초기화 (한 번만)
        self.storage = Storage()
        self.feature_engineer = FeatureEngineer(self.config.get("ml", {}).get("features"))
        self.ensemble = EnsembleSignalGenerator()
        self.rl_agent = RLAgent(self.config.get("rl", {}))
        self.risk_manager = RiskManager(self.config["risk"])
        self.strategy_manager = StrategyManager(self.config.get("trading", {}))
        self.adaptive = AdaptiveOptimizer()
        self.feedback = TradeFeedbackAnalyzer()
        self.anomaly_detector = AnomalyDetector()

        # 외부 데이터 매니저 (뉴스/센티먼트/온체인/매크로/공포탐욕)
        self.external_manager = ExternalDataManager(self.config.get("external", {}))

        # Paper 트레이더 (paper / dual 모드에서 사용)
        self.paper_trader = PaperTrader(
            initial_capital=self.config.get("backtest", {}).get("initial_capital", 10000),
            commission=self.config.get("backtest", {}).get("commission_pct", 0.0004),
            trailing_config=self.config.get("trailing_stop", {}),
        )
        self.exchange_clients: dict[str, ExchangeClient] = {}
        self.order_managers: dict[str, OrderManager] = {}

        # StrategyOptimizer — Paper/Live 각각 독립 추적
        self.strategy_optimizer_paper = StrategyOptimizer()
        self.strategy_optimizer_live = StrategyOptimizer()

        # 최소 주문 notional (거래소 최소수량 충족용)
        self.min_order_notional = self.config["risk"].get("min_order_notional", 100)

        # 상장 시그널 감지기
        self.listing_detector = ListingDetector()

        # 상태
        self.equity = self.config.get("backtest", {}).get("initial_capital", 10000)
        self.initial_capital = self.equity
        self.total_pnl = 0.0
        self.last_signals = {}
        self.last_external = {}
        self.is_running = False

        # 자가진단 LIVE 일시정지 상태
        self._live_paused = False
        self._live_pause_reason = ""
        self._live_pause_time = None

    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            config = yaml.safe_load(f)
        # 환경변수 치환
        for ex_name, ex_cfg in config.get("exchanges", {}).items():
            for key in ["api_key", "secret"]:
                val = ex_cfg.get(key, "")
                if val.startswith("${") and val.endswith("}"):
                    env_var = val[2:-1]
                    ex_cfg[key] = os.environ.get(env_var, "")
        return config

    async def initialize(self):
        """거래소 연결 및 모델 로드"""
        logger.info(f"AutoTrader 초기화 - 모드: {self.mode}")

        # 데이터 수집기
        self.collector = DataCollector(self.config["exchanges"])
        await self.collector.initialize()

        # 실거래 모드 시 거래소 클라이언트 초기화 (live 또는 dual)
        if self.mode in ("live", "dual"):
            for name, cfg in self.config["exchanges"].items():
                client = ExchangeClient(name, cfg)
                self.exchange_clients[name] = client
                self.order_managers[name] = OrderManager(
                    client, self.config["risk"],
                    trailing_config=self.config.get("trailing_stop", {}),
                )

        # 기존 모델 로드 시도
        if self.ensemble.load_all():
            logger.info("기존 ML 모델 로드 성공")
        if self.rl_agent.load():
            logger.info("기존 RL 모델 로드 성공")

        # 외부 데이터 초기 수집
        if self.external_manager.enabled:
            logger.info("외부 데이터 초기 수집 시작...")
            await self.external_manager.update()
            logger.info("외부 데이터 초기 수집 완료")

        # NLTK VADER 데이터 다운로드 (첫 실행 시)
        try:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
        except Exception:
            pass

        self.risk_manager.initialize(self.equity)

        # 상장 감지기에 거래소 클라이언트 연결
        if self.exchange_clients:
            first_client = list(self.exchange_clients.values())[0]
            self.listing_detector.exchange = first_client

        # SL/TP 콜백 등록 (체결 시 피드백 학습 + 텔레그램 알림)
        def _make_trade_callback(close_type: str):
            """SL/TP 공통 콜백 생성"""
            def on_triggered(result: dict):
                symbol = result.get("symbol", "?")
                pnl = result.get("pnl", 0)
                side = result.get("side", "?")

                # 현재 시장 레짐 가져오기
                regime = self.adaptive.current_regime if hasattr(self, 'adaptive') else "unknown"

                # 피드백 학습에 기록 (v4: exit_reason 추가)
                self.feedback.record_trade(
                    {"pnl": pnl, "side": side, "symbol": symbol},
                    {"regime": regime, "signal": 0, "confidence": 0,
                     "external_score": 0, "external_direction": "neutral",
                     "exit_reason": close_type.lower(),
                     "confirming_sources": [],
                     "entry_path": "callback"},
                )
                # 손실/수익 학습에 전달 (StrategyManager 엄격화)
                if pnl < 0:
                    self.strategy_manager.record_loss()
                else:
                    self.strategy_manager.record_win()
                self.risk_manager.record_pnl(pnl)
                self.storage.save_trade({
                    "exchange": "binance", "symbol": symbol, "side": "close",
                    "price": result.get("exit_price", 0),
                    "amount": result.get("size", 0),
                    "pnl": pnl, "fee": 0,
                    "strategy": f"{close_type.lower()}_triggered",
                    "mode": "LIVE",
                })

                # StrategyOptimizer 기록
                l_hash = self.strategy_optimizer_live._config_to_hash(
                    self.strategy_optimizer_live.current_config
                )
                self.strategy_optimizer_live.record_trade(l_hash, {
                    "pnl": pnl, "timestamp": datetime.utcnow(),
                    "symbol": symbol, "hour": datetime.utcnow().hour,
                })

                emoji = "🛑" if close_type == "SL" else "🎯"
                logger.info(f"[{close_type}학습] {symbol} {side} {close_type} 체결 → PnL: ${pnl:.2f} → 학습기록 완료 (regime={regime})")
                tg_notify(
                    f"{emoji} <b>{close_type} 체결</b>\n"
                    f"━━━━━━━━━━━━━\n"
                    f"종목: {symbol}\n"
                    f"방향: {side}\n"
                    f"PnL: ${pnl:+.2f}\n"
                    f"레짐: {regime}\n"
                    f"📝 피드백 학습에 기록됨"
                )
            return on_triggered

        for om in self.order_managers.values():
            om.set_sl_callback(_make_trade_callback("SL"))
            om.set_tp_callback(_make_trade_callback("TP"))

        # 시스템 재시작 시 기존 포지션 복구 + SL 재설정
        trading_symbols = self.config.get("trading", {}).get("symbols", [])
        for om in self.order_managers.values():
            recovered = await om.recover_positions(trading_symbols)
            if recovered:
                for pos in recovered:
                    logger.info(f"[복구완료] {pos.symbol} {pos.side} {pos.size}개 @ {pos.entry_price:.4f}")
                    tg_notify(
                        f"🔄 <b>포지션 복구</b>\n"
                        f"━━━━━━━━━━━━━\n"
                        f"종목: {pos.symbol}\n"
                        f"방향: {pos.side}\n"
                        f"수량: {pos.size}\n"
                        f"진입가: ${pos.entry_price:.4f}\n"
                        f"SL: ${pos.stop_loss:.4f}\n"
                        f"TP: ${pos.take_profit:.4f}"
                    )

        logger.info("AutoTrader 초기화 완료")

    async def run_backtest(self):
        """백테스트 모드 실행"""
        logger.info("=== 백테스트 모드 시작 ===")
        bt_config = self.config["backtest"]
        engine = BacktestEngine(bt_config)

        exchange_name = list(self.config["exchanges"].keys())[0]

        for symbol in self.config["trading"]["symbols"]:
            logger.info(f"백테스트: {symbol}")

            # 데이터 수집
            df = await self.collector.fetch_all_ohlcv(
                exchange_name, symbol, "1h",
                days=self.config.get("ml", {}).get("lookback_days", 90),
            )
            df = self.feature_engineer.generate(df)
            feature_cols = self.feature_engineer.get_feature_columns(df)

            # 모델 학습
            self.ensemble.train_all(df, feature_cols)

            # 시그널 생성
            signals = []
            for i in range(len(df)):
                chunk = df.iloc[:i+1]
                if len(chunk) < 60:
                    signals.append({"action": "hold", "size": 0})
                    continue
                pred = self.ensemble.predict(chunk)
                action = pred["direction"] if pred["direction"] != "neutral" else "hold"
                if action == "long":
                    action = "long"
                elif action == "short":
                    action = "short"
                signals.append({"action": action, "size": pred["confidence"]})

            result = engine.run(df, signals)
            logger.info(f"{symbol} 백테스트 완료")

    async def run_trading_loop(self):
        """실시간 트레이딩 루프 (페이퍼/실거래)"""
        logger.info(f"=== {self.mode.upper()} 트레이딩 시작 ===")
        self.is_running = True

        exchange_name = list(self.config["exchanges"].keys())[0]
        symbols = self.config["trading"]["symbols"]
        timeframes = self.config["trading"]["timeframes"]  # 멀티타임프레임
        primary_tf = timeframes[0]  # 메인 타임프레임 (5m)

        # 자기학습 트레이너
        trainer = SelfLearningTrainer(
            self.collector, self.storage, self.ensemble, self.rl_agent, self.config,
        )

        # 모델이 없으면 초기 학습
        if not self.ensemble.load_all():
            logger.info("모델 없음 - 초기 학습 시작")
            for symbol in symbols:
                await trainer.train_cycle(exchange_name, symbol, primary_tf)

        loop_count = 0

        while self.is_running:
            try:
                # 재학습 체크 (일반 + stuck 감지)
                diag = self.strategy_manager.get_diagnostics()
                needs_retrain = trainer.should_retrain()

                # 자기진단: 200회 연속 hold (~100분) → 강제 재학습
                if diag["is_stuck"] and diag["consecutive_holds"] % 200 == 0:
                    logger.warning(
                        f"[자기진단] {diag['consecutive_holds']}회 연속 HOLD → 강제 재학습 트리거 "
                        f"(min_conf: {diag['current_min_confidence']:.3f})"
                    )
                    needs_retrain = True
                    tg_notify(
                        f"⚠️ <b>자기진단 알림</b>\n"
                        f"━━━━━━━━━━━━━\n"
                        f"🔄 {diag['consecutive_holds']}회 연속 HOLD 감지\n"
                        f"📊 min_conf: {diag['current_min_confidence']:.3f}\n"
                        f"🤖 강제 재학습 시작",
                        silent=True,
                    )

                if needs_retrain:
                    add_live_log({
                        "time": datetime.utcnow().strftime("%H:%M:%S"),
                        "type": "retrain",
                        "message": f"자기학습 재훈련 시작 (holds: {diag['consecutive_holds']})",
                    })
                    for symbol in symbols:
                        await trainer.train_cycle(exchange_name, symbol, primary_tf)
                    add_live_log({
                        "time": datetime.utcnow().strftime("%H:%M:%S"),
                        "type": "retrain",
                        "message": "자기학습 재훈련 완료",
                    })

                # 외부 데이터 업데이트 (뉴스/센티먼트/파생상품/계절 등)
                if self.external_manager.enabled:
                    for symbol in symbols:
                        await self.external_manager.update(symbol)
                        self.last_external = self.external_manager.get_report()

                        # 외부 피처를 FeatureEngineer에 주입
                        ext_features = self.external_manager.get_all_features()
                        self.feature_engineer.set_external_features(ext_features)

                # 멀티타임프레임 분석 (각 타임프레임 데이터 수집 & 분석)
                for symbol in symbols:
                    for tf in timeframes:
                        try:
                            tf_df = await self.collector.fetch_ohlcv(exchange_name, symbol, tf, limit=200)
                            if len(tf_df) > 50:
                                self.external_manager.update_multi_timeframe(tf_df, tf)
                        except Exception as e:
                            logger.debug(f"MTF {tf} 수집 실패: {e}")

                    # 멀티타임프레임 합류 계산
                    mtf_result = self.external_manager.get_multi_tf_confluence()

                # === 집중 매매 모드: 모든 심볼 분석 후 최강 시그널 1개만 LIVE 진입 ===
                concentration = self.config["trading"].get("concentration_mode", False)
                max_live = self.config["trading"].get("max_concurrent_live", 1)
                scalp_profiles = self.config["risk"].get("scalp_profiles", {})

                if concentration:
                    # 모든 심볼의 시그널 수집
                    candidates = []
                    for symbol in symbols:
                        result = await self._analyze_symbol(exchange_name, symbol, primary_tf)
                        if result and result.get("action") in ("long", "short"):
                            # 종목별 프로파일로 TP/SL 동적 설정
                            coin = symbol.split("/")[0]
                            profile = scalp_profiles.get(coin, {})
                            result["tp_pct"] = profile.get("tp_pct", self.config["risk"]["take_profit_pct"])
                            result["sl_pct"] = profile.get("sl_pct", self.config["risk"]["stop_loss_pct"])
                            result["priority"] = profile.get("priority", 5)
                            candidates.append(result)

                    # PAPER: 모든 시그널 실행 (학습용)
                    for c in candidates:
                        await self._execute_paper(c)

                    # LIVE: 기존 포지션 트레일링/SL/TP 관리 (매 루프)
                    live_positions = sum(
                        len(om.positions) for om in self.order_managers.values()
                    )
                    if live_positions > 0:
                        for om in self.order_managers.values():
                            await om.update_positions()

                    # LIVE: 포지션 여유 있을 때만 신규 진입
                    if live_positions < max_live and candidates:
                        # 확신도 × (1 - priority/10) 로 최종 순위 결정
                        best = max(
                            candidates,
                            key=lambda c: c["confidence"] * (1 - c["priority"] / 10),
                        )
                        await self._execute_live(exchange_name, best)
                else:
                    # 기존 모드: 각 심볼 독립 매매
                    for symbol in symbols:
                        await self._process_symbol(exchange_name, symbol, primary_tf)

                # === 상장 시그널 스캔 (15분마다) ===
                if self.listing_detector.exchange and loop_count % 30 == 0:
                    try:
                        listing_signals = await self.listing_detector.scan_signals()
                        if listing_signals:
                            tradeable = self.listing_detector.get_tradeable_signals(min_score=0.65)
                            for sig in tradeable[:2]:  # 상위 2개만 거래 시도
                                sym = sig["symbol"]
                                if sym not in symbols:
                                    # 동적으로 심볼 추가하여 처리
                                    logger.info(
                                        f"[ListingDetector] 🎯 {sig['coin']} ({sig['tier']}) "
                                        f"score={sig['confidence']:.2f} → {sig['action']} "
                                        f"사유: {', '.join(sig['reasons'])}"
                                    )
                                    add_live_log({
                                        "time": datetime.utcnow().strftime("%H:%M:%S"),
                                        "type": "listing_signal",
                                        "message": f"🎯 상장시그널: {sig['coin']} ({sig['tier']}) "
                                                   f"score={sig['confidence']:.2f} {', '.join(sig['reasons'])}",
                                    })
                                    # 시그널이 강하면 텔레그램 알림
                                    if sig["confidence"] >= 0.75:
                                        tg_notify(
                                            f"🎯 <b>상장 시그널 감지</b>\n"
                                            f"━━━━━━━━━━━━━\n"
                                            f"코인: <b>{sig['coin']}</b> ({sig['tier']})\n"
                                            f"점수: {sig['confidence']:.2f}\n"
                                            f"방향: {sig['action']}\n"
                                            f"사유: {', '.join(sig['reasons'])}"
                                        )
                            # 대시보드에 리포트 반영
                            self.last_external["listing"] = self.listing_detector.get_report()
                    except Exception as e:
                        logger.debug(f"[ListingDetector] 스캔 실패: {e}")

                loop_count += 1
                if loop_count % 10 == 0:
                    ext_report = self.external_manager.get_report()
                    cs = ext_report.get("composite_signal", {})
                    deriv = ext_report.get("derivatives", {}).get("composite", {})
                    seasonal = ext_report.get("seasonal", {})
                    mtf = ext_report.get("multi_timeframe", {}).get("confluence", {})
                    logger.info(
                        f"[Loop {loop_count}] 종합: {cs.get('score', 0):.2f}({cs.get('direction', '?')}) | "
                        f"파생: {deriv.get('score', 0):.2f} | "
                        f"계절: {seasonal.get('direction', '?')}({seasonal.get('score', 0):.2f}) "
                        f"[{seasonal.get('halving_phase', '?')}] | "
                        f"MTF합류: {mtf.get('score', 0):.2f}({mtf.get('agreement', 0):.0%})"
                    )

                # 자동 전략 최적화 (각 모드별 독립)
                for label, optimizer in [("PAPER", self.strategy_optimizer_paper),
                                         ("LIVE", self.strategy_optimizer_live)]:
                    total = sum(len(v) for v in optimizer.performance_history.values())
                    last_key = f"_last_opt_{label}"
                    if total >= 30 and (
                        loop_count % 2880 == 0 or
                        total - getattr(self, last_key, 0) >= 30
                    ):
                        all_trades = []
                        for trades_list in optimizer.performance_history.values():
                            all_trades.extend(trades_list)
                        if len(all_trades) >= 30:
                            optimizer.optimize_daily(all_trades)
                            setattr(self, last_key, total)
                            report = optimizer.get_report()
                            logger.info(
                                f"[{label} Optimizer] 최적화 완료 | "
                                f"승률: {report['current_win_rate']:.1%} | "
                                f"거래: {report['total_trades']}"
                            )

                # === 30분마다 긴급 자가진단 (60 루프 × 30초 ≈ 30분) ===
                if loop_count % 60 == 0 and loop_count > 0:
                    await self._critical_health_check()

                # === 2시간마다 종합 자가진단 (240 루프 × 30초 ≈ 2시간) ===
                if loop_count % 240 == 0 and loop_count > 0:
                    await self._learning_health_check()

                # === 4시간마다 전략 자체 리뷰 (480 루프 × 30초 ≈ 4시간) ===
                if loop_count % 480 == 0 and loop_count > 0:
                    await self._strategic_self_review()

                # 대기
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"트레이딩 루프 에러: {e}")
                await asyncio.sleep(60)

    async def _process_symbol(self, exchange_name: str, symbol: str, timeframe: str):
        """개별 심볼 처리"""
        # 1. 최신 데이터 수집
        df = await self.collector.fetch_ohlcv(exchange_name, symbol, timeframe, limit=200)
        df = self.feature_engineer.generate(df)
        feature_cols = self.feature_engineer.get_feature_columns(df)

        if len(df) < 60:
            return

        # 2. 시장 레짐 감지
        prices = df["close"].values
        volumes = df["volume"].values
        adaptive_params = self.adaptive.update(prices, volumes)

        # 3. ML 시그널
        ml_signal = self.ensemble.predict(df)

        # 4. RL 행동 결정 (기본 피처만 사용 - 차원 고정 보장)
        base_feature_cols = self.feature_engineer.get_base_feature_columns(df)
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        rl_obs_data = df[base_feature_cols].values[-1].astype(np.float32)
        rl_obs_data = np.nan_to_num(rl_obs_data, nan=0.0)

        position_info = np.array([0.0, 0.0, self.equity / self.initial_capital, 0.0], dtype=np.float32)
        obs = np.concatenate([rl_obs_data, position_info])

        rl_action, rl_confidence = self.rl_agent.predict(obs)

        # 4.5. 외부 신호 가져오기
        ext_signal = self.external_manager.get_signal_for_strategy()

        # 4.6. 멀티타임프레임 합류 시그널
        mtf_signal = self.external_manager.multi_tf.get_signal_for_strategy()

        # 4.7. 모멘텀 시그널 계산 (ML/RL 무반응 시 fallback)
        momentum_signal = self._calculate_momentum(df)

        # 5. 전략 결정 (ML + RL + 외부 요인 + MTF + 모멘텀 + 레짐 바이어스 통합)
        ext_features_all = self.external_manager.get_all_features()
        funding_rate = ext_features_all.get("deriv_funding_rate", 0)
        fear_greed = ext_features_all.get("fg_value", 50)

        current_position = 0.0
        if self.mode in ("paper", "dual"):
            current_position = 1.0 if symbol in self.paper_trader.positions else 0.0
        decision = self.strategy_manager.decide(
            ml_signal, rl_action, rl_confidence, current_position,
            adaptive_params["regime"], external_signal=ext_signal,
            momentum=momentum_signal,
            funding_rate=funding_rate,
            fear_greed_index=fear_greed,
        )

        # 5.1. MTF 합류 필터 - 상위 타임프레임과 반대면 진입 차단
        if decision.action in ["long", "short"]:
            mtf_agreement = mtf_signal.get("agreement", 0)
            mtf_dir = mtf_signal.get("direction", "neutral")
            action_opposes_mtf = (
                (decision.action == "long" and mtf_dir == "bearish") or
                (decision.action == "short" and mtf_dir == "bullish")
            )
            action_agrees_mtf = (
                (decision.action == "long" and mtf_dir == "bullish") or
                (decision.action == "short" and mtf_dir == "bearish")
            )

            # 강한 MTF 반대 (합의 75%+) → 진입 차단
            if action_opposes_mtf and mtf_agreement >= 0.75:
                logger.info(
                    f"[MTF] {symbol} {decision.action} 차단 — "
                    f"MTF {mtf_dir} 합의 {mtf_agreement:.0%} (conf={decision.confidence:.2f})"
                )
                decision.action = "hold"
                decision.confidence = 0.0
                decision.size = 0.0
                decision.reason = f"MTF 강반대 차단 ({mtf_dir} {mtf_agreement:.0%})"

            # 약한 MTF 반대 (50~75%) → 확신도 40% 감소
            elif action_opposes_mtf and mtf_agreement >= 0.5:
                decision.confidence *= 0.6
                decision.reason += f" ! MTF반대({mtf_dir} {mtf_agreement:.0%})"
                if decision.confidence < self.strategy_manager.min_confidence:
                    decision.action = "hold"
                    decision.size = 0.0
                    decision.reason += " → 확신도부족"

            # MTF 합류 확인 시 확신도 부스트
            elif action_agrees_mtf and mtf_agreement > 0.7:
                decision.confidence = min(decision.confidence * 1.15, 1.0)
                decision.reason += " + MTF합류"

            # 상위 TF 간 충돌 (4h vs 1h)
            elif mtf_signal.get("higher_tf_conflict"):
                decision.confidence *= 0.85
                decision.reason += " ! TF충돌"

        self.last_signals = {
            "symbol": symbol,
            "direction": decision.action,
            "confidence": decision.confidence,
            "signal": ml_signal.get("signal", 0),
            "regime": adaptive_params["regime"],
            "reason": decision.reason,
            "external": {
                "score": ext_signal.get("score", 0),
                "direction": ext_signal.get("direction", "neutral"),
                "strength": ext_signal.get("strength", "weak"),
            },
            "mtf": {
                "score": mtf_signal.get("score", 0),
                "agreement": mtf_signal.get("agreement", 0),
                "direction": mtf_signal.get("direction", "neutral"),
            },
            "momentum": momentum_signal,
            "diagnostics": self.strategy_manager.get_diagnostics(),
        }

        # 가격 히스토리 업데이트 (상관관계 계산용)
        price = float(df["close"].iloc[-1])
        self.risk_manager.update_price_history(symbol, price)

        # 실시간 로그 기록
        add_live_log({
            "time": datetime.utcnow().strftime("%H:%M:%S"),
            "type": "analysis",
            "symbol": symbol,
            "price": price,
            "action": decision.action,
            "confidence": round(decision.confidence, 3),
            "ml_dir": ml_signal.get("direction", "?"),
            "ml_conf": round(ml_signal.get("confidence", 0), 3),
            "rl_action": ["hold", "long", "short", "close"][rl_action],
            "rl_conf": round(rl_confidence, 3),
            "ext_score": round(ext_signal.get("score", 0), 3),
            "ext_dir": ext_signal.get("direction", "neutral"),
            "mtf_score": round(mtf_signal.get("score", 0), 3),
            "mtf_agree": round(mtf_signal.get("agreement", 0), 2),
            "mom_dir": momentum_signal.get("direction", "?"),
            "mom_str": round(momentum_signal.get("strength", 0), 3),
            "mom_rsi": round(momentum_signal.get("rsi", 50), 1),
            "regime": adaptive_params["regime"],
            "reason": decision.reason,
            "features_count": len(feature_cols),
            "min_conf": round(self.strategy_manager.min_confidence, 3),
            "holds": self.strategy_manager._consecutive_holds,
        })

        # 5.5. 이상 시장 감지
        price = float(df["close"].iloc[-1])
        volume = float(df["volume"].iloc[-1])
        alerts = self.anomaly_detector.update(price, volume)
        if alerts:
            self.last_signals["anomalies"] = [a["message"] for a in alerts]
            high_alerts = [a for a in alerts if a["severity"] == "high"]
            if high_alerts and decision.action in ["long", "short"]:
                logger.warning(f"[Anomaly] 이상 감지로 진입 보류: {high_alerts[0]['message']}")
                return

        # 6. 리스크 체크
        num_positions = len(self.paper_trader.positions) if self.mode in ("paper", "dual") else 0
        can_trade, risk_msg = self.risk_manager.check_can_trade(self.equity, num_positions)

        if decision.action in ["long", "short"] and not can_trade:
            logger.info(f"{symbol} 거래 차단: {risk_msg}")
            return

        # 6.5. 피드백 필터 (과거 거래 결과 기반)
        if decision.action in ["long", "short"]:
            from datetime import datetime as dt
            fb_ok, fb_reason = self.feedback.should_trade_now(
                hour=dt.utcnow().hour,
                regime=adaptive_params["regime"],
                side=decision.action,
                signal_strength=ml_signal.get("signal", 0),
            )
            if not fb_ok:
                logger.info(f"[Feedback] {symbol} 거래 차단: {fb_reason}")
                return

        # 7. 주문 실행
        if decision.action in ["long", "short"]:
            volatility = df["returns_1"].std() if "returns_1" in df.columns else 0.01

            # 7.1. 포지션 상관관계 체크
            current_positions = {
                s: {"side": p.side} for s, p in self.paper_trader.positions.items()
            }
            corr_ok, corr_reason, corr_mult = self.risk_manager.check_correlation(
                symbol, decision.action, current_positions,
            )

            # 7.2. 동적 레버리지 계산
            ext_agrees = (
                (decision.action == "long" and ext_signal.get("direction") == "bullish") or
                (decision.action == "short" and ext_signal.get("direction") == "bearish")
            )
            dynamic_lev = self.risk_manager.calculate_dynamic_leverage(
                confidence=decision.confidence,
                volatility=volatility,
                regime=adaptive_params["regime"],
                external_agreement=ext_agrees,
            )

            # 7.3. 피드백 기반 포지션 크기 조정
            fb_scale = self.feedback.get_position_scale(adaptive_params["regime"], decision.action)
            size = self.risk_manager.calculate_position_size(
                self.equity, decision.confidence, volatility,
                adaptive_params["position_scale"] * fb_scale * corr_mult,
            )

            # 7.4. 최소 주문 notional 보장 (Binance 최소 $100 + 여유분)
            notional = size * dynamic_lev
            min_notional_with_margin = self.min_order_notional * 1.05  # 5% 여유
            if notional < min_notional_with_margin:
                size = max(size, min_notional_with_margin / dynamic_lev)
                # 자본의 50%를 넘지 않도록 캡
                size = min(size, self.equity * 0.50)
                notional = size * dynamic_lev

            price = float(df["close"].iloc[-1])

            # === PAPER 실행 (paper / dual 모드) ===
            if self.mode in ("paper", "dual"):
                if symbol not in self.paper_trader.positions:
                    self.paper_trader.open_position(
                        symbol, decision.action, size, price,
                        leverage=dynamic_lev,
                        sl_pct=self.config["risk"]["stop_loss_pct"] * adaptive_params["stop_loss_mult"],
                        tp_pct=self.config["risk"]["take_profit_pct"],
                        atr_pct=atr_pct,
                    )
                    logger.info(
                        f"[PAPER] {decision.action.upper()} {symbol} | "
                        f"마진: ${size:.2f} × {dynamic_lev}x = ${notional:.2f} | "
                        f"사유: {decision.reason}"
                    )
                    add_live_log({
                        "time": datetime.utcnow().strftime("%H:%M:%S"),
                        "type": "trade_open",
                        "mode": "PAPER",
                        "symbol": symbol,
                        "action": decision.action,
                        "size": round(size, 2),
                        "price": price,
                        "leverage": dynamic_lev,
                        "notional": round(notional, 2),
                        "reason": decision.reason,
                    })
                    tg_notify(format_trade_open("PAPER", symbol, decision.action, price, notional, dynamic_lev, decision.reason), silent=True)

            # === LIVE 실행 (live / dual 모드) ===
            if self.mode in ("live", "dual"):
                om = self.order_managers.get(exchange_name)
                if om and symbol not in getattr(om, "positions", {}):
                    result = await om.open_position(
                        symbol, decision.action, size, dynamic_lev, atr_pct=atr_pct,
                    )
                    if result:
                        logger.info(
                            f"[LIVE] {decision.action.upper()} {symbol} | "
                            f"마진: ${size:.2f} × {dynamic_lev}x = ${notional:.2f} | "
                            f"사유: {decision.reason}"
                        )
                        add_live_log({
                            "time": datetime.utcnow().strftime("%H:%M:%S"),
                            "type": "trade_open",
                            "mode": "LIVE",
                            "symbol": symbol,
                            "action": decision.action,
                            "size": round(size, 2),
                            "price": price,
                            "leverage": dynamic_lev,
                            "notional": round(notional, 2),
                            "reason": decision.reason,
                        })
                        tg_notify(format_trade_open("🔥LIVE", symbol, decision.action, price, notional, dynamic_lev, decision.reason))
                    else:
                        logger.warning(f"[LIVE] {symbol} 주문 실패 (마진 ${size:.2f})")

        elif decision.action == "close":
            price = float(df["close"].iloc[-1])
            volatility = df["returns_1"].std() if "returns_1" in df.columns else 0
            trade_context = {
                "regime": adaptive_params["regime"],
                "signal": ml_signal.get("signal", 0),
                "confidence": decision.confidence,
                "volatility": volatility,
                "external_score": ext_signal.get("score", 0),
                "external_direction": ext_signal.get("direction", "neutral"),
            }

            # === PAPER 청산 ===
            if self.mode in ("paper", "dual"):
                result = self.paper_trader.close_position(symbol, price, decision.reason)
                if result:
                    self.risk_manager.record_pnl(result["pnl"])
                    self.storage.save_trade({
                        "exchange": exchange_name, "symbol": symbol, "side": "close",
                        "price": price, "amount": result["size"], "pnl": result["pnl"],
                        "fee": result["fee"], "strategy": "hybrid",
                        "mode": "PAPER",
                    })
                    trade_context["exit_reason"] = "strategy_close"
                    trade_context["confirming_sources"] = getattr(decision, "confirming_sources", [])
                    trade_context["entry_path"] = "+".join(sorted(getattr(decision, "confirming_sources", [])))
                    self.feedback.record_trade(result, trade_context)
                    if result["pnl"] < 0:
                        self.strategy_manager.record_loss()
                    else:
                        self.strategy_manager.record_win()
                    # Paper StrategyOptimizer 기록
                    p_hash = self.strategy_optimizer_paper._config_to_hash(
                        self.strategy_optimizer_paper.current_config
                    )
                    self.strategy_optimizer_paper.record_trade(p_hash, {
                        "pnl": result["pnl"],
                        "timestamp": datetime.utcnow(),
                        "symbol": symbol,
                        "duration_minutes": result.get("duration_minutes", 0),
                        "hour": datetime.utcnow().hour,
                    })
                    add_live_log({
                        "time": datetime.utcnow().strftime("%H:%M:%S"),
                        "type": "trade_close",
                        "mode": "PAPER",
                        "symbol": symbol,
                        "pnl": round(result["pnl"], 2),
                        "reason": decision.reason,
                    })
                    logger.info(f"[PAPER] 청산 {symbol} | PnL: ${result['pnl']:.2f}")
                    tg_notify(format_trade_close("PAPER", symbol, result["pnl"], decision.reason, result.get("duration_minutes", 0)), silent=True)

            # === LIVE 청산 ===
            if self.mode in ("live", "dual"):
                om = self.order_managers.get(exchange_name)
                if om and symbol in om.positions:
                    live_result = await om.close_position(symbol, decision.reason)
                    if live_result:
                        # Live StrategyOptimizer 기록
                        l_hash = self.strategy_optimizer_live._config_to_hash(
                            self.strategy_optimizer_live.current_config
                        )
                        live_pnl = live_result.get("pnl", 0) if isinstance(live_result, dict) else 0

                        # 학습 기록 (feedback + storage + risk)
                        self.risk_manager.record_pnl(live_pnl)
                        self.feedback.record_trade(
                            {"pnl": live_pnl, "side": live_result.get("side", ""), "symbol": symbol},
                            {"regime": adaptive_params.get("regime", "unknown"),
                             "signal": ml_signal, "confidence": decision.confidence,
                             "external_score": ext_signal.get("score", 0) if isinstance(ext_signal, dict) else 0,
                             "external_direction": ext_signal.get("direction", "neutral") if isinstance(ext_signal, dict) else "neutral",
                             "exit_reason": "strategy_close",
                             "confirming_sources": getattr(decision, "confirming_sources", []),
                             "entry_path": "+".join(sorted(getattr(decision, "confirming_sources", [])))},
                        )
                        if live_pnl < 0:
                            self.strategy_manager.record_loss()
                        else:
                            self.strategy_manager.record_win()
                        self.storage.save_trade({
                            "exchange": exchange_name, "symbol": symbol, "side": "close",
                            "price": live_result.get("exit_price", 0),
                            "amount": live_result.get("size", 0),
                            "pnl": live_pnl, "fee": 0, "strategy": "live_signal_close",
                            "mode": "LIVE",
                        })

                        # StrategyOptimizer 기록
                        self.strategy_optimizer_live.record_trade(l_hash, {
                            "pnl": live_pnl,
                            "timestamp": datetime.utcnow(),
                            "symbol": symbol,
                            "duration_minutes": 0,
                            "hour": datetime.utcnow().hour,
                        })
                        add_live_log({
                            "time": datetime.utcnow().strftime("%H:%M:%S"),
                            "type": "trade_close",
                            "mode": "LIVE",
                            "symbol": symbol,
                            "pnl": round(live_pnl, 2),
                            "reason": decision.reason,
                        })
                        logger.info(f"[LIVE] 청산 {symbol} | PnL: ${live_pnl:.2f} → 학습기록 완료")
                        tg_notify(format_trade_close("🔥LIVE", symbol, live_pnl, decision.reason))

        # 페이퍼 트레이더 가격 업데이트 (paper / dual)
        if self.mode in ("paper", "dual"):
            price = float(df["close"].iloc[-1])
            self.paper_trader.update_prices({symbol: price})
            self.equity = self.paper_trader.equity
            self.total_pnl = self.equity - self.initial_capital

            # 자기수정: stale 포지션 자동 청산 (4시간 이상 보유 + 수익 없으면)
            if symbol in self.paper_trader.positions:
                pos = self.paper_trader.positions[symbol]
                age_minutes = (datetime.utcnow() - pos.entry_time).total_seconds() / 60
                if age_minutes > 240 and pos.unrealized_pnl < 0:
                    result = self.paper_trader.close_position(symbol, price, f"자기수정: {age_minutes:.0f}분 보유 + 손실")
                    if result:
                        stale_pnl = result["pnl"]
                        self.risk_manager.record_pnl(stale_pnl)
                        self.feedback.record_trade(
                            {"pnl": stale_pnl, "side": result.get("side", ""), "symbol": symbol},
                            {"regime": self.adaptive.current_regime,
                             "signal": 0, "confidence": 0,
                             "external_score": 0, "external_direction": "neutral"},
                        )
                        self.storage.save_trade({
                            "exchange": exchange_name, "symbol": symbol, "side": "close",
                            "price": price, "amount": result["size"], "pnl": stale_pnl,
                            "fee": result["fee"], "strategy": "auto_close",
                            "mode": "PAPER",
                        })
                        add_live_log({
                            "time": datetime.utcnow().strftime("%H:%M:%S"),
                            "type": "trade_close",
                            "mode": "PAPER",
                            "symbol": symbol,
                            "pnl": round(stale_pnl, 2),
                            "reason": f"자기수정: {age_minutes:.0f}분 stale 포지션 청산",
                        })
                        logger.info(f"[자기수정] PAPER stale 청산 {symbol} | PnL: ${stale_pnl:.2f} | {age_minutes:.0f}분 보유 → 학습기록 완료")
                        tg_notify(format_trade_close("PAPER", symbol, stale_pnl, f"자기수정: stale 청산 ({age_minutes:.0f}분)"), silent=True)

    # =========================================================================
    # 집중 매매 모드 메서드 (concentration_mode=true)
    # =========================================================================

    async def _analyze_symbol(self, exchange_name: str, symbol: str, timeframe: str) -> dict | None:
        """심볼 분석만 수행하고 시그널 반환 (실행 X)"""
        try:
            df = await self.collector.fetch_ohlcv(exchange_name, symbol, timeframe, limit=200)
            df = self.feature_engineer.generate(df)
            feature_cols = self.feature_engineer.get_feature_columns(df)

            if len(df) < 60:
                return None

            prices = df["close"].values
            volumes = df["volume"].values
            adaptive_params = self.adaptive.update(prices, volumes)

            ml_signal = self.ensemble.predict(df)

            base_feature_cols = self.feature_engineer.get_base_feature_columns(df)
            rl_obs_data = df[base_feature_cols].values[-1].astype(np.float32)
            rl_obs_data = np.nan_to_num(rl_obs_data, nan=0.0)
            position_info = np.array([0.0, 0.0, self.equity / self.initial_capital, 0.0], dtype=np.float32)
            obs = np.concatenate([rl_obs_data, position_info])
            rl_action, rl_confidence = self.rl_agent.predict(obs)

            ext_signal = self.external_manager.get_signal_for_strategy()
            mtf_signal = self.external_manager.multi_tf.get_signal_for_strategy()
            momentum_signal = self._calculate_momentum(df)

            # ATR 값 추출 (동적 SL/TP용)
            atr_pct = float(df["atr_pct"].iloc[-1]) if "atr_pct" in df.columns else 0.0
            if atr_pct != atr_pct:  # NaN 체크
                atr_pct = 0.0

            # 피드백 블랙리스트 조회
            fb_blacklist = self.feedback.get_entry_blacklist()

            # 펀딩비 + 공포탐욕 지수 (레짐 바이어스용)
            ext_features = self.external_manager.get_all_features()
            funding_rate = ext_features.get("deriv_funding_rate", 0)
            fear_greed = ext_features.get("fg_value", 50)

            current_position = 1.0 if symbol in self.paper_trader.positions else 0.0
            decision = self.strategy_manager.decide(
                ml_signal, rl_action, rl_confidence, current_position,
                adaptive_params["regime"], external_signal=ext_signal,
                momentum=momentum_signal,
                feedback_blacklist=fb_blacklist,
                funding_rate=funding_rate,
                fear_greed_index=fear_greed,
            )

            # MTF 필터 적용
            if decision.action in ["long", "short"]:
                mtf_agreement = mtf_signal.get("agreement", 0)
                mtf_dir = mtf_signal.get("direction", "neutral")
                action_opposes_mtf = (
                    (decision.action == "long" and mtf_dir == "bearish") or
                    (decision.action == "short" and mtf_dir == "bullish")
                )
                action_agrees_mtf = (
                    (decision.action == "long" and mtf_dir == "bullish") or
                    (decision.action == "short" and mtf_dir == "bearish")
                )
                if action_opposes_mtf and mtf_agreement >= 0.75:
                    decision.action = "hold"
                    decision.confidence = 0.0
                elif action_opposes_mtf and mtf_agreement >= 0.5:
                    decision.confidence *= 0.6
                elif action_agrees_mtf and mtf_agreement > 0.7:
                    decision.confidence = min(decision.confidence * 1.15, 1.0)

            price = float(df["close"].iloc[-1])
            volatility = df["returns_1"].std() if "returns_1" in df.columns else 0.01

            # 동적 레버리지
            ext_agrees = (
                (decision.action == "long" and ext_signal.get("direction") == "bullish") or
                (decision.action == "short" and ext_signal.get("direction") == "bearish")
            )
            dynamic_lev = self.risk_manager.calculate_dynamic_leverage(
                confidence=decision.confidence,
                volatility=volatility,
                regime=adaptive_params["regime"],
                external_agreement=ext_agrees,
            )

            # 포지션 크기 (풀시드)
            fb_scale = self.feedback.get_position_scale(adaptive_params["regime"], decision.action)
            size = self.risk_manager.calculate_position_size(
                self.equity, decision.confidence, volatility,
                adaptive_params["position_scale"] * fb_scale,
            )

            # 최소 notional 보장
            notional = size * dynamic_lev
            min_notional = self.min_order_notional * 1.05
            if notional < min_notional:
                size = max(size, min_notional / dynamic_lev)
                size = min(size, self.equity * 0.95)
                notional = size * dynamic_lev

            # 로그
            add_live_log({
                "time": datetime.utcnow().strftime("%H:%M:%S"),
                "type": "analysis",
                "symbol": symbol,
                "price": price,
                "action": decision.action,
                "confidence": round(decision.confidence, 3),
                "ml_dir": ml_signal.get("direction", "?"),
                "rl_action": ["hold", "long", "short", "close"][rl_action],
                "mom_dir": momentum_signal.get("direction", "?"),
                "regime": adaptive_params["regime"],
                "reason": decision.reason,
                "min_conf": round(self.strategy_manager.min_confidence, 3),
                "holds": self.strategy_manager._consecutive_holds,
            })

            self.last_signals = {
                "symbol": symbol,
                "direction": decision.action,
                "confidence": decision.confidence,
                "regime": adaptive_params["regime"],
                "reason": decision.reason,
                "momentum": momentum_signal,
                "diagnostics": self.strategy_manager.get_diagnostics(),
            }

            # 페이퍼 트레이더 가격 업데이트
            if self.mode in ("paper", "dual"):
                self.paper_trader.update_prices({symbol: price})
                self.equity = self.paper_trader.equity
                self.total_pnl = self.equity - self.initial_capital

            if decision.action not in ("long", "short", "close"):
                return None

            return {
                "symbol": symbol,
                "exchange_name": exchange_name,
                "action": decision.action,
                "confidence": decision.confidence,
                "size": size,
                "price": price,
                "dynamic_lev": dynamic_lev,
                "notional": notional,
                "reason": decision.reason,
                "volatility": volatility,
                "regime": adaptive_params["regime"],
                "adaptive_params": adaptive_params,
                "ml_signal": ml_signal,
                "ext_signal": ext_signal,
                "df": df,
                "atr_pct": atr_pct,  # ATR 기반 동적 SL/TP
                "confirming_sources": decision.confirming_sources,  # 다중확인 소스
                "signal_strength": decision.signal_strength,  # 시그널 강도
            }

        except Exception as e:
            logger.debug(f"[집중분석] {symbol} 분석 실패: {e}")
            return None

    async def _execute_paper(self, c: dict):
        """PAPER 포지션 실행"""
        symbol = c["symbol"]
        if self.mode not in ("paper", "dual"):
            return
        if c["action"] in ("long", "short"):
            if symbol not in self.paper_trader.positions:
                tp_pct = c.get("tp_pct", self.config["risk"]["take_profit_pct"])
                sl_pct = c.get("sl_pct", self.config["risk"]["stop_loss_pct"])
                self.paper_trader.open_position(
                    symbol, c["action"], c["size"], c["price"],
                    leverage=c["dynamic_lev"],
                    sl_pct=sl_pct * c["adaptive_params"]["stop_loss_mult"],
                    tp_pct=tp_pct,
                    atr_pct=c.get("atr_pct", 0),
                )
                logger.info(
                    f"[PAPER] {c['action'].upper()} {symbol} | "
                    f"${c['size']:.2f} × {c['dynamic_lev']}x = ${c['notional']:.2f} | "
                    f"{c['reason']}"
                )
                add_live_log({
                    "time": datetime.utcnow().strftime("%H:%M:%S"),
                    "type": "trade_open",
                    "mode": "PAPER",
                    "symbol": symbol,
                    "action": c["action"],
                    "size": round(c["size"], 2),
                    "price": c["price"],
                    "leverage": c["dynamic_lev"],
                    "notional": round(c["notional"], 2),
                    "reason": c["reason"],
                })
                tg_notify(format_trade_open("PAPER", symbol, c["action"], c["price"], c["notional"], c["dynamic_lev"], c["reason"]), silent=True)
        elif c["action"] == "close" and symbol in self.paper_trader.positions:
            result = self.paper_trader.close_position(symbol, c["price"], c["reason"])
            if result:
                paper_pnl = result["pnl"]
                self.risk_manager.record_pnl(paper_pnl)

                # 학습 기록 (feedback + storage + optimizer)
                self.feedback.record_trade(
                    {"pnl": paper_pnl, "side": c["action_before_close"] if "action_before_close" in c else result.get("side", ""), "symbol": symbol},
                    {"regime": c.get("adaptive_params", {}).get("regime", "unknown"),
                     "signal": c.get("ml_signal", 0), "confidence": c.get("confidence", 0),
                     "external_score": 0, "external_direction": "neutral"},
                )
                self.storage.save_trade({
                    "exchange": "paper", "symbol": symbol, "side": "close",
                    "price": c["price"], "amount": result.get("size", 0),
                    "pnl": paper_pnl, "fee": 0, "strategy": "paper_concentration",
                    "mode": "PAPER",
                })
                p_hash = self.strategy_optimizer_paper._config_to_hash(
                    self.strategy_optimizer_paper.current_config
                )
                self.strategy_optimizer_paper.record_trade(p_hash, {
                    "pnl": paper_pnl, "timestamp": datetime.utcnow(),
                    "symbol": symbol, "hour": datetime.utcnow().hour,
                })

                logger.info(f"[PAPER] 청산 {symbol} | PnL: ${paper_pnl:.2f} → 학습기록 완료")
                tg_notify(format_trade_close("PAPER", symbol, paper_pnl, c["reason"], result.get("duration_minutes", 0)), silent=True)

    async def _execute_live(self, exchange_name: str, c: dict):
        """LIVE 포지션 실행 — 가장 강한 시그널에만"""
        if self.mode not in ("live", "dual"):
            return
        # 자가진단에 의한 LIVE 일시정지 상태면 신규 진입 차단
        if getattr(self, '_live_paused', False):
            logger.info(f"[LIVE] 일시정지 상태 — 신규 진입 차단 (사유: {getattr(self, '_live_pause_reason', '?')})")
            return
        symbol = c["symbol"]
        om = self.order_managers.get(exchange_name)
        if not om:
            return

        if c["action"] in ("long", "short") and symbol not in om.positions:
            # StrategyOptimizer 조정값 적용 (시간대/종목별 스케일)
            opt_scale = self.strategy_optimizer_live.get_position_scale(
                symbol=symbol, hour=datetime.utcnow().hour
            )
            if opt_scale <= 0:
                logger.info(f"[LIVE] {symbol} 거래 차단 (optimizer: 시간대/종목 회피)")
                return

            # LIVE는 실제 거래소 잔고 기준 사이즈 조정
            try:
                balance = await om.exchange.get_balance()
                live_free = balance.get("free", 0)
                # 가용 잔고의 90% × optimizer 스케일
                live_size = live_free * 0.90 * opt_scale
                if live_size < self.min_order_notional / c["dynamic_lev"]:
                    logger.warning(f"[LIVE] 잔고 부족: ${live_free:.2f} × {opt_scale:.1f} (필요: ${self.min_order_notional / c['dynamic_lev']:.2f})")
                    return
                c["size"] = live_size
                c["notional"] = live_size * c["dynamic_lev"]
            except Exception as e:
                logger.warning(f"[LIVE] 잔고 조회 실패: {e}")

            result = await om.open_position(
                symbol, c["action"], c["size"], c["dynamic_lev"],
                sl_pct=c.get("sl_pct"), tp_pct=c.get("tp_pct"),
                atr_pct=c.get("atr_pct", 0),
            )
            if result:
                coin = symbol.split("/")[0]
                logger.info(
                    f"[LIVE🔥] {c['action'].upper()} {symbol} | "
                    f"${c['size']:.2f} × {c['dynamic_lev']}x = ${c['notional']:.2f} | "
                    f"TP: {c.get('tp_pct', 0.012)*100:.1f}% SL: {c.get('sl_pct', 0.008)*100:.1f}% | "
                    f"{c['reason']}"
                )
                add_live_log({
                    "time": datetime.utcnow().strftime("%H:%M:%S"),
                    "type": "trade_open",
                    "mode": "LIVE",
                    "symbol": symbol,
                    "action": c["action"],
                    "size": round(c["size"], 2),
                    "price": c["price"],
                    "leverage": c["dynamic_lev"],
                    "notional": round(c["notional"], 2),
                    "reason": f"[집중매매] {c['reason']}",
                })
                tg_notify(format_trade_open(
                    "🔥LIVE 집중", symbol, c["action"], c["price"],
                    c["notional"], c["dynamic_lev"],
                    f"TP:{c.get('tp_pct',0.012)*100:.1f}% SL:{c.get('sl_pct',0.008)*100:.1f}% | {c['reason']}"
                ))
            else:
                logger.warning(f"[LIVE] {symbol} 주문 실패")
        elif c["action"] == "close" and symbol in om.positions:
            live_result = await om.close_position(symbol, c["reason"])
            if live_result:
                live_pnl = live_result.get("pnl", 0) if isinstance(live_result, dict) else 0
                logger.info(f"[LIVE] 청산 {symbol} | PnL: ${live_pnl:.2f}")

                # LIVE도 쿨다운/연패 추적
                self.risk_manager.record_pnl(live_pnl)

                # LIVE 청산도 피드백 학습에 기록
                self.feedback.record_trade(
                    {"pnl": live_pnl, "side": live_result.get("side", ""), "symbol": symbol},
                    {"regime": c.get("adaptive_params", {}).get("regime", "unknown"),
                     "signal": c.get("ml_signal", 0),
                     "confidence": c.get("confidence", 0),
                     "external_score": c.get("ext_signal", {}).get("score", 0) if isinstance(c.get("ext_signal"), dict) else 0,
                     "external_direction": c.get("ext_signal", {}).get("direction", "neutral") if isinstance(c.get("ext_signal"), dict) else "neutral",
                     "exit_reason": "strategy_close",
                     "confirming_sources": c.get("confirming_sources", []),
                     "entry_path": "+".join(sorted(c.get("confirming_sources", [])))},
                )
                if live_pnl < 0:
                    self.strategy_manager.record_loss()
                else:
                    self.strategy_manager.record_win()
                self.storage.save_trade({
                    "exchange": exchange_name, "symbol": symbol, "side": "close",
                    "price": live_result.get("exit_price", 0),
                    "amount": live_result.get("size", 0),
                    "pnl": live_pnl, "fee": 0, "strategy": "live_concentration",
                    "mode": "LIVE",
                })

                # StrategyOptimizer 기록
                l_hash = self.strategy_optimizer_live._config_to_hash(
                    self.strategy_optimizer_live.current_config
                )
                self.strategy_optimizer_live.record_trade(l_hash, {
                    "pnl": live_pnl, "timestamp": datetime.utcnow(),
                    "symbol": symbol, "hour": datetime.utcnow().hour,
                })

                add_live_log({
                    "time": datetime.utcnow().strftime("%H:%M:%S"),
                    "type": "trade_close",
                    "mode": "LIVE",
                    "symbol": symbol,
                    "pnl": round(live_pnl, 2),
                    "reason": c["reason"],
                })
                tg_notify(format_trade_close("🔥LIVE", symbol, live_pnl, c["reason"]))

    # ========== 자가진단 시스템 v2 ==========

    async def _critical_health_check(self):
        """30분마다 실행 — 즉시 대응 필요한 치명적 문제만 검사"""
        issues = []
        fixes = []

        try:
            # ──── 1. 텔레그램 실제 전송 테스트 ────
            tg_working = await self._check_telegram()
            if not tg_working:
                issues.append("텔레그램 전송 불가")
                fix = await self._fix_telegram()
                if fix:
                    fixes.append(fix)

            # ──── 2. 거래소 고아 포지션 감지 (내부 추적 없는 포지션) ────
            orphan_fixes = await self._check_orphan_positions()
            for item in orphan_fixes:
                if item["type"] == "issue":
                    issues.append(item["msg"])
                else:
                    fixes.append(item["msg"])

            # ──── 3. 모든 LIVE 포지션에 SL/TP Algo 주문 존재 확인 ────
            algo_fixes = await self._check_algo_orders()
            for item in algo_fixes:
                if item["type"] == "issue":
                    issues.append(item["msg"])
                else:
                    fixes.append(item["msg"])

            # ──── 4. 연속 손실 / 승률 감시 → 자동 일시정지 ────
            loss_result = self._check_consecutive_losses()
            if loss_result:
                issues.append(loss_result)

            # ──── 5. SL/TP 콜백 등록 확인 ────
            cb_fixes = self._check_callbacks()
            for item in cb_fixes:
                if item["type"] == "issue":
                    issues.append(item["msg"])
                else:
                    fixes.append(item["msg"])

            # 리포트
            if issues:
                status = f"🚨 {len(issues)}건 치명적 문제"
                report_lines = [
                    f"🚨 <b>긴급 자가진단</b>",
                    f"━━━━━━━━━━━━━",
                    f"❌ 문제점:",
                ]
                for issue in issues:
                    report_lines.append(f"  • {issue}")
                if fixes:
                    report_lines.append(f"\n🔧 자동수정:")
                    for fix in fixes:
                        report_lines.append(f"  • {fix}")
                report_text = "\n".join(report_lines)
                logger.warning(f"[긴급진단] {status} | issues: {issues} | fixes: {fixes}")
                tg_notify(report_text)
            else:
                logger.info("[긴급진단] ✅ 치명적 문제 없음")

        except Exception as e:
            logger.error(f"[긴급진단] 실행 실패: {e}")

    async def _learning_health_check(self):
        """2시간마다 학습 시스템 종합 진단 + 리포트"""
        issues = []
        fixes = []

        try:
            # ──── 기본 진단 (기존 항목) ────
            fb_path = Path("data/feedback_history.json")
            if not fb_path.exists():
                issues.append("feedback_history.json 없음")
                self.feedback._save()
                fixes.append("feedback_history.json 재생성")

            fb_total = self.feedback.feedback.get("total_analyzed", 0)
            fb_recent = len(self.feedback.recent_trades)

            db_counts = self.storage.get_trade_counts_by_mode(hours=24)
            live_count_24h = db_counts.get("LIVE", 0)
            paper_count_24h = db_counts.get("PAPER", 0)

            live_opt_trades = sum(len(v) for v in self.strategy_optimizer_live.performance_history.values())
            paper_opt_trades = sum(len(v) for v in self.strategy_optimizer_paper.performance_history.values())

            total_db = sum(db_counts.values())
            if fb_total > 0 and total_db == 0:
                issues.append(f"feedback {fb_total}건, DB 0건 — save_trade 누락 가능")

            # ──── 텔레그램 체크 ────
            tg_working = await self._check_telegram()
            if not tg_working:
                issues.append("텔레그램 전송 불가")
                fix = await self._fix_telegram()
                if fix:
                    fixes.append(fix)

            # ──── 거래소 고아 포지션 ────
            orphan_fixes = await self._check_orphan_positions()
            for item in orphan_fixes:
                if item["type"] == "issue":
                    issues.append(item["msg"])
                else:
                    fixes.append(item["msg"])

            # ──── Algo 주문 확인 ────
            algo_fixes = await self._check_algo_orders()
            for item in algo_fixes:
                if item["type"] == "issue":
                    issues.append(item["msg"])
                else:
                    fixes.append(item["msg"])

            # ──── 콜백 확인 ────
            cb_fixes = self._check_callbacks()
            for item in cb_fixes:
                if item["type"] == "issue":
                    issues.append(item["msg"])
                else:
                    fixes.append(item["msg"])

            # ──── 연속 손실 / 승률 ────
            loss_result = self._check_consecutive_losses()
            if loss_result:
                issues.append(loss_result)

            # ──── 코드 버전 확인 (git commit vs 프로세스 시작 시간) ────
            version_issue = self._check_code_version()
            if version_issue:
                issues.append(version_issue)

            # ──── SL 적중률 분석 → 자동 조정 제안 ────
            sl_suggestion = self.feedback.get_sl_adjustment_suggestion()
            sl_note = ""
            if sl_suggestion and sl_suggestion.get("action"):
                sl_note = f"SL제안: {sl_suggestion['action']} (적중률 {sl_suggestion.get('sl_hit_rate', 0):.0%})"

            # ──── 프로세스 가동시간 ────
            uptime = datetime.utcnow() - self.start_time
            uptime_str = f"{uptime.days}일 {uptime.seconds // 3600}시간"

            # ──── 잔고 체크 ────
            balance_info = ""
            for name, client in self.exchange_clients.items():
                try:
                    bal = await client.get_balance()
                    balance_info = f"${bal.get('total', 0):,.2f} (가용: ${bal.get('free', 0):,.2f})"
                except Exception:
                    balance_info = "조회 실패"

            # === 리포트 생성 ===
            status = "✅ 정상" if not issues else f"⚠️ {len(issues)}건 문제"
            report_lines = [
                f"🔍 <b>종합 자가진단</b> {status}",
                f"━━━━━━━━━━━━━",
                f"⏱ 가동: {uptime_str}",
                f"💰 잔고: {balance_info}",
                f"📊 피드백: {fb_total}건 (메모리: {fb_recent}건)",
                f"💾 DB 24h: LIVE {live_count_24h} | PAPER {paper_count_24h}",
                f"📈 Optimizer: LIVE {live_opt_trades} | PAPER {paper_opt_trades}",
                f"📡 텔레그램: {'✅' if tg_working else '❌'}",
            ]
            if sl_note:
                report_lines.append(f"📐 {sl_note}")

            if issues:
                report_lines.append(f"\n❌ 문제점:")
                for issue in issues:
                    report_lines.append(f"  • {issue}")
            if fixes:
                report_lines.append(f"\n🔧 자동수정:")
                for fix in fixes:
                    report_lines.append(f"  • {fix}")

            # LIVE 포지션 상태
            live_pos_info = []
            for name, om in self.order_managers.items():
                for symbol, pos in om.positions.items():
                    live_pos_info.append(f"  {pos.side} {symbol} @ {pos.entry_price:.4f}")
            if live_pos_info:
                report_lines.append(f"\n📍 LIVE 포지션:")
                report_lines.extend(live_pos_info)

            # 연속손실 현황
            loss_penalty = getattr(self.strategy_manager, '_loss_penalty', 0)
            consec = getattr(self.strategy_manager, '_consecutive_losses', 0)
            if consec > 0:
                report_lines.append(f"\n⚡ 연속손실: {consec}회 (확신도 +{loss_penalty:.2f})")

            report_text = "\n".join(report_lines)
            logger.info(f"[종합진단] {status} | FB:{fb_total} DB_LIVE:{live_count_24h} DB_PAPER:{paper_count_24h} | issues:{len(issues)} fixes:{len(fixes)}")
            tg_notify(report_text, silent=len(issues) == 0)

        except Exception as e:
            logger.error(f"[종합진단] 실패: {e}")
            tg_notify(f"⚠️ 종합 자가진단 실패: {e}")

    # ──── 개별 진단 모듈들 ────

    async def _check_telegram(self) -> bool:
        """텔레그램 실제 전송 테스트 (빈 메시지 아닌 실제 API 호출)"""
        global _tg_ok
        try:
            # 1. 모듈 임포트 가능한지
            from scripts.telegram_bot import send_message as _test_send, BOT_TOKEN, CHAT_ID
            if not BOT_TOKEN or not CHAT_ID:
                logger.warning("[진단] 텔레그램 BOT_TOKEN 또는 CHAT_ID 미설정")
                return False

            # 2. 실제 API 호출 테스트 (getMe — 메시지 전송 없이 봇 상태 확인)
            import urllib.request
            import json
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=10)
            result = json.loads(resp.read())
            if result.get("ok"):
                _tg_ok = True
                return True
            else:
                logger.warning(f"[진단] 텔레그램 getMe 실패: {result}")
                return False
        except Exception as e:
            logger.warning(f"[진단] 텔레그램 테스트 실패: {e}")
            return False

    async def _fix_telegram(self) -> str:
        """텔레그램 복구 시도"""
        global _tg_ok
        try:
            # telegram_bot.py 파일 존재 확인
            tg_path = Path(__file__).parent / "scripts" / "telegram_bot.py"
            if not tg_path.exists():
                logger.error(f"[자가수정] telegram_bot.py 파일 없음: {tg_path}")
                return "telegram_bot.py 파일 없음 — 수동 복원 필요"

            # 모듈 리임포트 시도
            import importlib
            import scripts.telegram_bot as tg_mod
            importlib.reload(tg_mod)
            from scripts.telegram_bot import send_message as tg_send_new
            # 글로벌 참조 업데이트
            import builtins
            globals_main = sys.modules[__name__]
            if hasattr(globals_main, 'tg_send'):
                # tg_send를 직접 교체할 수는 없으므로, _tg_ok 플래그만 복원
                pass
            _tg_ok = True
            return "텔레그램 모듈 리로드 완료"
        except Exception as e:
            logger.error(f"[자가수정] 텔레그램 복구 실패: {e}")
            return f"텔레그램 복구 실패: {e}"

    async def _check_orphan_positions(self) -> list[dict]:
        """거래소에는 포지션이 있지만 내부 추적에 없는 '고아 포지션' 감지 + 복구"""
        results = []
        trading_symbols = self.config.get("trading", {}).get("symbols", [])

        for name, client in self.exchange_clients.items():
            try:
                exchange_positions = await client.get_all_positions()
                om = self.order_managers.get(name)
                if not om:
                    continue

                internal_symbols = set(om.positions.keys())

                for epos in exchange_positions:
                    sym = epos["symbol"]
                    if sym not in internal_symbols:
                        # 고아 포지션 발견!
                        results.append({
                            "type": "issue",
                            "msg": f"고아 포지션 발견: {sym} {epos['side']} {epos['size']}개 @ {epos['entry_price']:.4f} (내부 추적 없음)"
                        })

                        # 자가수정: recover_positions으로 복구
                        try:
                            if sym in trading_symbols:
                                recovered = await om.recover_positions([sym])
                                if recovered:
                                    results.append({
                                        "type": "fix",
                                        "msg": f"{sym} 고아 포지션 복구 완료 (SL/TP 재설정)"
                                    })
                                    tg_notify(
                                        f"🔄 <b>고아 포지션 자동복구</b>\n"
                                        f"━━━━━━━━━━━━━\n"
                                        f"종목: {sym}\n"
                                        f"방향: {epos['side']}\n"
                                        f"수량: {epos['size']}\n"
                                        f"진입가: ${epos['entry_price']:.4f}\n"
                                        f"⚠️ SL/TP 자동 재설정됨"
                                    )
                            else:
                                # 추적 대상이 아닌 심볼 → 경고만
                                results.append({
                                    "type": "issue",
                                    "msg": f"{sym} 비추적 심볼 고아 포지션 — 수동 확인 필요"
                                })
                        except Exception as e:
                            results.append({
                                "type": "issue",
                                "msg": f"{sym} 고아 포지션 복구 실패: {e}"
                            })

                # 반대도 확인: 내부에는 있지만 거래소에 없는 유령 포지션
                exchange_symbols = {epos["symbol"] for epos in exchange_positions}
                for internal_sym in internal_symbols:
                    if internal_sym not in exchange_symbols:
                        results.append({
                            "type": "issue",
                            "msg": f"유령 포지션: {internal_sym} 내부 추적 있지만 거래소에 없음"
                        })
                        # 자가수정: 내부 포지션 제거
                        try:
                            del om.positions[internal_sym]
                            results.append({
                                "type": "fix",
                                "msg": f"{internal_sym} 유령 포지션 내부 추적 제거"
                            })
                        except Exception:
                            pass

            except Exception as e:
                results.append({
                    "type": "issue",
                    "msg": f"{name} 거래소 포지션 전체 조회 실패: {e}"
                })

        return results

    async def _check_algo_orders(self) -> list[dict]:
        """모든 LIVE 포지션의 SL/TP Algo 주문 존재 여부 확인 + 자가수정"""
        results = []
        for name, om in self.order_managers.items():
            for symbol, pos in list(om.positions.items()):
                try:
                    algo_orders = await om.exchange.get_algo_orders(symbol)
                    if len(algo_orders) < 2:
                        results.append({
                            "type": "issue",
                            "msg": f"{symbol} Algo 주문 {len(algo_orders)}건 (2건 필요)"
                        })
                        # 자가수정
                        await om.exchange.cancel_all_orders(symbol)
                        close_side = "sell" if pos.side == "long" else "buy"
                        await om.exchange.create_stop_loss(symbol, close_side, pos.size, pos.stop_loss)
                        await om.exchange.create_take_profit(symbol, close_side, pos.size, pos.take_profit)
                        results.append({
                            "type": "fix",
                            "msg": f"{symbol} SL({pos.stop_loss:.4f})/TP({pos.take_profit:.4f}) 재설정"
                        })
                except Exception as e:
                    results.append({
                        "type": "issue",
                        "msg": f"{symbol} Algo 확인 실패: {e}"
                    })
        return results

    def _check_callbacks(self) -> list[dict]:
        """SL/TP 콜백 등록 확인 + 자가수정"""
        results = []
        for name, om in self.order_managers.items():
            sl_ok = getattr(om, '_on_sl_callback', None) is not None
            tp_ok = getattr(om, '_on_tp_callback', None) is not None

            if not sl_ok or not tp_ok:
                missing = []
                if not sl_ok:
                    missing.append("SL")
                if not tp_ok:
                    missing.append("TP")
                results.append({
                    "type": "issue",
                    "msg": f"{name} {'/'.join(missing)} 콜백 미등록"
                })

                # 자가수정: 콜백 재등록
                def _make_trade_callback_fix(close_type: str):
                    def on_triggered(result: dict):
                        symbol = result.get("symbol", "?")
                        pnl = result.get("pnl", 0)
                        side = result.get("side", "?")
                        regime = self.adaptive.current_regime if hasattr(self, 'adaptive') else "unknown"
                        self.feedback.record_trade(
                            {"pnl": pnl, "side": side, "symbol": symbol},
                            {"regime": regime, "signal": 0, "confidence": 0,
                             "external_score": 0, "external_direction": "neutral",
                             "exit_reason": close_type.lower(),
                             "confirming_sources": [],
                             "entry_path": "callback"},
                        )
                        if pnl < 0:
                            self.strategy_manager.record_loss()
                        else:
                            self.strategy_manager.record_win()
                        self.risk_manager.record_pnl(pnl)
                        self.storage.save_trade({
                            "exchange": "binance", "symbol": symbol, "side": "close",
                            "price": result.get("exit_price", 0),
                            "amount": result.get("size", 0),
                            "pnl": pnl, "fee": 0,
                            "strategy": f"{close_type.lower()}_triggered",
                            "mode": "LIVE",
                        })
                        l_hash = self.strategy_optimizer_live._config_to_hash(
                            self.strategy_optimizer_live.current_config
                        )
                        self.strategy_optimizer_live.record_trade(l_hash, {
                            "pnl": pnl, "timestamp": datetime.utcnow(),
                            "symbol": symbol, "hour": datetime.utcnow().hour,
                        })
                        logger.info(f"[{close_type}학습] {symbol} PnL: ${pnl:.2f} → 학습기록 완료")
                        tg_notify(f"{'🛑' if close_type == 'SL' else '🎯'} {close_type} 체결 {symbol} PnL: ${pnl:+.2f}")
                    return on_triggered

                om.set_sl_callback(_make_trade_callback_fix("SL"))
                om.set_tp_callback(_make_trade_callback_fix("TP"))
                results.append({
                    "type": "fix",
                    "msg": f"{name} SL/TP 콜백 재등록 완료"
                })

        return results

    def _check_consecutive_losses(self) -> str | None:
        """연속 손실 / 승률 감시 → LIVE 자동 일시정지"""
        # 최근 LIVE 거래 승률 확인
        try:
            recent_live = self.storage.get_recent_trades(mode="LIVE", limit=10)
            if len(recent_live) >= 5:
                wins = sum(1 for t in recent_live if (t.get("pnl") or 0) > 0)
                win_rate = wins / len(recent_live)

                if win_rate == 0 and len(recent_live) >= 5:
                    # 최근 5건 이상 전패 → LIVE 일시정지
                    if not getattr(self, '_live_paused', False):
                        self._live_paused = True
                        self._live_pause_reason = f"최근 {len(recent_live)}건 전패 (승률 0%)"
                        self._live_pause_time = datetime.utcnow()
                        logger.warning(f"[자가진단] ⛔ LIVE 자동 일시정지: {self._live_pause_reason}")
                        tg_notify(
                            f"⛔ <b>LIVE 자동 일시정지</b>\n"
                            f"━━━━━━━━━━━━━\n"
                            f"사유: {self._live_pause_reason}\n"
                            f"최근 {len(recent_live)}건 PnL: {[round(t.get('pnl', 0), 2) for t in recent_live]}\n"
                            f"📝 PAPER 모드는 계속 학습 중\n"
                            f"🔄 승률 개선 시 자동 재개"
                        )
                    return f"LIVE 일시정지: 최근 {len(recent_live)}건 전패"

                elif win_rate > 0 and getattr(self, '_live_paused', False):
                    # 승률 회복 → 재개
                    self._live_paused = False
                    pause_duration = datetime.utcnow() - getattr(self, '_live_pause_time', datetime.utcnow())
                    logger.info(f"[자가진단] ✅ LIVE 재개 (승률 {win_rate:.0%}, 정지 {pause_duration})")
                    tg_notify(
                        f"✅ <b>LIVE 자동 재개</b>\n"
                        f"━━━━━━━━━━━━━\n"
                        f"승률 회복: {win_rate:.0%} ({wins}/{len(recent_live)}건)\n"
                        f"정지 기간: {pause_duration}"
                    )
                    return None

                elif win_rate < 0.25 and len(recent_live) >= 8:
                    return f"LIVE 저조: 최근 {len(recent_live)}건 승률 {win_rate:.0%}"

        except Exception as e:
            logger.debug(f"[진단] 연속 손실 체크 실패: {e}")

        return None

    async def _strategic_self_review(self):
        """4시간마다 실행 — 자체 거래 패턴 분석 + 전략 자동 조정

        분석 항목:
        1. 펀딩비 vs 방향별 승패 패턴
        2. 레짐별 승률 분석
        3. 시간대별 성과
        4. 패턴 기반 자동 블랙리스트 / 파라미터 조정
        """
        try:
            recent_trades = self.storage.get_recent_trades(limit=50)
            if len(recent_trades) < 5:
                logger.info("[SelfReview] 거래 수 부족 (5건 미만) — 스킵")
                return

            # === 1. 방향별 승패 분석 ===
            long_trades = [t for t in recent_trades if t.get("side") == "long" or t.get("strategy", "").startswith("long")]
            short_trades = [t for t in recent_trades if t.get("side") == "short" or t.get("strategy", "").startswith("short")]

            # entry 거래만 분석 (close 거래의 pnl로 판단)
            close_trades = [t for t in recent_trades if t.get("side") == "close" and t.get("pnl") is not None]

            if not close_trades:
                logger.info("[SelfReview] 청산 거래 없음 — 스킵")
                return

            wins = [t for t in close_trades if (t.get("pnl") or 0) > 0]
            losses = [t for t in close_trades if (t.get("pnl") or 0) < 0]
            total_pnl = sum(t.get("pnl", 0) for t in close_trades)
            win_rate = len(wins) / len(close_trades) if close_trades else 0

            # === 2. 모드별 분석 ===
            live_closes = [t for t in close_trades if t.get("mode") == "LIVE"]
            paper_closes = [t for t in close_trades if t.get("mode") == "PAPER"]

            live_wr = (sum(1 for t in live_closes if (t.get("pnl") or 0) > 0) / len(live_closes)) if live_closes else 0
            paper_wr = (sum(1 for t in paper_closes if (t.get("pnl") or 0) > 0) / len(paper_closes)) if paper_closes else 0
            live_pnl = sum(t.get("pnl", 0) for t in live_closes)
            paper_pnl = sum(t.get("pnl", 0) for t in paper_closes)

            # === 3. 전략별 손실 패턴 분석 ===
            strategy_stats = {}
            for t in close_trades:
                strat = t.get("strategy", "unknown")
                if strat not in strategy_stats:
                    strategy_stats[strat] = {"wins": 0, "losses": 0, "pnl": 0}
                if (t.get("pnl") or 0) > 0:
                    strategy_stats[strat]["wins"] += 1
                elif (t.get("pnl") or 0) < 0:
                    strategy_stats[strat]["losses"] += 1
                strategy_stats[strat]["pnl"] += t.get("pnl", 0)

            # === 4. 시간대별 분석 ===
            hour_stats = {}
            for t in close_trades:
                try:
                    ts = t.get("timestamp", "")
                    if ts:
                        hour = datetime.fromisoformat(ts.replace("Z", "+00:00")).hour if "T" in ts else int(ts.split(" ")[1].split(":")[0])
                    else:
                        hour = -1
                except Exception:
                    hour = -1
                if hour >= 0:
                    if hour not in hour_stats:
                        hour_stats[hour] = {"wins": 0, "losses": 0, "pnl": 0}
                    if (t.get("pnl") or 0) > 0:
                        hour_stats[hour]["wins"] += 1
                    else:
                        hour_stats[hour]["losses"] += 1
                    hour_stats[hour]["pnl"] += t.get("pnl", 0)

            # === 5. 자동 조정 로직 ===
            adjustments = []

            # 5a. SL 적중률 너무 높으면 SL 여유 확대 제안
            sl_trades = [t for t in close_trades if "sl" in (t.get("strategy") or "").lower()]
            if len(sl_trades) >= 3 and len(close_trades) > 0:
                sl_ratio = len(sl_trades) / len(close_trades)
                if sl_ratio > 0.6:
                    adjustments.append(f"SL 적중률 {sl_ratio:.0%} — SL 여유 확대 필요")

            # 5b. LIVE vs PAPER 괴리 감지
            if len(live_closes) >= 3 and len(paper_closes) >= 3:
                gap = abs(live_wr - paper_wr)
                if gap > 0.25:
                    adjustments.append(
                        f"LIVE({live_wr:.0%}) vs PAPER({paper_wr:.0%}) 승률 괴리 {gap:.0%} — 실행 차이 점검"
                    )

            # 5c. 특정 시간대 손실 집중
            bad_hours = []
            for h, stats in hour_stats.items():
                total = stats["wins"] + stats["losses"]
                if total >= 3 and stats["losses"] > stats["wins"] * 2:
                    bad_hours.append(f"{h}시({stats['losses']}패/{total}건)")
            if bad_hours:
                adjustments.append(f"손실 집중 시간대: {', '.join(bad_hours)}")

            # 5d. 연속 손실 후 min_confidence 자동 상향
            consecutive_losses = 0
            for t in close_trades[:10]:  # 최근 10건
                if (t.get("pnl") or 0) < 0:
                    consecutive_losses += 1
                else:
                    break
            if consecutive_losses >= 3:
                old_conf = self.strategy_manager.base_min_confidence
                new_conf = min(old_conf + 0.05, 0.60)
                if new_conf != old_conf:
                    self.strategy_manager.base_min_confidence = new_conf
                    adjustments.append(
                        f"연속 {consecutive_losses}패 → min_confidence {old_conf:.2f}→{new_conf:.2f} 상향"
                    )

            # 5e. 승률 좋으면 min_confidence 완화
            if win_rate > 0.65 and len(close_trades) >= 10 and consecutive_losses == 0:
                old_conf = self.strategy_manager.base_min_confidence
                new_conf = max(old_conf - 0.02, 0.35)
                if new_conf != old_conf:
                    self.strategy_manager.base_min_confidence = new_conf
                    adjustments.append(
                        f"승률 {win_rate:.0%} 양호 → min_confidence {old_conf:.2f}→{new_conf:.2f} 완화"
                    )

            # === 6. 리포트 생성 ===
            report_lines = [
                f"🧠 <b>전략 자체 리뷰</b>",
                f"━━━━━━━━━━━━━",
                f"📊 최근 {len(close_trades)}건 | 승률: {win_rate:.0%} | PnL: ${total_pnl:+.2f}",
            ]
            if live_closes:
                report_lines.append(f"🔥 LIVE: {len(live_closes)}건 승률 {live_wr:.0%} PnL ${live_pnl:+.2f}")
            if paper_closes:
                report_lines.append(f"📝 PAPER: {len(paper_closes)}건 승률 {paper_wr:.0%} PnL ${paper_pnl:+.2f}")

            # 전략별 요약
            if strategy_stats:
                report_lines.append(f"\n📐 <b>전략별 성과</b>")
                for strat, stats in sorted(strategy_stats.items(), key=lambda x: x[1]["pnl"]):
                    total = stats["wins"] + stats["losses"]
                    wr = stats["wins"] / total if total > 0 else 0
                    report_lines.append(f"  {strat}: {total}건 승률 {wr:.0%} PnL ${stats['pnl']:+.2f}")

            if adjustments:
                report_lines.append(f"\n🔧 <b>자동 조정</b>")
                for adj in adjustments:
                    report_lines.append(f"  • {adj}")
            else:
                report_lines.append(f"\n✅ 특이 패턴 없음 — 현행 유지")

            # 현재 전략 파라미터
            report_lines.append(
                f"\n⚙️ min_conf: {self.strategy_manager.base_min_confidence:.2f} | "
                f"min_confirm: {self.strategy_manager.min_confirming}"
            )

            report_text = "\n".join(report_lines)
            logger.info(f"[SelfReview] 완료 | {len(close_trades)}건 승률 {win_rate:.0%} PnL ${total_pnl:+.2f} | 조정 {len(adjustments)}건")
            tg_notify(report_text, silent=len(adjustments) == 0)

        except Exception as e:
            logger.error(f"[SelfReview] 실패: {e}")

    def _check_code_version(self) -> str | None:
        """git commit 시간 vs 프로세스 시작 시간 비교"""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ct"],
                capture_output=True, text=True, timeout=5,
                cwd=str(Path(__file__).parent)
            )
            if result.returncode == 0:
                last_commit_ts = int(result.stdout.strip())
                last_commit_time = datetime.utcfromtimestamp(last_commit_ts)
                process_start = self.start_time

                if last_commit_time > process_start:
                    diff = last_commit_time - process_start
                    diff_min = diff.total_seconds() / 60
                    if diff_min > 5:  # 5분 이상 차이
                        msg = (
                            f"코드 버전 불일치! 커밋: {last_commit_time.strftime('%H:%M')} > "
                            f"프로세스: {process_start.strftime('%H:%M')} "
                            f"({diff_min:.0f}분 차이) — 재시작 필요"
                        )
                        logger.warning(f"[진단] {msg}")
                        tg_notify(
                            f"⚠️ <b>코드 버전 불일치</b>\n"
                            f"━━━━━━━━━━━━━\n"
                            f"최신 커밋: {last_commit_time.strftime('%Y-%m-%d %H:%M')}\n"
                            f"프로세스 시작: {process_start.strftime('%Y-%m-%d %H:%M')}\n"
                            f"⏱ {diff_min:.0f}분 차이\n"
                            f"🔄 새 코드 적용 위해 재시작 필요"
                        )
                        return msg
        except Exception:
            pass
        return None

    @staticmethod
    def _calculate_momentum(df) -> dict:
        """가격 모멘텀 시그널 계산 (ML/RL fallback용)"""
        try:
            close = df["close"].values
            if len(close) < 30:
                return {"direction": "neutral", "strength": 0, "rsi": 50, "trend_aligned": False}

            # EMA 크로스오버
            ema_fast = df["close"].ewm(span=8).mean().values
            ema_slow = df["close"].ewm(span=21).mean().values
            ema_trend = df["close"].ewm(span=50).mean().values

            ema_cross = (ema_fast[-1] - ema_slow[-1]) / ema_slow[-1]  # 양수=bullish

            # 최근 N봉 모멘텀
            returns_5 = (close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0
            returns_10 = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0
            returns_20 = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0

            # RSI (14)
            rsi = 50.0
            if "rsi_14" in df.columns:
                rsi_val = df["rsi_14"].iloc[-1]
                if not np.isnan(rsi_val):
                    rsi = float(rsi_val)
            else:
                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-10)
                rsi = float(100 - (100 / (1 + rs.iloc[-1])))

            # 볼륨 스파이크
            vol_avg = df["volume"].rolling(20).mean().iloc[-1] if len(df) >= 20 else df["volume"].mean()
            vol_spike = float(df["volume"].iloc[-1] / (vol_avg + 1e-10))

            # 종합 모멘텀 점수
            momentum_score = (
                ema_cross * 10 * 0.3 +   # EMA 크로스 (큰 값으로 스케일)
                returns_5 * 5 * 0.3 +     # 5봉 수익률
                returns_10 * 3 * 0.2 +    # 10봉 수익률
                returns_20 * 2 * 0.2      # 20봉 수익률
            )

            # 볼륨 스파이크 부스트
            if vol_spike > 1.5:
                momentum_score *= (1 + min((vol_spike - 1.5) * 0.3, 0.5))

            # 방향 결정
            if momentum_score > 0.1:
                direction = "long"
            elif momentum_score < -0.1:
                direction = "short"
            else:
                direction = "neutral"

            # 트렌드 일치 (50 EMA 위/아래)
            trend_aligned = (
                (direction == "long" and close[-1] > ema_trend[-1]) or
                (direction == "short" and close[-1] < ema_trend[-1])
            )

            return {
                "direction": direction,
                "strength": round(float(momentum_score), 4),
                "rsi": round(rsi, 1),
                "trend_aligned": bool(trend_aligned),
                "ema_cross": round(float(ema_cross), 6),
                "returns_5": round(float(returns_5), 6),
                "vol_spike": round(float(vol_spike), 2),
            }
        except Exception as e:
            logger.debug(f"모멘텀 계산 실패: {e}")
            return {"direction": "neutral", "strength": 0, "rsi": 50, "trend_aligned": False}

    def get_positions(self) -> list[dict]:
        positions = []
        if self.mode in ("paper", "dual"):
            for p in self.paper_trader.positions.values():
                positions.append({
                    "mode": "PAPER",
                    "symbol": p.symbol, "side": p.side, "size": p.size,
                    "entry_price": p.entry_price, "unrealized_pnl": p.unrealized_pnl,
                    "stop_loss": p.stop_loss, "take_profit": p.take_profit,
                })
        if self.mode in ("live", "dual"):
            for name, om in self.order_managers.items():
                for sym, pos in getattr(om, "positions", {}).items():
                    positions.append({
                        "mode": "LIVE",
                        "symbol": sym, "side": pos.get("side", "?"),
                        "size": pos.get("size", 0),
                        "entry_price": pos.get("entry_price", 0),
                        "unrealized_pnl": pos.get("unrealized_pnl", 0),
                    })
        return positions

    @property
    def uptime(self):
        return datetime.utcnow() - self.start_time

    async def shutdown(self):
        """종료 처리"""
        self.is_running = False
        logger.info("AutoTrader 종료 중...")
        await self.collector.close()
        for client in self.exchange_clients.values():
            await client.close()
        self.storage.close()
        logger.info("AutoTrader 종료 완료")


async def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    trader = AutoTrader(config_path)

    mode_label = "DUAL (Paper + Live)" if trader.dual_mode else trader.mode.upper()
    logger.info(f"{'='*60}")
    logger.info(f"AutoTrader AI 시작 — 모드: {mode_label}")
    logger.info(f"{'='*60}")

    # 대시보드 상태 연결
    set_state(trader, trader.storage)

    # 대시보드 서버 (별도 스레드)
    dash_config = trader.config.get("dashboard", {})
    dash_thread = threading.Thread(
        target=uvicorn.run,
        args=(dashboard_app,),
        kwargs={"host": dash_config.get("host", "0.0.0.0"),
                "port": dash_config.get("port", 8888),
                "log_level": "warning"},
        daemon=True,
    )
    dash_thread.start()
    logger.info(f"대시보드: http://localhost:{dash_config.get('port', 8888)}")

    # 초기화
    await trader.initialize()

    if trader.dual_mode:
        logger.info("Paper + Live 동시 학습 모드 활성화")
        logger.info(f"  자본: ${trader.initial_capital}")
        logger.info(f"  레버리지: {trader.config['risk']['dynamic_leverage']['base']}x "
                     f"(min {trader.config['risk']['dynamic_leverage']['min']}x ~ "
                     f"max {trader.config['risk']['dynamic_leverage']['max']}x)")
        logger.info(f"  최소 주문: ${trader.min_order_notional} notional")

    # 실행
    try:
        if trader.mode == "backtest":
            await trader.run_backtest()
        else:
            await trader.run_trading_loop()
    except KeyboardInterrupt:
        pass
    finally:
        await trader.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
