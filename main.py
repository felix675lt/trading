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
                self.order_managers[name] = OrderManager(client, self.config["risk"])

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

        # 5. 전략 결정 (ML + RL + 외부 요인 + MTF + 모멘텀 통합)
        current_position = 0.0
        if self.mode in ("paper", "dual"):
            current_position = 1.0 if symbol in self.paper_trader.positions else 0.0
        decision = self.strategy_manager.decide(
            ml_signal, rl_action, rl_confidence, current_position,
            adaptive_params["regime"], external_signal=ext_signal,
            momentum=momentum_signal,
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
                    result = await om.open_position(symbol, decision.action, size, dynamic_lev)
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
                    })
                    self.feedback.record_trade(result, trade_context)
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
                        logger.info(f"[LIVE] 청산 {symbol} | PnL: ${live_pnl:.2f}")
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
                        self.risk_manager.record_pnl(result["pnl"])
                        self.storage.save_trade({
                            "exchange": exchange_name, "symbol": symbol, "side": "close",
                            "price": price, "amount": result["size"], "pnl": result["pnl"],
                            "fee": result["fee"], "strategy": "auto_close",
                        })
                        add_live_log({
                            "time": datetime.utcnow().strftime("%H:%M:%S"),
                            "type": "trade_close",
                            "mode": "PAPER",
                            "symbol": symbol,
                            "pnl": round(result["pnl"], 2),
                            "reason": f"자기수정: {age_minutes:.0f}분 stale 포지션 청산",
                        })
                        logger.info(f"[자기수정] PAPER stale 청산 {symbol} | PnL: ${result['pnl']:.2f} | {age_minutes:.0f}분 보유")
                        tg_notify(format_trade_close("PAPER", symbol, result["pnl"], f"자기수정: stale 청산 ({age_minutes:.0f}분)"), silent=True)

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
