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
from core.strategy.adaptive import AdaptiveOptimizer
from core.strategy.manager import StrategyManager
from dashboard.app import app as dashboard_app, set_state, add_live_log


class AutoTrader:
    """자기학습 선물 트레이딩 시스템 메인 클래스"""

    def __init__(self, config_path: str = "config/default.yaml"):
        load_dotenv()
        self.config = self._load_config(config_path)
        self.mode = self.config["trading"]["mode"]
        self.start_time = datetime.utcnow()

        # 로깅 설정
        log_cfg = self.config.get("logging", {})
        log_file = log_cfg.get("file", "logs/autotrader.log")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, rotation="10 MB", level=log_cfg.get("level", "INFO"))

        # 컴포넌트 초기화
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

        # 페이퍼/실거래 트레이더
        self.paper_trader = PaperTrader(
            initial_capital=self.config.get("backtest", {}).get("initial_capital", 10000),
            commission=self.config.get("backtest", {}).get("commission_pct", 0.0004),
            trailing_config=self.config.get("trailing_stop", {}),
        )
        self.exchange_clients: dict[str, ExchangeClient] = {}
        self.order_managers: dict[str, OrderManager] = {}

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

        # 실거래 모드 시 거래소 클라이언트 초기화
        if self.mode == "live":
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
                # 재학습 체크
                if trainer.should_retrain():
                    add_live_log({
                        "timestamp": datetime.utcnow().isoformat(),
                        "type": "retrain",
                        "message": "자기학습 재훈련 시작",
                    })
                    for symbol in symbols:
                        await trainer.train_cycle(exchange_name, symbol, primary_tf)
                    add_live_log({
                        "timestamp": datetime.utcnow().isoformat(),
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

        # 5. 전략 결정 (ML + RL + 외부 요인 + MTF 통합)
        current_position = 0.0
        if self.mode == "paper":
            current_position = 1.0 if symbol in self.paper_trader.positions else 0.0
        decision = self.strategy_manager.decide(
            ml_signal, rl_action, rl_confidence, current_position,
            adaptive_params["regime"], external_signal=ext_signal,
        )

        # 5.1. MTF 합류 필터 - 상위 타임프레임과 반대면 진입 차단
        if decision.action in ["long", "short"] and mtf_signal.get("higher_tf_conflict"):
            logger.info(f"[MTF] {symbol} 상위 TF 충돌 → 진입 보류")
            decision = self.strategy_manager.decide(
                ml_signal, 0, 0, current_position,
                adaptive_params["regime"], external_signal=ext_signal,
            )  # hold로 전환

        # 5.2. MTF 합류 확인 시 확신도 부스트
        if decision.action in ["long", "short"]:
            mtf_agreement = mtf_signal.get("agreement", 0)
            mtf_dir = mtf_signal.get("direction", "neutral")
            action_agrees_mtf = (
                (decision.action == "long" and mtf_dir == "bullish") or
                (decision.action == "short" and mtf_dir == "bearish")
            )
            if action_agrees_mtf and mtf_agreement > 0.7:
                decision.confidence = min(decision.confidence * 1.15, 1.0)
                decision.reason += " + MTF합류"
            elif not action_agrees_mtf and mtf_dir != "neutral":
                decision.confidence *= 0.8
                decision.reason += f" ! MTF반대({mtf_dir})"

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
        }

        # 가격 히스토리 업데이트 (상관관계 계산용)
        price = float(df["close"].iloc[-1])
        self.risk_manager.update_price_history(symbol, price)

        # 실시간 로그 기록
        add_live_log({
            "timestamp": datetime.utcnow().isoformat(),
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
            "regime": adaptive_params["regime"],
            "reason": decision.reason,
            "features_count": len(feature_cols),
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
        num_positions = len(self.paper_trader.positions) if self.mode == "paper" else 0
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
            current_positions = {}
            if self.mode == "paper":
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
            price = float(df["close"].iloc[-1])

            if self.mode == "paper":
                self.paper_trader.open_position(
                    symbol, decision.action, size, price,
                    leverage=dynamic_lev,
                    sl_pct=self.config["risk"]["stop_loss_pct"] * adaptive_params["stop_loss_mult"],
                    tp_pct=self.config["risk"]["take_profit_pct"],
                )
            elif self.mode == "live":
                om = self.order_managers.get(exchange_name)
                if om:
                    await om.open_position(symbol, decision.action, size, dynamic_lev)

            logger.info(
                f"[{self.mode.upper()}] {decision.action.upper()} {symbol} | "
                f"크기: ${size:.2f} | 레버리지: {dynamic_lev}x | "
                f"사유: {decision.reason}"
            )
            add_live_log({
                "timestamp": datetime.utcnow().isoformat(),
                "type": "trade_open",
                "symbol": symbol,
                "action": decision.action,
                "size": round(size, 2),
                "price": price,
                "leverage": dynamic_lev,
                "reason": decision.reason,
            })

        elif decision.action == "close":
            price = float(df["close"].iloc[-1])
            if self.mode == "paper":
                result = self.paper_trader.close_position(symbol, price, decision.reason)
                if result:
                    self.risk_manager.record_pnl(result["pnl"])
                    self.storage.save_trade({
                        "exchange": exchange_name, "symbol": symbol, "side": "close",
                        "price": price, "amount": result["size"], "pnl": result["pnl"],
                        "fee": result["fee"], "strategy": "hybrid",
                    })
                    add_live_log({
                        "timestamp": datetime.utcnow().isoformat(),
                        "type": "trade_close",
                        "symbol": symbol,
                        "pnl": round(result["pnl"], 2),
                        "reason": decision.reason,
                    })
                    # 피드백 기록 → 자기학습 (외부 신호 정보 포함)
                    volatility = df["returns_1"].std() if "returns_1" in df.columns else 0
                    self.feedback.record_trade(result, {
                        "regime": adaptive_params["regime"],
                        "signal": ml_signal.get("signal", 0),
                        "confidence": decision.confidence,
                        "volatility": volatility,
                        "external_score": ext_signal.get("score", 0),
                        "external_direction": ext_signal.get("direction", "neutral"),
                    })
            elif self.mode == "live":
                om = self.order_managers.get(exchange_name)
                if om:
                    await om.close_position(symbol, decision.reason)

        # 페이퍼 트레이더 가격 업데이트
        if self.mode == "paper":
            price = float(df["close"].iloc[-1])
            self.paper_trader.update_prices({symbol: price})
            self.equity = self.paper_trader.equity
            self.total_pnl = self.equity - self.initial_capital

    def get_positions(self) -> list[dict]:
        if self.mode == "paper":
            return [
                {
                    "symbol": p.symbol, "side": p.side, "size": p.size,
                    "entry_price": p.entry_price, "unrealized_pnl": p.unrealized_pnl,
                    "stop_loss": p.stop_loss, "take_profit": p.take_profit,
                }
                for p in self.paper_trader.positions.values()
            ]
        return []

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

    # 대시보드 상태 연결
    set_state(trader, trader.storage)

    # 대시보드 서버 (별도 스레드)
    dash_config = trader.config.get("dashboard", {})
    dash_thread = threading.Thread(
        target=uvicorn.run,
        args=(dashboard_app,),
        kwargs={"host": dash_config.get("host", "0.0.0.0"), "port": dash_config.get("port", 8888), "log_level": "warning"},
        daemon=True,
    )
    dash_thread.start()
    logger.info(f"대시보드: http://localhost:{dash_config.get('port', 8888)}")

    # 초기화
    await trader.initialize()

    # 모드별 실행
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
