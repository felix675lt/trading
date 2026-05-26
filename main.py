"""AutoTrader AI - 자기학습 선물 트레이딩 시스템 메인 실행"""

import asyncio
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
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
from core.quant_signals import QuantSignals
from core.capital_tiers import CapitalTierManager
from core.treasury.btc_reserve import BTCReserve
from core.learning.ic_tracker import ICTracker, SignalWeightOptimizer
from core.learning.meta_labeler import MetaLabeler
from core.strategy.regime_hmm import HMMRegimeClassifier
from core.strategy.cointegration import CointegrationTester
from core.strategy.pairs_trading import PairsTradingStrategy
from core.portfolio.hrp import HRPAllocator
from core.execution.smart_router import SmartRouter
from dashboard.app import app as dashboard_app, set_state, add_live_log

# Telegram 알림 (실패해도 트레이딩에 영향 없음)
try:
    from scripts.telegram_bot import (
        send_message as tg_send, format_trade_open, format_trade_close,
        format_system_alert, format_external_alert,
    )
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
    def format_external_alert(alert_type, data):
        return f"📢 {alert_type}: {data}"

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
        # [Patch K+, 2026-04-28] retention/compression 추가 — 1개월 무인 운영 시 디스크 보호
        logger.add(
            log_file,
            rotation="10 MB",
            retention="14 days",   # 14일 이상 옛 로그 자동 삭제
            compression="gz",      # 회전된 로그 gz 압축 (10MB → ~1MB)
            level=log_cfg.get("level", "INFO"),
        )

        # 컴포넌트 초기화 (한 번만)
        self.storage = Storage()
        self.feature_engineer = FeatureEngineer(self.config.get("ml", {}).get("features"))
        # [Patch M, 2026-04-28] Pattern Memory Bank — Retrieval-Augmented Trading (Phase 1)
        # 사용자 통찰: 데이터를 백업해서 보존했으니, ML 모델로 압축 외울 필요 없이
        # 필요할 때 직접 retrieve. ML 모델은 보조 역할로 작게 유지.
        # Phase 1: shadow mode (로그만 출력, 결정에 영향 없음 — 신호 품질 검증 후 Phase 2에서 통합)
        self.pattern_banks: dict = {}  # symbol → PatternMemoryBank
        self.ensemble = EnsembleSignalGenerator()
        self.rl_agent = RLAgent(self.config.get("rl", {}))
        self.risk_manager = RiskManager(self.config["risk"])
        self.strategy_manager = StrategyManager(
            self.config.get("trading", {}),
            trade_profiles=self.config.get("trade_profiles", {}),
        )
        self.adaptive = AdaptiveOptimizer()
        # === Feedback — A/B variant 격리 (2026-04-21) ===
        # self.feedback: MACRO_ON 기본 정책 — data/feedback_history_macro_on.json
        # self.feedback_off: MACRO_OFF 섀도우 정책 — data/feedback_history_macro_off.json
        # 각 분기의 losing_streak/regime_scale/direction_scale 조정이 상대에 오염되지 않음.
        # LIVE는 MACRO_ON 기본 정책과 동일 feedback 사용 (LIVE는 MACRO_ON 결과로 운영).
        self.feedback = TradeFeedbackAnalyzer(variant="macro_on")
        self.feedback_off = TradeFeedbackAnalyzer(variant="macro_off")
        self.anomaly_detector = AnomalyDetector()

        # 외부 데이터 매니저 (뉴스/센티먼트/온체인/매크로/공포탐욕)
        self.external_manager = ExternalDataManager(self.config.get("external", {}))

        # === Capital Tier System ===
        # 시드 구간별 기능 자동 활성화. PAPER는 가상 시드로 상위 티어 선행 검증.
        self.tier_manager = CapitalTierManager(self.config)

        # PAPER 초기 자본 = 가상 시드 (상위 티어 기능 검증용)
        # LIVE 잔고는 별도(order_manager가 실 거래소에서 조회)
        paper_initial = self.tier_manager.paper_virtual_seed

        # === A/B 듀얼북 Paper 트레이더 (2026-04-21) ===
        # 같은 시장 입력을 두 정책(MACRO_ON / MACRO_OFF)으로 병렬 실행 → 통계적 비교.
        # 수학적 격리 원칙:
        #   1) 자본(equity)/포지션 완전 분리 — capital contamination 없음
        #   2) feedback(학습 이력) 분리 — 상호 간섭 없음
        #   3) storage 저장 시 variant 태그 — 표본 오염 없음
        #   4) 동일 가상시드에서 시작 — 초기 조건 편향 없음
        #   5) 동일 slippage/commission 모델 — 실행 편향 없음
        # 주의: paper_trader(MACRO_ON)는 기본/운영 기준 — BTC reserve/LIVE 연계도 이쪽 사용.
        self.paper_trader = PaperTrader(
            initial_capital=paper_initial,
            commission=self.config.get("backtest", {}).get("commission_pct", 0.0004),
            trailing_config=self.config.get("trailing_stop", {}),
            trade_profiles=self.config.get("trade_profiles", {}),
            variant="PAPER_MACRO_ON",
        )
        # 섀도우 variant — 같은 조건에서 macro_block OFF로 실행 → 순수 정책 A/B
        self.paper_trader_off = PaperTrader(
            initial_capital=paper_initial,
            commission=self.config.get("backtest", {}).get("commission_pct", 0.0004),
            trailing_config=self.config.get("trailing_stop", {}),
            trade_profiles=self.config.get("trade_profiles", {}),
            variant="PAPER_MACRO_OFF",
        )
        # PAPER 자동청산 콜백 (SL/TP/트레일링 → DB 저장 + 학습)
        # 콜백은 trade["variant"]로 분기해서 올바른 feedback/optimizer에 기록
        self.paper_trader.set_auto_close_callback(self._on_paper_auto_close)
        self.paper_trader_off.set_auto_close_callback(self._on_paper_auto_close)

        # === BTC Treasury Reserve (2026-04-18) ===
        # 선물 실현수익의 일부를 현물 BTC로 자동 적립. 티어별 적립률 적용.
        # collector는 initialize()에서 생성되므로 여기선 None으로 만들고 나중에 주입.
        self.btc_reserve = BTCReserve(
            config=self.config,
            tier_manager=self.tier_manager,
            collector=None,  # initialize()에서 주입
        )
        # PAPER 청산 → 가상 BTC 적립 (sync 콜백)
        self.paper_trader.set_profit_callback(self.btc_reserve.on_paper_close)

        self.exchange_clients: dict[str, ExchangeClient] = {}
        self.order_managers: dict[str, OrderManager] = {}
        # Binance spot 클라이언트 (BTC Reserve 실매수용) — initialize()에서 생성
        self.spot_exchange_client = None

        # StrategyOptimizer — Paper/Live 각각 독립 추적 + PAPER_OFF 변종
        self.strategy_optimizer_paper = StrategyOptimizer()
        self.strategy_optimizer_paper_off = StrategyOptimizer()
        self.strategy_optimizer_live = StrategyOptimizer()

        # 퀀트 시그널 (오더북, VPIN, 베이시스, 크래시보호, 알파, 레짐)
        self.quant_signals = QuantSignals()

        # === 고급 퀀트 모듈 (tier 기반 활성화) ===
        # IC Tracker — 시그널 품질 추적 (always-on, 저비용)
        self.ic_tracker = ICTracker()
        # 레짐별 시그널 가중치 자동화 (2026-04-24 C)
        # ML/RL/MOM/RSI_extreme/EXT/BREAKOUT 각 소스의 (regime, source) IC를
        # 기반으로 vote weight multiplier를 동적 조정. 1h 주기 갱신.
        self.signal_weight_opt = SignalWeightOptimizer(
            min_samples=20, smoothing=0.25, mult_min=0.3, mult_max=1.5,
        )
        # StrategyManager에 주입 — _count_signal_votes()가 getattr로 꺼내 씀
        try:
            self.strategy_manager.signal_weight_optimizer = self.signal_weight_opt
        except Exception:
            pass
        # HMM Regime Classifier — tier=large+ 활성 (학습/추론은 별도 트리거)
        self.hmm_regime = HMMRegimeClassifier(n_states=3)
        self.hmm_regime.load()  # 있으면 로드
        # Meta-Labeler — tier=large+ 활성 (학습은 trainer에서)
        self.meta_labeler = MetaLabeler()
        self.meta_labeler.load()
        # HRP Allocator — tier=pro 활성 (포트폴리오 가중치 산출)
        self.hrp = HRPAllocator()
        # Cointegration + Pairs Trading — tier=pro 활성
        self.coint_tester = CointegrationTester()
        self.pairs_strategy = PairsTradingStrategy(
            exchange=None,  # 첫 번째 exchange 주입됨 (runtime)
            coint_tester=self.coint_tester,
        )
        # Smart Router — tier=pro 활성 (여러 거래소 주입 시)
        self.smart_router: SmartRouter | None = None
        # 일일 pairs 탐색 타이머
        self._last_pairs_discovery: datetime | None = None
        self._last_hmm_fit: datetime | None = None
        self._last_ic_log: datetime | None = None

        # 최소 주문 notional (거래소 최소수량 충족용)
        self.min_order_notional = self.config["risk"].get("min_order_notional", 100)

        # 상태 — PAPER 가상 시드 기준 (dual/paper 모드에서 포트폴리오 지표 산출용)
        self.equity = paper_initial
        self.initial_capital = paper_initial
        self.live_equity: float = 0.0  # 실 거래소 잔고 (LIVE 루프에서 업데이트)
        self.total_pnl = 0.0
        self.last_signals = {}
        self.last_external = {}
        self.is_running = False

        # 자가진단 LIVE 일시정지 상태
        self._live_paused = False
        self._live_pause_reason = ""
        self._live_pause_time = None

        # [2026-04-21] 일일 MaxDD 하드캡 상태 — LIVE 전용 (PAPER는 학습 위해 무제한 지속)
        self._daily_dd_paused = False
        self._daily_dd_reason = ""
        self._daily_dd_trigger_time = None
        self._daily_dd_threshold_pct = -3.0  # LIVE 일 누적 PnL / 초기자본 < -3% → 24h LIVE 정지

        # [Patch C, 2026-04-26] LIVE EV (Expected Value) 자동 정지 상태 — LIVE 전용
        # 무인 1개월 운영 안전장치: 최근 50 LIVE 거래 EV<0이면 LIVE만 무기한 정지.
        # 시간 기반 자동 재개 없음 — EV>0 회복(또는 모델 재학습 후 신규 신호) 시에만 해제.
        # PAPER는 영향 없음 (학습 데이터 계속 수집).
        self._live_ev_paused = False
        self._live_ev_pause_reason = ""
        self._live_ev_pause_time = None
        self._live_ev_lookback = 50    # 최근 N LIVE 트레이드 (HIPPO 제외)
        self._live_ev_threshold = -0.5  # USD/trade — EV < -0.5$ 면 정지 (잡음 마진)
        self._live_ev_min_samples = 20  # 표본 N개 이상일 때만 판단

        # 외부 요인 알림 상태 추적 (공포탐욕 제거됨)
        self._ext_alert_state = {
            "last_composite_score": 0.0,
            "last_macro_score": 0.0,
            "last_oil_price": 0.0,
            "last_dxy": 0.0,
            "last_vix": 0.0,
            "last_sentiment": 0.0,
            "last_alert_time": {},       # alert_type → timestamp
            "alerted_headlines": set(),  # 이미 알린 뉴스 제목
        }

    def _check_external_alerts(self):
        """외부 요인 변동 감지 → 텔레그램 알림 발송"""
        try:
            st = self._ext_alert_state
            now = datetime.utcnow()

            # 쿨다운 체크 함수 (같은 유형 알림 최소 30분 간격)
            def _can_alert(alert_type, cooldown_min=30):
                last = st["last_alert_time"].get(alert_type)
                if last and (now - last).total_seconds() < cooldown_min * 60:
                    return False
                return True

            def _mark_alerted(alert_type):
                st["last_alert_time"][alert_type] = now

            # === 1. 매크로 지표 급변 (유가/DXY/VIX) ===
            rm = self.external_manager.real_macro
            rm_sig = rm.get_signal()
            rm_score = rm.get_features().get("real_macro_composite_score", 0)
            oil = rm_sig.get("oil", 0)
            dxy = rm_sig.get("dxy", 0)
            vix = rm_sig.get("vix", 0)
            gold = rm_sig.get("gold", 0)

            # 유가 5%+ 급변
            if oil > 0 and st["last_oil_price"] > 0 and _can_alert("oil_move"):
                oil_change = (oil - st["last_oil_price"]) / st["last_oil_price"]
                if abs(oil_change) >= 0.03:  # 3%+
                    brent = rm.data.get("brent_oil_price", 0)
                    impl = ""
                    if oil_change < -0.03:
                        impl = "유가 하락 → 지정학 긴장 완화 or 수요 부진. 크립토 불확실성 감소 가능"
                    elif oil_change > 0.05:
                        impl = "유가 급등 → 지정학 리스크 확대. 위험자산 회피 가능"
                    elif oil_change > 0.03:
                        impl = "유가 상승 → 인플레이션 우려. 금리인하 기대 약화 가능"
                    tg_notify(format_external_alert("oil_move", {
                        "price": oil, "change": oil_change, "brent": brent,
                        "implication": impl,
                    }))
                    _mark_alerted("oil_move")

            # DXY 0.5%+ 급변
            if dxy > 0 and st["last_dxy"] > 0 and _can_alert("dxy_move"):
                dxy_change = (dxy - st["last_dxy"]) / st["last_dxy"]
                if abs(dxy_change) >= 0.004:  # 0.4%+
                    tg_notify(format_external_alert("dxy_move", {
                        "dxy": dxy, "change": dxy_change,
                    }))
                    _mark_alerted("dxy_move")

            # VIX 급등 (25+ 이상 진입 or 30+ 돌파)
            if vix > 0 and _can_alert("vix_spike", cooldown_min=60):
                if vix >= 30 and st.get("last_vix", 0) < 30:
                    tg_notify(format_external_alert("macro_shift", {
                        "score": rm_score,
                        "changes": [
                            f"⚠️ VIX {vix:.1f} — 극단적 공포 구간 진입",
                            f"  WTI: ${oil:.1f} | Gold: ${gold:.0f} | DXY: {dxy:.1f}",
                            f"  💡 VIX 30+ = 시장 패닉. 단기 변동성 극대화 주의",
                        ],
                    }))
                    _mark_alerted("vix_spike")
                elif vix >= 25 and (st.get("last_vix", 0) < 25 or st.get("last_vix", 0) == 0):
                    tg_notify(format_external_alert("macro_shift", {
                        "score": rm_score,
                        "changes": [
                            f"🟡 VIX {vix:.1f} — 경계 구간 진입",
                            f"  WTI: ${oil:.1f} | Gold: ${gold:.0f} | DXY: {dxy:.1f}",
                        ],
                    }))
                    _mark_alerted("vix_spike")

            # 매크로 종합 점수 큰 변화 (0.3+ 변동)
            if abs(rm_score - st["last_macro_score"]) >= 0.25 and _can_alert("macro_shift"):
                changes = []
                if oil > 0:
                    changes.append(f"🛢️ WTI: ${oil:.1f}")
                if gold > 0:
                    changes.append(f"🥇 Gold: ${gold:.0f}")
                if dxy > 0:
                    changes.append(f"💵 DXY: {dxy:.1f}")
                if vix > 0:
                    changes.append(f"😰 VIX: {vix:.1f}")

                sp_chg = rm.data.get("sp500_change_1d", 0)
                nq_chg = rm.data.get("nasdaq_change_1d", 0)
                if sp_chg:
                    changes.append(f"📈 S&P500: {sp_chg:+.1%}")
                if nq_chg:
                    changes.append(f"📈 나스닥: {nq_chg:+.1%}")

                tg_notify(format_external_alert("macro_shift", {
                    "score": rm_score, "changes": changes,
                }))
                _mark_alerted("macro_shift")

            # === 2. 뉴스 기반 알림 — 쿨다운 없음 (속보는 즉시 전달) ===
            # 중복 방지는 alerted_headlines(제목 기반)으로만 처리
            news_list = self.external_manager.news.news
            geo_headlines = []
            crypto_headlines = []

            GEO_KEYWORDS = [
                "war", "iran", "sanctions", "tariff", "missile", "ceasefire",
                "peace", "invasion", "nuclear", "troops", "military",
                "oil price", "opec", "embargo", "strait", "hormuz",
                "conflict", "airstrike", "escalat", "de-escalat",
                "negotiate", "deal", "treaty", "withdraw",
                "oman", "houthi", "yemen", "hezbollah", "gaza", "israel",
                "taiwan", "china", "nato", "russia", "ukraine",
                "oil surge", "oil crash", "crude", "brent",
                "protocol", "shipping", "blockade", "naval",
                # 한국어 키워드 (한국 뉴스 소스용)
                "전쟁", "제재", "미사일", "휴전", "평화", "핵",
                "호르무즈", "이란", "유가", "원유", "관세",
            ]
            CRYPTO_IMPACT_KEYWORDS = [
                "etf approved", "etf approval", "etf rejected", "etf denied",
                "sec", "ban", "regulation", "crackdown",
                "hack", "hacked", "exploit", "vulnerability",
                "bankruptcy", "bankrupt", "liquidation", "insolvent",
                "rate cut", "rate hike", "fed", "fomc", "powell",
                "bitcoin reserve", "stablecoin bill", "crypto bill",
                "whale", "dump", "crash", "surge", "soar", "plunge",
                "delisting", "listing", "blackrock",
            ]

            for n in news_list:  # 전체 뉴스 스캔 ([:50] 제한 제거 — 지정학 뉴스가 뒤에 있을 수 있음)
                title = n.get("title", "")
                title_lower = title.lower()
                title_key = title[:80]

                # 이미 알린 뉴스는 건너뛰기 (같은 제목 중복만 방지)
                if title_key in st["alerted_headlines"]:
                    continue

                # 지정학 키워드 감지
                geo_hit = [kw for kw in GEO_KEYWORDS if kw in title_lower]
                if geo_hit:
                    geo_headlines.append(f"{title} [{', '.join(geo_hit[:3])}]")
                    st["alerted_headlines"].add(title_key)

                # 크립토 직접 영향 키워드 감지
                crypto_hit = [kw for kw in CRYPTO_IMPACT_KEYWORDS if kw in title_lower]
                if crypto_hit:
                    crypto_headlines.append(f"{title} [{', '.join(crypto_hit[:3])}]")
                    st["alerted_headlines"].add(title_key)

            # 지정학 뉴스 → 즉시 발송 (건별)
            if geo_headlines:
                geo_risk = rm.get_features().get("real_macro_geo_risk", 0)
                tg_notify(format_external_alert("geopolitical", {
                    "events": geo_headlines[:8],
                    "geo_risk": geo_risk,
                }))

            # 크립토 영향 뉴스 → 즉시 발송 (건별)
            if crypto_headlines:
                tg_notify(format_external_alert("breaking_news", {
                    "headlines": crypto_headlines[:8],
                    "impact": "high",
                }))

            # 알림 히스토리 정리 (너무 커지면 오래된 것 제거)
            if len(st["alerted_headlines"]) > 500:
                st["alerted_headlines"] = set(list(st["alerted_headlines"])[-300:])

            # === 3. 종합 신호 급변 (방향 전환) ===
            cs = self.external_manager.composite_signal
            new_cs_score = cs.get("score", 0)
            old_cs_score = st["last_composite_score"]

            # 방향 전환 감지 (음→양 or 양→음, 0.15+ 변동)
            if _can_alert("composite_shift", cooldown_min=60):
                direction_changed = (old_cs_score < -0.05 and new_cs_score > 0.05) or \
                                    (old_cs_score > 0.05 and new_cs_score < -0.05)
                big_move = abs(new_cs_score - old_cs_score) >= 0.15

                if direction_changed or big_move:
                    tg_notify(format_external_alert("composite_shift", {
                        "old_score": old_cs_score,
                        "new_score": new_cs_score,
                        "components": cs.get("components", {}),
                    }))
                    _mark_alerted("composite_shift")

            # === 4. [제거됨] 공포탐욕 극단값 — 후행 지표라 알림 가치 없음 ===

            # === 5. 센티먼트 급변 ===
            sent = self.external_manager.all_features.get("sentiment_avg", 0)
            if abs(sent - st["last_sentiment"]) >= 0.20 and _can_alert("sentiment_shift", cooldown_min=60):
                details = ""
                if sent > st["last_sentiment"]:
                    details = "뉴스/소셜 긍정 심리 급증. 매수 심리 강화 가능"
                else:
                    details = "뉴스/소셜 부정 심리 급증. 매도 압력 증가 가능"
                tg_notify(format_external_alert("sentiment_shift", {
                    "old_score": st["last_sentiment"],
                    "new_score": sent,
                    "details": details,
                }))
                _mark_alerted("sentiment_shift")

            # 상태 업데이트
            st["last_composite_score"] = new_cs_score
            st["last_macro_score"] = rm_score
            if oil > 0:
                st["last_oil_price"] = oil
            if dxy > 0:
                st["last_dxy"] = dxy
            if vix > 0:
                st["last_vix"] = vix
            st["last_sentiment"] = sent

        except Exception as e:
            logger.debug(f"[ExtAlert] 외부 알림 체크 실패: {e}")

    def _save_trade_with_context(self, trade: dict, context: dict = None):
        """save_trade 래퍼 — 퀀트 시그널/레짐/확신도 등 메타데이터 자동 포함

        [2026-04-21] A/B variant 기본값 자동 설정:
            - mode=LIVE → variant="LIVE"
            - mode=PAPER, variant 미지정 → "PAPER_MACRO_ON" (기본 정책)
            - mode=PAPER, variant="PAPER_MACRO_OFF" → 섀도우 variant
        호출부가 trade["variant"]를 명시하면 그대로 사용.
        """
        # A/B variant 기본값 — 표본 분류의 신뢰성을 위해 반드시 채움
        if "variant" not in trade or not trade.get("variant"):
            mode_upper = str(trade.get("mode", "")).upper()
            if mode_upper == "LIVE":
                trade["variant"] = "LIVE"
            elif mode_upper == "PAPER":
                trade["variant"] = "PAPER_MACRO_ON"
            else:
                trade["variant"] = ""
        meta = {}
        if context:
            meta.update(context)
        # 현재 퀀트 상태 자동 수집
        try:
            ext_feats = self.feature_engineer.external_features
            meta["quant_score"] = ext_feats.get("quant_score", 0)
            meta["quant_risk_scale"] = ext_feats.get("quant_risk_scale", 1.0)
            meta["quant_ob_imbalance"] = ext_feats.get("quant_ob_imbalance", 0)
            meta["quant_vpin"] = ext_feats.get("quant_vpin", 0)
            meta["quant_basis_pct"] = ext_feats.get("quant_basis_pct", 0)
            meta["quant_crash_prob"] = ext_feats.get("quant_crash_prob", 0)
            meta["quant_alpha_score"] = ext_feats.get("quant_alpha_score", 0)
            meta["regime"] = self.adaptive.current_regime if hasattr(self, 'adaptive') else "?"
        except Exception:
            pass
        trade["metadata"] = meta
        self.storage.save_trade(trade)

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

        # BTC Reserve에 collector 주입 (가격 조회용)
        self.btc_reserve.collector = self.collector
        # Telegram 알림 주입
        try:
            self.btc_reserve.set_notifier(tg_notify)
        except Exception:
            pass

        # 실거래 모드 시 거래소 클라이언트 초기화 (live 또는 dual)
        if self.mode in ("live", "dual"):
            for name, cfg in self.config["exchanges"].items():
                client = ExchangeClient(name, cfg)
                self.exchange_clients[name] = client
                self.order_managers[name] = OrderManager(
                    client, self.config["risk"],
                    trailing_config=self.config.get("trailing_stop", {}),
                    trade_profiles=self.config.get("trade_profiles", {}),
                )
                # LIVE 청산 → BTC 적립 콜백 주입 (async — 내부에서 create_task)
                self.order_managers[name].set_profit_callback(self.btc_reserve.on_live_close)

            # === Binance Spot 클라이언트 (BTC Reserve 실매수용) ===
            # 선물(defaultType=future)과 분리된 spot 모드로 별도 ExchangeClient 생성.
            # 같은 API 키 사용 — Binance는 단일 키로 spot/futures 모두 가능.
            binance_cfg = self.config["exchanges"].get("binance")
            if binance_cfg and self.config.get("treasury", {}).get("btc_reserve", {}).get("enabled", False):
                spot_cfg = dict(binance_cfg)
                spot_options = dict(binance_cfg.get("options", {}) or {})
                spot_options["defaultType"] = "spot"
                spot_cfg["options"] = spot_options
                try:
                    self.spot_exchange_client = ExchangeClient("binance", spot_cfg)
                    self.btc_reserve.set_spot_exchange(self.spot_exchange_client)
                    logger.info("[BTCReserve] Binance Spot 클라이언트 생성 완료 — LIVE 실매수 활성")
                except Exception as e:
                    logger.warning(f"[BTCReserve] Spot 클라이언트 생성 실패 → 가상 기록만: {e}")

            # Smart Router (tier=pro) — 다거래소 있으면 자동 주입
            if len(self.exchange_clients) >= 2:
                self.smart_router = SmartRouter(
                    exchanges=dict(self.exchange_clients),
                    taker_fees={n: self.config["exchanges"][n].get("taker_fee", 0.0004)
                                for n in self.exchange_clients},
                )
                logger.info(f"[SmartRouter] 활성화 — {list(self.exchange_clients.keys())}")

            # Pairs Strategy에 첫 번째 exchange 주입
            if self.exchange_clients:
                self.pairs_strategy.exchange = next(iter(self.exchange_clients.values()))

        # 기존 모델 로드 시도
        if self.ensemble.load_all():
            logger.info("기존 ML 모델 로드 성공")
        if self.rl_agent.load():
            logger.info("기존 RL 모델 로드 성공")

        # [Patch D, 2026-04-26] 부팅 직후 스키마 검증 — silent breakage 조기 발견
        try:
            schema_issues = self._check_schema_health(sample=True)
            if schema_issues:
                logger.warning(f"[Schema-Boot] 발견된 문제 {len(schema_issues)}건")
                try:
                    body = "\n".join(f"  • {it}" for it in schema_issues)
                    tg_notify(
                        f"⚠️ <b>부팅 스키마 검증 경고 (Patch D)</b>\n"
                        f"━━━━━━━━━━━━━\n{body}\n"
                        f"📝 학습/거래 계속 진행 (자동 재학습 사이클이 정정)"
                    )
                except Exception:
                    pass
            else:
                logger.info("[Schema-Boot] ✅ 모델/DB/피처 스키마 정합")
        except Exception as e:
            logger.debug(f"[Schema-Boot] 검증 호출 실패: {e}")

        # 외부 데이터 초기 수집
        if self.external_manager.enabled:
            # Storage 주입 — 통찰 #3: 파생 스냅샷 시계열 누적용 (2026-04-20)
            try:
                self.external_manager.set_storage(self.storage)
                logger.info("[External] Storage 주입 완료 — derivatives_snapshots 시계열 누적 활성")
            except Exception as e:
                logger.debug(f"[External] set_storage 실패: {e}")
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
                self._save_trade_with_context({
                    "exchange": "binance", "symbol": symbol, "side": result.get("side", "close"),
                    "price": result.get("exit_price", 0),
                    "amount": result.get("size", 0),
                    "pnl": pnl, "fee": result.get("fee", 0),
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

                # 손실 시 즉시 원인분석 리포트
                if pnl < 0:
                    self._generate_loss_report(result, mode="LIVE")
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

        # [Patch M, 2026-04-28] Pattern Memory Bank 디스크 로드 (있으면)
        try:
            from core.patterns.memory_bank import PatternMemoryBank
            bank_dir = Path("data/pattern_bank")
            if bank_dir.exists():
                loaded = 0
                for npz_file in sorted(bank_dir.glob("*.npz")):
                    try:
                        bank = PatternMemoryBank.load(npz_file)
                        # 파일명에서 symbol 복원 (BTC_USDT_USDT_5m.npz → BTC/USDT:USDT)
                        sym_part = npz_file.stem.rsplit("_", 1)[0]
                        # _USDT_USDT → /USDT:USDT 패턴
                        sym = sym_part.replace("_USDT_USDT", "/USDT:USDT")
                        self.pattern_banks[sym] = bank
                        loaded += 1
                    except Exception as e:
                        logger.debug(f"[PatternBank] {npz_file.name} 로드 실패: {e}")
                if loaded > 0:
                    total_patterns = sum(len(b.embeddings) for b in self.pattern_banks.values())
                    logger.info(
                        f"[PatternBank] {loaded}개 심볼 인덱스 로드 — 총 {total_patterns:,}개 패턴 "
                        f"(retrieval-augmented Phase 1 shadow mode 활성)"
                    )
        except Exception as e:
            logger.debug(f"[PatternBank] 초기화 실패 (무시): {e}")

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
        # === 심볼 유니버스: 티어 기반 (PAPER+LIVE 합집합) ===
        # 데이터 수집/분석은 union으로 수행. 실제 주문은 각 mode의 티어에서 허용된 심볼만.
        # 티어 심볼이 있으면 티어, 없으면 config.trading.symbols fallback.
        tier_symbols = self.tier_manager.union_symbols()
        if tier_symbols:
            symbols = tier_symbols
            logger.info(
                f"[CapitalTier] 심볼 유니버스 ({len(symbols)}): {symbols} | "
                f"LIVE 티어={self.tier_manager.get_tier('live').name} "
                f"PAPER 티어={self.tier_manager.get_tier('paper').name}"
            )
        else:
            symbols = self.config["trading"]["symbols"]
        timeframes = self.config["trading"]["timeframes"]  # 멀티타임프레임
        primary_tf = timeframes[0]  # 메인 타임프레임 (5m)

        # 자기학습 트레이너 (tier_manager + meta/hmm 주입 → Walk-Forward CV + Meta/HMM 자동 학습)
        trainer = SelfLearningTrainer(
            self.collector, self.storage, self.ensemble, self.rl_agent, self.config,
            tier_manager=self.tier_manager,
            meta_labeler=self.meta_labeler,
            hmm_regime=self.hmm_regime,
        )

        # 모델이 없으면 초기 학습
        if not self.ensemble.load_all():
            logger.info("모델 없음 - 초기 학습 시작")
            for symbol in symbols:
                await trainer.train_cycle(exchange_name, symbol, primary_tf)

        loop_count = 0

        while self.is_running:
            try:
                # === 자본 티어 업데이트 (매 루프) ===
                # LIVE 잔고 조회 (실 거래소) + PAPER equity → 티어 결정
                live_eq = 0.0
                try:
                    om = self.order_managers.get(exchange_name)
                    if om and self.mode in ("live", "dual"):
                        bal = await om.exchange.get_balance()
                        live_eq = float(bal.get("total", bal.get("free", 0)) or 0)
                        self.live_equity = live_eq
                except Exception as e:
                    logger.debug(f"[CapitalTier] LIVE 잔고 조회 실패: {e}")

                tier_changes = self.tier_manager.update(
                    live_equity=live_eq,
                    paper_equity=self.paper_trader.equity,
                )
                if tier_changes:
                    # 티어 변경 시 텔레그램 알림
                    for which, (old, new) in tier_changes.items():
                        if old is not None:  # 최초 init 제외
                            tg_notify(
                                f"🎯 <b>자본 티어 변경 ({which.upper()})</b>\n"
                                f"━━━━━━━━━━━━━\n"
                                f"{old} → <b>{new}</b>\n"
                                f"설명: {self.tier_manager.get_tier(which).description}\n"
                                f"활성 기능: {list(self.tier_manager.get_tier(which).features.keys())}",
                                silent=False,
                            )

                # === 티어 기반 주문 라우팅 적용 (매 루프, idempotent) ===
                # order_routing: market_only / limit_first / twap / smart
                paper_routing = self.tier_manager.get_feature(
                    "order_routing", mode="paper", default="market_only"
                )
                live_routing = self.tier_manager.get_feature(
                    "order_routing", mode="live", default="market_only"
                )
                # PaperTrader (limit_first 시뮬만 가능 — TWAP/smart는 live 전용)
                self.paper_trader.set_routing(
                    limit_first=(paper_routing in ("limit_first", "twap", "smart"))
                )
                # OrderManager (LIVE) — mode별 세부 설정 전달
                for om_obj in self.order_managers.values():
                    om_obj.set_routing(
                        mode=live_routing,
                        limit_first=(live_routing in ("limit_first", "twap", "smart")),
                        wait_seconds=20,
                        offset_pct=0.0001,
                        twap_slices=5,
                        twap_duration_s=60,
                        smart_router=getattr(self, "smart_router", None),
                    )

                # === LIVE → Paper 피드백: 실측 maker 체결률을 Paper에 주입 ===
                # 샘플 충분(≥5)하면 실측값으로 덮어쓰기, 부족하면 기본값 유지
                try:
                    rates = []
                    for om_obj in self.order_managers.values():
                        r = om_obj.get_maker_fill_rate()
                        if r is not None:
                            rates.append(r)
                    if rates:
                        self.paper_trader.set_maker_fill_rate(sum(rates) / len(rates))
                except Exception as e:
                    logger.debug(f"[Paper-Feedback] maker_fill_rate 동기화 실패: {e}")

                # === BTC 현재가 hint → BTC Reserve (cost basis/미실현 PnL 계산용) ===
                try:
                    btc_ticker = await self.collector.fetch_ticker(
                        exchange_name, "BTC/USDT:USDT",
                    )
                    btc_price = float(btc_ticker.get("last") or 0)
                    if btc_price > 0:
                        self.btc_reserve.set_btc_price_hint(btc_price)
                except Exception as e:
                    logger.debug(f"[BTCReserve] BTC 가격 hint 실패: {e}")

                # 재학습 체크 (일반 + stuck 감지 + SmartScheduler 보완 게이트)
                diag = self.strategy_manager.get_diagnostics()
                # SmartScheduler 통합: 24h 게이트 + perf_decline + regime_changed 등
                # current_accuracy: 현재 ensemble XGB 정확도 (있으면)
                # regime_changed: 직전 사이클 대비 레짐 변화 여부
                _cur_acc = float(getattr(self.ensemble.xgb, "accuracy", 0.0) or 0.0)
                _regime_changed = bool(getattr(self, "_last_regime_changed", False))
                try:
                    needs_retrain, retrain_reason = trainer.should_retrain_smart(
                        current_accuracy=_cur_acc,
                        regime_changed=_regime_changed,
                    )
                    if needs_retrain:
                        logger.info(f"[재학습] 트리거 사유: {retrain_reason}")
                except Exception:
                    # 신규 메서드 실패 시 기존 동작으로 폴백
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

                        # === Paper 피드백: 심볼별 funding rate 주입 (raw 8h rate) ===
                        try:
                            fr_raw = (
                                self.last_external.get("derivatives", {})
                                .get("funding_rate", {})
                                .get("current_rate", 0.0)
                            )
                            if fr_raw is not None:
                                self.paper_trader.set_funding_rate(symbol, float(fr_raw))
                        except Exception as e:
                            logger.debug(f"[Paper-Feedback] {symbol} funding rate 주입 실패: {e}")

                        # 외부 피처를 FeatureEngineer에 주입
                        ext_features = self.external_manager.get_all_features()

                        # 퀀트 시그널을 ML 피처로 사전 주입 (예측 전에 계산)
                        # v2 (2026-04-16): 실제 OHLCV/returns/volatility/spot-futures 전달
                        #   이전에는 returns=[], df=None, futures=spot 로 넘겨서
                        #   basis/crash/alpha가 항상 0 → ML 피처로 무의미했음
                        try:
                            ob_data = await self.collector.fetch_orderbook(exchange_name, symbol, limit=20)
                            ticker = await self.collector.fetch_ticker(exchange_name, symbol)
                            _futures = ticker.get("last", 0)  # 선물 가격 (ccxt futures 모드)
                            _taker_vol = ticker.get("quoteVolume", 0) or 0
                            _bid_v = ob_data.get("bid_volume", 0)
                            _ask_v = ob_data.get("ask_volume", 0)
                            _total_v = _bid_v + _ask_v if (_bid_v + _ask_v) > 0 else 1
                            _buy_v = _taker_vol * (_bid_v / _total_v)
                            _sell_v = _taker_vol * (_ask_v / _total_v)

                            # 실제 spot 가격 조회 (basis_spread 계산용)
                            _spot = _futures  # 기본값 (실패 시 basis=0)
                            try:
                                spot_symbol = symbol.replace(":USDT", "")  # 선물→현물 심볼
                                spot_ticker = await self.collector.fetch_ticker(exchange_name, spot_symbol)
                                _spot = spot_ticker.get("last", _futures)
                            except Exception:
                                pass

                            # 짧은 OHLCV 1회 조회 (returns/df/volatility용)
                            _df_pre = await self.collector.fetch_ohlcv(
                                exchange_name, symbol, primary_tf, limit=60,
                            )
                            if len(_df_pre) >= 20:
                                _closes = _df_pre["close"].values
                                _rets = [(_closes[i] - _closes[i-1]) / _closes[i-1]
                                         for i in range(1, len(_closes))]
                                _cur_vol = float(np.std(_rets[-5:])) if len(_rets) >= 5 else 0.01
                                _avg_vol = float(np.std(_rets[-30:])) if len(_rets) >= 30 else _cur_vol
                                _df_for_alpha = _df_pre
                            else:
                                _rets = []
                                _cur_vol = 0.01
                                _avg_vol = 0.01
                                _df_for_alpha = None

                            qs_pre = self.quant_signals.get_all_signals(
                                orderbook=ob_data, spot_price=_spot,
                                futures_price=_futures,
                                trades_volume=_taker_vol, buy_volume=_buy_v,
                                sell_volume=_sell_v, returns=_rets,
                                current_vol=_cur_vol, avg_vol=_avg_vol, df=_df_for_alpha,
                            )
                            # 퀀트 피처를 ext_features에 병합 → ML이 학습 가능
                            ext_features["quant_score"] = qs_pre.get("combined_score", 0)
                            ext_features["quant_confidence"] = qs_pre.get("combined_confidence", 0)
                            ext_features["quant_risk_scale"] = qs_pre.get("risk_scale", 1.0)
                            ob_sig = qs_pre.get("orderbook", {})
                            ext_features["quant_ob_imbalance"] = ob_sig.get("imbalance", 0)
                            ext_features["quant_ob_score"] = ob_sig.get("score", 0)
                            vpin_sig = qs_pre.get("vpin", {})
                            ext_features["quant_vpin"] = vpin_sig.get("vpin", 0)
                            ext_features["quant_vpin_risk"] = 1.0 - vpin_sig.get("position_scale", 1.0)
                            basis_sig = qs_pre.get("basis", {})
                            ext_features["quant_basis_pct"] = basis_sig.get("basis_pct", 0)
                            ext_features["quant_basis_score"] = basis_sig.get("score", 0)
                            crash_sig = qs_pre.get("crash", {})
                            ext_features["quant_crash_prob"] = crash_sig.get("crash_risk", 0)
                            ext_features["quant_crash_scale"] = crash_sig.get("position_scale", 1.0)
                            alpha_sig = qs_pre.get("alpha", {})
                            ext_features["quant_alpha_score"] = alpha_sig.get("score", 0)
                            ext_features["quant_alpha_vwap"] = alpha_sig.get("alphas", {}).get("vwap_dev", 0)
                            ext_features["quant_alpha_obv"] = alpha_sig.get("alphas", {}).get("obv_roc", 0)
                            regime_sig = qs_pre.get("regime", {})
                            ext_features["quant_regime_adx"] = regime_sig.get("adx", 0)
                            ext_features["quant_regime_bb_width"] = regime_sig.get("bb_width", 0)
                            ext_features["quant_regime_vol_ratio"] = regime_sig.get("vol_ratio", 0)
                        except Exception as e:
                            logger.debug(f"[Quant-ML] {symbol} 사전 피처 수집 실패: {e}")

                        self.feature_engineer.set_external_features(ext_features)

                    # 외부 요인 변동 → 텔레그램 알림 체크 (심볼 루프 끝난 후 1회)
                    self._check_external_alerts()

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

                # === 집중 매매 모드 결정 — LIVE 티어 기준 ===
                # LIVE 티어가 concentration 모드이면 활성화 (micro 티어에서만 true 기본값)
                concentration = self.tier_manager.get_feature(
                    "concentration_mode", mode="live",
                    default=self.config["trading"].get("concentration_mode", False),
                )
                max_live = self.tier_manager.get_feature(
                    "max_positions", mode="live",
                    default=self.config["trading"].get("max_concurrent_live", 1),
                )
                scalp_profiles = self.config["risk"].get("scalp_profiles", {})

                # === Cross-Asset BTC Reference 갱신 (통찰 #2, 2026-04-20) ===
                # 심볼 루프마다 새로 계산하면 중복 → 루프당 1회 캐시
                # _process_symbol / _analyze_symbol 내부에서 alt 심볼은 이 캐시를 사용
                await self._refresh_btc_reference(exchange_name, primary_tf)

                if concentration:
                    # 모든 심볼의 시그널 수집
                    candidates = []
                    for symbol in symbols:
                        result = await self._analyze_symbol(exchange_name, symbol, primary_tf)
                        if result and result.get("action") in ("long", "short"):
                            # trade_type이 scalp인 경우만 종목별 스캘핑 프로파일 오버라이드
                            trade_type = result.get("trade_type", "scalp")
                            if trade_type == "scalp":
                                coin = symbol.split("/")[0]
                                coin_profile = scalp_profiles.get(coin, {})
                                if coin_profile:
                                    result["tp_pct"] = coin_profile.get("tp_pct", result["tp_pct"])
                                    result["sl_pct"] = coin_profile.get("sl_pct", result["sl_pct"])
                                result["priority"] = coin_profile.get("priority", 5) if coin_profile else 5
                            else:
                                # swing은 trade_profiles에서 이미 설정됨
                                result["priority"] = 3  # swing은 중간 우선순위
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

                    # 스나이핑 포지션 펀비 기반 익절 체크
                    await self._check_funding_rate_exit()

                    # LIVE: 포지션 여유 있을 때만 신규 진입
                    if live_positions < max_live and candidates:
                        # [LIVE_LONG_ONLY] LIVE 후보에서 숏 제외 (PAPER는 위에서 둘 다 실행됨)
                        live_candidates = candidates
                        if getattr(self.strategy_manager, "live_long_only", False):
                            live_candidates = [c for c in candidates if c.get("action") == "long"]
                        if live_candidates:
                            # 확신도 × (1 - priority/10) 로 최종 순위 결정
                            best = max(
                                live_candidates,
                                key=lambda c: c["confidence"] * (1 - c["priority"] / 10),
                            )
                            await self._execute_live(exchange_name, best)
                else:
                    # 기존 모드: 각 심볼 독립 매매
                    for symbol in symbols:
                        await self._process_symbol(exchange_name, symbol, primary_tf)

                # [2026-04-20 제거] 상장 스나이핑 / VC 언락 펌프 진입 로직 삭제
                # 사유: 18일 실증 결과 -$47.24 (WR 39.5%, 후행 시그널 구조),
                # 퀀트 앙상블(XGB/LSTM/PPO)과 분리된 독립 엔트리로 리스크 관리 우회.
                # 상세 리포트: commit log 참조.

                # === 고급 퀀트 주기적 작업 ===
                now_utc = datetime.utcnow()

                # 1) Pairs Trading — tier=pro 에서만 (stat_arb_pairs 플래그), 24h마다 탐색
                try:
                    pairs_on = self.tier_manager.feature_enabled("stat_arb_pairs", mode="live") \
                        or self.tier_manager.feature_enabled("stat_arb_pairs", mode="paper")
                    if pairs_on:
                        need_discover = (
                            self._last_pairs_discovery is None
                            or (now_utc - self._last_pairs_discovery).total_seconds() > 86400
                        )

                        async def _fetch_prices_for_pair(sym):
                            try:
                                d = await self.collector.fetch_ohlcv(exchange_name, sym, primary_tf, limit=500)
                                return d["close"] if d is not None and len(d) > 0 else pd.Series(dtype=float)
                            except Exception:
                                import pandas as _pd
                                return _pd.Series(dtype=float)

                        import pandas as pd  # local rebind (outer import 있음)
                        if need_discover:
                            try:
                                await self.pairs_strategy.discover_pairs(
                                    symbols=list(symbols), fetch_prices_fn=_fetch_prices_for_pair,
                                )
                                self._last_pairs_discovery = now_utc
                            except Exception as e:
                                logger.debug(f"[Pairs] discover 실패: {e}")

                        # 매 루프 시그널 스캔 (order_manager는 첫 LIVE 거래소)
                        try:
                            om_first = next(iter(self.order_managers.values())) if self.order_managers else None
                            equity_for_pairs = max(self.live_equity, self.paper_trader.equity, 1.0)
                            await self.pairs_strategy.scan_and_trade(
                                equity=equity_for_pairs,
                                fetch_prices_fn=_fetch_prices_for_pair,
                                order_manager=om_first if self.mode in ("live", "dual") else None,
                            )
                        except Exception as e:
                            logger.debug(f"[Pairs] scan 실패: {e}")
                except Exception as e:
                    logger.debug(f"[Pairs] 주기 체크 실패: {e}")

                # 2) IC summary log (1h마다)
                try:
                    if (
                        self._last_ic_log is None
                        or (now_utc - self._last_ic_log).total_seconds() > 3600
                    ):
                        self.ic_tracker.log_summary()
                        # LLM 가중치 자동 튜닝 — Claude 시그널 IC 기반으로
                        # external_manager.llm_weight를 동적 조정. 수동 설정 불필요.
                        try:
                            if hasattr(self, "external_manager") and self.external_manager is not None:
                                self.external_manager.auto_tune_llm_weight(self.ic_tracker)
                        except Exception as ee:
                            logger.debug(f"[LLM-AutoTune] 주기 실행 실패: {ee}")
                        # CPCV/DSR 오버핏 자동 검증 — 하루 1회 강제
                        try:
                            if hasattr(self, "strategy_optimizer_paper"):
                                self.strategy_optimizer_paper.validate_configs_dsr()
                            if hasattr(self, "strategy_optimizer_live"):
                                self.strategy_optimizer_live.validate_configs_dsr()
                        except Exception as ee:
                            logger.debug(f"[DSR] 주기 검증 실패: {ee}")
                        # 레짐별 시그널 가중치 자동 갱신 (C) — 매 1h IC 매트릭스 → multiplier
                        try:
                            if hasattr(self, "signal_weight_opt"):
                                summary = self.signal_weight_opt.update_from_tracker(self.ic_tracker)
                                if summary:
                                    logger.info(
                                        f"[SignalWeight] {len(summary)}개 (regime,source) 조합 갱신"
                                    )
                        except Exception as ee:
                            logger.debug(f"[SignalWeight] 주기 실행 실패: {ee}")
                        # [Phase K, 2026-04-25] 앙상블 모델별 IC 가중치 갱신 — 매 1h
                        # ensemble.py의 정확도 기반 가중치를 IC 기반으로 교체.
                        # 샘플 부족 시 자동 보류 → safe.
                        try:
                            if hasattr(self, "ensemble") and self.ensemble is not None:
                                _ic_apply = self.ensemble.apply_ic_weights(
                                    self.ic_tracker, min_samples=20, smoothing=0.5,
                                )
                                # apply_ic_weights() 자체가 적용 여부와 IC를 로깅함 — 추가 로그 불필요
                                _ = _ic_apply  # noqa
                        except Exception as ee:
                            logger.debug(f"[Ensemble-IC] 주기 실행 실패: {ee}")
                        # [Phase J, 2026-04-25] 차단 사유 카운터 — 매 1h 분포 출력 + 리셋
                        try:
                            if hasattr(self, "strategy_manager") and self.strategy_manager is not None:
                                self.strategy_manager.log_block_stats(force=True)
                        except Exception as ee:
                            logger.debug(f"[BlockStats] 주기 실행 실패: {ee}")
                        # Paper↔Live 체결 모델 동기화 — LIVE 실측 슬리피지/메이커율 피드백
                        # 백테스트·페이퍼가 실거래와 다른 체결비용으로 돌아가면 전략 신호가
                        # 왜곡되므로, OrderManager가 누적한 중앙값 통계를 PaperTrader로
                        # 지수평활 주입해 괴리를 0으로 수렴시킨다.
                        try:
                            if self.mode in ("paper", "dual") and self.order_managers:
                                merged_stats: dict = {}
                                total_n = 0
                                for _om in self.order_managers.values():
                                    try:
                                        _s = _om.get_execution_stats()
                                    except Exception:
                                        continue
                                    _n = int(_s.get("n_samples", 0) or 0)
                                    if _n <= 0:
                                        continue
                                    # 가중합 (샘플 수 비례) — 복수 거래소 가중 중앙값 근사
                                    for _k in ("entry_slip_bps_med", "exit_slip_bps_med",
                                               "sl_slip_bps_med", "maker_fill_rate"):
                                        _v = _s.get(_k)
                                        if _v is None:
                                            continue
                                        merged_stats[_k] = merged_stats.get(_k, 0.0) + float(_v) * _n
                                    total_n += _n
                                if total_n > 0:
                                    for _k in list(merged_stats.keys()):
                                        merged_stats[_k] = merged_stats[_k] / total_n
                                    merged_stats["n_samples"] = total_n
                                    delta = self.paper_trader.sync_from_live_execution(merged_stats)
                                    # delta.entry_before/after, sl_extra_before/after 등은
                                    # PaperTrader 내부에서 이미 info 레벨로 로깅됨.
                                    # 여기서는 skip 사유만 debug로 남긴다.
                                    if delta and "skipped" in delta:
                                        logger.debug(f"[Paper-LiveSync] skipped: {delta['skipped']}")
                        except Exception as ee:
                            logger.debug(f"[Paper-LiveSync] 주기 실행 실패: {ee}")
                        self._last_ic_log = now_utc
                except Exception as e:
                    logger.debug(f"[IC] 주기 summary 실패: {e}")

                loop_count += 1

                # === equity_curve 기록 (5분마다 = 10루프) ===
                if loop_count % 10 == 0:
                    try:
                        risk_status = self.risk_manager.get_status()
                        dd = risk_status.get("current_drawdown", 0)
                        pos_info = {}
                        if self.mode in ("paper", "dual"):
                            pos_info["paper"] = {s: {"side": p.side, "pnl": getattr(p, 'unrealized_pnl', 0)}
                                                  for s, p in self.paper_trader.positions.items()}
                        for name, om in self.order_managers.items():
                            pos_info[f"live_{name}"] = {s: {"side": p.side}
                                                         for s, p in om.positions.items()}
                        self.storage.save_equity(self.equity, dd, pos_info)
                    except Exception as e:
                        logger.debug(f"[DB] equity_curve 저장 실패: {e}")

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

                    # === $100K 타겟 진행률 (2026-04-18) ===
                    target = float(self.config.get("trading", {}).get("target_equity", 100000))
                    live_eq_now = float(getattr(self, "live_equity", 0.0))
                    paper_eq_now = float(self.paper_trader.equity)
                    total_eq = live_eq_now + paper_eq_now
                    if target > 0:
                        pct = min(total_eq / target * 100, 100.0)
                        filled = int(pct / 5)  # 20칸 스케일
                        bar = "█" * filled + "░" * (20 - filled)
                        live_tier_name = self.tier_manager.get_tier("live").name
                        paper_tier_name = self.tier_manager.get_tier("paper").name
                        logger.info(
                            f"[Target] 🎯 $100K 진행 {pct:5.2f}% [{bar}] | "
                            f"LIVE ${live_eq_now:,.2f}({live_tier_name}) + "
                            f"PAPER ${paper_eq_now:,.2f}({paper_tier_name}) = ${total_eq:,.2f} / ${target:,.0f}"
                        )

                        # === BTC Treasury 누적 현황 (활성 시만) ===
                        try:
                            if getattr(self, "btc_reserve", None) and self.btc_reserve.enabled:
                                rs = self.btc_reserve.get_status()
                                live_btc = rs["live"]["total_btc"]
                                paper_btc = rs["paper"]["total_btc"]
                                if live_btc > 0 or paper_btc > 0:
                                    live_val = rs["live"].get("current_value") or 0
                                    paper_val = rs["paper"].get("current_value") or 0
                                    live_pnl_pct = rs["live"].get("pnl_pct")
                                    paper_pnl_pct = rs["paper"].get("pnl_pct")
                                    logger.info(
                                        f"[BTCReserve] 🏛️ 금고 "
                                        f"LIVE {live_btc:.8f}BTC (${live_val:,.2f}"
                                        f"{f' {live_pnl_pct:+.1f}%' if live_pnl_pct is not None else ''}) + "
                                        f"PAPER {paper_btc:.8f}BTC (${paper_val:,.2f}"
                                        f"{f' {paper_pnl_pct:+.1f}%' if paper_pnl_pct is not None else ''})"
                                    )
                        except Exception as e:
                            logger.debug(f"[BTCReserve] 로그 출력 실패: {e}")

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

    async def _refresh_btc_reference(self, exchange_name: str, timeframe: str):
        """Cross-Asset BTC 선행 피처용 reference 캐시 (통찰 #2)

        루프당 1회 호출 → BTC OHLCV + 최소 피처 계산 → self._btc_ref_cache 저장.
        이후 _process_symbol / _analyze_symbol이 alt 심볼 처리 시 이 캐시를
        feature_engineer.btc_reference 로 주입.
        """
        try:
            import ta as _ta
            btc_df = await self.collector.fetch_ohlcv(
                exchange_name, "BTC/USDT:USDT", timeframe, limit=200
            )
            if btc_df is None or len(btc_df) < 20:
                self._btc_ref_cache = None
                return
            btc_df["returns_1"] = btc_df["close"].pct_change(1)
            btc_df["returns_5"] = btc_df["close"].pct_change(5)
            btc_df["returns_20"] = btc_df["close"].pct_change(20)
            btc_df["rsi_14"] = _ta.momentum.RSIIndicator(btc_df["close"], window=14).rsi()
            btc_df["volatility_20"] = btc_df["returns_1"].rolling(20).std()
            self._btc_ref_cache = btc_df
        except Exception as e:
            logger.debug(f"[CrossAsset] BTC reference refresh 실패: {e}")
            self._btc_ref_cache = None

    def _apply_btc_reference(self, symbol: str):
        """BTC reference 주입 — 모든 심볼이 동일한 피처 차원(39개) 유지.

        [2026-04-20 수정] BTC 자신도 자기 자신을 reference로 사용.
        btc_returns_1 == returns_1 (동어반복)이 되지만 단일 모델이 모든 심볼에서
        같은 피처 수로 동작 (ETH/SOL/DOGE 학습 시 LSTM input_size 불일치 방지).
        """
        self.feature_engineer.set_btc_reference(getattr(self, "_btc_ref_cache", None))

    async def _process_symbol(self, exchange_name: str, symbol: str, timeframe: str):
        """개별 심볼 처리"""
        # 0. Cross-Asset BTC reference 주입 (통찰 #2)
        self._apply_btc_reference(symbol)

        # 1. 최신 데이터 수집
        df = await self.collector.fetch_ohlcv(exchange_name, symbol, timeframe, limit=200)
        df = self.feature_engineer.generate(df)
        feature_cols = self.feature_engineer.get_feature_columns(df)

        if len(df) < 60:
            return

        # 1.1 HRP용 OHLCV 캐시 주입 (다자산 전환 시 HRPAllocator가 읽음)
        if not hasattr(self, "_last_ohlcv_cache"):
            self._last_ohlcv_cache = {}
        self._last_ohlcv_cache[symbol] = df

        # 1.5. 퀀트 ML 피처 확인 (최초 1회 로깅)
        quant_cols = [c for c in df.columns if "quant" in c]
        if quant_cols and not getattr(self, '_quant_feat_logged', False):
            self._quant_feat_logged = True
            sample = {c: round(float(df[c].iloc[-1]), 4) for c in quant_cols[:8]}
            logger.info(f"[ML-Features] 퀀트 피처 {len(quant_cols)}개 주입됨: {sample}")

        # 2. 시장 레짐 감지 — 룰 기반 + HMM(tier=large+)
        prices = df["close"].values
        volumes = df["volume"].values
        adaptive_params = self.adaptive.update(prices, volumes)

        # 2b. HMM regime (tier=large+) — posterior 확률 기반 overlay
        if self.tier_manager.feature_enabled("hmm_regime", mode="paper") and self.hmm_regime.fitted:
            try:
                hmm_label = self.hmm_regime.predict(prices)
                hmm_proba = self.hmm_regime.predict_proba(prices)
                # HMM 결과를 adaptive_params에 주입 (룰 기반 우선, HMM은 overlay)
                adaptive_params["hmm_regime"] = hmm_label
                adaptive_params["hmm_proba"] = hmm_proba
                # HMM이 bull/bear를 확신(>0.7)하면 룰 기반 regime 오버라이드
                max_prob = max(hmm_proba.values()) if hmm_proba else 0
                if max_prob > 0.7 and hmm_label != "normal":
                    mapped = self.hmm_regime.regime_to_adaptive(hmm_label)
                    if mapped != adaptive_params.get("regime"):
                        logger.info(
                            f"[HMM] regime 오버라이드: {adaptive_params['regime']} → {mapped} "
                            f"(p={max_prob:.2f})"
                        )
                        adaptive_params["regime"] = mapped
            except Exception as e:
                logger.debug(f"[HMM] predict 실패: {e}")

        # 3. ML 시그널 (레짐 전달 — strong_uptrend WR 13.8% 학습 기반 가중치 적용)
        ml_signal = self.ensemble.predict(df, regime=adaptive_params.get("regime"))

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
        # [공포탐욕 제거] 후행 지표라 예측력 없음 — 매크로/파생상품/센티먼트로 대체
        ext_features_all = self.external_manager.get_all_features()
        funding_rate = ext_features_all.get("deriv_funding_rate", 0)

        current_position = 0.0
        if self.mode in ("paper", "dual"):
            current_position = 1.0 if symbol in self.paper_trader.positions else 0.0

        # 피드백 블랙리스트 조회 (반복 실패 시그널 조합 차단)
        fb_blacklist = self.feedback.get_entry_blacklist()

        # LIVE/dual에서는 "live" mode로 → LIVE 공격 롱 정책 적용
        # 순수 paper에서는 "paper" mode로 → 기존 보수적 정책
        decide_mode = "live" if self.mode in ("live", "dual") else "paper"
        decision = self.strategy_manager.decide(
            ml_signal, rl_action, rl_confidence, current_position,
            adaptive_params["regime"], external_signal=ext_signal,
            momentum=momentum_signal,
            feedback_blacklist=fb_blacklist,
            funding_rate=funding_rate,
            mode=decide_mode,
            ohlcv_df=df,
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

            # [재활성화] 강한 MTF 반대 — 진입 차단 (상위 TF 반대 시 승률 급락)
            if action_opposes_mtf and mtf_agreement >= 0.75:
                logger.warning(
                    f"[MTF차단] {decision.action} → hold "
                    f"(상위 TF {mtf_dir} 합의 {mtf_agreement:.0%})"
                )
                decision.action = "hold"
                decision.reason = f"MTF강반대 차단 ({mtf_dir} {mtf_agreement:.0%})"
                decision.confidence = 0.0
                decision.size = 0.0

            # [재활성화 — 약한 반대] 확신도 감소 (차단하진 않음)
            elif action_opposes_mtf and mtf_agreement >= 0.5:
                decision.confidence *= 0.75
                decision.reason += f" !MTF약반대({mtf_dir} {mtf_agreement:.0%})"

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
        # volatility / atr_pct — A/B 섀도우도 primary가 hold일 때 필요 → 항상 선계산
        volatility = df["returns_1"].std() if "returns_1" in df.columns else 0.01
        atr_pct = float(df["atr_pct"].iloc[-1]) if "atr_pct" in df.columns and df["atr_pct"].iloc[-1] == df["atr_pct"].iloc[-1] else 0.0
        alerts = self.anomaly_detector.update(price, volume)
        if alerts:
            self.last_signals["anomalies"] = [a["message"] for a in alerts]
            high_alerts = [a for a in alerts if a["severity"] == "high"]
            if high_alerts and decision.action in ["long", "short"]:
                logger.info(f"[리스크해제] 이상 감지 무시: {high_alerts[0]['message']}")

        # 6. 리스크 체크
        num_positions = len(self.paper_trader.positions) if self.mode in ("paper", "dual") else 0
        can_trade, risk_msg = self.risk_manager.check_can_trade(self.equity, num_positions)

        if decision.action in ["long", "short"] and not can_trade:
            logger.info(f"{symbol} 거래 차단: {risk_msg}")
            return

        # 6.5. 피드백 필터 [해제됨] — 로그만 남김
        if decision.action in ["long", "short"]:
            from datetime import datetime as dt
            fb_ok, fb_reason = self.feedback.should_trade_now(
                hour=dt.utcnow().hour,
                regime=adaptive_params["regime"],
                side=decision.action,
                signal_strength=ml_signal.get("signal", 0),
            )
            if not fb_ok:
                logger.info(f"[리스크해제] 피드백 차단 무시: {fb_reason}")

        # 7. 주문 실행
        if decision.action in ["long", "short"]:
            # volatility/atr_pct 는 위에서 선계산됨 (A/B 섀도우 일관성 보장)
            # Paper 슬리피지 모델에 ATR 주입 (포지션 유지 중 close 슬리피지도 최신 ATR 사용)
            try:
                self.paper_trader.set_atr(symbol, atr_pct)
            except Exception:
                pass

            # 7.0. 퀀트 시그널 계산 (오더북, VPIN, 베이시스, 크래시보호, 알파, 레짐)
            try:
                ob_data = await self.collector.fetch_orderbook(exchange_name, symbol, limit=20)
                ticker = await self.collector.fetch_ticker(exchange_name, symbol)
                spot_price = ticker.get("last", price)
                futures_price = float(df["close"].iloc[-1])
                returns_list = df["returns_1"].dropna().tolist() if "returns_1" in df.columns else []
                avg_vol = df["returns_1"].std() if "returns_1" in df.columns else 0.01
                taker_vol = ticker.get("quoteVolume", 0) or 0
                # 간이 buy/sell 추정 (오더북 불균형 기반)
                bid_vol = ob_data.get("bid_volume", 0)
                ask_vol = ob_data.get("ask_volume", 0)
                total_vol = bid_vol + ask_vol if (bid_vol + ask_vol) > 0 else 1
                buy_vol = taker_vol * (bid_vol / total_vol)
                sell_vol = taker_vol * (ask_vol / total_vol)

                qs = self.quant_signals.get_all_signals(
                    orderbook=ob_data, spot_price=spot_price,
                    futures_price=futures_price,
                    trades_volume=taker_vol, buy_volume=buy_vol,
                    sell_volume=sell_vol, returns=returns_list,
                    current_vol=volatility, avg_vol=avg_vol, df=df,
                )
                quant_risk_scale = qs.get("risk_scale", 1.0)
                quant_regime = qs.get("regime", {})
                quant_score = qs.get("combined_score", 0)

                logger.info(
                    f"[Quant] {symbol} | score={quant_score:+.3f} | "
                    f"regime={quant_regime.get('regime','?')} | "
                    f"risk_scale={quant_risk_scale:.2f} | "
                    f"OB={qs.get('orderbook',{}).get('imbalance',0):.3f} | "
                    f"VPIN={qs.get('vpin',{}).get('vpin',0):.3f} | "
                    f"basis={qs.get('basis',{}).get('basis_pct',0):.4f}"
                )

                # 퀀트 레짐이 ranging이면 추가 확신도 감소
                if quant_regime.get("regime") == "ranging":
                    decision.confidence *= 0.6
                    decision.reason += " !퀀트횡보장"
                    if decision.confidence < self.strategy_manager.min_confidence:
                        decision.action = "hold"
                        decision.size = 0.0
                        decision.reason += " → 진입차단"
                        return

                # 퀀트 시그널 방향 불일치 시 감액
                if quant_score != 0:
                    quant_agrees = (
                        (decision.action == "long" and quant_score > 0) or
                        (decision.action == "short" and quant_score < 0)
                    )
                    if not quant_agrees and abs(quant_score) > 0.3:
                        decision.confidence *= 0.7
                        decision.reason += f" !퀀트반대({quant_score:+.2f})"
                    elif quant_agrees and abs(quant_score) > 0.3:
                        decision.confidence = min(decision.confidence * 1.15, 1.0)
                        decision.reason += f" +퀀트합류({quant_score:+.2f})"

            except Exception as e:
                logger.warning(f"[Quant] {symbol} 시그널 수집 실패: {e}")
                quant_risk_scale = 1.0

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

            # === SL×레버리지 리스크 캡핑 ===
            tp_profile = self.config.get("trade_profiles", {}).get(decision.trade_type, {})
            sl_pct_for_cap = tp_profile.get("sl_pct", self.config["risk"]["stop_loss_pct"])
            max_risk_pct = tp_profile.get("max_risk_pct", 0.05)
            dynamic_lev = self.risk_manager.cap_leverage_by_risk(
                dynamic_lev, sl_pct_for_cap, max_risk_pct
            )

            # 7.3. 피드백 기반 포지션 크기 조정 + 퀀트 리스크 스케일
            fb_scale = self.feedback.get_position_scale(adaptive_params["regime"], decision.action)
            # === Kelly sizing (tier=mid+ 활성화, PAPER 가상시드 $5K → mid) ===
            # 첫 번째 call-site는 paper/dual 공통 분석 단계 → PAPER 티어 기준으로 Kelly
            kelly_enabled = self.tier_manager.feature_enabled("kelly_enabled", mode="paper")
            kelly_fraction = self.tier_manager.get_feature("kelly_fraction", mode="paper", default=0.25) or 0.25
            kelly_stats = (
                self.feedback.get_kelly_stats(
                    regime=adaptive_params["regime"],
                    side=decision.action,
                )
                if kelly_enabled else None
            )
            # === HRP 다자산 가중치 (2026-04-24) — max_positions>1 & concentration_off일 때만 ===
            hrp_scale = self._hrp_weight_for_symbol(symbol, mode="paper")
            size = self.risk_manager.calculate_position_size(
                self.equity, decision.confidence, volatility,
                adaptive_params["position_scale"] * fb_scale * corr_mult * quant_risk_scale * hrp_scale,
                kelly_enabled=kelly_enabled,
                kelly_fraction=float(kelly_fraction),
                kelly_stats=kelly_stats,
                atr_pct=atr_pct,
                leverage=dynamic_lev,
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
                        sl_pct=tp_profile.get("sl_pct", self.config["risk"]["stop_loss_pct"]) * adaptive_params["stop_loss_mult"],
                        tp_pct=tp_profile.get("tp_pct", self.config["risk"]["take_profit_pct"]),
                        atr_pct=atr_pct,
                        trade_type=decision.trade_type,
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
                    # [Patch K] PAPER 진입은 알림 ON (silent=False) — 사용자가 무인 운영 중 진입 인지 가능
                    tg_notify(format_trade_open("PAPER", symbol, decision.action, price, notional, dynamic_lev, decision.reason), silent=False)

            # === LIVE 실행 (live / dual 모드) ===
            # [LIVE_LONG_ONLY] LIVE만 숏 차단 — PAPER는 위에서 이미 실행됐으므로 학습 데이터 수집 계속
            live_block_short = (
                getattr(self.strategy_manager, "live_long_only", False)
                and decision.action == "short"
            )
            if live_block_short:
                logger.info(
                    f"[LIVE_LONG_ONLY] {symbol} LIVE 숏 차단 (PAPER는 실행됨) | "
                    f"원신호: {decision.reason}"
                )
            if self.mode in ("live", "dual") and not live_block_short:
                om = self.order_managers.get(exchange_name)
                if om and symbol not in getattr(om, "positions", {}):
                    result = await om.open_position(
                        symbol, decision.action, size, dynamic_lev,
                        atr_pct=atr_pct, trade_type=decision.trade_type,
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
            # [Phase K, 2026-04-25] per-model 시그널 캡처 — IC 기반 앙상블 가중치 재료.
            # ensemble.predict() 가 마지막 호출의 per-model 출력을 last_per_model_signals에 보관.
            try:
                per_model_signals = dict(getattr(self.ensemble, "last_per_model_signals", {}) or {})
            except Exception:
                per_model_signals = {}
            trade_context = {
                "regime": adaptive_params["regime"],
                "signal": ml_signal.get("signal", 0),
                "confidence": decision.confidence,
                "volatility": volatility,
                "external_score": ext_signal.get("score", 0),
                "external_direction": ext_signal.get("direction", "neutral"),
                # Claude-native LLM 전용 IC 평가용 엔트리 시그널 (auto-tune 재료)
                "llm_score": float(ext_signal.get("llm_score", 0) or 0),
                # [Phase K] 모델별 시그널 — 청산 시 per-model IC 기록용
                "per_model_signals": per_model_signals,
            }

            # === PAPER 청산 ===
            if self.mode in ("paper", "dual"):
                result = self.paper_trader.close_position(symbol, price, decision.reason)
                if result:
                    self.risk_manager.record_pnl(result["pnl"])
                    self._save_trade_with_context({
                        "exchange": exchange_name, "symbol": symbol, "side": result.get("side", "close"),
                        "price": price, "amount": result["size"], "pnl": result["pnl"],
                        "fee": result["fee"], "strategy": "hybrid",
                        "mode": "PAPER",
                    })
                    trade_context["exit_reason"] = "strategy_close"
                    trade_context["confirming_sources"] = getattr(decision, "confirming_sources", [])
                    trade_context["entry_path"] = "+".join(sorted(getattr(decision, "confirming_sources", [])))
                    self.feedback.record_trade(result, trade_context)
                    # IC Tracker — 실현 수익률 vs 예측 시그널 기록 (알파 품질 모니터)
                    try:
                        entry_signal = trade_context.get("signal", 0.0)
                        realized = result["pnl"] / max(result.get("notional", 1.0), 1.0)
                        _regime = trade_context.get("regime", "normal")
                        self.ic_tracker.record(
                            signal=entry_signal,
                            realized_return=realized,
                            source="ensemble",
                            regime=_regime,
                        )
                        # 소스별 per-regime 기록 (C: SignalWeightOptimizer 재료)
                        self.ic_tracker.record(
                            signal=float(ml_signal.get("signal", 0) or 0),
                            realized_return=realized,
                            source="ml",
                            regime=_regime,
                        )
                        self.ic_tracker.record(
                            signal=float(trade_context.get("external_score", 0) or 0),
                            realized_return=realized,
                            source="ext",
                            regime=_regime,
                        )
                        # Claude-native LLM 단독 IC (llm_weight auto-tune 재료)
                        llm_entry = float(trade_context.get("llm_score", 0) or 0)
                        if abs(llm_entry) > 1e-6:
                            self.ic_tracker.record(
                                signal=llm_entry,
                                realized_return=realized,
                                source="llm",
                                regime=_regime,
                            )
                        # [Phase K, 2026-04-25] per-model IC 기록 — ensemble.apply_ic_weights() 재료
                        # entry 시점의 per-model 시그널을 trade_context에서 꺼내 기록.
                        per_model = trade_context.get("per_model_signals", {}) or {}
                        _src_map = {
                            "xgboost": "model_xgb",
                            "lstm": "model_lstm",
                            "lightgbm": "model_lgb",
                            "cnn_attention": "model_cnn",
                        }
                        for _mn, _sig in per_model.items():
                            _src = _src_map.get(_mn, f"model_{_mn}")
                            try:
                                self.ic_tracker.record(
                                    signal=float(_sig or 0.0),
                                    realized_return=realized,
                                    source=_src,
                                    regime=_regime,
                                )
                            except Exception as _e:
                                logger.debug(f"[IC-PerModel] {_mn} 기록 실패: {_e}")
                    except Exception as e:
                        logger.debug(f"[IC] 기록 실패: {e}")
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
                    if result["pnl"] < 0:
                        self._generate_loss_report(result, mode="PAPER")

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
                        # IC Tracker — LIVE 실현 vs 예측
                        try:
                            entry_signal = ml_signal.get("signal", 0.0) if isinstance(ml_signal, dict) else float(ml_signal or 0.0)
                            notional = live_result.get("notional") or (live_result.get("size", 0) * max(live_result.get("entry_price", 0), 1e-9))
                            realized = live_pnl / max(notional, 1.0)
                            _regime_live = adaptive_params.get("regime", "normal")
                            self.ic_tracker.record(
                                signal=entry_signal,
                                realized_return=realized,
                                source="ensemble_live",
                                regime=_regime_live,
                            )
                            # 소스별 per-regime — LIVE에서도 SignalWeightOpt 재료 수집
                            self.ic_tracker.record(
                                signal=entry_signal,
                                realized_return=realized,
                                source="ml",
                                regime=_regime_live,
                            )
                            if isinstance(ext_signal, dict):
                                self.ic_tracker.record(
                                    signal=float(ext_signal.get("score", 0) or 0),
                                    realized_return=realized,
                                    source="ext",
                                    regime=_regime_live,
                                )
                            # LLM 단독 IC (LIVE도 "llm" 소스로 통합 — auto-tune에 합산)
                            llm_entry = float(ext_signal.get("llm_score", 0) or 0) if isinstance(ext_signal, dict) else 0.0
                            if abs(llm_entry) > 1e-6:
                                self.ic_tracker.record(
                                    signal=llm_entry,
                                    realized_return=realized,
                                    source="llm",
                                    regime=_regime_live,
                                )
                            # [Phase K, 2026-04-25] LIVE per-model IC 기록 — 가중치 재료
                            try:
                                _per_model = dict(getattr(self.ensemble, "last_per_model_signals", {}) or {})
                                _src_map = {
                                    "xgboost": "model_xgb",
                                    "lstm": "model_lstm",
                                    "lightgbm": "model_lgb",
                                    "cnn_attention": "model_cnn",
                                }
                                for _mn, _sig in _per_model.items():
                                    self.ic_tracker.record(
                                        signal=float(_sig or 0.0),
                                        realized_return=realized,
                                        source=_src_map.get(_mn, f"model_{_mn}"),
                                        regime=_regime_live,
                                    )
                            except Exception as _e:
                                logger.debug(f"[IC-PerModel-LIVE] 실패: {_e}")
                        except Exception as e:
                            logger.debug(f"[IC-LIVE] 기록 실패: {e}")
                        if live_pnl < 0:
                            self.strategy_manager.record_loss()
                        else:
                            self.strategy_manager.record_win()
                        self._save_trade_with_context({
                            "exchange": exchange_name, "symbol": symbol, "side": live_result.get("side", "close"),
                            "price": live_result.get("exit_price", 0),
                            "amount": live_result.get("size", 0),
                            "pnl": live_pnl, "fee": live_result.get("fee", 0), "strategy": "live_signal_close",
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
                        if live_pnl < 0 and isinstance(live_result, dict):
                            self._generate_loss_report(live_result, mode="LIVE")

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
                        self._save_trade_with_context({
                            "exchange": exchange_name, "symbol": symbol, "side": result.get("side", "close"),
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

        # === A/B 섀도우 variant (MACRO_OFF) 병렬 실행 (2026-04-21) ===
        # 같은 시장 입력을 매크로 차단 OFF 정책으로 paper_trader_off 에 동시 실행.
        # LIVE에 흐르지 않는 순수 관측용 표본 — A/B 비교용 독립 표본 수집.
        if self.mode in ("paper", "dual"):
            try:
                # 예외 경로 대비 — locals()에 없으면 안전 기본값
                _local = locals()
                _q_regime_obj = _local.get("quant_regime", {})
                _q_regime_name = _q_regime_obj.get("regime", "?") if isinstance(_q_regime_obj, dict) else "?"
                await self._run_shadow_macro_off(
                    symbol=symbol,
                    exchange_name=exchange_name,
                    df=df,
                    ml_signal=ml_signal,
                    rl_action=rl_action,
                    rl_confidence=rl_confidence,
                    ext_signal=ext_signal,
                    momentum_signal=momentum_signal,
                    mtf_signal=mtf_signal,
                    quant_score=_local.get("quant_score", 0.0),
                    quant_regime_name=_q_regime_name,
                    quant_risk_scale=_local.get("quant_risk_scale", 1.0),
                    adaptive_params=adaptive_params,
                    funding_rate=funding_rate,
                    atr_pct=_local.get("atr_pct", 0.0),
                    volatility=_local.get("volatility", 0.01),
                )
            except Exception as e:
                logger.warning(f"[A/B-Shadow] {symbol} 실행 생략: {e}")

    # =========================================================================
    # 집중 매매 모드 메서드 (concentration_mode=true)
    # =========================================================================

    async def _analyze_symbol(self, exchange_name: str, symbol: str, timeframe: str) -> dict | None:
        """심볼 분석만 수행하고 시그널 반환 (실행 X)"""
        try:
            # 0. Cross-Asset BTC reference 주입 (통찰 #2)
            self._apply_btc_reference(symbol)

            df = await self.collector.fetch_ohlcv(exchange_name, symbol, timeframe, limit=200)
            df = self.feature_engineer.generate(df)
            feature_cols = self.feature_engineer.get_feature_columns(df)

            if len(df) < 60:
                return None

            # 퀀트 ML 피처 확인 (최초 1회)
            quant_cols = [c for c in df.columns if "quant" in c]
            if quant_cols and not getattr(self, '_quant_feat_logged_analyze', False):
                self._quant_feat_logged_analyze = True
                sample = {c: round(float(df[c].iloc[-1]), 4) for c in quant_cols[:8]}
                logger.info(f"[ML-Features] {symbol} 퀀트 피처 {len(quant_cols)}개: {sample}")

            prices = df["close"].values
            volumes = df["volume"].values
            adaptive_params = self.adaptive.update(prices, volumes)

            # 레짐 기반 시그널 가중치 (ensemble.REGIME_SIGNAL_WEIGHT 참고)
            ml_signal = self.ensemble.predict(df, regime=adaptive_params.get("regime"))

            # [Patch M, 2026-04-28] Pattern Memory Bank — Retrieval-Augmented (Phase 1: shadow mode)
            # 현재 캔들과 유사한 과거 패턴 100개의 forward return 통계.
            # ML signal과 비교하여 일치/불일치 검증 → Phase 2에서 결정 통합.
            pattern_signal = None
            try:
                bank = self.pattern_banks.get(symbol)
                if bank is not None:
                    pstats = bank.predict(df.iloc[[-1]])
                    if pstats is not None:
                        pattern_signal = pstats.to_signal()
                        ml_dir = ml_signal.get("direction", "neutral")
                        agree = pattern_signal["direction"] == ml_dir
                        agree_emoji = "🟢" if agree else "🔴"
                        if not getattr(self, '_pattern_log_throttle', None):
                            self._pattern_log_throttle = {}
                        # 심볼당 5분에 1번만 로그 (소음 방지)
                        last_ts = self._pattern_log_throttle.get(symbol, 0)
                        now_ts = time.time()
                        if now_ts - last_ts > 300:
                            logger.info(
                                f"[PatternBank] {symbol} {agree_emoji} ML={ml_dir}/{ml_signal.get('signal',0):+.3f} "
                                f"vs Pattern={pattern_signal['direction']}/{pattern_signal['signal']:+.3f} "
                                f"| n={pattern_signal['n_neighbors']} WR_1h={pattern_signal['winrate_1h']*100:.0f}% "
                                f"EV_1h={pattern_signal['ev_1h_pct']:+.3f}% sim={pattern_signal['similarity']:.3f}"
                            )
                            self._pattern_log_throttle[symbol] = now_ts
            except Exception as pat_e:
                logger.debug(f"[PatternBank] {symbol} 추론 실패 (shadow mode 무시): {pat_e}")

            base_feature_cols = self.feature_engineer.get_base_feature_columns(df)
            rl_obs_data = df[base_feature_cols].values[-1].astype(np.float32)
            rl_obs_data = np.nan_to_num(rl_obs_data, nan=0.0)
            position_info = np.array([0.0, 0.0, self.equity / self.initial_capital, 0.0], dtype=np.float32)
            obs = np.concatenate([rl_obs_data, position_info])
            # [Patch I, 2026-04-28] RL observation shape 불일치 시 거래 분석 자체가 죽지 않도록 격리
            try:
                rl_action, rl_confidence = self.rl_agent.predict(obs)
            except Exception as rl_e:
                if not getattr(self, '_rl_shape_warned', False):
                    logger.warning(
                        f"[RL] obs shape 불일치 또는 예측 실패 → neutral fallback: {rl_e} "
                        f"(다음 RL 재학습 시 자동 복구)"
                    )
                    self._rl_shape_warned = True
                rl_action, rl_confidence = 1, 0.0  # 1=neutral/hold

            ext_signal = self.external_manager.get_signal_for_strategy()
            mtf_signal = self.external_manager.multi_tf.get_signal_for_strategy()
            momentum_signal = self._calculate_momentum(df)

            # ATR 값 추출 (동적 SL/TP용)
            atr_pct = float(df["atr_pct"].iloc[-1]) if "atr_pct" in df.columns else 0.0
            if atr_pct != atr_pct:  # NaN 체크
                atr_pct = 0.0

            # 피드백 블랙리스트 조회
            fb_blacklist = self.feedback.get_entry_blacklist()

            # 펀딩비 (레짐 바이어스용) — 공포탐욕 제거됨
            ext_features = self.external_manager.get_all_features()
            funding_rate = ext_features.get("deriv_funding_rate", 0)

            current_position = 1.0 if symbol in self.paper_trader.positions else 0.0
            # concentration 모드에서 _analyze_symbol은 주로 LIVE 후보 선정 목적
            decide_mode = "live" if self.mode in ("live", "dual") else "paper"
            decision = self.strategy_manager.decide(
                ml_signal, rl_action, rl_confidence, current_position,
                adaptive_params["regime"], external_signal=ext_signal,
                momentum=momentum_signal,
                feedback_blacklist=fb_blacklist,
                funding_rate=funding_rate,
                mode=decide_mode,
                ohlcv_df=df,
            )

            # === [Patch O, 2026-05-22] Pattern Bank Decision Fusion (Phase 2) ===
            # 데이터 근거 (24일 운영): ML 모델 WR 17%, ML-Pattern 일치율 35.7%.
            # → Pattern Bank(실제 과거 통계)에 veto power 부여.
            #   ML이 진입하려는데 유사 과거 패턴이 명확히 반대면 차단.
            #   Pattern도 같은 방향 + 높은 WR이면 confidence 강화.
            if pattern_signal is not None and decision.action in ("long", "short"):
                try:
                    pat_dir = pattern_signal["direction"]
                    pat_wr = float(pattern_signal["winrate_1h"])
                    pat_sim = float(pattern_signal["similarity"])
                    pat_ev = float(pattern_signal["ev_1h_pct"])

                    # 1) 충돌 veto — ML 진입 방향과 Pattern 방향이 정반대
                    conflict = (
                        (decision.action == "long" and pat_dir == "short")
                        or (decision.action == "short" and pat_dir == "long")
                    )
                    # 2) Pattern이 진입 방향과 반대 EV (long인데 과거 패턴 EV<0 등)
                    adverse_ev = (
                        (decision.action == "long" and pat_ev < -0.05)
                        or (decision.action == "short" and pat_ev > 0.05)
                    )

                    if conflict and pat_sim >= 0.90:
                        logger.info(
                            f"[Fusion] {symbol} 🛑 Pattern VETO — "
                            f"ML={decision.action} vs Pattern={pat_dir} "
                            f"(sim {pat_sim:.3f}, WR {pat_wr*100:.0f}%)"
                        )
                        decision.action = "hold"
                        decision.confidence = 0.0
                        decision.reason = (decision.reason or "") + " | Pattern VETO(충돌)"
                    elif adverse_ev and pat_sim >= 0.92:
                        logger.info(
                            f"[Fusion] {symbol} 🛑 Pattern VETO — "
                            f"{decision.action} 인데 과거 패턴 EV_1h={pat_ev:+.3f}% (sim {pat_sim:.3f})"
                        )
                        decision.action = "hold"
                        decision.confidence = 0.0
                        decision.reason = (decision.reason or "") + " | Pattern VETO(역EV)"
                    elif pat_dir == decision.action and pat_wr >= 0.58 and pat_sim >= 0.92:
                        # 3) 확증 — Pattern도 같은 방향 + 높은 WR + 높은 유사도
                        old_conf = decision.confidence
                        decision.confidence = min(decision.confidence * 1.25, 1.0)
                        decision.reason = (decision.reason or "") + (
                            f" | Pattern 확증(WR{pat_wr*100:.0f}% sim{pat_sim:.2f})"
                        )
                        logger.info(
                            f"[Fusion] {symbol} ✅ Pattern 확증 — {decision.action} "
                            f"conf {old_conf:.2f}→{decision.confidence:.2f}"
                        )
                except Exception as fus_e:
                    logger.debug(f"[Fusion] {symbol} fusion 실패 (무시): {fus_e}")

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

            # === 퀀트 시그널 계산 ===
            quant_risk_scale = 1.0
            quant_score = 0.0
            quant_regime = {}
            try:
                ob_data = await self.collector.fetch_orderbook(exchange_name, symbol, limit=20)
                ticker = await self.collector.fetch_ticker(exchange_name, symbol)
                spot_price = ticker.get("last", price)
                returns_list = df["returns_1"].dropna().tolist() if "returns_1" in df.columns else []
                avg_vol = volatility
                taker_vol = ticker.get("quoteVolume", 0) or 0
                bid_vol = ob_data.get("bid_volume", 0)
                ask_vol = ob_data.get("ask_volume", 0)
                total_vol = bid_vol + ask_vol if (bid_vol + ask_vol) > 0 else 1
                buy_vol = taker_vol * (bid_vol / total_vol)
                sell_vol = taker_vol * (ask_vol / total_vol)

                qs = self.quant_signals.get_all_signals(
                    orderbook=ob_data, spot_price=spot_price,
                    futures_price=price,
                    trades_volume=taker_vol, buy_volume=buy_vol,
                    sell_volume=sell_vol, returns=returns_list,
                    current_vol=volatility, avg_vol=avg_vol, df=df,
                )
                quant_risk_scale = qs.get("risk_scale", 1.0)
                quant_regime = qs.get("regime", {})
                quant_score = qs.get("combined_score", 0)

                logger.info(
                    f"[Quant] {symbol} | score={quant_score:+.3f} | "
                    f"regime={quant_regime.get('regime','?')} | "
                    f"risk_scale={quant_risk_scale:.2f} | "
                    f"OB={qs.get('orderbook',{}).get('imbalance',0):.3f} | "
                    f"VPIN={qs.get('vpin',{}).get('vpin',0):.3f} | "
                    f"basis={qs.get('basis',{}).get('basis_pct',0):.4f}"
                )

                # 퀀트 레짐 횡보 → 진입 차단
                if quant_regime.get("regime") == "ranging" and decision.action in ("long", "short"):
                    decision.confidence *= 0.6
                    decision.reason += " !퀀트횡보장"

                # 퀀트 시그널 방향 확인
                if quant_score != 0 and decision.action in ("long", "short"):
                    quant_agrees = (
                        (decision.action == "long" and quant_score > 0) or
                        (decision.action == "short" and quant_score < 0)
                    )
                    if not quant_agrees and abs(quant_score) > 0.3:
                        decision.confidence *= 0.7
                        decision.reason += f" !퀀트반대({quant_score:+.2f})"
                    elif quant_agrees and abs(quant_score) > 0.3:
                        decision.confidence = min(decision.confidence * 1.15, 1.0)
                        decision.reason += f" +퀀트합류({quant_score:+.2f})"
            except Exception as e:
                logger.warning(f"[Quant] {symbol} 시그널 실패: {e}")

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

            # === SL×레버리지 리스크 캡핑 ===
            trade_type = decision.trade_type
            tp_profile = self.config.get("trade_profiles", {}).get(trade_type, {})
            sl_pct_for_cap = tp_profile.get("sl_pct", self.config["risk"]["stop_loss_pct"])
            max_risk_pct = tp_profile.get("max_risk_pct", 0.05)
            dynamic_lev = self.risk_manager.cap_leverage_by_risk(
                dynamic_lev, sl_pct_for_cap, max_risk_pct
            )

            # 포지션 크기 (풀시드) + 퀀트 리스크 스케일
            fb_scale = self.feedback.get_position_scale(adaptive_params["regime"], decision.action)
            # === Kelly sizing: 집중매매 분석 (PAPER 가상시드 티어 기준) ===
            kelly_enabled = self.tier_manager.feature_enabled("kelly_enabled", mode="paper")
            kelly_fraction = self.tier_manager.get_feature("kelly_fraction", mode="paper", default=0.25) or 0.25
            kelly_stats = (
                self.feedback.get_kelly_stats(
                    regime=adaptive_params["regime"],
                    side=decision.action,
                )
                if kelly_enabled else None
            )
            size = self.risk_manager.calculate_position_size(
                self.equity, decision.confidence, volatility,
                adaptive_params["position_scale"] * fb_scale * quant_risk_scale,
                kelly_enabled=kelly_enabled,
                kelly_fraction=float(kelly_fraction),
                kelly_stats=kelly_stats,
                atr_pct=atr_pct,
                leverage=dynamic_lev,
                mode="paper",  # [Patch Q] _analyze_symbol은 PAPER 가상시드 기준 분석 → 3% target_risk
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

            # 시그널 DB 기록 (hold 포함 모든 결정)
            try:
                _ml_score = ml_signal.get("signal", 0) if isinstance(ml_signal, dict) else float(ml_signal)
                self.storage.save_signal(
                    symbol=symbol, model="ensemble",
                    signal=float(_ml_score), confidence=float(decision.confidence),
                    metadata={
                        "action": decision.action,
                        "regime": adaptive_params["regime"],
                        "quant_score": round(quant_score, 4),
                        "quant_risk_scale": round(quant_risk_scale, 4),
                        "quant_regime": quant_regime.get("regime", "?"),
                        "sources": getattr(decision, 'confirming_sources', []),
                        "reason": decision.reason[:200],
                    },
                )
            except Exception as e:
                logger.warning(f"[DB] signal 저장 실패: {e}")

            # === A/B 섀도우 variant (MACRO_OFF) — 집중모드에서도 병렬 실행 ===
            # primary가 hold여도 shadow는 독립 결정 → 매 tick 병렬 관측
            if self.mode in ("paper", "dual"):
                try:
                    await self._run_shadow_macro_off(
                        symbol=symbol,
                        exchange_name=exchange_name,
                        df=df,
                        ml_signal=ml_signal,
                        rl_action=rl_action,
                        rl_confidence=rl_confidence,
                        ext_signal=ext_signal,
                        momentum_signal=momentum_signal,
                        mtf_signal=mtf_signal,
                        quant_score=quant_score,
                        quant_regime_name=quant_regime.get("regime", "?") if isinstance(quant_regime, dict) else "?",
                        quant_risk_scale=quant_risk_scale,
                        adaptive_params=adaptive_params,
                        funding_rate=funding_rate,
                        atr_pct=atr_pct,
                        volatility=volatility,
                    )
                except Exception as e:
                    logger.warning(f"[A/B-Shadow-집중] {symbol} 실행 생략: {e}")

            if decision.action not in ("long", "short", "close"):
                return None

            # trade_type별 SL/TP 프로파일 적용
            trade_type = decision.trade_type
            tp_profile = self.config.get("trade_profiles", {}).get(trade_type, {})

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
                "atr_pct": atr_pct,
                "confirming_sources": decision.confirming_sources,
                "signal_strength": decision.signal_strength,
                "trade_type": trade_type,
                "tp_pct": tp_profile.get("tp_pct", self.config["risk"]["take_profit_pct"]),
                "sl_pct": tp_profile.get("sl_pct", self.config["risk"]["stop_loss_pct"]),
                "quant_score": quant_score,
                "quant_risk_scale": quant_risk_scale,
                "quant_regime": quant_regime.get("regime", "?"),
            }

        except Exception as e:
            logger.warning(f"[집중분석] {symbol} 분석 실패: {e}")
            return None

    async def _execute_paper(self, c: dict):
        """PAPER 포지션 실행.

        [2026-04-21] PAPER는 학습 데이터 수집이 목적 → 손실도 부정 샘플로 가치 있음.
        _check_daily_drawdown / _check_consecutive_losses 의 정지 플래그는 PAPER에
        적용하지 않는다 (LIVE 전용). 실제 돈 손실 없음.
        """
        symbol = c["symbol"]
        if self.mode not in ("paper", "dual"):
            return
        # PAPER 분기 진입 — 리스크 매니저 mode 복원 (LIVE 분기에서 'live'로 변경됐을 수 있음)
        try:
            self.risk_manager.set_trading_mode("paper")
        except Exception:
            pass
        # === 티어 심볼 필터 ===
        if not self.tier_manager.allowed_symbol(symbol, mode="paper"):
            logger.debug(
                f"[Tier-PAPER] {symbol} 차단 — PAPER 티어({self.tier_manager.get_tier('paper').name}) 심볼 아님"
            )
            return
        # === 티어 max_positions 제한 ===
        paper_max = self.tier_manager.get_feature(
            "max_positions", mode="paper",
            default=self.config["trading"].get("max_concurrent_paper", 4),
        )
        if len(self.paper_trader.positions) >= paper_max:
            logger.debug(
                f"[Tier-PAPER] {symbol} 차단 — 최대 포지션({paper_max}) 도달 "
                f"[현재: {list(self.paper_trader.positions.keys())}]"
            )
            return
        if c["action"] in ("long", "short"):
            if symbol not in self.paper_trader.positions:
                # === CVaR tail risk 체크 (PAPER 가상시드 mid+ 티어에서 활성화) ===
                if self.tier_manager.feature_enabled("cvar_risk", mode="paper"):
                    passed, cvar_pct, reason = self.risk_manager.check_cvar_limit(
                        proposed_notional=c["notional"],
                        equity=max(self.paper_trader.equity, 1.0),
                        threshold_pct=0.05,
                    )
                    if not passed:
                        logger.warning(f"[CVaR-PAPER] {symbol} 진입 차단: {reason}")
                        add_live_log({
                            "time": datetime.utcnow().strftime("%H:%M:%S"),
                            "type": "cvar_block",
                            "mode": "PAPER",
                            "symbol": symbol,
                            "cvar_pct": round(cvar_pct, 4),
                            "reason": reason,
                        })
                        return

                # === Meta-Labeler 2차 결정 (tier=large+ 활성화) ===
                if self.tier_manager.feature_enabled("meta_labeling", mode="paper") and self.meta_labeler.model is not None:
                    try:
                        df_c = c.get("df")
                        if df_c is not None and len(df_c) > 0:
                            feature_cols = [col for col in self.meta_labeler.feature_columns[:-2] if col in df_c.columns]
                            if feature_cols:
                                row = df_c[feature_cols].iloc[-1].values.astype(float)
                                primary_sig = c["ml_signal"].get("signal", 0.0) if isinstance(c["ml_signal"], dict) else 0.0
                                primary_conf = c["confidence"]
                                meta = self.meta_labeler.predict(row, primary_sig, primary_conf)
                                if not meta["take"]:
                                    logger.info(
                                        f"[Meta-PAPER] {symbol} 진입 SKIP (prob={meta['prob']:.2f} < {meta['threshold']})"
                                    )
                                    return
                    except Exception as e:
                        logger.debug(f"[Meta-PAPER] 체크 실패 → 통과: {e}")

                tp_pct = c.get("tp_pct", self.config["risk"]["take_profit_pct"])
                sl_pct = c.get("sl_pct", self.config["risk"]["stop_loss_pct"])
                self.paper_trader.open_position(
                    symbol, c["action"], c["size"], c["price"],
                    leverage=c["dynamic_lev"],
                    sl_pct=sl_pct * c["adaptive_params"]["stop_loss_mult"],
                    tp_pct=tp_pct,
                    atr_pct=c.get("atr_pct", 0),
                    trade_type=c.get("trade_type", "scalp"),
                )
                logger.info(
                    f"[PAPER] {c['action'].upper()} {symbol} [{c.get('trade_type','scalp').upper()}] | "
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
                # [Patch K] PAPER 진입은 알림 ON (사용자 무인 운영 인지)
                tg_notify(format_trade_open("PAPER", symbol, c["action"], c["price"], c["notional"], c["dynamic_lev"], c["reason"]), silent=False)
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
                self._save_trade_with_context({
                    "exchange": "paper", "symbol": symbol, "side": result.get("side", "close"),
                    "price": c["price"], "amount": result.get("size", 0),
                    "pnl": paper_pnl, "fee": result.get("fee", 0), "strategy": "paper_concentration",
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

    # [2026-04-20 제거] _execute_listing_snipe / _execute_vc_pump_entry 제거.
    # 사유: 상장 스나이핑 18일 실증 -$47.24 / WR 39.5% / 후행 시그널 구조.
    # 퀀트 앙상블(XGB/LSTM/PPO) 외부의 독립 엔트리 경로로 리스크 관리 우회했음.

    async def _check_funding_rate_exit(self):
        """극단 펀딩비(숏 최대치) 익절 — 매 루프 모든 포지션 대상.

        펀비 -0.5% 이하 + 수익 1%+ → 즉시 익절
        펀비 -1.0% 이하 + 수익 > 0 → 긴급 탈출
        (과거 상장 스나이핑 전용이었으나, 펀비 극단은 일반 포지션에도 유효한 신호라 유지)
        """
        for name, om in self.order_managers.items():
            for symbol, pos in list(om.positions.items()):
                try:
                    # 현재 펀딩비 조회
                    import json, urllib.request
                    binance_sym = symbol.replace("/", "").replace(":USDT", "")
                    url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={binance_sym}"
                    data = json.loads(urllib.request.urlopen(url, timeout=5).read())
                    funding_rate = float(data.get("lastFundingRate", 0))

                    # 현재가
                    price = await om.exchange.get_ticker_price(symbol)

                    # 수익률 계산
                    if pos.side == "long":
                        profit_pct = (price - pos.entry_price) / pos.entry_price
                    else:
                        profit_pct = (pos.entry_price - price) / pos.entry_price

                    # === 펀비 익절 조건 ===
                    # 펀비 -0.5% 이하 (숏 극단) + 수익 1% 이상 → 즉시 익절
                    if funding_rate <= -0.005 and profit_pct >= 0.01:
                        logger.info(
                            f"[펀비익절] {symbol} 펀비={funding_rate*100:.2f}% 극단 숏 + "
                            f"수익 {profit_pct:.1%} → 즉시 익절"
                        )
                        result = await om.close_position(symbol, f"펀비 익절 (FR={funding_rate*100:.2f}%)")
                        if result:
                            pnl = result.get("pnl", 0)
                            tg_notify(
                                f"💰 <b>펀비 익절 완료</b>\n"
                                f"━━━━━━━━━━━━━\n"
                                f"코인: {symbol}\n"
                                f"수익: ${pnl:.2f} ({profit_pct:.1%})\n"
                                f"펀딩비: {funding_rate*100:.2f}% (숏 극단)\n"
                                f"📝 숏 최대치 → 덤프 전 탈출"
                            )

                    # 펀비 -1% 이하 = 초극단 → 수익 0%만 넘어도 바로 탈출
                    elif funding_rate <= -0.01 and profit_pct > 0:
                        logger.info(
                            f"[긴급탈출] {symbol} 펀비={funding_rate*100:.2f}% 초극단 + "
                            f"수익 {profit_pct:.1%} → 긴급 탈출"
                        )
                        result = await om.close_position(symbol, f"긴급 탈출 (FR={funding_rate*100:.2f}%)")
                        if result:
                            pnl = result.get("pnl", 0)
                            tg_notify(
                                f"🚨 <b>긴급 탈출 완료</b>\n"
                                f"코인: {symbol} | PnL: ${pnl:.2f}\n"
                                f"펀딩비: {funding_rate*100:.2f}% — 초극단 숏"
                            )

                except Exception as e:
                    logger.debug(f"[펀비체크] {symbol} 실패: {e}")

    async def _execute_live(self, exchange_name: str, c: dict):
        """LIVE 포지션 실행 — 가장 강한 시그널에만"""
        if self.mode not in ("live", "dual"):
            return
        symbol = c.get("symbol", "?")
        # === LIVE 심볼 화이트리스트 (수학적 최적화, 2026-04-20) ===
        # 알트 투기 차단 — 엣지 검증된 심볼(BTC/ETH/SOL/DOGE)만 실거래 허용
        live_whitelist = self.config.get("trading", {}).get("live_symbol_whitelist")
        if live_whitelist and symbol not in live_whitelist:
            logger.info(
                f"[LIVE-Whitelist] {symbol} 차단 — 허용된 심볼 아님 "
                f"(허용: {live_whitelist})"
            )
            return
        # === 티어 심볼 필터 (LIVE) ===
        if not self.tier_manager.allowed_symbol(symbol, mode="live"):
            logger.info(
                f"[Tier-LIVE] {symbol} 차단 — LIVE 티어({self.tier_manager.get_tier('live').name}) "
                f"심볼 아님 (허용: {self.tier_manager.get_symbols('live')})"
            )
            return
        # === LIVE 리스크 게이트 (2026-04-25 — Manus v3 부분 채택) ===
        # PAPER는 학습 데이터 수집을 위해 게이트 해제, LIVE만 DD/일일손실/쿨다운 enforce.
        # risk_gates_mode='smart' & live_only=true 설정 시 활성화.
        # _execute_paper 진입 시 다시 'paper'로 복원되므로 단방향 set OK.
        try:
            self.risk_manager.set_trading_mode("live")
            num_live_pos = sum(len(om.positions) for om in self.order_managers.values())
            can_live, live_reason = self.risk_manager.check_can_trade(self.equity, num_live_pos)
            if not can_live:
                logger.warning(f"[리스크-LIVE] {symbol} 진입 차단: {live_reason}")
                return
        except Exception as _gate_err:
            logger.debug(f"[리스크-LIVE] 게이트 체크 실패(무시): {_gate_err}")
        # === Quant score 게이트 (엣지 없는 구간 차단, 2026-04-20) ===
        # |quant_alpha_score| < 0.25 면 진입 포기 — 최근 LIVE 손실 대부분 |score|<0.2 구간
        quant_min = self.config.get("trading", {}).get("live_quant_score_min", 0.25)
        if quant_min > 0:
            qs = c.get("quant_alpha_score")
            if qs is None:
                # 분석결과에서 quant score 추출 — 없으면 0으로 보수적 처리
                qs_dict = c.get("quant", {}) or {}
                qs = qs_dict.get("alpha_score", 0.0)
            if abs(float(qs)) < quant_min:
                logger.info(
                    f"[Quant-Gate] {symbol} LIVE 차단 — |score|={abs(float(qs)):.3f} "
                    f"< 최소 {quant_min:.2f} (엣지 부족)"
                )
                return
        # [LIVE_LONG_ONLY] LIVE만 숏 차단 (PAPER는 이미 상위에서 실행됨)
        if (
            getattr(self.strategy_manager, "live_long_only", False)
            and c.get("action") == "short"
        ):
            logger.info(
                f"[LIVE_LONG_ONLY] {c.get('symbol')} LIVE 숏 차단 "
                f"(PAPER 학습용 실행은 유지) | 원신호: {c.get('reason','?')}"
            )
            return
        # [2026-04-21 복원] LIVE 자가진단 일시정지 — 실제로 엔트리 차단
        # 이전에는 _live_paused=False 강제 해제라 pause 플래그가 장식용이었음.
        # 이제 pause = 정지. 해제는 _check_consecutive_losses 의 resume 조건에서만.
        if getattr(self, '_live_paused', False):
            logger.info(
                f"[LIVE일시정지] {c.get('symbol')} 엔트리 차단 "
                f"(사유: {getattr(self, '_live_pause_reason', '?')})"
            )
            return
        # 일일 DD 하드캡 정지
        if getattr(self, '_daily_dd_paused', False):
            logger.info(
                f"[일일DD정지] {c.get('symbol')} 엔트리 차단 "
                f"(사유: {getattr(self, '_daily_dd_reason', '?')})"
            )
            return
        # [Patch C, 2026-04-26] LIVE EV 음수 정지 — LIVE 전용
        if getattr(self, '_live_ev_paused', False):
            logger.info(
                f"[EV정지] {c.get('symbol')} LIVE 엔트리 차단 "
                f"(사유: {getattr(self, '_live_ev_pause_reason', '?')})"
            )
            return
        # symbol 은 함수 상단에서 이미 c.get("symbol") 로 설정됨
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

                # === Kelly 분수 사이징 (수학적 최적화, 2026-04-20) ===
                # f* = (p·b − q) / b, b=win/loss_ratio, p=WR, q=1-p
                # 최근 50 LIVE 트레이드 empirical로 계산 → 1/4-Kelly cap
                # f*≤0 (negative edge)이면 min_size로 축소 (탐색 유지)
                kelly_cfg = self.config.get("trading", {}).get("kelly_sizing", {}) or {}
                base_alloc = 0.90  # 기본 잔고 90%
                if kelly_cfg.get("enabled", False):
                    try:
                        f_frac = self._compute_live_kelly(
                            lookback=int(kelly_cfg.get("lookback_trades", 50)),
                            fraction=float(kelly_cfg.get("fraction", 0.25)),
                        )
                        min_s = float(kelly_cfg.get("min_size_pct", 0.10))
                        max_s = float(kelly_cfg.get("max_size_pct", 0.50))
                        base_alloc = max(min_s, min(max_s, f_frac))
                        logger.info(
                            f"[Kelly-LIVE] {symbol} f*={f_frac:.3f} → alloc={base_alloc:.1%} "
                            f"(min={min_s:.0%}, max={max_s:.0%})"
                        )
                    except Exception as e:
                        logger.debug(f"[Kelly-LIVE] 계산 실패, 기본값 사용: {e}")

                # 가용 잔고의 base_alloc × optimizer 스케일
                live_size = live_free * base_alloc * opt_scale
                if live_size < self.min_order_notional / c["dynamic_lev"]:
                    logger.warning(f"[LIVE] 잔고 부족: ${live_free:.2f} × {opt_scale:.1f} × {base_alloc:.2f} (필요: ${self.min_order_notional / c['dynamic_lev']:.2f})")
                    return
                c["size"] = live_size
                c["notional"] = live_size * c["dynamic_lev"]
            except Exception as e:
                logger.warning(f"[LIVE] 잔고 조회 실패: {e}")
                return  # 잔고 확인 안 되면 LIVE 진입 차단

            # === CVaR tail risk 체크 (tier=mid+ 활성화) ===
            if self.tier_manager.feature_enabled("cvar_risk", mode="live"):
                notional_check = c["size"] * c["dynamic_lev"]
                passed, cvar_pct, reason = self.risk_manager.check_cvar_limit(
                    proposed_notional=notional_check,
                    equity=max(self.live_equity, 1.0),
                    threshold_pct=0.05,
                )
                if not passed:
                    logger.warning(f"[CVaR-LIVE] {symbol} 진입 차단: {reason}")
                    add_live_log({
                        "time": datetime.utcnow().strftime("%H:%M:%S"),
                        "type": "cvar_block",
                        "mode": "LIVE",
                        "symbol": symbol,
                        "cvar_pct": round(cvar_pct, 4),
                        "reason": reason,
                    })
                    return

            # === Meta-Labeler 2차 결정 (tier=large+ 활성화) ===
            if self.tier_manager.feature_enabled("meta_labeling", mode="live") and self.meta_labeler.model is not None:
                try:
                    df_c = c.get("df")
                    if df_c is not None and len(df_c) > 0:
                        feature_cols = [col for col in self.meta_labeler.feature_columns[:-2] if col in df_c.columns]
                        if feature_cols:
                            row = df_c[feature_cols].iloc[-1].values.astype(float)
                            primary_sig = c["ml_signal"].get("signal", 0.0) if isinstance(c["ml_signal"], dict) else 0.0
                            primary_conf = c["confidence"]
                            meta = self.meta_labeler.predict(row, primary_sig, primary_conf)
                            if not meta["take"]:
                                logger.info(
                                    f"[Meta-LIVE] {symbol} 진입 SKIP (prob={meta['prob']:.2f} < {meta['threshold']})"
                                )
                                add_live_log({
                                    "time": datetime.utcnow().strftime("%H:%M:%S"),
                                    "type": "meta_block",
                                    "mode": "LIVE",
                                    "symbol": symbol,
                                    "prob": round(meta["prob"], 3),
                                })
                                return
                except Exception as e:
                    logger.debug(f"[Meta-LIVE] 체크 실패 → 통과: {e}")

            result = await om.open_position(
                symbol, c["action"], c["size"], c["dynamic_lev"],
                sl_pct=c.get("sl_pct"), tp_pct=c.get("tp_pct"),
                atr_pct=c.get("atr_pct", 0),
                trade_type=c.get("trade_type", "scalp"),
            )
            if result:
                coin = symbol.split("/")[0]
                trade_type = c.get("trade_type", "scalp")
                logger.info(
                    f"[LIVE🔥] {c['action'].upper()} {symbol} [{trade_type.upper()}] | "
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
                # IC Tracker — LIVE 실현 수익률 vs 예측 시그널 (알파 품질 모니터)
                try:
                    ml_sig = c.get("ml_signal", 0)
                    entry_signal = ml_sig.get("signal", 0.0) if isinstance(ml_sig, dict) else float(ml_sig or 0.0)
                    notional = live_result.get("notional") or (live_result.get("size", 0) * max(live_result.get("entry_price", 0), 1e-9))
                    realized = live_pnl / max(notional, 1.0)
                    self.ic_tracker.record(
                        signal=entry_signal,
                        realized_return=realized,
                        source="ensemble_live",
                    )
                    # LLM 단독 IC — ext_signal(dict)에서 llm_score 추출
                    ext_c = c.get("ext_signal") if isinstance(c, dict) else None
                    llm_entry = float(ext_c.get("llm_score", 0) or 0) if isinstance(ext_c, dict) else 0.0
                    if abs(llm_entry) > 1e-6:
                        self.ic_tracker.record(
                            signal=llm_entry,
                            realized_return=realized,
                            source="llm",
                        )
                except Exception as e:
                    logger.debug(f"[IC-LIVE] 기록 실패: {e}")
                if live_pnl < 0:
                    self.strategy_manager.record_loss()
                else:
                    self.strategy_manager.record_win()
                self._save_trade_with_context({
                    "exchange": exchange_name, "symbol": symbol, "side": live_result.get("side", "close"),
                    "price": live_result.get("exit_price", 0),
                    "amount": live_result.get("size", 0),
                    "pnl": live_pnl, "fee": live_result.get("fee", 0), "strategy": "live_concentration",
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

                # 손실 시 즉시 원인분석 리포트
                if live_pnl < 0 and isinstance(live_result, dict):
                    self._generate_loss_report(live_result, mode="LIVE")

    # ========== PAPER 자동청산 콜백 ==========

    def _on_paper_auto_close(self, trade: dict):
        """PaperTrader SL/TP/트레일링 자동 청산 시 호출 → DB 저장 + 학습

        [2026-04-21] variant-aware:
            trade["variant"]에 따라 해당 variant의 feedback/optimizer에 기록.
            MACRO_ON → self.feedback, self.strategy_optimizer_paper  (기본 정책, BTC 리저브 적립 대상)
            MACRO_OFF → self.feedback_off, self.strategy_optimizer_paper_off  (섀도우 정책)
        risk_manager.record_pnl / strategy_manager 연패 기록은 MACRO_ON만 반영 — LIVE 운영 기준.
        """
        try:
            symbol = trade.get("symbol", "?")
            pnl = trade.get("pnl", 0)
            reason = trade.get("reason", "auto")
            side = trade.get("side", "")
            variant = trade.get("variant", "PAPER_MACRO_ON")
            is_shadow = (variant == "PAPER_MACRO_OFF")

            # DB 저장 — variant 명시 태그
            self._save_trade_with_context({
                "exchange": "paper", "symbol": symbol, "side": side or "close",
                "price": trade.get("exit_price", 0),
                "amount": trade.get("size", 0),
                "pnl": pnl, "fee": trade.get("fee", 0),
                "strategy": f"paper_{reason.replace(' ', '_')}",
                "mode": "PAPER",
                "variant": variant,
            })

            # 학습 기록 — variant별 격리
            regime = self.adaptive.current_regime if hasattr(self, 'adaptive') else "unknown"
            target_feedback = self.feedback_off if is_shadow else self.feedback
            target_feedback.record_trade(
                {"pnl": pnl, "side": side, "symbol": symbol},
                {"regime": regime, "signal": 0, "confidence": 0,
                 "external_score": 0, "external_direction": "neutral",
                 "exit_reason": reason},
            )

            # LIVE 운영 기준(MACRO_ON)만 risk_manager/strategy_manager 상태 반영.
            # 섀도우 variant는 관찰 전용 — 공통 리스크 상태 오염 금지.
            if not is_shadow:
                self.risk_manager.record_pnl(pnl)
                if pnl < 0:
                    self.strategy_manager.record_loss()
                else:
                    self.strategy_manager.record_win()

            # StrategyOptimizer 기록 — variant별 격리
            target_opt = self.strategy_optimizer_paper_off if is_shadow else self.strategy_optimizer_paper
            p_hash = target_opt._config_to_hash(target_opt.current_config)
            target_opt.record_trade(p_hash, {
                "pnl": pnl, "timestamp": datetime.utcnow(),
                "symbol": symbol, "hour": datetime.utcnow().hour,
            })

            variant_tag = "PAPER-OFF" if is_shadow else "PAPER"
            add_live_log({
                "time": datetime.utcnow().strftime("%H:%M:%S"),
                "type": "trade_close", "mode": variant_tag,
                "symbol": symbol, "pnl": round(pnl, 2),
                "reason": reason,
            })
            logger.info(f"[{variant_tag}-Auto] {reason} {symbol} {side} | PnL: ${pnl:+.2f} → DB+학습 저장")
            # 알림 — MACRO_ON만 (MACRO_OFF는 관찰 전용, 알림 스팸 방지)
            if not is_shadow:
                tg_notify(format_trade_close("PAPER", symbol, pnl, reason), silent=True)

            # 손실 시 즉시 원인분석 리포트 — MACRO_ON만 (운영 기준)
            if pnl < 0 and not is_shadow:
                self._generate_loss_report(trade, mode="PAPER")

        except Exception as e:
            logger.error(f"[PAPER-Auto] 콜백 실패: {e}")

    # ========== A/B 섀도우 variant 실행 (MACRO_OFF) ==========

    async def _run_shadow_macro_off(
        self,
        *,
        symbol: str,
        exchange_name: str,
        df,
        ml_signal: dict,
        rl_action: int,
        rl_confidence: float,
        ext_signal: dict,
        momentum_signal: dict,
        mtf_signal: dict,
        quant_score: float,
        quant_regime_name: str,
        quant_risk_scale: float,
        adaptive_params: dict,
        funding_rate: float,
        atr_pct: float,
        volatility: float,
    ) -> None:
        """A/B 섀도우 variant (MACRO_OFF) 실행.

        수학적 격리 원칙 (2026-04-21):
            - 같은 시장 입력(ML/RL/EXT/MTF/Quant/ATR/Vol)을 primary와 동일하게 수용
            - strategy_manager.decide()에 variant_override={"disable_macro_block": True} 주입
            - 동일한 post-processing(MTF/Quant) 로직 적용 → 정책 차이만이 결과 차이를 설명
            - 자체 paper_trader_off(equity/positions) + feedback_off(learning) → 완전 격리
            - risk_manager.check_can_trade는 호출하지 않음 — shadow는 학습용,
              primary의 MaxDD/연패 상태에 오염되면 A/B 독립성 훼손
            - LIVE에 흘러가지 않음 — 순수 관찰/비교 목적
        """
        if self.mode not in ("paper", "dual"):
            return
        try:
            price = float(df["close"].iloc[-1])
            pt = self.paper_trader_off
            fb = self.feedback_off
            opt = self.strategy_optimizer_paper_off

            # ATR 슬리피지 모델 갱신 + 가격 업데이트(SL/TP 자동체크)
            try:
                pt.set_atr(symbol, atr_pct)
            except Exception:
                pass
            pt.update_prices({symbol: price})

            current_position = 1.0 if symbol in pt.positions else 0.0
            fb_blacklist = fb.get_entry_blacklist()

            # variant 결정: macro_block OFF
            decision = self.strategy_manager.decide(
                ml_signal, rl_action, rl_confidence, current_position,
                adaptive_params["regime"], external_signal=ext_signal,
                momentum=momentum_signal,
                feedback_blacklist=fb_blacklist,
                funding_rate=funding_rate,
                mode="paper",
                variant_override={"disable_macro_block": True},
                ohlcv_df=df,
            )

            # MTF 필터 — primary와 동일 로직 (양쪽 동일 적용해야 macro 차이만 분리됨)
            if decision.action in ["long", "short"]:
                mtf_agreement = mtf_signal.get("agreement", 0) if isinstance(mtf_signal, dict) else 0
                mtf_dir = mtf_signal.get("direction", "neutral") if isinstance(mtf_signal, dict) else "neutral"
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
                    decision.size = 0.0
                elif action_opposes_mtf and mtf_agreement >= 0.5:
                    decision.confidence *= 0.75
                elif action_agrees_mtf and mtf_agreement > 0.7:
                    decision.confidence = min(decision.confidence * 1.15, 1.0)

            # 퀀트 필터 — primary와 동일 로직
            if quant_regime_name == "ranging" and decision.action in ("long", "short"):
                decision.confidence *= 0.6
                if decision.confidence < self.strategy_manager.min_confidence:
                    decision.action = "hold"
                    decision.size = 0.0
            if quant_score != 0 and decision.action in ("long", "short"):
                quant_agrees = (
                    (decision.action == "long" and quant_score > 0) or
                    (decision.action == "short" and quant_score < 0)
                )
                if not quant_agrees and abs(quant_score) > 0.3:
                    decision.confidence *= 0.7
                elif quant_agrees and abs(quant_score) > 0.3:
                    decision.confidence = min(decision.confidence * 1.15, 1.0)

            if decision.action not in ("long", "short", "close"):
                return

            # Sizing — shadow equity 독립, feedback_off의 Kelly stats 사용
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
            trade_type = decision.trade_type
            tp_profile = self.config.get("trade_profiles", {}).get(trade_type, {})
            sl_pct_for_cap = tp_profile.get("sl_pct", self.config["risk"]["stop_loss_pct"])
            max_risk_pct = tp_profile.get("max_risk_pct", 0.05)
            dynamic_lev = self.risk_manager.cap_leverage_by_risk(
                dynamic_lev, sl_pct_for_cap, max_risk_pct
            )
            fb_scale = fb.get_position_scale(adaptive_params["regime"], decision.action)
            kelly_enabled = self.tier_manager.feature_enabled("kelly_enabled", mode="paper")
            kelly_fraction = self.tier_manager.get_feature("kelly_fraction", mode="paper", default=0.25) or 0.25
            kelly_stats = (
                fb.get_kelly_stats(regime=adaptive_params["regime"], side=decision.action)
                if kelly_enabled else None
            )
            size = self.risk_manager.calculate_position_size(
                pt.equity, decision.confidence, volatility,
                adaptive_params["position_scale"] * fb_scale * quant_risk_scale,
                kelly_enabled=kelly_enabled,
                kelly_fraction=float(kelly_fraction),
                kelly_stats=kelly_stats,
                atr_pct=atr_pct,
                leverage=dynamic_lev,
            )
            notional = size * dynamic_lev
            min_notional = self.min_order_notional * 1.05
            if notional < min_notional:
                size = max(size, min_notional / dynamic_lev)
                size = min(size, pt.equity * 0.50)
                notional = size * dynamic_lev

            # === 실행 ===
            if decision.action in ("long", "short"):
                if symbol not in pt.positions:
                    pt.open_position(
                        symbol, decision.action, size, price,
                        leverage=dynamic_lev,
                        sl_pct=tp_profile.get("sl_pct", self.config["risk"]["stop_loss_pct"])
                        * adaptive_params["stop_loss_mult"],
                        tp_pct=tp_profile.get("tp_pct", self.config["risk"]["take_profit_pct"]),
                        atr_pct=atr_pct,
                        trade_type=decision.trade_type,
                    )
                    logger.info(
                        f"[PAPER-OFF] {decision.action.upper()} {symbol} | "
                        f"${size:.2f} × {dynamic_lev}x = ${notional:.2f} | {decision.reason}"
                    )
                    add_live_log({
                        "time": datetime.utcnow().strftime("%H:%M:%S"),
                        "type": "trade_open",
                        "mode": "PAPER-OFF",
                        "symbol": symbol,
                        "action": decision.action,
                        "size": round(size, 2),
                        "price": price,
                        "leverage": dynamic_lev,
                        "notional": round(notional, 2),
                        "reason": decision.reason,
                    })
            elif decision.action == "close" and symbol in pt.positions:
                result = pt.close_position(symbol, price, decision.reason)
                if result:
                    pnl = result["pnl"]
                    self._save_trade_with_context({
                        "exchange": exchange_name, "symbol": symbol, "side": result.get("side", "close"),
                        "price": price, "amount": result["size"], "pnl": pnl,
                        "fee": result["fee"], "strategy": "hybrid_shadow",
                        "mode": "PAPER", "variant": "PAPER_MACRO_OFF",
                    })
                    fb.record_trade(result, {
                        "regime": adaptive_params["regime"],
                        "signal": ml_signal.get("signal", 0) if isinstance(ml_signal, dict) else 0,
                        "confidence": decision.confidence,
                        "volatility": volatility,
                        "external_score": ext_signal.get("score", 0) if isinstance(ext_signal, dict) else 0,
                        "external_direction": ext_signal.get("direction", "neutral") if isinstance(ext_signal, dict) else "neutral",
                        "exit_reason": "strategy_close_shadow",
                        "confirming_sources": getattr(decision, "confirming_sources", []),
                        "entry_path": "+".join(sorted(getattr(decision, "confirming_sources", []))),
                    })
                    p_hash = opt._config_to_hash(opt.current_config)
                    opt.record_trade(p_hash, {
                        "pnl": pnl, "timestamp": datetime.utcnow(),
                        "symbol": symbol, "hour": datetime.utcnow().hour,
                    })
                    logger.info(f"[PAPER-OFF] 청산 {symbol} | PnL: ${pnl:+.2f} → shadow 학습저장")
                    add_live_log({
                        "time": datetime.utcnow().strftime("%H:%M:%S"),
                        "type": "trade_close",
                        "mode": "PAPER-OFF",
                        "symbol": symbol,
                        "pnl": round(pnl, 2),
                        "reason": decision.reason,
                    })

            # stale 포지션 자기수정 — shadow도 동일 기준
            if symbol in pt.positions:
                pos = pt.positions[symbol]
                age_minutes = (datetime.utcnow() - pos.entry_time).total_seconds() / 60
                if age_minutes > 240 and pos.unrealized_pnl < 0:
                    result = pt.close_position(symbol, price, f"자기수정: {age_minutes:.0f}분 stale (shadow)")
                    if result:
                        pnl = result["pnl"]
                        self._save_trade_with_context({
                            "exchange": exchange_name, "symbol": symbol, "side": result.get("side", "close"),
                            "price": price, "amount": result["size"], "pnl": pnl,
                            "fee": result["fee"], "strategy": "auto_close_shadow",
                            "mode": "PAPER", "variant": "PAPER_MACRO_OFF",
                        })
                        fb.record_trade(result, {
                            "regime": adaptive_params["regime"],
                            "signal": 0, "confidence": 0,
                            "external_score": 0, "external_direction": "neutral",
                            "exit_reason": "stale_shadow",
                        })
                        logger.info(f"[PAPER-OFF-자기수정] stale 청산 {symbol} | PnL: ${pnl:+.2f}")
        except Exception as e:
            logger.warning(f"[PAPER-OFF] shadow variant 실행 실패: {e}")

    # ========== 손실 자동 피드백 리포트 ==========

    def _generate_loss_report(self, trade: dict, mode: str = "LIVE"):
        """큰 손실 발생 시 즉시 원인 분석 리포트 → 텔레그램 전송

        trade: close_position 반환값 (symbol, side, entry_price, exit_price, pnl, reason, ...)
        equity 대비 2% 이상 손실 시 자동 발동
        """
        try:
            pnl = trade.get("pnl", 0)
            if pnl >= 0:
                return  # 수익이면 스킵

            # 손실 비율 계산
            equity = self.equity if self.equity > 0 else self.initial_capital
            loss_pct = abs(pnl) / equity * 100

            # 2% 미만 소액 손실은 간략 로그만
            if loss_pct < 2.0:
                return

            symbol = trade.get("symbol", "?")
            side = trade.get("side", "?")
            entry_price = trade.get("entry_price", 0)
            exit_price = trade.get("exit_price", 0)
            reason = trade.get("reason", "?")
            size = trade.get("size", 0)

            # === 1. 가격 변동 분석 ===
            if entry_price > 0 and exit_price > 0:
                price_move_pct = abs(exit_price - entry_price) / entry_price * 100
            else:
                price_move_pct = 0

            # === 2. 레버리지 역산 (pnl / price_move로 추정) ===
            if price_move_pct > 0 and size > 0:
                notional = size * entry_price
                expected_pnl_1x = notional * (price_move_pct / 100)
                est_leverage = abs(pnl) / expected_pnl_1x if expected_pnl_1x > 0 else 0
            else:
                est_leverage = 0

            # === 3. 최근 거래 패턴 (연패 확인) ===
            recent = self.storage.get_recent_trades(mode=mode, limit=10)
            recent_pnls = [t.get("pnl", 0) for t in recent if t.get("pnl") is not None]
            consecutive_losses = 0
            for p in recent_pnls:
                if p < 0:
                    consecutive_losses += 1
                else:
                    break

            total_recent_pnl = sum(recent_pnls)
            recent_wins = sum(1 for p in recent_pnls if p > 0)
            recent_wr = recent_wins / len(recent_pnls) * 100 if recent_pnls else 0

            # === 4. 현재 시장 상태 ===
            regime = self.adaptive.current_regime if hasattr(self, 'adaptive') else "?"
            risk_status = self.risk_manager.get_status()
            drawdown = risk_status.get("current_drawdown", 0) * 100

            # === 5. 원인 진단 ===
            diagnosis = []
            recommendations = []

            if price_move_pct < 2.0 and loss_pct > 5.0:
                diagnosis.append(f"가격 {price_move_pct:.1f}% 변동에 손실 {loss_pct:.1f}% → 과다 레버리지")
                recommendations.append("레버리지 축소 필요")

            if "SL" in reason:
                diagnosis.append("SL 도달로 청산")
                if price_move_pct < 1.5:
                    diagnosis.append("SL이 너무 타이트했을 가능성")
                    recommendations.append("ATR 기반 SL 확인 필요")

            if consecutive_losses >= 3:
                diagnosis.append(f"{consecutive_losses}연패 진행 중")
                recommendations.append("쿨다운 또는 포지션 축소 권장")

            if regime in ("extreme_volatility", "high_volume_breakout"):
                diagnosis.append(f"고변동 장세({regime})에서 손실")
                recommendations.append("변동성 장세에서 진입 자제")

            if side == "short" and "bullish" in str(self.last_signals.get("momentum", {})):
                diagnosis.append("숏 포지션 vs 상승 모멘텀 불일치")
                recommendations.append("모멘텀 방향 확인 후 진입")

            if not diagnosis:
                diagnosis.append("구체적 원인 추가 분석 필요")

            if not recommendations:
                recommendations.append("다음 진입 시 확신도 기준 상향")

            # === 6. 자동 조치 실행 (리포트 + 실제 파라미터 변경) ===
            auto_actions = []

            # 6a. 과다 레버리지 → 다음 진입 포지션 축소 (feedback adjustments 반영)
            if price_move_pct < 2.0 and loss_pct > 5.0:
                adj = self.feedback.feedback.get("adjustments", {})
                old_scale = adj.get("overlev_scale", 1.0)
                new_scale = max(0.3, old_scale * 0.7)
                adj["overlev_scale"] = new_scale
                self.feedback._save()
                auto_actions.append(f"과다레버리지 감지 → 포지션 스케일 {old_scale:.0%}→{new_scale:.0%}")

            # 6b. 연패 → strategy_manager에 즉시 반영
            if consecutive_losses >= 2:
                self.strategy_manager.record_loss()
                auto_actions.append(
                    f"연패 {consecutive_losses}회 → 확신도 +{self.strategy_manager._loss_confidence_boost:.2f} 상향"
                )

            # 6c. 고변동 레짐 손실 → 해당 레짐 스케일 축소
            if regime in ("extreme_volatility", "high_volume_breakout"):
                adj = self.feedback.feedback.get("adjustments", {})
                key = f"regime_scale_{regime}"
                old_scale = adj.get(key, 1.0)
                new_scale = max(0.2, old_scale * 0.6)
                adj[key] = new_scale
                self.feedback._save()
                auto_actions.append(f"고변동 레짐 '{regime}' → 스케일 {old_scale:.0%}→{new_scale:.0%}")

            # 6d. 큰 손실(5%+) → 즉시 쿨다운 (3분간 진입 차단)
            if loss_pct >= 5.0:
                self.strategy_manager._loss_penalty_holds = max(
                    self.strategy_manager._loss_penalty_holds, 6  # 6루프 = 3분
                )
                self.strategy_manager._loss_confidence_boost = max(
                    self.strategy_manager._loss_confidence_boost, 0.10
                )
                auto_actions.append(f"큰 손실 {loss_pct:.1f}% → 3분 쿨다운 + 확신도 +0.10")

            # === 리포트 생성 ===
            report = (
                f"🚨 <b>손실 분석 리포트</b> [{mode}]\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"종목: {symbol} ({side})\n"
                f"진입: ${entry_price:,.2f} → 청산: ${exit_price:,.2f}\n"
                f"가격변동: {price_move_pct:.2f}%\n"
                f"손실: <b>${pnl:+.2f}</b> (자본의 {loss_pct:.1f}%)\n"
                f"추정 레버리지: {est_leverage:.1f}x\n"
                f"청산 사유: {reason}\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>원인 진단</b>\n"
            )
            for d in diagnosis:
                report += f"  • {d}\n"

            report += f"━━━━━━━━━━━━━━━━━━\n"
            report += f"🔧 <b>개선 권고</b>\n"
            for r in recommendations:
                report += f"  • {r}\n"

            if auto_actions:
                report += f"━━━━━━━━━━━━━━━━━━\n"
                report += f"⚡ <b>자동 조치 실행</b>\n"
                for a in auto_actions:
                    report += f"  ✅ {a}\n"

            report += (
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📈 최근 {len(recent_pnls)}건: 승률 {recent_wr:.0f}% | "
                f"누적 ${total_recent_pnl:+.2f}\n"
                f"연패: {consecutive_losses}회 | 레짐: {regime}\n"
                f"DD: {drawdown:.1f}% | 잔고: ${equity:.2f}"
            )

            logger.warning(f"[손실리포트] {symbol} ${pnl:+.2f} ({loss_pct:.1f}%) | 진단: {'; '.join(diagnosis)}")
            tg_notify(report)

        except Exception as e:
            logger.error(f"[손실리포트] 생성 실패: {e}")

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

            # ──── 3. Algo 주문 체크 비활성화 (내부 SL/TP 모니터링 방식 사용) ────
            # 거래소에 SL/TP Algo 주문을 설정하지 않으므로 체크 불필요
            # _check_algo_orders()가 잘못된 사이즈로 주문 생성하여 충돌 유발

            # ──── 4. 연속 손실 / 승률 감시 → 자동 일시정지 ────
            loss_result = self._check_consecutive_losses()
            if loss_result:
                issues.append(loss_result)

            # ──── 4b. 일일 MaxDD 하드캡 (2026-04-21) ────
            dd_result = self._check_daily_drawdown()
            if dd_result:
                issues.append(dd_result)

            # ──── 4c. LIVE EV 모니터 (Patch C, 2026-04-26) ────
            ev_result = self._check_live_ev_monitor()
            if ev_result:
                issues.append(ev_result)

            # ──── 4d. 스키마 정합성 (Patch D, 2026-04-26) ────
            for sch_issue in self._check_schema_health(sample=False):
                issues.append(sch_issue)

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

            # ──── save_trade 누락 진단 (cumulative vs cumulative) ────
            # [2026-04-25 수정] 기존: 24h DB == 0 + fb_total > 0 → "save_trade 누락"
            #   → 단순히 최근 24h에 거래가 없었을 뿐인 케이스도 잘못 트리거됨.
            # 수정: 누적 fb_total 과 누적 DB 카운트를 비교하여 'save 메커니즘 자체'가
            # 깨진 것인지 판단. 그리고 '거래 활동 부재'는 별도 시그널로 보고.
            try:
                db_total_counts = self.storage.get_trade_counts_total()
                total_db_cum = sum(db_total_counts.values())
            except Exception:
                total_db_cum = sum(db_counts.values())  # fallback
            total_db_24h = sum(db_counts.values())

            if fb_total > 50 and total_db_cum == 0:
                # 누적 피드백은 있는데 DB는 텅 비어있다 → save 진짜 깨짐
                issues.append(f"feedback {fb_total}건, DB 누적 0건 — save_trade 진짜 누락")
            elif fb_total > 0 and total_db_cum > 0 and fb_total - total_db_cum > 50:
                # gap 50 이상 → save가 부분적으로 깨졌을 가능성
                gap = fb_total - total_db_cum
                issues.append(f"feedback {fb_total} vs DB {total_db_cum} (gap {gap}) — save 일부 누락 의심")

            # ──── 거래 활동 부재 (별도 신호) ────
            try:
                last_age_h = self.storage.get_last_trade_age_hours()
            except Exception:
                last_age_h = None
            if total_db_cum > 0 and total_db_24h == 0:
                if last_age_h is not None and last_age_h >= 12:
                    issues.append(
                        f"거래 활동 정지: 마지막 거래 {last_age_h:.1f}시간 전 — "
                        f"PAPER/LIVE 게이팅 점검 필요"
                    )

            # ──── 퀀트 시그널 + ML 피처 통합 체크 ────
            quant_status = self._check_quant_integration()
            if quant_status.get("issues"):
                for qi in quant_status["issues"]:
                    issues.append(qi)
            if quant_status.get("fixes"):
                for qf in quant_status["fixes"]:
                    fixes.append(qf)

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

            # ──── Algo 주문 확인 비활성화 (내부 SL/TP 모니터링 방식) ────

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
            if not self.exchange_clients:
                balance_info = "Paper 전용 (거래소 미연결)"
            for name, client in self.exchange_clients.items():
                try:
                    bal = await client.get_balance()
                    total = bal.get("total", 0) or 0
                    free = bal.get("free", 0) or 0
                    if total > 0:
                        balance_info = f"${total:,.2f} (가용: ${free:,.2f})"
                    else:
                        balance_info = f"$0.00 (가용: $0.00)"
                except Exception as e:
                    err_msg = str(e)
                    if "Invalid Api-Key" in err_msg or "invalid" in err_msg.lower():
                        balance_info = f"❌ {name} API 키 무효 — 키 재발급 필요"
                        issues.append(f"{name} API 키 인증 실패")
                    elif "permission" in err_msg.lower():
                        balance_info = f"❌ {name} API 권한 부족 — 선물 권한 확인"
                        issues.append(f"{name} API 권한 부족")
                    else:
                        balance_info = f"❌ {name} 잔고조회 실패: {err_msg[:60]}"
                    logger.warning(f"[진단] {name} 잔고 조회 실패: {e}")

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
                f"🧬 퀀트: ML피처 {quant_status['status'].get('ml_quant_features', 0)}개 | "
                f"OB={quant_status['status'].get('ob_history', 0)} "
                f"VPIN={quant_status['status'].get('vpin_buckets', 0)} "
                f"Basis={quant_status['status'].get('basis_history', 0)}",
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
                        self._save_trade_with_context({
                            "exchange": "binance", "symbol": symbol, "side": result.get("side", "close"),
                            "price": result.get("exit_price", 0),
                            "amount": result.get("size", 0),
                            "pnl": pnl, "fee": result.get("fee", 0),
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
        """연속 손실 / 승률 감시 → LIVE 자동 일시정지.

        [2026-04-21 개정]
        - Pause threshold: 최근 5건 중 전패 → 정지 (기존 동일)
        - Resume 조건 강화: 최근 5건 중 ≥ 2승 AND 정지 후 최소 30분 경과

        [2026-04-25 핵심 버그픽스: Resume Deadlock]
        - 문제: LIVE 일시정지 중 → LIVE 거래 0건 → recent_live[:5]는 정지 직전의
          전패 5건 그대로 → wins5는 영원히 0 → 영구 정지(deadlock).
        - 해결책 A: Time-based force-resume — 정지 후 6시간 경과 시 PAPER 최근 성과를
          참고하여 LIVE 자동 재개. PAPER가 회복했다면 LIVE도 재시도할 가치가 있음.
        - 해결책 B: 12시간 hard-resume — PAPER 성과와 무관하게 무조건 재개 시도
          (재개 즉시 또 전패하면 다시 자동 정지되므로 안전).
        - 해결책 C: 'PAPER 최근 5건 ≥ 3승' 충족 시 LIVE 즉시 재개 (정지 30분+ 후).
        """
        try:
            # [Patch N, 2026-05-22] 7일 윈도우 필터 — deadlock 근본 해소.
            # 기존 버그: get_recent_trades는 시간 무관 최근 10건 → LIVE 거래가 끊기면
            #   한 달 전 전패 5건을 영원히 보고 정지/재개/정지 무한 반복.
            #   (실측: LIVE 04-20 마지막 거래 후 30일간 0건, deadlock 지속 확인)
            _recent_live_raw = self.storage.get_recent_trades(mode="LIVE", limit=10)
            _cutoff_7d = datetime.utcnow() - timedelta(days=7)
            recent_live = []
            for _t in _recent_live_raw:
                _ts = _t.get("timestamp")
                try:
                    _dt = datetime.fromisoformat(
                        str(_ts).replace("Z", "").split("+")[0].split(".")[0]
                    )
                    if _dt >= _cutoff_7d:
                        recent_live.append(_t)
                except Exception:
                    # 파싱 실패 시 보수적으로 포함
                    recent_live.append(_t)

            # ──── 정지 중일 때: PAPER 성과 + 시간 기반 재개 검사 (Resume Deadlock 방지) ────
            if getattr(self, '_live_paused', False):
                # [Patch N] 7일 윈도우에 LIVE 거래가 5건 미만이면 stale-pause 자동 해제.
                # 옛날 전패 데이터로 영구 정지되는 deadlock 차단.
                if len(recent_live) < 5:
                    self._live_paused = False
                    _pd = datetime.utcnow() - getattr(self, '_live_pause_time', datetime.utcnow())
                    logger.warning(
                        f"[자가진단] ✅ LIVE stale-pause 자동 해제 — 7일 내 LIVE 거래 "
                        f"{len(recent_live)}건(<5) → 옛 전패 데이터 만료 (정지 {_pd})"
                    )
                    try:
                        tg_notify(
                            f"✅ <b>LIVE 정지 자동 해제 (Patch N)</b>\n"
                            f"━━━━━━━━━━━━━\n"
                            f"7일 윈도우 LIVE 거래 {len(recent_live)}건 (필요 5건)\n"
                            f"옛 전패 데이터 만료 → LIVE 거래 재개\n"
                            f"정지 기간: {_pd}"
                        )
                    except Exception:
                        pass
                    return None
            if getattr(self, '_live_paused', False):
                pause_dur = datetime.utcnow() - getattr(self, '_live_pause_time', datetime.utcnow())
                pause_sec = pause_dur.total_seconds()
                min_dur_ok = pause_sec >= 1800  # 최소 30분

                # 1) 기존 LIVE-기반 재개 (LIVE 거래가 새로 발생한 경우)
                if len(recent_live) >= 5:
                    last5 = recent_live[:5]
                    wins5 = sum(1 for t in last5 if (t.get("pnl") or 0) > 0)
                    wr5 = wins5 / len(last5)
                    if min_dur_ok and wins5 >= 2:
                        self._live_paused = False
                        logger.info(f"[자가진단] ✅ LIVE 재개 (LIVE 5건 WR {wr5:.0%}, 정지 {pause_dur})")
                        tg_notify(
                            f"✅ <b>LIVE 자동 재개</b> (LIVE 회복)\n"
                            f"━━━━━━━━━━━━━\n"
                            f"최근 5건 WR: {wr5:.0%} ({wins5}/5)\n"
                            f"정지 기간: {pause_dur}"
                        )
                        return None

                # 2) PAPER 회복 기반 재개 (LIVE는 정지 중이므로 LIVE는 안 늘어남)
                #    → PAPER 최근 5건 중 ≥ 3승이면 재개 (정지 30분+ 후)
                if min_dur_ok:
                    try:
                        recent_paper = self.storage.get_recent_trades(mode="PAPER", limit=5)
                    except Exception:
                        recent_paper = []
                    if len(recent_paper) >= 5:
                        paper_wins = sum(1 for t in recent_paper if (t.get("pnl") or 0) > 0)
                        if paper_wins >= 3:
                            self._live_paused = False
                            logger.info(
                                f"[자가진단] ✅ LIVE 재개 (PAPER 회복 신호: 5건 중 {paper_wins}승)"
                            )
                            tg_notify(
                                f"✅ <b>LIVE 자동 재개</b> (PAPER 회복)\n"
                                f"━━━━━━━━━━━━━\n"
                                f"PAPER 최근 5건: {paper_wins}승/5\n"
                                f"정지 기간: {pause_dur}"
                            )
                            return None

                # 3) Time-based soft-resume — 6시간 경과 시 자동 재개 (재실패하면 또 정지됨)
                if pause_sec >= 6 * 3600:
                    self._live_paused = False
                    logger.warning(
                        f"[자가진단] ⏰ LIVE 시간 기반 강제 재개 (정지 {pause_dur} 경과)"
                    )
                    tg_notify(
                        f"⏰ <b>LIVE 시간 기반 자동 재개</b>\n"
                        f"━━━━━━━━━━━━━\n"
                        f"정지 기간: {pause_dur} (6시간 한계)\n"
                        f"⚠️ 재진입 후 다시 5연패 시 자동 재정지\n"
                        f"📝 Resume Deadlock 방지 안전장치"
                    )
                    return None

                # 정지 유지 — 진단 메시지 갱신
                remain_sec = max(0, 6 * 3600 - int(pause_sec))
                return (
                    f"LIVE 정지 중 (정지 {int(pause_sec/60)}분, "
                    f"강제 재개 {remain_sec//60}분 후)"
                )

            # ──── 정지 상태가 아닐 때: 신규 정지 판단 ────
            if len(recent_live) >= 5:
                last5 = recent_live[:5]
                wins5 = sum(1 for t in last5 if (t.get("pnl") or 0) > 0)
                wr5 = wins5 / len(last5)
                wins10 = sum(1 for t in recent_live if (t.get("pnl") or 0) > 0)
                wr10 = wins10 / len(recent_live)

                # === Pause ===
                if wr5 == 0:
                    self._live_paused = True
                    self._live_pause_reason = f"최근 5건 전패 (전체 10건 WR {wr10:.0%})"
                    self._live_pause_time = datetime.utcnow()
                    logger.warning(f"[자가진단] ⛔ LIVE 자동 일시정지: {self._live_pause_reason}")
                    tg_notify(
                        f"⛔ <b>LIVE 자동 일시정지</b>\n"
                        f"━━━━━━━━━━━━━\n"
                        f"사유: {self._live_pause_reason}\n"
                        f"최근 5건 PnL: {[round(t.get('pnl', 0), 2) for t in last5]}\n"
                        f"📝 PAPER 모드는 계속 학습 중\n"
                        f"🔄 재개 조건:\n"
                        f"   • LIVE 최근 5건 ≥ 2승 + 30분 경과, 또는\n"
                        f"   • PAPER 최근 5건 ≥ 3승 + 30분 경과, 또는\n"
                        f"   • 6시간 경과 시 강제 재개"
                    )
                    return f"LIVE 일시정지: 최근 5건 전패"

                # === Soft warning ===
                if wr10 <= 0.10 and len(recent_live) >= 8:
                    return f"LIVE 저조: 최근 {len(recent_live)}건 WR {wr10:.0%}"

        except Exception as e:
            logger.debug(f"[진단] 연속 손실 체크 실패: {e}")

        return None

    def _check_daily_drawdown(self) -> str | None:
        """일일 MaxDD 하드캡 — LIVE 전용 정지.

        [2026-04-21 개정] 사용자 원칙 반영: PAPER는 학습 데이터 수집 목적이므로
        손실도 부정 샘플로 가치 있음 → PAPER는 DD 체크 대상에서 제외.
        LIVE 일 누적 PnL / initial_capital ≤ -3% 면 자정(UTC)까지 LIVE만 정지.
        """
        try:
            from datetime import datetime as _dt
            today_utc = _dt.utcnow().strftime("%Y-%m-%d")
            # LIVE 거래만 집계 (PAPER 제외)
            cursor = self.storage.conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM trades "
                "WHERE DATE(timestamp) = ? AND mode = 'LIVE'",
                (today_utc,),
            )
            today_pnl = float(cursor.fetchone()[0] or 0)
            base_cap = max(float(self.initial_capital), 1.0)
            dd_pct = (today_pnl / base_cap) * 100.0

            # Trigger
            if dd_pct <= self._daily_dd_threshold_pct and not self._daily_dd_paused:
                self._daily_dd_paused = True
                self._daily_dd_reason = (
                    f"LIVE 일 누적 PnL ${today_pnl:.2f} = {dd_pct:+.2f}% "
                    f"(한계 {self._daily_dd_threshold_pct:.1f}%)"
                )
                self._daily_dd_trigger_time = _dt.utcnow()
                logger.warning(f"[일일DD] 🛑 LIVE 정지: {self._daily_dd_reason}")
                tg_notify(
                    f"🛑 <b>일일 MaxDD 하드캡 (LIVE)</b>\n"
                    f"━━━━━━━━━━━━━\n"
                    f"{self._daily_dd_reason}\n"
                    f"LIVE 만 00:00 UTC 까지 정지 (PAPER 는 학습 지속)\n"
                    f"📝 학습 사이클도 계속 실행 (모델 개선)"
                )
                return f"일일 DD 정지 (LIVE only): {dd_pct:+.2f}%"

            # Auto release at UTC midnight rollover
            if self._daily_dd_paused and self._daily_dd_trigger_time:
                trigger_day = self._daily_dd_trigger_time.strftime("%Y-%m-%d")
                if today_utc != trigger_day:
                    self._daily_dd_paused = False
                    logger.info(f"[일일DD] ✅ 신규 UTC 일자 — LIVE 정지 해제 (오늘 LIVE PnL {today_pnl:+.2f})")
                    tg_notify(
                        f"✅ <b>일일 DD 정지 해제</b>\n"
                        f"신규 UTC 일자 시작 — LIVE 재개"
                    )

            if self._daily_dd_paused:
                return f"일일 DD 정지 지속 (LIVE only, {dd_pct:+.2f}%)"

        except Exception as e:
            logger.debug(f"[진단] 일일 DD 체크 실패: {e}")

        return None

    def _check_schema_health(self, sample: bool = True) -> list[str]:
        """[Patch D, 2026-04-26] 스키마 정합성 검증 — silent breakage 재발 방지.

        2026-04-21 사고 재현 방지:
        - external_manager가 DB에 ext_llm_confidence/_ev/_max_severity 로 insert
        - 학습 코드는 ext_llm_conviction/_expected_value/_max_risk_severity 로 SELECT
        - 5시간 동안 1,594건 silent log 만 누적, 거래 0건

        검사 항목:
        (1) llm_signal_snapshots 테이블의 컬럼 ⊇ external_manager 가 사용하는 컬럼
        (2) XGBoost 모델의 feature_columns ⊆ FeatureEngineer가 생성하는 컬럼 (drift 검출)
        (3) 모델이 학습된 컬럼이 inference 시점 DataFrame에 모두 존재하는지

        Returns: 발견된 문제점 list (빈 리스트면 정상)
        """
        issues = []
        try:
            # === (1) llm_signal_snapshots 스키마 ↔ external_manager 사용 컬럼 ===
            # 행이 1개 이상 존재할 때만 검증 — 빈 테이블은 스키마가 동적 생성되므로 false-positive.
            try:
                row_cnt = self.storage.conn.execute(
                    "SELECT COUNT(*) FROM llm_signal_snapshots"
                ).fetchone()[0]
                if row_cnt > 0:
                    cur = self.storage.conn.execute(
                        "SELECT name FROM pragma_table_info('llm_signal_snapshots')"
                    )
                    db_cols = {r[0] for r in cur.fetchall()}
                    # DB는 bare column 사용 (score, conviction, …); ext_llm_* prefix는
                    # trainer.py 가 add_prefix("ext_llm_") 로 in-memory 추가한다.
                    # 따라서 bare name 의 존재 여부를 검사해야 정합.
                    expected_bare = {
                        "score", "conviction", "expected_value",
                        "max_risk_severity", "direction",
                    }
                    missing = expected_bare - db_cols
                    if missing:
                        issues.append(
                            f"[Schema] llm_signal_snapshots bare 컬럼 누락: {sorted(missing)} "
                            f"(external_manager.py insert ↔ DB 스키마 불일치)"
                        )
            except Exception as e:
                logger.debug(f"[Schema] llm_signal_snapshots 점검 실패: {e}")

            # === (2/3) XGBoost feature_columns vs FeatureEngineer 출력 ===
            xgb = getattr(self.ensemble, "xgb", None)
            if xgb is not None and getattr(xgb, "model", None) is not None:
                model_feats = list(getattr(xgb, "feature_columns", []) or [])
                if model_feats and sample:
                    try:
                        # 샘플 DF 생성으로 FeatureEngineer가 만들어낼 컬럼 셋 검사
                        import numpy as _np
                        import pandas as _pd
                        idx = _pd.date_range("2025-01-01", periods=300, freq="5min", tz="UTC")
                        sdf = _pd.DataFrame({
                            "open": _np.random.rand(300) * 100 + 50,
                            "high": _np.random.rand(300) * 100 + 55,
                            "low": _np.random.rand(300) * 100 + 45,
                            "close": _np.random.rand(300) * 100 + 50,
                            "volume": _np.random.rand(300) * 1000,
                        }, index=idx)
                        gen = self.feature_engineer.generate(sdf)
                        cur_feats = set(self.feature_engineer.get_feature_columns(gen))
                        # 동적 주입 피처 (학습 시점에만 set_btc_reference / external_features 로 추가) 제외
                        # btc_*: cross-asset BTC 선행 / ext_*: 외부 데이터 매니저 / deriv_*: 파생 스냅샷
                        DYN_PREFIXES = ("btc_", "ext_", "deriv_")
                        missing = [
                            c for c in model_feats
                            if c not in cur_feats and not c.startswith(DYN_PREFIXES)
                        ]
                        if missing:
                            issues.append(
                                f"[Schema] 모델학습 컬럼 {len(missing)}개가 현재 FeatureEngineer "
                                f"출력에 없음 (예시: {missing[:5]}) — 재학습 필요"
                            )
                    except Exception as e:
                        logger.debug(f"[Schema] FeatureEngineer 샘플 점검 실패: {e}")

            if not issues:
                logger.debug("[Schema] ✅ 검증 통과 — 모델/DB/피처 스키마 정합")
            else:
                for it in issues:
                    logger.warning(it)
        except Exception as e:
            logger.debug(f"[Schema] 검증 실행 실패: {e}")
        return issues

    def _check_live_ev_monitor(self) -> str | None:
        """[Patch C, 2026-04-26] LIVE EV 자동 정지 감시.

        무인 운영 핵심 안전장치:
        - 최근 N LIVE 트레이드(HIPPO 제외)의 expected value (mean PnL/trade) 계산.
        - EV ≤ threshold(기본 -$0.5/trade) 면 LIVE 만 정지 (PAPER 무관).
        - 재개 조건: EV가 threshold 초과로 회복 (시간 기반 재개 없음).
                     → 모델 재학습 후 새 신호로 EV 회복 시에만 자연스럽게 해제.

        분리 이유:
        - _check_consecutive_losses 는 5건 전패라는 단기 노이즈 정지.
        - _check_live_ev_monitor 는 50건 표본 기반 통계적 엣지 정지.
          (WR 60% 라도 RR<<1 이면 EV<0 → 의미없는 거래로 자본 잠식.)

        반환: 정지 메시지 (issues 리포트용) 또는 None.
        """
        try:
            # [Patch I, 2026-04-28] 시간 윈도우 필터 추가 — 7일 이상 묵은 트레이드는 EV 판단에서 제외.
            # 기존: 최근 N건 (시간 무관) → 거래 정지 시 새 트레이드 0건이라 옛날 데이터로 영원히 정지 (catch-22).
            # 변경: 최근 7일 내 LIVE 트레이드 중 최대 N건. 7일 이상 새 트레이드 없으면 표본 부족으로 자연 해제.
            cur = self.storage.conn.execute(
                """
                SELECT pnl, timestamp FROM trades
                WHERE mode='LIVE' AND symbol != 'HIPPO/USDT:USDT'
                  AND timestamp > datetime('now', '-7 days')
                ORDER BY timestamp DESC LIMIT ?
                """,
                (int(self._live_ev_lookback),),
            )
            pnls = [float(r[0]) for r in cur.fetchall() if r and r[0] is not None]
            n = len(pnls)
            if n < self._live_ev_min_samples:
                # 표본 부족 — 판단 보류. 단, 이미 정지된 상태로 표본이 부족해진 경우(데드락) 자동 해제.
                if self._live_ev_paused:
                    pause_dur = datetime.utcnow() - (self._live_ev_pause_time or datetime.utcnow())
                    self._live_ev_paused = False
                    logger.info(
                        f"[EV모니터] ✅ LIVE 자동 해제 — 7일 윈도우 표본 부족({n}건<{self._live_ev_min_samples}건). "
                        f"정지 기간 {pause_dur} → 신규 LIVE 트레이드로 EV 재평가 시작"
                    )
                    try:
                        tg_notify(
                            f"✅ <b>LIVE EV 정지 자동 해제</b>\n"
                            f"━━━━━━━━━━━━━\n"
                            f"7일 윈도우에 LIVE 트레이드 {n}건 (필요 {self._live_ev_min_samples}건)\n"
                            f"옛날 표본 만료 → 새 트레이드로 EV 재평가 시작\n"
                            f"정지 기간: {pause_dur}"
                        )
                    except Exception:
                        pass
                return None

            ev = sum(pnls) / n
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / n

            # === 신규 정지 ===
            if not self._live_ev_paused and ev <= self._live_ev_threshold:
                self._live_ev_paused = True
                self._live_ev_pause_time = datetime.utcnow()
                self._live_ev_pause_reason = (
                    f"최근 {n}건 LIVE EV={ev:+.2f}$/trade ≤ {self._live_ev_threshold:+.2f}$ "
                    f"(WR {wr:.0%})"
                )
                logger.warning(f"[EV모니터] 🛑 LIVE 자동 정지: {self._live_ev_pause_reason}")
                try:
                    tg_notify(
                        f"🛑 <b>LIVE EV 자동 정지 (Patch C)</b>\n"
                        f"━━━━━━━━━━━━━\n"
                        f"최근 {n}건 평균 PnL: ${ev:+.2f}/거래\n"
                        f"승률: {wr:.0%} ({wins}/{n})\n"
                        f"한계: ${self._live_ev_threshold:+.2f}/거래\n"
                        f"━━━━━━━━━━━━━\n"
                        f"📝 PAPER 학습 계속 진행\n"
                        f"📝 모델 재학습 후 EV 회복 시 자동 재개\n"
                        f"⚠️ 시간 기반 강제 재개 없음 — 엣지 회복까지 정지"
                    )
                except Exception:
                    pass
                return f"LIVE EV 정지: {self._live_ev_pause_reason}"

            # === 재개 (EV 회복) ===
            if self._live_ev_paused and ev > self._live_ev_threshold:
                pause_dur = datetime.utcnow() - (self._live_ev_pause_time or datetime.utcnow())
                self._live_ev_paused = False
                logger.info(
                    f"[EV모니터] ✅ LIVE 재개 — EV={ev:+.2f}$/trade > {self._live_ev_threshold:+.2f}$ "
                    f"(정지 기간 {pause_dur})"
                )
                try:
                    tg_notify(
                        f"✅ <b>LIVE EV 자동 재개</b>\n"
                        f"━━━━━━━━━━━━━\n"
                        f"최근 {n}건 평균 PnL: ${ev:+.2f}/거래\n"
                        f"승률: {wr:.0%}\n"
                        f"정지 기간: {pause_dur}"
                    )
                except Exception:
                    pass
                return None

            # === 정지 유지 메시지 ===
            if self._live_ev_paused:
                return f"LIVE EV 정지 유지 (현 {n}건 EV={ev:+.2f}$)"

        except Exception as e:
            logger.debug(f"[EV모니터] 체크 실패: {e}")
        return None

    def _compute_live_kelly(self, lookback: int = 50, fraction: float = 0.25) -> float:
        """Kelly 분수 계산 — LIVE empirical WR/RR 기반.

        공식: f* = (p·b − q) / b
          p = WR (win rate), q = 1-p, b = avg_win / avg_loss

        최근 N LIVE 트레이드(HIPPO 과거 노이즈 제외)에서 p, b 추정 →
        fractional Kelly (1/4) 적용 → 최종 allocation 반환.

        Returns: 0.0~1.0 (잔고 대비 포지션 비율). 엣지 없으면 0 반환 (콜러가 min_size 적용).
        """
        try:
            cursor = self.storage.conn.execute(
                """
                SELECT pnl FROM trades
                WHERE mode='LIVE' AND symbol != 'HIPPO/USDT:USDT'
                ORDER BY timestamp DESC LIMIT ?
                """,
                (int(lookback),),
            )
            pnls = [float(r[0]) for r in cursor.fetchall() if r and r[0] is not None]
            if len(pnls) < 10:
                # 표본 부족 — 보수적으로 낮은 값 반환 (min_size로 fallback)
                return 0.0
            wins = [p for p in pnls if p > 0]
            losses = [abs(p) for p in pnls if p < 0]
            if not wins or not losses:
                return 0.0
            p_win = len(wins) / len(pnls)
            q_loss = 1.0 - p_win
            avg_win = sum(wins) / len(wins)
            avg_loss = sum(losses) / len(losses)
            b = avg_win / avg_loss  # payoff ratio
            # Kelly fraction
            f_star_full = (p_win * b - q_loss) / b if b > 0 else 0.0
            # 분수 Kelly (보수: 1/4-Kelly 표준)
            f_star = f_star_full * float(fraction)
            logger.debug(
                f"[Kelly] n={len(pnls)} WR={p_win:.2%} RR(b)={b:.2f} "
                f"full_Kelly={f_star_full:.3f} frac({fraction:.2f})={f_star:.3f}"
            )
            # 음수 엣지 → 0 반환 (min_size 로 안전망)
            return max(0.0, f_star)
        except Exception as e:
            logger.debug(f"[Kelly] 계산 오류: {e}")
            return 0.0

    def _hrp_weight_for_symbol(self, symbol: str, mode: str = "paper") -> float:
        """HRP 가중치 적용 (2026-04-24, 다자산 전환 시만 활성).

        활성 조건:
          - tier.features['max_positions'] >= 2  (단일 집중 모드면 의미 없음)
          - concentration_mode == False
          - tier.name in ('mid','large','pro')    (소시드 집중 모드는 스킵)

        로직:
          - 티어의 전체 심볼 유니버스를 구하고 최근 N봉 수익률 DataFrame 구성
          - HRPAllocator.allocate(returns) → {symbol: weight}
          - 해당 symbol의 weight (미포함이면 1/N)를 반환
          - 반환값을 position_scale에 곱해 사이즈를 HRP-aware로 축소

        Returns: 0.1~1.0 (equal-weight 대비 비율 — 1.0 = 동일 가중)
        """
        try:
            max_pos = self.tier_manager.get_feature("max_positions", mode=mode, default=1) or 1
            if max_pos < 2:
                return 1.0
            concentrate = self.tier_manager.get_feature("concentration_mode", mode=mode, default=True)
            if concentrate:
                return 1.0
            tier = self.tier_manager.get_tier(mode)
            if tier.name not in ("mid", "large", "pro"):
                return 1.0

            universe = list(self.tier_manager.get_symbols(mode) or [])
            if len(universe) < 2 or symbol not in universe:
                return 1.0

            # 최근 수익률 수집 — paper_trader/live_trader 중 어디든 caching된 df 활용
            # self.last_signals 안에 {symbol: {"df": ohlcv}} 같은 캐시가 있을 수 있으므로
            # 일단 있는 심볼만 가져와 공통 최소 길이로 정렬
            import pandas as pd
            cached = getattr(self, "_last_ohlcv_cache", None) or {}
            returns = {}
            for s in universe:
                df = cached.get(s)
                if df is None or len(df) < 50:
                    continue
                returns[s] = df["close"].pct_change().dropna().tail(100).reset_index(drop=True)
            if len(returns) < 2 or symbol not in returns:
                # 공평한 equal-weight fallback → 단일 심볼 비중 1/N
                n = max(max_pos, len(universe))
                return round(1.0 / n, 4)

            df_r = pd.DataFrame(returns).dropna()
            if len(df_r) < 20:
                return round(1.0 / max(len(returns), 1), 4)
            weights = self.hrp.allocate(df_r)
            w = float(weights.get(symbol, 1.0 / max(len(returns), 1)))
            # Equal-weight 기준(=1/N) 대비 비율 계산해 반환 (equal=1.0, 과비중=1.x, 저비중=0.x)
            eq = 1.0 / max(len(returns), 1)
            ratio = w / max(eq, 1e-9)
            # 0.3~2.0 범위로 clip
            ratio = max(0.3, min(ratio, 2.0))
            logger.debug(
                f"[HRP-{mode.upper()}] {symbol} weight={w:.3f} eq={eq:.3f} ratio={ratio:.2f} "
                f"(universe={list(returns.keys())})"
            )
            return ratio
        except Exception as e:
            logger.debug(f"[HRP] {symbol} 가중치 계산 실패 → 1.0: {e}")
            return 1.0

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

            # === 7. A/B 비교 (MACRO_ON vs MACRO_OFF) — 2026-04-21 ===
            # 동일 시장 · 동일 ML/RL 입력에서 매크로 차단 정책만 변경한 변종.
            # Welch's t + Cohen's d + 부트스트랩 CI — pre-registered 정지규칙.
            try:
                from core.learning.ab_tester import (
                    compare_variants,
                    load_variant_pnls,
                )
                pnls_on = load_variant_pnls(self.storage, "PAPER_MACRO_ON", limit=5000)
                pnls_off = load_variant_pnls(self.storage, "PAPER_MACRO_OFF", limit=5000)
                if len(pnls_on) >= 5 and len(pnls_off) >= 5:
                    ab = compare_variants(
                        "PAPER_MACRO_ON", pnls_on,
                        "PAPER_MACRO_OFF", pnls_off,
                        alpha=0.05, n_min=100, d_min=0.3,
                    )
                    report_lines.append("\n🧪 <b>A/B: MACRO_ON vs MACRO_OFF</b>")
                    a, b = ab.variant_a, ab.variant_b
                    report_lines.append(
                        f"  N: {a.n} / {b.n} | WR: {a.wr:.0%} / {b.wr:.0%}"
                    )
                    report_lines.append(
                        f"  mean PnL: {a.mean_pnl:+.2f} / {b.mean_pnl:+.2f} "
                        f"(Δ={ab.mean_diff:+.2f}, CI [{ab.mean_diff_ci95[0]:+.2f}, {ab.mean_diff_ci95[1]:+.2f}])"
                    )
                    report_lines.append(
                        f"  Sharpe/trade: {a.sharpe_pt:.2f} / {b.sharpe_pt:.2f}"
                    )
                    report_lines.append(
                        f"  p={ab.p_welch:.4f} | d={ab.cohens_d:+.2f} ({ab.significance})"
                    )
                    if ab.stopping_rule_met:
                        report_lines.append(
                            f"  ✅ <b>정지규칙 충족 → winner={ab.winner}</b>"
                        )
                        report_lines.append(f"  해석: {ab.reasoning}")
                        logger.warning(
                            f"[A/B] 정지규칙 충족 — winner={ab.winner} "
                            f"p={ab.p_welch:.4f} d={ab.cohens_d:+.2f}"
                        )
                    else:
                        report_lines.append(
                            f"  🔬 수집 중 ({ab.significance}): {ab.reasoning}"
                        )
                else:
                    report_lines.append(
                        f"\n🧪 A/B 표본: ON={len(pnls_on)} / OFF={len(pnls_off)} "
                        f"(각 5건 이상 필요)"
                    )
            except Exception as e:
                logger.debug(f"[A/B] 리포트 생성 실패: {e}")

            report_text = "\n".join(report_lines)
            logger.info(f"[SelfReview] 완료 | {len(close_trades)}건 승률 {win_rate:.0%} PnL ${total_pnl:+.2f} | 조정 {len(adjustments)}건")
            tg_notify(report_text, silent=len(adjustments) == 0)

        except Exception as e:
            logger.error(f"[SelfReview] 실패: {e}")

    def _check_quant_integration(self) -> dict:
        """퀀트 시그널 10개 전략 + ML 피처 주입 통합 체크"""
        result = {"issues": [], "fixes": [], "status": {}}
        try:
            # 1. QuantSignals 인스턴스 확인
            if not hasattr(self, 'quant_signals'):
                result["issues"].append("QuantSignals 미초기화")
                return result

            # 2. 오더북 히스토리 쌓이는지 (시그널이 실제로 호출되는 증거)
            ob_len = len(self.quant_signals._ob_history)
            if ob_len == 0:
                result["issues"].append("오더북 임밸런스 히스토리 0건 — 시그널 미호출")
            result["status"]["ob_history"] = ob_len

            # 3. VPIN 버킷 쌓이는지
            vpin_len = len(self.quant_signals._vpin_buckets)
            result["status"]["vpin_buckets"] = vpin_len

            # 4. 베이시스 히스토리
            basis_len = len(self.quant_signals._basis_history)
            result["status"]["basis_history"] = basis_len

            # 5. ML 피처에 퀀트 컬럼 존재 확인
            ext_feats = self.feature_engineer.external_features
            quant_keys = [k for k in ext_feats if "quant" in k]
            if len(quant_keys) < 10:
                result["issues"].append(
                    f"ML 퀀트 피처 {len(quant_keys)}/17개만 주입됨 — 사전 수집 실패 가능"
                )
            result["status"]["ml_quant_features"] = len(quant_keys)

            # 6. 퀀트 피처 값이 전부 0이면 문제
            all_zero = all(ext_feats.get(k, 0) == 0 for k in quant_keys)
            if quant_keys and all_zero:
                result["issues"].append("퀀트 ML 피처 전부 0 — 오더북/티커 수집 실패 가능")

            # 로그
            if not result["issues"]:
                logger.info(
                    f"[퀀트체크] ✅ 정상 | OB={ob_len} VPIN={vpin_len} "
                    f"Basis={basis_len} ML피처={len(quant_keys)}개"
                )
            else:
                logger.warning(f"[퀀트체크] ⚠️ {result['issues']}")

        except Exception as e:
            result["issues"].append(f"퀀트 체크 실패: {e}")
        return result

    def _check_code_version(self) -> str | None:
        """git commit 시간 vs 프로세스 시작 시간 비교.

        [2026-04-25 개정]
        - 동일 커밋 SHA에 대해 텔레그램 알림은 1회만 발송 (중복 스팸 방지)
        - 알림 본문에 구체적 재시작 명령 포함
        - 자동 재시작 옵션은 운영 안전을 위해 명시적으로 OFF
          (config.diagnostics.auto_restart_on_version_mismatch=true 시 활성)
        """
        try:
            import subprocess
            sha_res = subprocess.run(
                ["git", "log", "-1", "--format=%H"],
                capture_output=True, text=True, timeout=5,
                cwd=str(Path(__file__).parent)
            )
            ts_res = subprocess.run(
                ["git", "log", "-1", "--format=%ct"],
                capture_output=True, text=True, timeout=5,
                cwd=str(Path(__file__).parent)
            )
            if sha_res.returncode != 0 or ts_res.returncode != 0:
                return None

            last_commit_sha = sha_res.stdout.strip()
            last_commit_ts = int(ts_res.stdout.strip())
            last_commit_time = datetime.utcfromtimestamp(last_commit_ts)
            process_start = self.start_time

            if last_commit_time <= process_start:
                # 정상 — 알림 SHA 캐시 리셋 (다음 커밋부터 새로 알림)
                self._version_alerted_sha = None
                return None

            diff = last_commit_time - process_start
            diff_min = diff.total_seconds() / 60
            if diff_min <= 5:
                return None

            short_sha = last_commit_sha[:8]
            msg = (
                f"코드 버전 불일치! 커밋: {last_commit_time.strftime('%H:%M')} > "
                f"프로세스: {process_start.strftime('%H:%M')} "
                f"({diff_min:.0f}분 차이) — 재시작 필요 ({short_sha})"
            )

            # 동일 커밋에 대해 알림 1회만 (스팸 방지)
            already_alerted_sha = getattr(self, '_version_alerted_sha', None)
            if already_alerted_sha != last_commit_sha:
                logger.warning(f"[진단] {msg}")
                tg_notify(
                    f"⚠️ <b>코드 버전 불일치</b>\n"
                    f"━━━━━━━━━━━━━\n"
                    f"최신 커밋: <code>{short_sha}</code> "
                    f"({last_commit_time.strftime('%Y-%m-%d %H:%M')} UTC)\n"
                    f"프로세스 시작: {process_start.strftime('%Y-%m-%d %H:%M')} UTC\n"
                    f"⏱ {diff_min:.0f}분 차이\n"
                    f"━━━━━━━━━━━━━\n"
                    f"🔄 <b>재시작 명령</b> (서버에서 실행):\n"
                    f"<code>cd ~/trading && git pull && "
                    f"pkill -f 'python.*main.py' && "
                    f"nohup python main.py &gt; logs/run.log 2&gt;&amp;1 &amp;</code>\n"
                    f"━━━━━━━━━━━━━\n"
                    f"💡 본 알림은 동일 커밋({short_sha})에 대해 1회만 발송됩니다."
                )
                self._version_alerted_sha = last_commit_sha

                # 옵션: 자동 재시작 (config 활성 시)
                try:
                    auto_restart = (
                        self.config.get("diagnostics", {})
                        .get("auto_restart_on_version_mismatch", False)
                    )
                except Exception:
                    auto_restart = False
                if auto_restart and diff_min >= 30:
                    logger.warning("[진단] 🔄 auto_restart_on_version_mismatch=true → 자가 재시작 시도")
                    tg_notify("🔄 <b>자동 재시작 진행 중...</b>")
                    try:
                        import os, sys
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                    except Exception as e:
                        logger.error(f"[진단] 자동 재시작 실패: {e}")

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
        if self.spot_exchange_client:
            try:
                await self.spot_exchange_client.close()
            except Exception:
                pass
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
