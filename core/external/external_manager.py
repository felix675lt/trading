"""외부 데이터 통합 매니저 - 모든 외부 요인을 하나로 통합

데이터 소스:
1. 기존: 뉴스, 센티먼트, 공포탐욕, 온체인, 매크로, 소셜(Reddit)
2. 신규: 계절 사이클, 파생상품, 멀티타임프레임
3. 최신: 크립토 트위터/Stocktwits (실시간 소셜 센티먼트)
"""

import asyncio
from datetime import datetime

from loguru import logger

from .fear_greed import FearGreedCollector
from .macro_collector import MacroCollector
from .news_collector import NewsCollector
from .onchain_collector import OnchainCollector
from .sentiment_analyzer import SentimentAnalyzer
from .social_collector import SocialCollector
from .seasonal_cycle import SeasonalCycleAnalyzer
from .derivatives_data import DerivativesDataCollector
from .multi_timeframe import MultiTimeframeAnalyzer
from .crypto_twitter import CryptoTwitterCollector
from .polymarket import PolymarketCollector
from .real_macro_collector import RealMacroCollector


class ExternalDataManager:
    """
    모든 외부 데이터 소스를 통합 관리

    데이터 흐름:
    1. 뉴스/소셜/온체인/매크로/공포탐욕/파생상품 수집 (비동기 병렬)
    2. 뉴스+소셜 텍스트 → 센티먼트 분석
    3. 계절 사이클 → 반감기/시즌 분석
    4. 멀티타임프레임 → 합류 분석 (별도 호출)
    5. 모든 피처를 통합하여 ML 피처로 변환
    6. 종합 외부 신호 점수 계산
    """

    def __init__(self, config: dict | None = None):
        config = config or {}

        # 기존 수집기
        self.news = NewsCollector()
        self.sentiment = SentimentAnalyzer()
        self.fear_greed = FearGreedCollector()
        self.onchain = OnchainCollector()
        self.macro = MacroCollector()
        self.social = SocialCollector()

        # 신규 수집기
        self.seasonal = SeasonalCycleAnalyzer()
        self.derivatives = DerivativesDataCollector()
        self.multi_tf = MultiTimeframeAnalyzer()

        # 실시간 소셜 (Stocktwits + CryptoNews AI)
        self.crypto_twitter = CryptoTwitterCollector()

        # 예측 시장 (Polymarket 내부자 베팅 패턴)
        self.polymarket = PolymarketCollector()

        # 실제 매크로 (Yahoo Finance: 유가/금/DXY/VIX/국채/S&P500)
        self.real_macro = RealMacroCollector()

        self.enabled = config.get("enabled", True)
        self.weight = config.get("weight", 0.3)

        self.last_update: datetime | None = None
        self.composite_signal: dict = {}
        self.all_features: dict = {}

    async def update(self, symbol: str = "BTC/USDT:USDT") -> dict:
        """모든 외부 데이터 수집 및 분석"""
        if not self.enabled:
            return self._empty_signal()

        try:
            # 심볼에서 Binance 형식 추출 (BTC/USDT:USDT → BTCUSDT)
            binance_symbol = symbol.replace("/", "").replace(":USDT", "")

            # 심볼 단축 (BTCUSDT → BTC)
            short_symbol = binance_symbol.replace("USDT", "")

            # 병렬 데이터 수집 (모든 소스 동시)
            await asyncio.gather(
                self.news.fetch(),
                self.fear_greed.fetch(),
                self.onchain.fetch(),
                self.macro.fetch(),
                self.social.fetch(),
                self.derivatives.collect(binance_symbol),
                self.crypto_twitter.fetch(short_symbol),
                self.polymarket.fetch(),
                self.real_macro.fetch(),
                return_exceptions=True,
            )

            # 센티먼트 분석 (뉴스 + 소셜 텍스트 통합)
            texts = [n["title"] for n in self.news.news]
            texts.extend(self.social.get_titles())
            sentiment_features = self.sentiment.analyze_batch(texts, symbol)

            # 각 모듈의 피처 수집
            fg_features = self.fear_greed.get_features()
            onchain_features = self.onchain.get_features()
            macro_features = self.macro.get_features()
            news_features = self.news.get_features(symbol)
            social_features = self.social.get_features(symbol)
            seasonal_features = self.seasonal.get_features()
            derivatives_features = self.derivatives.get_features()
            twitter_features = self.crypto_twitter.fetch.__self__._cache.get(
                f"twitter_{short_symbol}", {}
            ) if hasattr(self.crypto_twitter, '_cache') else {}
            # 더 안전한 방법으로 트위터 피처 가져오기
            if not twitter_features:
                twitter_features = {}
                for k, v in self.crypto_twitter._cache.items():
                    if isinstance(v, dict) and "twitter_composite" in v:
                        twitter_features = v
                        break

            # Polymarket 시그널
            polymarket_features = self.polymarket.get_features()

            # 실제 매크로 피처 (유가/금/DXY/VIX/국채/S&P500)
            real_macro_features = self.real_macro.get_features()

            # 계절 시그널 업데이트
            self.seasonal.get_seasonal_signal()

            # 모든 피처 통합
            self.all_features = {
                **fg_features,
                **onchain_features,
                **macro_features,
                **news_features,
                **social_features,
                **sentiment_features,
                **seasonal_features,
                **derivatives_features,
                **twitter_features,
                **polymarket_features,
                **real_macro_features,
            }

            # 종합 외부 신호 계산
            self.composite_signal = self._compute_composite_signal(
                fg_features, onchain_features, macro_features,
                sentiment_features, news_features, social_features,
                seasonal_features, derivatives_features, twitter_features,
                polymarket_features, real_macro_features,
            )

            self.last_update = datetime.utcnow()

            # 파생상품 리포트
            deriv = self.derivatives.get_signal_for_strategy()
            seasonal_sig = self.seasonal.current_signal

            twitter_sig = self.crypto_twitter.get_signal()
            poly_sig = self.polymarket.get_signal()
            poly_markets = poly_sig.get("polymarket_markets_monitored", 0)
            poly_sc = poly_sig.get("polymarket_score", 0)
            rm_sig = self.real_macro.get_signal()
            logger.info(
                f"[External] 종합신호: {self.composite_signal['score']:.2f} | "
                f"공포탐욕: {fg_features.get('fg_value', 50):.0f} | "
                f"센티먼트: {sentiment_features.get('sentiment_avg', 0):.2f} | "
                f"트위터: {twitter_features.get('twitter_composite', 0):.2f} | "
                f"펀딩비: {derivatives_features.get('deriv_funding_rate', 0):.1f}bp | "
                f"롱숏비: {derivatives_features.get('deriv_global_ls_ratio', 1):.2f} | "
                f"계절: {seasonal_sig.get('direction', '?')}({seasonal_sig.get('score', 0):.2f}) | "
                f"반감기: {seasonal_sig.get('halving_phase', '?')} | "
                f"PM: {poly_sc:.2f}({poly_markets}건) | "
                f"매크로: WTI${rm_sig.get('oil', 0):.0f} DXY{rm_sig.get('dxy', 0):.1f} VIX{rm_sig.get('vix', 0):.1f} 점수{rm_sig.get('score', 0):.2f}"
            )

            return self.composite_signal

        except Exception as e:
            logger.error(f"[External] 업데이트 실패: {e}")
            return self._empty_signal()

    def update_multi_timeframe(self, df, timeframe: str):
        """멀티타임프레임 데이터 업데이트 (main.py에서 각 tf마다 호출)"""
        self.multi_tf.analyze_timeframe(df, timeframe)

    def get_multi_tf_confluence(self) -> dict:
        """멀티타임프레임 합류 결과"""
        return self.multi_tf.calculate_confluence()

    def _compute_composite_signal(
        self, fg: dict, onchain: dict, macro: dict,
        sentiment: dict, news: dict, social: dict,
        seasonal: dict, derivatives: dict, twitter: dict | None = None,
        polymarket: dict | None = None, real_macro: dict | None = None,
    ) -> dict:
        """
        종합 외부 신호 계산 (v5 - 실제 매크로 데이터 추가)

        가중치 (총 100%):
        - 파생상품 (펀딩비/OI/롱숏): 16% ← 선물 핵심
        - 크립토 트위터/Stocktwits:   14% ← 실시간 소셜
        - 실제 매크로 (유가/DXY/VIX): 12% ← 전통 금융 (신규)
        - Polymarket 예측 시장:        7% ← 내부자 베팅 패턴
        - 계절 사이클:                 8% ← 10년 검증 패턴
        - Fear & Greed:              10% (역발상 + 순방향)
        - NLP 센티먼트:               9%
        - 매크로 (CoinGecko):         6%
        - 온체인:                      8%
        - Reddit 소셜:                6%
        - 뉴스:                        4%
        """
        twitter = twitter or {}
        polymarket = polymarket or {}
        real_macro = real_macro or {}

        # 각 카테고리 점수 (-1 ~ 1)
        # [제거됨] 공포탐욕 지수 — 후행 지표, 예측력 없음, 노이즈만 추가
        fg_score = 0.0
        fg_contrarian = 0.0

        sentiment_score = sentiment.get("sentiment_avg", 0)
        macro_score = macro.get("macro_composite_score", 0)
        onchain_score = onchain.get("onchain_composite_score", 0)

        # Reddit 소셜 버즈 점수
        social_score = 0.0
        if social.get("social_engagement", 0) > 0.5:
            social_score = (social.get("social_sentiment_ratio", 0.5) - 0.5) * 2

        # 파생상품 시그널
        deriv_signal = self.derivatives.get_signal_for_strategy()
        deriv_score = deriv_signal.get("score", 0)

        # 계절 시그널
        seasonal_score = seasonal.get("seasonal_score", 0)

        # 크립토 트위터 시그널 (Stocktwits + AI 센티먼트 + 이벤트)
        twitter_score = twitter.get("twitter_composite", 0)

        # Polymarket 시그널 (예측 시장 내부자 베팅 패턴)
        poly_score = polymarket.get("polymarket_score", 0)
        poly_confidence = polymarket.get("polymarket_confidence", 0)

        # 실제 매크로 시그널 (유가/금/DXY/VIX/국채/S&P500)
        real_macro_score = real_macro.get("real_macro_composite_score", 0)
        geo_risk = real_macro.get("real_macro_geo_risk", 0)

        # 가중 평균 (v6 - 공포탐욕 제거, 비중 재분배)
        composite = (
            deriv_score * 0.18 +             # 파생상품 (16→18)
            twitter_score * 0.15 +           # 크립토 트위터 (14→15)
            real_macro_score * 0.14 +        # 실제 매크로 (12→14)
            poly_score * 0.08 +              # Polymarket 예측 시장 (7→8)
            seasonal_score * 0.08 +          # 계절 사이클 (유지)
            sentiment_score * 0.10 +         # NLP 센티먼트 (9→10)
            macro_score * 0.07 +             # 매크로 CoinGecko (6→7)
            onchain_score * 0.10 +           # 온체인 (8→10)
            social_score * 0.06 +            # Reddit 소셜 (유지)
            0.0 * 0.04                       # 뉴스 (이벤트 임팩트로 반영)
        )

        # 지정학 리스크가 높으면 변동성 증가 → 포지션 축소 시그널
        if geo_risk > 0.6:
            composite *= (1 - geo_risk * 0.2)  # 지정학 리스크 높으면 시그널 약화

        # Polymarket 강한 시그널 시 추가 부스트
        # (확신도 높고 방향 일치하면 composite 강화)
        if poly_confidence > 0.5 and abs(poly_score) > 0.3:
            if (poly_score > 0 and composite > 0) or (poly_score < 0 and composite < 0):
                composite *= 1.15  # 같은 방향이면 15% 부스트
            elif (poly_score > 0 and composite < -0.1) or (poly_score < 0 and composite > 0.1):
                composite *= 0.85  # 반대 방향이면 약화

        # 이벤트 임팩트 보정
        high_impact = sentiment.get("high_impact_count", 0)
        if high_impact > 0:
            composite *= (1 + min(high_impact * 0.2, 0.5))

        # 계절 패턴 강화: Dec-Feb 반등 구간에서 확신도 부스트
        seasonal_conf = seasonal.get("seasonal_confidence", 0)
        is_dec_feb = seasonal.get("is_dec_feb_bounce", 0)
        if is_dec_feb and seasonal_conf > 0.7:
            if composite > 0:
                composite *= 1.2  # 불리시 시그널 강화
            # Dec-Feb 반등 구간에서 숏은 약화
            elif composite < 0:
                composite *= 0.7

        # 극단값 클리핑
        composite = max(-1, min(1, composite))

        # 방향 및 강도 결정
        if composite > 0.3:
            direction = "bullish"
            strength = "strong"
        elif composite > 0.1:
            direction = "bullish"
            strength = "moderate"
        elif composite < -0.3:
            direction = "bearish"
            strength = "strong"
        elif composite < -0.1:
            direction = "bearish"
            strength = "moderate"
        else:
            direction = "neutral"
            strength = "weak"

        return {
            "score": round(composite, 4),
            "direction": direction,
            "strength": strength,
            "components": {
                "derivatives": round(deriv_score, 3),
                "twitter": round(twitter_score, 3),
                "real_macro": round(real_macro_score, 3),
                "polymarket": round(poly_score, 3),
                "seasonal": round(seasonal_score, 3),
                "fear_greed": round(fg_score, 3),
                "sentiment": round(sentiment_score, 3),
                "macro": round(macro_score, 3),
                "onchain": round(onchain_score, 3),
                "social": round(social_score, 3),
                "geo_risk": round(geo_risk, 3),
            },
            "high_impact_events": high_impact > 0,
            "confidence": min(abs(composite) * 2, 1.0),
        }

    def _empty_signal(self) -> dict:
        return {
            "score": 0.0,
            "direction": "neutral",
            "strength": "weak",
            "components": {},
            "high_impact_events": False,
            "confidence": 0.0,
        }

    def get_all_features(self) -> dict:
        """ML 모델에 입력할 전체 외부 피처 반환 (멀티TF 피처 포함)"""
        features = dict(self.all_features)
        features.update(self.multi_tf.get_features())
        return features

    def get_signal_for_strategy(self) -> dict:
        """전략 매니저에 전달할 외부 신호"""
        return self.composite_signal

    def get_report(self) -> dict:
        """대시보드용 리포트"""
        return {
            "enabled": self.enabled,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "composite_signal": self.composite_signal,
            "fear_greed": self.fear_greed.current,
            "news_count": len(self.news.news),
            "social_posts": len(self.social.posts),
            "sentiment_trend": self.sentiment.get_sentiment_trend(),
            "onchain_data": self.onchain.data,
            "macro_data": self.macro.data,
            "crypto_twitter": self.crypto_twitter.get_signal(),
            "derivatives": self.derivatives.get_report(),
            "seasonal": self.seasonal.get_report(),
            "multi_timeframe": self.multi_tf.get_report(),
            "polymarket": self.polymarket.get_report(),
            "real_macro": self.real_macro.get_report(),
            "feature_count": len(self.all_features),
        }
