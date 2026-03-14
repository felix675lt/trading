"""외부 데이터 통합 매니저 - 모든 외부 요인을 하나로 통합"""

import asyncio
from datetime import datetime

from loguru import logger

from .fear_greed import FearGreedCollector
from .macro_collector import MacroCollector
from .news_collector import NewsCollector
from .onchain_collector import OnchainCollector
from .sentiment_analyzer import SentimentAnalyzer
from .social_collector import SocialCollector


class ExternalDataManager:
    """
    모든 외부 데이터 소스를 통합 관리

    데이터 흐름:
    1. 뉴스/소셜/온체인/매크로/공포탐욕 수집 (비동기 병렬)
    2. 뉴스+소셜 텍스트 → 센티먼트 분석
    3. 모든 피처를 통합하여 ML 피처로 변환
    4. 종합 외부 신호 점수 계산
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.news = NewsCollector()
        self.sentiment = SentimentAnalyzer()
        self.fear_greed = FearGreedCollector()
        self.onchain = OnchainCollector()
        self.macro = MacroCollector()
        self.social = SocialCollector()

        self.enabled = config.get("enabled", True)
        self.weight = config.get("weight", 0.3)  # 전체 결정에서 외부 요인 가중치

        self.last_update: datetime | None = None
        self.composite_signal: dict = {}
        self.all_features: dict = {}

    async def update(self, symbol: str = "BTC/USDT:USDT") -> dict:
        """모든 외부 데이터 수집 및 분석"""
        if not self.enabled:
            return self._empty_signal()

        try:
            # 병렬 데이터 수집
            await asyncio.gather(
                self.news.fetch(),
                self.fear_greed.fetch(),
                self.onchain.fetch(),
                self.macro.fetch(),
                self.social.fetch(),
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

            # 모든 피처 통합
            self.all_features = {
                **fg_features,
                **onchain_features,
                **macro_features,
                **news_features,
                **social_features,
                **sentiment_features,
            }

            # 종합 외부 신호 계산
            self.composite_signal = self._compute_composite_signal(
                fg_features, onchain_features, macro_features,
                sentiment_features, news_features, social_features,
            )

            self.last_update = datetime.utcnow()

            logger.info(
                f"[External] 종합신호: {self.composite_signal['score']:.2f} | "
                f"공포탐욕: {fg_features.get('fg_value', 50):.0f} | "
                f"센티먼트: {sentiment_features.get('sentiment_avg', 0):.2f} | "
                f"온체인: {onchain_features.get('onchain_composite_score', 0):.2f} | "
                f"매크로: {macro_features.get('macro_composite_score', 0):.2f}"
            )

            return self.composite_signal

        except Exception as e:
            logger.error(f"[External] 업데이트 실패: {e}")
            return self._empty_signal()

    def _compute_composite_signal(
        self, fg: dict, onchain: dict, macro: dict,
        sentiment: dict, news: dict, social: dict,
    ) -> dict:
        """
        종합 외부 신호 계산

        가중치:
        - Fear & Greed: 25% (시장 심리의 가장 직접적 지표)
        - 센티먼트: 25% (뉴스/소셜 감성)
        - 매크로: 20% (거시경제 방향)
        - 온체인: 20% (블록체인 펀더멘털)
        - 소셜 버즈: 10% (화제성/모멘텀)
        """
        # 각 카테고리 점수 (-1 ~ 1)
        fg_score = fg.get("fg_normalized", 0)
        # Fear & Greed는 반전 해석: 극도의 공포 = 매수 기회
        # → 낮을수록 매수 시그널이므로 부호 반전
        fg_contrarian = -fg_score * 0.5  # 역발상 요소

        sentiment_score = sentiment.get("sentiment_avg", 0)
        macro_score = macro.get("macro_composite_score", 0)
        onchain_score = onchain.get("onchain_composite_score", 0)

        # 소셜 버즈 점수
        social_score = 0.0
        if social.get("social_engagement", 0) > 0.5:
            social_score = (social.get("social_sentiment_ratio", 0.5) - 0.5) * 2

        # 가중 평균
        composite = (
            fg_contrarian * 0.15 +         # 역발상 공포탐욕
            fg_score * 0.10 +              # 순방향 공포탐욕
            sentiment_score * 0.25 +        # 센티먼트
            macro_score * 0.20 +            # 매크로
            onchain_score * 0.20 +          # 온체인
            social_score * 0.10             # 소셜
        )

        # 이벤트 임팩트 보정
        high_impact = sentiment.get("high_impact_count", 0)
        if high_impact > 0:
            # 고임팩트 이벤트 시 외부 신호 가중치 증가
            composite *= (1 + min(high_impact * 0.2, 0.5))

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
            "score": composite,
            "direction": direction,
            "strength": strength,
            "components": {
                "fear_greed": fg_score,
                "sentiment": sentiment_score,
                "macro": macro_score,
                "onchain": onchain_score,
                "social": social_score,
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
        """ML 모델에 입력할 전체 외부 피처 반환"""
        return self.all_features

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
            "feature_count": len(self.all_features),
        }
