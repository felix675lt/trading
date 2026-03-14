"""뉴스 센티먼트 분석 - VADER (경량, GPU 불필요) + 키워드 스코어링"""

import re
from collections import defaultdict

from loguru import logger

# VADER 임포트 (nltk 의존)
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False


# 크립토 특화 감성 사전 (VADER 보강)
CRYPTO_LEXICON = {
    # 극도의 긍정
    "moon": 3.0, "mooning": 3.5, "bullish": 2.5, "bull run": 3.0,
    "ath": 2.5, "all-time high": 3.0, "breakout": 2.0, "pump": 2.0,
    "rally": 2.0, "surge": 2.5, "soar": 2.5, "skyrocket": 3.0,
    "adoption": 2.0, "institutional": 1.5, "approval": 2.5,
    "etf approved": 3.5, "etf approval": 3.0, "mainstream": 1.5,
    "partnership": 1.5, "upgrade": 1.5, "halving": 1.5,

    # 극도의 부정
    "crash": -3.0, "dump": -2.5, "bear": -2.0, "bearish": -2.5,
    "scam": -3.5, "hack": -3.0, "hacked": -3.5, "exploit": -3.0,
    "rug pull": -4.0, "rugpull": -4.0, "ponzi": -3.5,
    "ban": -3.0, "banned": -3.0, "regulation": -1.5, "crackdown": -2.5,
    "sec": -1.0, "lawsuit": -2.5, "fraud": -3.0, "collapse": -3.5,
    "bankrupt": -4.0, "bankruptcy": -4.0, "liquidation": -2.0,
    "fud": -1.5, "panic": -2.5, "sell-off": -2.0, "selloff": -2.0,
    "delisting": -3.0, "delist": -3.0, "vulnerability": -2.5,

    # 중립적이지만 중요
    "whale": 0.5, "accumulation": 1.0, "distribution": -0.5,
    "resistance": -0.3, "support": 0.3, "consolidation": 0.0,
    "volatile": -0.5, "volatility": -0.3,
}

# 시장 이벤트 키워드 (감성과 별개로 중요도 측정)
EVENT_KEYWORDS = {
    "high_impact": [
        "etf", "halving", "fed", "fomc", "interest rate", "cpi", "inflation",
        "ban", "regulation", "hack", "exploit", "bankruptcy", "collapse",
        "war", "sanction", "default", "recession",
    ],
    "medium_impact": [
        "partnership", "upgrade", "fork", "mainnet", "testnet", "airdrop",
        "listing", "delisting", "whale", "accumulation", "sec", "lawsuit",
        "earnings", "gdp", "unemployment", "pmi",
    ],
    "low_impact": [
        "update", "release", "roadmap", "community", "developer",
        "staking", "defi", "nft", "layer2", "bridge",
    ],
}


class SentimentAnalyzer:
    """뉴스 및 소셜 미디어 텍스트의 감성 분석"""

    def __init__(self):
        self.analyzer = None
        self._init_vader()
        self.recent_scores: list[dict] = []

    def _init_vader(self):
        """VADER 초기화 + 크립토 사전 추가"""
        if _VADER_AVAILABLE:
            try:
                self.analyzer = SentimentIntensityAnalyzer()
                # 크립토 특화 사전 추가
                self.analyzer.lexicon.update(CRYPTO_LEXICON)
                logger.info("[Sentiment] VADER 초기화 완료 (크립토 사전 추가)")
            except Exception as e:
                logger.warning(f"[Sentiment] VADER 초기화 실패, 키워드 방식 사용: {e}")
                self.analyzer = None
        else:
            logger.info("[Sentiment] VADER 미설치, 키워드 기반 분석 사용")

    def analyze_text(self, text: str) -> dict:
        """
        단일 텍스트 감성 분석

        Returns:
            compound: -1 ~ 1 (종합 감성)
            positive/negative/neutral: 0 ~ 1
            impact_level: high/medium/low/none
            events: 감지된 이벤트 키워드들
        """
        text_lower = text.lower()

        # VADER 분석
        if self.analyzer:
            scores = self.analyzer.polarity_scores(text)
        else:
            scores = self._keyword_sentiment(text_lower)

        # 이벤트 임팩트 분석
        impact_level = "none"
        detected_events = []

        for level, keywords in EVENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    detected_events.append(kw)
                    if level == "high_impact":
                        impact_level = "high"
                    elif level == "medium_impact" and impact_level != "high":
                        impact_level = "medium"
                    elif impact_level == "none":
                        impact_level = "low"

        result = {
            "compound": scores.get("compound", 0),
            "positive": scores.get("pos", 0),
            "negative": scores.get("neg", 0),
            "neutral": scores.get("neu", 1),
            "impact_level": impact_level,
            "events": detected_events,
        }
        return result

    def _keyword_sentiment(self, text: str) -> dict:
        """VADER 없을 때 키워드 기반 감성 분석"""
        score = 0.0
        count = 0
        for word, val in CRYPTO_LEXICON.items():
            if word in text:
                score += val
                count += 1

        if count == 0:
            return {"compound": 0, "pos": 0, "neg": 0, "neu": 1}

        compound = max(-1, min(1, score / (count * 2)))
        pos = max(0, compound)
        neg = abs(min(0, compound))
        neu = 1 - pos - neg

        return {"compound": compound, "pos": pos, "neg": neg, "neu": max(0, neu)}

    def analyze_batch(self, texts: list[str], symbol: str = "") -> dict:
        """
        여러 텍스트 일괄 분석 → 종합 감성 피처

        Returns:
            avg_sentiment, sentiment_std, bullish_ratio, bearish_ratio,
            high_impact_count, event_density
        """
        if not texts:
            return self._empty_features()

        # 심볼 관련 텍스트 우선 필터
        coin = symbol.split("/")[0].lower() if "/" in symbol else symbol.lower()
        relevant = [t for t in texts if coin in t.lower()] if coin else texts
        if not relevant:
            relevant = texts  # 관련 뉴스 없으면 전체 사용

        sentiments = [self.analyze_text(t) for t in relevant[:50]]

        compounds = [s["compound"] for s in sentiments]
        impacts = [s["impact_level"] for s in sentiments]
        all_events = []
        for s in sentiments:
            all_events.extend(s["events"])

        import numpy as np

        features = {
            "sentiment_avg": float(np.mean(compounds)),
            "sentiment_std": float(np.std(compounds)),
            "sentiment_max": float(max(compounds)),
            "sentiment_min": float(min(compounds)),
            "bullish_ratio": sum(1 for c in compounds if c > 0.1) / len(compounds),
            "bearish_ratio": sum(1 for c in compounds if c < -0.1) / len(compounds),
            "high_impact_count": float(impacts.count("high")),
            "medium_impact_count": float(impacts.count("medium")),
            "event_density": len(all_events) / max(len(relevant), 1),
            "unique_events": float(len(set(all_events))),
        }

        # 히스토리 기록
        self.recent_scores.append({
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
            "avg": features["sentiment_avg"],
            "count": len(relevant),
        })
        if len(self.recent_scores) > 100:
            self.recent_scores = self.recent_scores[-100:]

        return features

    def _empty_features(self) -> dict:
        return {
            "sentiment_avg": 0.0,
            "sentiment_std": 0.0,
            "sentiment_max": 0.0,
            "sentiment_min": 0.0,
            "bullish_ratio": 0.5,
            "bearish_ratio": 0.5,
            "high_impact_count": 0.0,
            "medium_impact_count": 0.0,
            "event_density": 0.0,
            "unique_events": 0.0,
        }

    def get_sentiment_trend(self) -> float:
        """최근 센티먼트 트렌드 (-1 하락 ~ 1 상승)"""
        if len(self.recent_scores) < 3:
            return 0.0
        recent = [s["avg"] for s in self.recent_scores[-5:]]
        older = [s["avg"] for s in self.recent_scores[-10:-5]] if len(self.recent_scores) >= 10 else [0]
        import numpy as np
        return float(np.mean(recent) - np.mean(older))
