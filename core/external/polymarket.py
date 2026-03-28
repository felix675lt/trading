"""Polymarket 예측 시장 모니터링 — 내부자 베팅 패턴 감지

핵심 아이디어:
- 크립토 관련 Polymarket 마켓의 급격한 확률 변동을 감지
- 대규모 베팅이 들어오면 내부자 행동 가능성 → 크립토 선행 시그널
- 확률 변화 속도 + 거래량 급증을 결합하여 시그널 강도 산출

모니터링 대상:
- Bitcoin Reserve / 비트코인 관련 정책
- SEC / 규제 관련
- ETF 승인/거부
- 연준 금리 결정
- 크립토 우호적/적대적 정책
"""

import asyncio
import time
from datetime import datetime, timedelta

import aiohttp
from loguru import logger


class PolymarketCollector:
    """Polymarket 예측 시장 크립토 시그널 수집기"""

    GAMMA_BASE = "https://gamma-api.polymarket.com"
    CLOB_BASE = "https://clob.polymarket.com"

    # 크립토 영향 키워드 (검색용)
    CRYPTO_KEYWORDS = [
        "bitcoin",
        "crypto",
        "SEC crypto",
        "bitcoin reserve",
        "ethereum ETF",
        "bitcoin ETF",
        "stablecoin",
    ]

    # 크립토에 영향을 미치는 정치/경제 키워드
    MACRO_KEYWORDS = [
        "fed rate",
        "interest rate",
        "tariff",
        "trump crypto",
    ]

    # 마켓 제목 → 크립토 방향 매핑 (Yes 승리 시 방향)
    # 양수 = crypto bullish, 음수 = crypto bearish
    IMPACT_MAP = {
        # Bitcoin/crypto 직접 관련
        "bitcoin reserve": 1.0,
        "bitcoin price": 0.8,
        "btc": 0.8,
        "crypto regulation": -0.5,  # 규제 = 보통 bearish
        "sec approve": 0.9,
        "etf approv": 0.9,
        "ban crypto": -1.0,
        "crypto ban": -1.0,
        "stablecoin": 0.5,
        # 간접 영향
        "rate cut": 0.6,   # 금리 인하 = bullish
        "rate hike": -0.6,  # 금리 인상 = bearish
        "tariff": -0.4,     # 관세 = 불확실성 = bearish
        "recession": -0.7,
    }

    def __init__(self):
        self._cache: dict = {}
        self._markets: list[dict] = []  # 모니터링 중인 마켓 목록
        self._price_history: dict[str, list[dict]] = {}  # market_id → [{time, price, volume}]
        self._last_search_time: float = 0
        self._search_interval = 3600  # 1시간마다 마켓 재검색
        self._last_fetch_time: float = 0
        self._fetch_interval = 60  # 1분마다 가격 체크 (외부에서 호출 시)
        self.current_signal: dict = self._empty_signal()

    def _empty_signal(self) -> dict:
        return {
            "polymarket_score": 0.0,
            "polymarket_direction": "neutral",
            "polymarket_confidence": 0.0,
            "polymarket_alerts": [],
            "polymarket_markets_monitored": 0,
        }

    async def fetch(self) -> dict:
        """메인 수집 루프 — ExternalDataManager에서 호출"""
        now = time.time()

        try:
            # 1. 주기적으로 크립토 관련 마켓 검색 (1시간마다)
            if now - self._last_search_time > self._search_interval:
                await self._discover_markets()
                self._last_search_time = now

            # 2. 모니터링 중인 마켓 가격/거래량 업데이트
            if self._markets:
                await self._update_prices()

            # 3. 시그널 계산
            self.current_signal = self._compute_signal()
            self._last_fetch_time = now

            return self.current_signal

        except Exception as e:
            logger.debug(f"[Polymarket] 수집 실패: {e}")
            return self._empty_signal()

    async def _discover_markets(self):
        """크립토 관련 활성 마켓 검색"""
        discovered = []

        async with aiohttp.ClientSession() as session:
            for keyword in self.CRYPTO_KEYWORDS + self.MACRO_KEYWORDS:
                try:
                    url = f"{self.GAMMA_BASE}/events"
                    params = {
                        "title_contains": keyword,
                        "active": "true",
                        "closed": "false",
                        "limit": 5,
                        "order": "volume24hr",
                        "ascending": "false",
                    }
                    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status != 200:
                            continue
                        events = await resp.json()

                        for event in events:
                            markets = event.get("markets", [])
                            for mkt in markets:
                                if not mkt.get("active") or mkt.get("closed"):
                                    continue

                                market_id = mkt.get("id", "")
                                title = (mkt.get("question") or mkt.get("groupItemTitle") or "").lower()

                                # 크립토 영향도 판별
                                impact = self._assess_crypto_impact(title)
                                if abs(impact) < 0.1:
                                    continue

                                # 중복 제거
                                if any(m["id"] == market_id for m in discovered):
                                    continue

                                outcome_prices = mkt.get("outcomePrices", "[]")
                                if isinstance(outcome_prices, str):
                                    import json
                                    try:
                                        outcome_prices = json.loads(outcome_prices)
                                    except Exception:
                                        outcome_prices = []

                                yes_price = float(outcome_prices[0]) if outcome_prices else 0.5

                                discovered.append({
                                    "id": market_id,
                                    "title": title,
                                    "event_title": event.get("title", ""),
                                    "impact": impact,
                                    "yes_price": yes_price,
                                    "volume_24h": float(mkt.get("volume24hr", 0) or 0),
                                    "volume_total": float(mkt.get("volumeNum", 0) or 0),
                                    "one_hour_change": float(mkt.get("oneHourPriceChange", 0) or 0),
                                    "one_day_change": float(mkt.get("oneDayPriceChange", 0) or 0),
                                    "clob_token_ids": mkt.get("clobTokenIds", []),
                                    "condition_id": mkt.get("conditionId", ""),
                                })

                    await asyncio.sleep(0.2)  # rate limit 배려

                except Exception as e:
                    logger.debug(f"[Polymarket] 검색 실패 ({keyword}): {e}")

        # 거래량 기준 상위 20개만 유지
        discovered.sort(key=lambda m: m["volume_24h"], reverse=True)
        self._markets = discovered[:20]

        if self._markets:
            logger.info(
                f"[Polymarket] {len(self._markets)}개 크립토 관련 마켓 감지 | "
                f"Top: {self._markets[0]['event_title'][:40]}... "
                f"(24h vol: ${self._markets[0]['volume_24h']:,.0f})"
            )

    async def _update_prices(self):
        """모니터링 중인 마켓의 최신 가격/거래량 업데이트"""
        market_ids = [m["id"] for m in self._markets[:10]]  # 상위 10개만

        async with aiohttp.ClientSession() as session:
            for market_id in market_ids:
                try:
                    url = f"{self.GAMMA_BASE}/markets/{market_id}"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()

                    outcome_prices = data.get("outcomePrices", "[]")
                    if isinstance(outcome_prices, str):
                        import json
                        try:
                            outcome_prices = json.loads(outcome_prices)
                        except Exception:
                            outcome_prices = []

                    new_price = float(outcome_prices[0]) if outcome_prices else 0.5
                    new_vol_24h = float(data.get("volume24hr", 0) or 0)
                    one_hour_change = float(data.get("oneHourPriceChange", 0) or 0)
                    one_day_change = float(data.get("oneDayPriceChange", 0) or 0)

                    # 기존 마켓 데이터 업데이트
                    for mkt in self._markets:
                        if mkt["id"] == market_id:
                            old_price = mkt["yes_price"]
                            old_vol = mkt["volume_24h"]

                            mkt["prev_price"] = old_price
                            mkt["prev_volume_24h"] = old_vol
                            mkt["yes_price"] = new_price
                            mkt["volume_24h"] = new_vol_24h
                            mkt["one_hour_change"] = one_hour_change
                            mkt["one_day_change"] = one_day_change
                            mkt["price_delta"] = new_price - old_price
                            mkt["last_updated"] = time.time()
                            break

                    # 가격 히스토리 기록
                    if market_id not in self._price_history:
                        self._price_history[market_id] = []
                    self._price_history[market_id].append({
                        "time": time.time(),
                        "price": new_price,
                        "volume_24h": new_vol_24h,
                    })
                    # 최근 100개만 유지
                    if len(self._price_history[market_id]) > 100:
                        self._price_history[market_id] = self._price_history[market_id][-100:]

                except Exception as e:
                    logger.debug(f"[Polymarket] 가격 업데이트 실패 ({market_id[:8]}): {e}")

                await asyncio.sleep(0.1)

    def _assess_crypto_impact(self, title: str) -> float:
        """마켓 제목으로 크립토 영향도 판단 (-1.0 ~ +1.0)"""
        title_lower = title.lower()
        max_impact = 0.0

        for keyword, impact in self.IMPACT_MAP.items():
            if keyword in title_lower:
                if abs(impact) > abs(max_impact):
                    max_impact = impact

        return max_impact

    def _compute_signal(self) -> dict:
        """모든 모니터링 마켓의 움직임을 종합하여 시그널 산출"""
        if not self._markets:
            return self._empty_signal()

        alerts = []
        weighted_signals = []

        for mkt in self._markets:
            impact = mkt.get("impact", 0)
            yes_price = mkt.get("yes_price", 0.5)
            one_hour_change = abs(mkt.get("one_hour_change", 0))
            one_day_change = abs(mkt.get("one_day_change", 0))
            vol_24h = mkt.get("volume_24h", 0)

            # === 급변 감지: 1시간 내 확률 변화가 크면 시그널 ===
            # 확률이 한 방향으로 강하게 이동 = 대규모 베팅 유입
            hour_change_raw = mkt.get("one_hour_change", 0)

            if one_hour_change > 0.03:  # 1시간에 3%p 이상 변동
                # Yes 확률 상승 + impact 양수 = crypto bullish
                # Yes 확률 상승 + impact 음수 = crypto bearish
                direction = 1.0 if hour_change_raw > 0 else -1.0
                crypto_direction = direction * impact

                # 시그널 강도 = 변화량 × 거래량 정규화 × 영향도
                vol_factor = min(vol_24h / 100000, 2.0)  # $100K 기준
                strength = min(one_hour_change * 5, 1.0) * vol_factor * abs(impact)

                weighted_signals.append({
                    "direction": crypto_direction,
                    "strength": strength,
                    "market": mkt.get("event_title", "")[:50],
                })

                # 강한 움직임은 알림
                if one_hour_change > 0.05 and vol_24h > 50000:
                    yes_pct = yes_price * 100
                    change_pct = hour_change_raw * 100
                    alert = (
                        f"{'🟢' if crypto_direction > 0 else '🔴'} "
                        f"{mkt.get('event_title', '')[:40]} | "
                        f"Yes: {yes_pct:.0f}% ({change_pct:+.1f}%/1h) | "
                        f"Vol: ${vol_24h:,.0f}"
                    )
                    alerts.append(alert)

            # === 극단적 확률 (>85% 또는 <15%) + 최근 급변 = 강한 확신 ===
            if (yes_price > 0.85 or yes_price < 0.15) and one_day_change > 0.05:
                direction = 1.0 if yes_price > 0.85 else -1.0
                crypto_direction = direction * impact
                strength = 0.3 * abs(impact)
                weighted_signals.append({
                    "direction": crypto_direction,
                    "strength": strength,
                    "market": mkt.get("event_title", "")[:50],
                })

        # === 종합 시그널 계산 ===
        if not weighted_signals:
            result = self._empty_signal()
            result["polymarket_markets_monitored"] = len(self._markets)
            return result

        # 가중 평균: strength가 큰 시그널이 더 큰 영향
        total_weight = sum(s["strength"] for s in weighted_signals)
        if total_weight > 0:
            composite = sum(s["direction"] * s["strength"] for s in weighted_signals) / total_weight
            confidence = min(total_weight, 1.0)
        else:
            composite = 0.0
            confidence = 0.0

        composite = max(-1.0, min(1.0, composite))

        if composite > 0.15:
            direction = "bullish"
        elif composite < -0.15:
            direction = "bearish"
        else:
            direction = "neutral"

        if alerts:
            for alert in alerts[:3]:
                logger.info(f"[Polymarket] {alert}")

        return {
            "polymarket_score": round(composite, 4),
            "polymarket_direction": direction,
            "polymarket_confidence": round(confidence, 4),
            "polymarket_alerts": alerts[:5],
            "polymarket_markets_monitored": len(self._markets),
            "polymarket_active_signals": len(weighted_signals),
        }

    def get_signal(self) -> dict:
        """현재 시그널 반환"""
        return self.current_signal

    def get_features(self) -> dict:
        """ML 피처용 반환"""
        sig = self.current_signal
        return {
            "polymarket_score": sig.get("polymarket_score", 0),
            "polymarket_confidence": sig.get("polymarket_confidence", 0),
            "polymarket_markets": sig.get("polymarket_markets_monitored", 0),
        }

    def get_report(self) -> dict:
        """대시보드용 상세 리포트"""
        top_markets = []
        for mkt in self._markets[:5]:
            top_markets.append({
                "title": mkt.get("event_title", "")[:60],
                "yes_price": f"{mkt.get('yes_price', 0) * 100:.0f}%",
                "1h_change": f"{mkt.get('one_hour_change', 0) * 100:+.1f}%",
                "24h_vol": f"${mkt.get('volume_24h', 0):,.0f}",
                "impact": mkt.get("impact", 0),
            })

        return {
            "signal": self.current_signal,
            "monitored_markets": len(self._markets),
            "top_markets": top_markets,
        }
