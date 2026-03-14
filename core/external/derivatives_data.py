"""파생상품 데이터 수집기 - 펀딩비, 미결제약정(OI), 롱숏비율

선물 트레이딩에서 가장 중요한 데이터:
- 펀딩비 > 0.05%: 롱 과열 → 숏 유리 (대중의 반대편)
- 펀딩비 < -0.05%: 숏 과열 → 롱 유리
- OI 급증 + 가격 상승: 진짜 상승 (신규 진입)
- OI 급증 + 가격 하락: 강한 하락 (숏 진입)
- OI 감소 + 가격 변동: 청산 중 (추세 약화)
- 롱숏비율 > 2.0: 롱 과열 → 역방향 가능
- 롱숏비율 < 0.5: 숏 과열 → 반등 가능

모든 API 무료, 키 불필요 (Binance 공개 API)
"""

import asyncio
from datetime import datetime, timedelta

import aiohttp
from loguru import logger


class DerivativesDataCollector:
    """Binance 선물 파생상품 데이터 수집"""

    BASE_URL = "https://fapi.binance.com"

    def __init__(self):
        self._cache: dict = {}
        self._last_update: datetime | None = None
        self._history: list[dict] = []  # OI 변화 추적용

    async def collect(self, symbol: str = "BTCUSDT") -> dict:
        """모든 파생상품 데이터 수집"""
        results = {}

        tasks = {
            "funding_rate": self._fetch_funding_rate(symbol),
            "open_interest": self._fetch_open_interest(symbol),
            "long_short_ratio": self._fetch_long_short_ratio(symbol),
            "taker_volume": self._fetch_taker_buy_sell(symbol),
        }

        for name, coro in tasks.items():
            try:
                results[name] = await coro
            except Exception as e:
                logger.debug(f"파생상품 데이터 수집 실패 ({name}): {e}")
                results[name] = {}

        # 복합 시그널 계산
        results["composite"] = self._calculate_composite(results)

        self._cache = results
        self._last_update = datetime.utcnow()

        # 히스토리 저장 (OI 변화 추적)
        self._history.append({
            "time": datetime.utcnow(),
            "oi": results.get("open_interest", {}).get("open_interest", 0),
            "funding": results.get("funding_rate", {}).get("current_rate", 0),
        })
        if len(self._history) > 200:
            self._history = self._history[-200:]

        return results

    async def _fetch_funding_rate(self, symbol: str) -> dict:
        """현재 + 최근 펀딩비"""
        async with aiohttp.ClientSession() as session:
            # 현재 펀딩비
            url = f"{self.BASE_URL}/fapi/v1/premiumIndex"
            async with session.get(url, params={"symbol": symbol}, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()

            current_rate = float(data.get("lastFundingRate", 0))
            mark_price = float(data.get("markPrice", 0))

            # 최근 펀딩비 히스토리 (8시간 간격)
            url_hist = f"{self.BASE_URL}/fapi/v1/fundingRate"
            async with session.get(url_hist, params={"symbol": symbol, "limit": 10}, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    hist_data = []
                else:
                    hist_data = await resp.json()

            rates = [float(h["fundingRate"]) for h in hist_data] if hist_data else [current_rate]
            avg_rate = sum(rates) / len(rates) if rates else 0

            # 펀딩비 해석
            # 양수: 롱이 숏에게 지불 (롱 과열)
            # 음수: 숏이 롱에게 지불 (숏 과열)
            is_extreme_long = current_rate > 0.0005  # 0.05% 이상
            is_extreme_short = current_rate < -0.0005
            is_very_extreme = abs(current_rate) > 0.001  # 0.1% 이상

            return {
                "current_rate": current_rate,
                "current_rate_pct": round(current_rate * 100, 4),
                "avg_rate_24h": round(avg_rate, 6),
                "mark_price": mark_price,
                "is_extreme_long": is_extreme_long,
                "is_extreme_short": is_extreme_short,
                "is_very_extreme": is_very_extreme,
                "annualized_pct": round(current_rate * 3 * 365 * 100, 2),  # 연환산
                "interpretation": self._interpret_funding(current_rate),
            }

    async def _fetch_open_interest(self, symbol: str) -> dict:
        """미결제약정"""
        async with aiohttp.ClientSession() as session:
            # 현재 OI
            url = f"{self.BASE_URL}/fapi/v1/openInterest"
            async with session.get(url, params={"symbol": symbol}, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()

            current_oi = float(data.get("openInterest", 0))

            # OI 히스토리 (5분봉)
            url_hist = f"{self.BASE_URL}/futures/data/openInterestHist"
            async with session.get(url_hist, params={
                "symbol": symbol, "period": "5m", "limit": 48,  # 최근 4시간
            }, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    hist_data = []
                else:
                    hist_data = await resp.json()

            oi_values = [float(h["sumOpenInterest"]) for h in hist_data] if hist_data else [current_oi]

            # OI 변화율 계산
            if len(oi_values) >= 2:
                oi_change_1h = (oi_values[-1] - oi_values[-12]) / oi_values[-12] if len(oi_values) >= 12 else 0
                oi_change_4h = (oi_values[-1] - oi_values[0]) / oi_values[0] if oi_values[0] > 0 else 0
            else:
                oi_change_1h = 0
                oi_change_4h = 0

            # OI 급변 감지
            is_oi_surge = oi_change_1h > 0.03  # 1시간에 3% 이상 증가
            is_oi_drop = oi_change_1h < -0.03  # 1시간에 3% 이상 감소

            return {
                "open_interest": current_oi,
                "oi_change_1h_pct": round(oi_change_1h * 100, 2),
                "oi_change_4h_pct": round(oi_change_4h * 100, 2),
                "is_oi_surge": is_oi_surge,
                "is_oi_drop": is_oi_drop,
            }

    async def _fetch_long_short_ratio(self, symbol: str) -> dict:
        """롱숏비율 (Top Trader + Global)"""
        async with aiohttp.ClientSession() as session:
            result = {}

            # Top Trader 롱숏비율
            url_top = f"{self.BASE_URL}/futures/data/topLongShortAccountRatio"
            async with session.get(url_top, params={
                "symbol": symbol, "period": "5m", "limit": 12,
            }, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        latest = data[-1]
                        result["top_long_ratio"] = float(latest.get("longAccount", 0.5))
                        result["top_short_ratio"] = float(latest.get("shortAccount", 0.5))
                        result["top_ls_ratio"] = float(latest.get("longShortRatio", 1.0))

            # Global 롱숏비율
            url_global = f"{self.BASE_URL}/futures/data/globalLongShortAccountRatio"
            async with session.get(url_global, params={
                "symbol": symbol, "period": "5m", "limit": 12,
            }, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        latest = data[-1]
                        result["global_long_ratio"] = float(latest.get("longAccount", 0.5))
                        result["global_short_ratio"] = float(latest.get("shortAccount", 0.5))
                        result["global_ls_ratio"] = float(latest.get("longShortRatio", 1.0))

            # 해석
            top_ls = result.get("top_ls_ratio", 1.0)
            global_ls = result.get("global_ls_ratio", 1.0)

            result["crowd_sentiment"] = "extreme_long" if global_ls > 2.0 else \
                                        "long_biased" if global_ls > 1.3 else \
                                        "extreme_short" if global_ls < 0.5 else \
                                        "short_biased" if global_ls < 0.77 else "neutral"

            # 스마트머니 vs 군중 괴리
            if top_ls > 0 and global_ls > 0:
                result["smart_crowd_divergence"] = round(top_ls - global_ls, 3)

            return result

    async def _fetch_taker_buy_sell(self, symbol: str) -> dict:
        """테이커 매수/매도 비율"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.BASE_URL}/futures/data/takerlongshortRatio"
            async with session.get(url, params={
                "symbol": symbol, "period": "5m", "limit": 12,
            }, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()

            if not data:
                return {}

            latest = data[-1]
            buy_vol = float(latest.get("buyVol", 0))
            sell_vol = float(latest.get("sellVol", 0))
            ratio = float(latest.get("buySellRatio", 1.0))

            # 최근 1시간 추세
            ratios = [float(d.get("buySellRatio", 1.0)) for d in data]
            avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0

            return {
                "buy_sell_ratio": ratio,
                "avg_ratio_1h": round(avg_ratio, 3),
                "buy_volume": buy_vol,
                "sell_volume": sell_vol,
                "is_aggressive_buying": ratio > 1.3,
                "is_aggressive_selling": ratio < 0.77,
            }

    def _interpret_funding(self, rate: float) -> str:
        """펀딩비 해석"""
        if rate > 0.001:
            return "극도의 롱 과열 → 강한 숏 시그널"
        elif rate > 0.0005:
            return "롱 과열 → 숏 우세"
        elif rate > 0.0001:
            return "약간의 롱 우세 → 중립"
        elif rate > -0.0001:
            return "균형 상태"
        elif rate > -0.0005:
            return "약간의 숏 우세 → 중립"
        elif rate > -0.001:
            return "숏 과열 → 롱 우세"
        else:
            return "극도의 숏 과열 → 강한 롱 시그널"

    def _calculate_composite(self, data: dict) -> dict:
        """복합 파생상품 시그널 계산

        핵심 로직: 대중의 반대편에 서기 (Contrarian)
        - 펀딩비 극단 → 반대 방향
        - 롱숏비율 극단 → 반대 방향
        - OI 급증 → 큰 움직임 예고 (방향은 다른 지표로 판단)
        """
        score = 0.0
        reasons = []

        # 1. 펀딩비 시그널 (가중치 40%)
        funding = data.get("funding_rate", {})
        fr = funding.get("current_rate", 0)
        if fr != 0:
            # 펀딩비의 반대 방향이 유리 (contrarian)
            funding_signal = -fr * 1000  # 스케일 조정
            funding_signal = max(-1, min(1, funding_signal))
            score += funding_signal * 0.4

            if funding.get("is_very_extreme"):
                reasons.append(f"펀딩비 극단 {fr*100:.3f}% → 역방향 강력 시그널")
            elif funding.get("is_extreme_long"):
                reasons.append(f"롱 과열 펀딩 {fr*100:.3f}%")
            elif funding.get("is_extreme_short"):
                reasons.append(f"숏 과열 펀딩 {fr*100:.3f}%")

        # 2. 롱숏비율 시그널 (가중치 30%)
        ls = data.get("long_short_ratio", {})
        global_ls = ls.get("global_ls_ratio", 1.0)
        if global_ls > 0:
            # 롱숏비율의 반대 (contrarian)
            ls_signal = -(global_ls - 1.0)  # 1.0이 균형, 편차의 반대
            ls_signal = max(-1, min(1, ls_signal))
            score += ls_signal * 0.3

            if ls.get("crowd_sentiment") in ["extreme_long", "extreme_short"]:
                reasons.append(f"군중 {ls['crowd_sentiment']} (LS={global_ls:.2f})")

        # 3. OI 시그널 (가중치 20%)
        oi = data.get("open_interest", {})
        oi_change = oi.get("oi_change_1h_pct", 0) / 100
        if oi.get("is_oi_surge"):
            reasons.append(f"OI 급증 {oi['oi_change_1h_pct']:+.1f}% → 큰 변동 예고")
            # OI 급증 방향은 펀딩비와 롱숏비율로 판단
            score *= 1.3  # 기존 시그널 강화
        elif oi.get("is_oi_drop"):
            reasons.append(f"OI 급감 {oi['oi_change_1h_pct']:+.1f}% → 청산 중, 추세 약화")
            score *= 0.5  # 시그널 약화 (방향 불확실)

        # 4. 테이커 매수/매도 (가중치 10%)
        taker = data.get("taker_volume", {})
        taker_ratio = taker.get("buy_sell_ratio", 1.0)
        if taker_ratio > 0:
            taker_signal = (taker_ratio - 1.0) * 0.5
            taker_signal = max(-1, min(1, taker_signal))
            score += taker_signal * 0.1

        score = max(-1, min(1, score))

        # 방향 판별
        if score > 0.2:
            direction = "bullish"
        elif score < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"

        strength = "strong" if abs(score) > 0.5 else "moderate" if abs(score) > 0.2 else "weak"

        return {
            "score": round(score, 3),
            "direction": direction,
            "strength": strength,
            "confidence": round(min(abs(score) * 1.5, 1.0), 3),
            "reasons": reasons,
        }

    def get_features(self) -> dict:
        """ML 모델용 피처"""
        d = self._cache
        if not d:
            return {}

        funding = d.get("funding_rate", {})
        oi = d.get("open_interest", {})
        ls = d.get("long_short_ratio", {})
        taker = d.get("taker_volume", {})
        composite = d.get("composite", {})

        return {
            "deriv_funding_rate": funding.get("current_rate", 0) * 10000,  # 스케일
            "deriv_funding_annualized": funding.get("annualized_pct", 0) / 100,
            "deriv_oi_change_1h": oi.get("oi_change_1h_pct", 0) / 100,
            "deriv_oi_change_4h": oi.get("oi_change_4h_pct", 0) / 100,
            "deriv_top_ls_ratio": ls.get("top_ls_ratio", 1.0),
            "deriv_global_ls_ratio": ls.get("global_ls_ratio", 1.0),
            "deriv_smart_crowd_div": ls.get("smart_crowd_divergence", 0),
            "deriv_taker_ratio": taker.get("buy_sell_ratio", 1.0),
            "deriv_composite_score": composite.get("score", 0),
        }

    def get_signal_for_strategy(self) -> dict:
        """전략 매니저용 시그널"""
        composite = self._cache.get("composite", {})
        return {
            "score": composite.get("score", 0),
            "direction": composite.get("direction", "neutral"),
            "strength": composite.get("strength", "weak"),
            "confidence": composite.get("confidence", 0),
        }

    def get_report(self) -> dict:
        """대시보드용 리포트"""
        d = self._cache
        if not d:
            return {"status": "no_data"}

        return {
            "funding_rate": d.get("funding_rate", {}),
            "open_interest": d.get("open_interest", {}),
            "long_short_ratio": d.get("long_short_ratio", {}),
            "taker_volume": d.get("taker_volume", {}),
            "composite": d.get("composite", {}),
        }
