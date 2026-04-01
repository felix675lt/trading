"""실제 매크로 경제 지표 수집 — Yahoo Finance API

기존 macro_collector.py는 CoinGecko 크립토 데이터만 수집.
이 모듈은 전통 금융시장의 실제 매크로 데이터를 수집:
- WTI/Brent 유가 (전쟁/지정학 시그널)
- 금 가격 (안전자산 흐름)
- DXY 달러 인덱스 (약달러 = 크립토 불리시)
- 미국채 10년 금리 (리스크온/오프)
- VIX 변동성 지수 (시장 공포)
- S&P 500 / 나스닥 (위험자산 상관관계)
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

_YF_AVAILABLE = None  # lazy init
yf = None


def _ensure_yfinance():
    global _YF_AVAILABLE, yf
    if _YF_AVAILABLE is not None:
        return _YF_AVAILABLE
    try:
        import sys
        logger.info(f"[RealMacro] sys.path: {sys.path[-3:]}")
        import yfinance as _yf
        yf = _yf
        _YF_AVAILABLE = True
        logger.info(f"[RealMacro] yfinance {_yf.__version__} 로드 성공")
    except Exception as e:
        _YF_AVAILABLE = False
        # 폴백: site-packages 직접 추가
        import sys
        site_pkg = "/opt/homebrew/lib/python3.13/site-packages"
        if site_pkg not in sys.path:
            sys.path.append(site_pkg)
            logger.info(f"[RealMacro] site-packages 경로 추가 후 재시도...")
            try:
                import yfinance as _yf
                yf = _yf
                _YF_AVAILABLE = True
                logger.info(f"[RealMacro] yfinance {_yf.__version__} 로드 성공 (경로 추가)")
            except Exception as e2:
                logger.warning(f"[RealMacro] yfinance 재시도 실패: {e2}")
        else:
            logger.warning(f"[RealMacro] yfinance 로드 실패: {type(e).__name__}: {e}")
    return _YF_AVAILABLE


class RealMacroCollector:
    """Yahoo Finance 기반 실제 매크로 데이터 수집기"""

    # Yahoo Finance 티커 매핑
    TICKERS = {
        "wti_oil": "CL=F",        # WTI 원유 선물
        "brent_oil": "BZ=F",      # Brent 원유 선물
        "gold": "GC=F",           # 금 선물
        "dxy": "DX-Y.NYB",       # 달러 인덱스
        "us10y": "^TNX",          # 미국채 10년 금리
        "vix": "^VIX",            # VIX 변동성 지수
        "sp500": "^GSPC",         # S&P 500
        "nasdaq": "^IXIC",        # 나스닥 종합
    }

    def __init__(self):
        self.data: dict = {}
        self.prev_data: dict = {}  # 이전 데이터 (변화율 계산용)
        self.last_fetch: Optional[datetime] = None
        self.fetch_interval = timedelta(minutes=15)  # 15분마다 갱신
        self.history: list[dict] = []
        self._fetch_count = 0

    async def fetch(self) -> dict:
        """매크로 데이터 수집 (비동기 래퍼)"""
        if not _ensure_yfinance():
            return self.data

        now = datetime.utcnow()
        if self.last_fetch and (now - self.last_fetch) < self.fetch_interval:
            return self.data

        logger.info("[RealMacro] 데이터 수집 시작...")

        # yfinance는 동기식이므로 executor에서 실행 (30초 타임아웃)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, self._fetch_sync),
                timeout=30.0,
            )
            self.last_fetch = now
            self._fetch_count += 1

            # 히스토리 기록
            self.history.append({**self.data, "timestamp": now.isoformat()})
            if len(self.history) > 200:
                self.history = self.history[-200:]

            logger.info(f"[RealMacro] 수집 완료: {len(self.data)}개 데이터 포인트")

        except asyncio.TimeoutError:
            logger.warning("[RealMacro] 수집 타임아웃 (30초)")
        except Exception as e:
            logger.warning(f"[RealMacro] 수집 실패: {type(e).__name__}: {e}")

        return self.data

    def _fetch_sync(self):
        """동기식 Yahoo Finance 데이터 수집"""
        logger.info("[RealMacro] Yahoo Finance 데이터 다운로드 시작...")
        # 이전 데이터 보존 (변화율 계산용)
        self.prev_data = dict(self.data)

        tickers_str = " ".join(self.TICKERS.values())
        try:
            # 1일 데이터 일괄 다운로드
            data = yf.download(
                tickers_str,
                period="5d",
                interval="1d",
                progress=False,
                threads=True,
            )

            if data.empty:
                logger.warning("[RealMacro] Yahoo Finance 데이터 비어있음")
                return

            for name, ticker in self.TICKERS.items():
                try:
                    if ticker in data["Close"].columns:
                        prices = data["Close"][ticker].dropna()
                    else:
                        # 단일 티커인 경우
                        prices = data["Close"].dropna()

                    if len(prices) >= 1:
                        current = float(prices.iloc[-1])
                        self.data[f"{name}_price"] = current

                        if len(prices) >= 2:
                            prev = float(prices.iloc[-2])
                            if prev > 0:
                                change_pct = (current - prev) / prev
                                self.data[f"{name}_change_1d"] = change_pct
                            else:
                                self.data[f"{name}_change_1d"] = 0.0
                        else:
                            self.data[f"{name}_change_1d"] = 0.0

                except Exception as e:
                    logger.debug(f"[RealMacro] {name} 파싱 실패: {e}")

            # 개별 티커로 재시도 (일괄 실패 시)
            for name, ticker in self.TICKERS.items():
                if f"{name}_price" not in self.data:
                    try:
                        t = yf.Ticker(ticker)
                        hist = t.history(period="5d")
                        if not hist.empty:
                            current = float(hist["Close"].iloc[-1])
                            self.data[f"{name}_price"] = current
                            if len(hist) >= 2:
                                prev = float(hist["Close"].iloc[-2])
                                self.data[f"{name}_change_1d"] = (current - prev) / prev if prev > 0 else 0
                            else:
                                self.data[f"{name}_change_1d"] = 0.0
                    except Exception:
                        pass

            self._log_summary()

        except Exception as e:
            logger.warning(f"[RealMacro] Yahoo Finance 다운로드 실패: {e}")
            # 개별 다운로드 폴백
            self._fetch_individual()

    def _fetch_individual(self):
        """개별 티커 하나씩 다운로드 (폴백)"""
        for name, ticker in self.TICKERS.items():
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="5d")
                if not hist.empty:
                    current = float(hist["Close"].iloc[-1])
                    self.data[f"{name}_price"] = current
                    if len(hist) >= 2:
                        prev = float(hist["Close"].iloc[-2])
                        self.data[f"{name}_change_1d"] = (current - prev) / prev if prev > 0 else 0
                    else:
                        self.data[f"{name}_change_1d"] = 0.0
            except Exception as e:
                logger.debug(f"[RealMacro] {name} 개별 수집 실패: {e}")

        self._log_summary()

    def _log_summary(self):
        """수집 결과 로깅"""
        oil = self.data.get("wti_oil_price", 0)
        gold = self.data.get("gold_price", 0)
        dxy = self.data.get("dxy_price", 0)
        vix = self.data.get("vix_price", 0)
        sp = self.data.get("sp500_change_1d", 0)

        logger.info(
            f"[RealMacro] WTI: ${oil:.1f} | Gold: ${gold:.0f} | "
            f"DXY: {dxy:.1f} | VIX: {vix:.1f} | S&P500: {sp:+.2%}"
        )

    def get_features(self) -> dict:
        """ML 피처로 변환"""
        features = {}

        # 유가 피처
        oil_price = self.data.get("wti_oil_price", 0)
        oil_change = self.data.get("wti_oil_change_1d", 0)
        features["real_macro_oil_price"] = oil_price
        features["real_macro_oil_change"] = oil_change

        # 금 피처
        gold_price = self.data.get("gold_price", 0)
        gold_change = self.data.get("gold_change_1d", 0)
        features["real_macro_gold_price"] = gold_price
        features["real_macro_gold_change"] = gold_change

        # DXY 피처 (약달러 = 크립토 불리시)
        dxy = self.data.get("dxy_price", 100)
        dxy_change = self.data.get("dxy_change_1d", 0)
        features["real_macro_dxy"] = dxy
        features["real_macro_dxy_change"] = dxy_change

        # VIX 피처
        vix = self.data.get("vix_price", 20)
        vix_change = self.data.get("vix_change_1d", 0)
        features["real_macro_vix"] = vix
        features["real_macro_vix_change"] = vix_change

        # 국채 금리
        us10y = self.data.get("us10y_price", 4.0)
        us10y_change = self.data.get("us10y_change_1d", 0)
        features["real_macro_us10y"] = us10y
        features["real_macro_us10y_change"] = us10y_change

        # 주식시장
        sp_change = self.data.get("sp500_change_1d", 0)
        nq_change = self.data.get("nasdaq_change_1d", 0)
        features["real_macro_sp500_change"] = sp_change
        features["real_macro_nasdaq_change"] = nq_change

        # ===== 종합 매크로 점수 계산 =====
        score = 0.0

        # 1. DXY 방향 (크립토와 역상관)
        if dxy_change < -0.005:       # 달러 0.5%+ 약세 → 크립토 불리시
            score += 0.3
        elif dxy_change < -0.002:
            score += 0.15
        elif dxy_change > 0.005:      # 달러 강세 → 크립토 베어리시
            score -= 0.3
        elif dxy_change > 0.002:
            score -= 0.15

        # 2. DXY 절대 수준
        if dxy < 98:
            score += 0.15  # 약달러 구간
        elif dxy > 105:
            score -= 0.15  # 강달러 구간

        # 3. VIX (공포 = 위험자산 회피)
        if vix > 30:
            score -= 0.25
        elif vix > 25:
            score -= 0.15
        elif vix < 15:
            score += 0.1

        # 4. 유가 급변 (지정학 리스크 프록시)
        if abs(oil_change) > 0.05:    # 5%+ 급변 → 시장 불확실성
            score -= 0.15             # 방향 불문 급변 자체가 리스크
        elif oil_change < -0.03:      # 유가 하락 → 전쟁해소 or 수요부진
            score += 0.1              # 약간 긍정적 (불확실성 해소)

        # 5. 금 가격 (안전자산 선호도)
        if gold_change > 0.02:        # 금 급등 → 리스크 오프
            score -= 0.1
        elif gold_change < -0.01:     # 금 하락 → 위험자산 선호
            score += 0.1

        # 6. 주식시장 상관관계
        if sp_change > 0.01:
            score += 0.15
        elif sp_change > 0.005:
            score += 0.08
        elif sp_change < -0.01:
            score -= 0.15
        elif sp_change < -0.005:
            score -= 0.08

        # 7. 국채 금리 (금리 하락 → 유동성 기대 → 크립토 불리시)
        if us10y_change < -0.02:      # 금리 하락
            score += 0.15
        elif us10y_change > 0.02:     # 금리 상승
            score -= 0.15

        features["real_macro_composite_score"] = max(-1.0, min(1.0, score))

        # 지정학 리스크 레벨 (유가 + VIX 기반)
        geo_risk = 0.0
        if oil_price > 100:
            geo_risk += 0.3
        if oil_price > 120:
            geo_risk += 0.3
        if vix > 25:
            geo_risk += 0.2
        if vix > 35:
            geo_risk += 0.2
        features["real_macro_geo_risk"] = min(1.0, geo_risk)

        return features

    def get_signal(self) -> dict:
        """전략 매니저용 시그널"""
        features = self.get_features()
        score = features.get("real_macro_composite_score", 0)

        if score > 0.2:
            direction = "bullish"
        elif score < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "score": score,
            "direction": direction,
            "geo_risk": features.get("real_macro_geo_risk", 0),
            "dxy": self.data.get("dxy_price", 0),
            "vix": self.data.get("vix_price", 0),
            "oil": self.data.get("wti_oil_price", 0),
            "gold": self.data.get("gold_price", 0),
        }

    def get_report(self) -> dict:
        """대시보드용 리포트"""
        return {
            "available": _YF_AVAILABLE,
            "last_fetch": self.last_fetch.isoformat() if self.last_fetch else None,
            "fetch_count": self._fetch_count,
            "data": dict(self.data),
            "features": self.get_features(),
        }
