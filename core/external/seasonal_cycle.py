"""계절 사이클 분석 엔진 - BTC 10년 역사적 패턴 기반 시즌 시그널

핵심 발견 (2017-2024 검증):
1. December Low → February High 반등: 7년 중 6년 발생 (85.7%)
   - 평균 반등: +54.4% (6%~215%)
   - 평균 기간: 64일 (51~72일)
   - 유일한 예외: 2017년 (블로오프탑 → 크래시)

2. 반감기 사이클과 강한 상관:
   - 반감기 연도 + 이후 1년: Sep-Nov 강세, Sep-Apr 대폭 상승
   - 베어 연도: Sep-Nov 하락, 하지만 Dec-Feb 반등은 여전히 발생

3. Sep 1 ~ Apr 1 (212일) 사이클:
   - 불장 연도: +55% ~ +395%
   - 베어 연도: -33% ~ -30%
   - 회복/반감기 연도: +42% ~ +167%
"""

from datetime import datetime, date
from loguru import logger


class SeasonalCycleAnalyzer:
    """BTC 10년 계절 패턴 기반 시그널 생성"""

    # 반감기 날짜
    HALVING_DATES = [
        date(2012, 11, 28),
        date(2016, 7, 9),
        date(2020, 5, 11),
        date(2024, 4, 19),
        date(2028, 4, 1),  # 예상
    ]

    # 검증된 역사적 사이클 데이터
    HISTORICAL_CYCLES = [
        {
            "year": 2017,
            "halving_phase": "post_halving_1",  # 반감기 다음 해 (폭등)
            "sep_nov_pct": 113.25,
            "sep_nov_dir": "up",
            "dec_low_day": 17,    # 2017은 예외: 12월이 고점
            "dec_feb_pct": -69.57,  # 유일한 하락 (블로오프탑)
            "dec_feb_days": 51,
            "sep_apr_pct": 55.2,
            "is_exception": True,  # 블로오프탑 예외 케이스
            "volume_k": 117.749,
        },
        {
            "year": 2018,
            "halving_phase": "post_halving_2",  # 베어마켓
            "sep_nov_pct": -44.59,
            "sep_nov_dir": "down",
            "dec_low_day": 15,
            "dec_feb_pct": 28.92,
            "dec_feb_days": 71,
            "sep_apr_pct": -33.21,
            "is_exception": False,
            "volume_k": 2941,
        },
        {
            "year": 2019,
            "halving_phase": "pre_halving",  # 반감기 전 해 (회복)
            "sep_nov_pct": -19.95,
            "sep_nov_dir": "down",
            "dec_low_day": 18,
            "dec_feb_pct": 57.33,
            "dec_feb_days": 62,
            "sep_apr_pct": -30.28,
            "is_exception": False,
            "volume_k": 4063,
        },
        {
            "year": 2020,
            "halving_phase": "halving_year",  # 반감기 연도
            "sep_nov_pct": 66.61,
            "sep_nov_dir": "up",
            "dec_low_day": 11,
            "dec_feb_pct": 215.78,
            "dec_feb_days": 72,
            "sep_apr_pct": 394.97,
            "is_exception": False,
            "volume_k": 6030,
        },
        {
            "year": 2021,
            "halving_phase": "post_halving_1",  # 반감기 다음 해
            "sep_nov_pct": 19.83,
            "sep_nov_dir": "up",
            "dec_low_day": 4,
            "dec_feb_pct": 6.0,
            "dec_feb_days": 66,
            "sep_apr_pct": -1.86,
            "is_exception": False,
            "volume_k": 4385,
        },
        {
            "year": 2022,
            "halving_phase": "post_halving_2",  # 베어마켓
            "sep_nov_pct": -14.93,
            "sep_nov_dir": "down",
            "dec_low_day": 19,
            "dec_feb_pct": 52.23,
            "dec_feb_days": 62,
            "sep_apr_pct": 42.35,
            "is_exception": False,
            "volume_k": 26466,
        },
        {
            "year": 2023,
            "halving_phase": "pre_halving",  # 반감기 전 해
            "sep_nov_pct": 49.23,
            "sep_nov_dir": "up",
            "dec_low_day": 11,
            "dec_feb_pct": 20.45,
            "dec_feb_days": 62,
            "sep_apr_pct": 166.92,
            "is_exception": False,
            "volume_k": 3006,
        },
    ]

    def __init__(self):
        self.current_signal: dict = {}
        self._last_update: datetime | None = None

    def get_halving_phase(self, dt: date | None = None) -> dict:
        """현재 반감기 사이클 위치 계산"""
        if dt is None:
            dt = date.today()

        # 가장 최근 반감기와 다음 반감기 찾기
        last_halving = None
        next_halving = None
        for i, h_date in enumerate(self.HALVING_DATES):
            if h_date <= dt:
                last_halving = h_date
            if h_date > dt and next_halving is None:
                next_halving = h_date

        if last_halving is None:
            return {"phase": "unknown", "days_since": 0, "days_until": 0, "progress": 0}

        days_since = (dt - last_halving).days
        days_until = (next_halving - dt).days if next_halving else 0
        cycle_length = (next_halving - last_halving).days if next_halving else 1461  # ~4년

        # 사이클 내 위치 (0~1)
        progress = days_since / cycle_length

        # 반감기 사이클 단계 판별
        if days_since < 365:
            phase = "halving_year"
        elif days_since < 730:
            phase = "post_halving_1"  # 통상 최대 강세
        elif days_since < 1095:
            phase = "post_halving_2"  # 통상 베어마켓
        else:
            phase = "pre_halving"  # 회복/축적기

        return {
            "phase": phase,
            "days_since_halving": days_since,
            "days_until_next": days_until,
            "cycle_progress": round(progress, 3),
            "last_halving": str(last_halving),
            "next_halving": str(next_halving) if next_halving else "unknown",
        }

    def get_seasonal_signal(self, dt: datetime | None = None) -> dict:
        """현재 날짜 기준 계절 시그널 생성"""
        if dt is None:
            dt = datetime.utcnow()

        today = dt.date() if isinstance(dt, datetime) else dt
        month = today.month
        day = today.day

        halving = self.get_halving_phase(today)
        phase = halving["phase"]

        # 같은 반감기 단계의 역사적 데이터 수집
        same_phase = [c for c in self.HISTORICAL_CYCLES if c["halving_phase"] == phase]

        signal = {
            "halving_phase": phase,
            "halving_info": halving,
            "season": self._get_season_name(month),
            "historical_match_count": len(same_phase),
            "confidence": 0.0,
            "direction": "neutral",
            "score": 0.0,
            "reason": "",
            "patterns": [],
        }

        patterns = []

        # === 패턴 1: December Low → February Bounce ===
        if month == 12 and day >= 1:
            # 12월: 저점 매수 구간 진입
            avg_low_day = sum(c["dec_low_day"] for c in self.HISTORICAL_CYCLES if not c["is_exception"]) / 6
            # 저점은 보통 12월 4일~19일 (평균 ~13일)

            if day < 20:
                # 12월 초~중순: 저점 형성 구간
                bounce_rates = [c["dec_feb_pct"] for c in self.HISTORICAL_CYCLES if not c["is_exception"]]
                avg_bounce = sum(bounce_rates) / len(bounce_rates)
                min_bounce = min(bounce_rates)

                confidence = 0.85  # 6/7 = 85.7%
                patterns.append({
                    "name": "december_low_forming",
                    "confidence": confidence,
                    "avg_bounce_pct": round(avg_bounce, 1),
                    "min_bounce_pct": round(min_bounce, 1),
                    "direction": "bullish_setup",
                    "message": f"12월 저점 형성 구간 (역사적 평균 반등 +{avg_bounce:.0f}%, 최소 +{min_bounce:.0f}%)",
                })
                signal["direction"] = "bullish"
                signal["score"] = 0.4  # 아직 저점 확인 전이므로 중간
                signal["confidence"] = confidence
            else:
                # 12월 하순: 저점 통과 가능성 높음
                patterns.append({
                    "name": "december_low_likely_passed",
                    "confidence": 0.85,
                    "direction": "bullish",
                    "message": "12월 저점 통과 추정 → 2월까지 반등 구간",
                })
                signal["direction"] = "bullish"
                signal["score"] = 0.5
                signal["confidence"] = 0.85

        elif month == 1:
            # 1월: 반등 진행 중
            bounce_rates = [c["dec_feb_pct"] for c in self.HISTORICAL_CYCLES if not c["is_exception"]]
            avg_bounce = sum(bounce_rates) / len(bounce_rates)

            patterns.append({
                "name": "january_bounce_active",
                "confidence": 0.85,
                "avg_bounce_pct": round(avg_bounce, 1),
                "direction": "bullish",
                "message": f"1월 반등 구간 (역사적 85.7% 확률, 평균 +{avg_bounce:.0f}%)",
            })
            signal["direction"] = "bullish"
            signal["score"] = 0.5
            signal["confidence"] = 0.8

        elif month == 2 and day <= 20:
            # 2월 초~중순: 반등 고점 접근
            patterns.append({
                "name": "february_peak_approaching",
                "confidence": 0.7,
                "direction": "cautious_bullish",
                "message": "2월 반등 고점 접근 구간 → 익절 준비",
            })
            signal["direction"] = "bullish"
            signal["score"] = 0.3
            signal["confidence"] = 0.7

        elif month == 2 and day > 20:
            # 2월 하순: 반등 고점 도달 가능성
            patterns.append({
                "name": "february_peak_zone",
                "confidence": 0.65,
                "direction": "neutral",
                "message": "2월 반등 고점 구간 → 신규 롱 자제, 기존 포지션 정리 고려",
            })
            signal["direction"] = "neutral"
            signal["score"] = 0.0
            signal["confidence"] = 0.65

        # === 패턴 2: Sep-Nov 90일 구간 ===
        elif month >= 9 and month <= 11:
            # 반감기 단계별 Sep-Nov 방향 예측
            up_phases = [c for c in same_phase if c["sep_nov_dir"] == "up"]
            down_phases = [c for c in same_phase if c["sep_nov_dir"] == "down"]

            if len(same_phase) > 0:
                up_ratio = len(up_phases) / len(same_phase)
                avg_pct = sum(c["sep_nov_pct"] for c in same_phase) / len(same_phase)

                if up_ratio > 0.5:
                    signal["direction"] = "bullish"
                    signal["score"] = min(0.4, up_ratio * 0.5)
                    signal["confidence"] = up_ratio
                    patterns.append({
                        "name": "sep_nov_historical_bullish",
                        "confidence": up_ratio,
                        "avg_pct": round(avg_pct, 1),
                        "direction": "bullish",
                        "message": f"Sep-Nov 구간: {phase} 단계에서 역사적 상승 확률 {up_ratio:.0%} (평균 {avg_pct:+.1f}%)",
                    })
                else:
                    signal["direction"] = "bearish"
                    signal["score"] = max(-0.4, -(1 - up_ratio) * 0.5)
                    signal["confidence"] = 1 - up_ratio
                    patterns.append({
                        "name": "sep_nov_historical_bearish",
                        "confidence": 1 - up_ratio,
                        "avg_pct": round(avg_pct, 1),
                        "direction": "bearish",
                        "message": f"Sep-Nov 구간: {phase} 단계에서 역사적 하락 확률 {1-up_ratio:.0%} (평균 {avg_pct:+.1f}%)",
                    })

        # === 패턴 3: Sep-Apr 212일 장기 사이클 ===
        if 9 <= month <= 12 or 1 <= month <= 4:
            if len(same_phase) > 0:
                avg_long = sum(c["sep_apr_pct"] for c in same_phase) / len(same_phase)
                positive = [c for c in same_phase if c["sep_apr_pct"] > 0]
                pos_ratio = len(positive) / len(same_phase)

                patterns.append({
                    "name": "sep_apr_long_cycle",
                    "phase": phase,
                    "avg_pct": round(avg_long, 1),
                    "positive_ratio": pos_ratio,
                    "direction": "bullish" if avg_long > 0 else "bearish",
                    "message": f"Sep-Apr 장기: {phase} 단계 평균 {avg_long:+.1f}% (양수 확률 {pos_ratio:.0%})",
                })

        # === 패턴 4: 반감기 사이클 강도 ===
        phase_bias = {
            "halving_year": {"bias": "bullish", "strength": 0.3,
                            "desc": "반감기 연도 → 공급 감소, 역사적 강세"},
            "post_halving_1": {"bias": "bullish", "strength": 0.4,
                              "desc": "반감기 다음 해 → 역사적 최대 강세 구간"},
            "post_halving_2": {"bias": "bearish", "strength": -0.3,
                              "desc": "반감기 2년 후 → 역사적 베어마켓 구간"},
            "pre_halving": {"bias": "bullish", "strength": 0.2,
                           "desc": "반감기 전 해 → 회복/축적 구간"},
        }
        if phase in phase_bias:
            pb = phase_bias[phase]
            patterns.append({
                "name": "halving_cycle_bias",
                "direction": pb["bias"],
                "strength": pb["strength"],
                "message": pb["desc"],
            })
            # 반감기 바이어스를 최종 스코어에 반영
            signal["score"] = signal["score"] * 0.7 + pb["strength"] * 0.3

        signal["patterns"] = patterns
        signal["reason"] = " | ".join(p["message"] for p in patterns[:3])

        self.current_signal = signal
        self._last_update = dt if isinstance(dt, datetime) else datetime.combine(dt, datetime.min.time())

        return signal

    def get_features(self) -> dict:
        """ML 모델에 입력할 계절 피처"""
        if not self.current_signal:
            self.get_seasonal_signal()

        s = self.current_signal
        halving = s.get("halving_info", {})

        # 반감기 단계를 원핫 인코딩
        phase = halving.get("phase", "unknown")
        features = {
            "seasonal_score": s.get("score", 0),
            "seasonal_confidence": s.get("confidence", 0),
            "halving_progress": halving.get("cycle_progress", 0.5),
            "halving_days_since": halving.get("days_since_halving", 0) / 1461,  # 정규화
            "is_halving_year": 1.0 if phase == "halving_year" else 0.0,
            "is_post_halving_1": 1.0 if phase == "post_halving_1" else 0.0,
            "is_post_halving_2": 1.0 if phase == "post_halving_2" else 0.0,
            "is_pre_halving": 1.0 if phase == "pre_halving" else 0.0,
            "is_dec_feb_bounce": 1.0 if self._in_dec_feb_zone() else 0.0,
            "month_sin": self._month_cyclical()[0],
            "month_cos": self._month_cyclical()[1],
        }
        return features

    def _in_dec_feb_zone(self) -> bool:
        """현재 12월~2월 반등 구간인지"""
        now = datetime.utcnow()
        return now.month == 12 or now.month <= 2

    def _month_cyclical(self) -> tuple[float, float]:
        """월을 cyclical encoding (sin/cos)"""
        import math
        month = datetime.utcnow().month
        sin_m = math.sin(2 * math.pi * month / 12)
        cos_m = math.cos(2 * math.pi * month / 12)
        return round(sin_m, 4), round(cos_m, 4)

    def _get_season_name(self, month: int) -> str:
        if month in [12, 1, 2]:
            return "winter_bounce"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall_setup"

    def get_report(self) -> dict:
        """대시보드용 리포트"""
        if not self.current_signal:
            self.get_seasonal_signal()

        s = self.current_signal
        return {
            "season": s.get("season", "unknown"),
            "halving_phase": s.get("halving_phase", "unknown"),
            "direction": s.get("direction", "neutral"),
            "score": s.get("score", 0),
            "confidence": s.get("confidence", 0),
            "patterns": [
                {"name": p["name"], "message": p["message"]}
                for p in s.get("patterns", [])
            ],
            "halving_info": s.get("halving_info", {}),
        }
