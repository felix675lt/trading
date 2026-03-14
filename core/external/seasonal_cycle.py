"""계절 사이클 분석 엔진 - BTC 10년 역사적 패턴 기반 시즌 시그널

사용자 검증 데이터 (2017-2024):
- Dec Low → Feb High 반등: 100% 발생 (7/7), 평균 +44.45%, 평균 64일
- 유일한 예외: 2017년 (블로오프탑 후 크래시)
- Sep-Nov 90일: 57% 상승 / 43% 하락 (반감기 사이클과 상관)

반감기 4년 사이클 (정확한 기준):
- 연도 기준이 아닌 "반감기 이후 경과일 수"로 계산
- halving_year: 반감기로부터 0~365일 (보상 감소 → 강세 준비)
- post_halving_1: 365~730일 (광산수익 감소 → 역사적 최대 강세, 평균 +210%)
- post_halving_2: 730~1095일 (강세 종료, 베어마켓, 평균 -30%)
- pre_halving: 1095~1460일 (회복/축적 구간)
"""

from datetime import datetime, date
import math
from loguru import logger


class SeasonalCycleAnalyzer:
    """BTC 10년 계절 패턴 기반 시그널 생성"""

    # 반감기 날짜 (정확함)
    HALVING_DATES = [
        date(2012, 11, 28),
        date(2016, 7, 9),
        date(2020, 5, 11),
        date(2024, 4, 19),  # 최신
        date(2028, 4, 1),   # 예상
    ]

    # 사용자 검증 데이터: 반감기 사이클별 실제 패턴
    VALIDATED_PATTERNS = {
        "halving_year": {
            "sep_nov_up_ratio": 0.5,      # 50% 상승
            "sep_nov_avg": 25.0,          # 평균 25%
            "dec_feb_avg": 60.8,          # Dec-Feb 평균 +60.8%
            "sep_apr_avg": 193.0,         # Sep-Apr 전체
            "confidence": 0.7,
            "note": "반감기 연도: 공급 감소 시작, 강세 준비 구간",
        },
        "post_halving_1": {
            "sep_nov_up_ratio": 0.667,    # 67% 상승 (2020, 2023 상승)
            "sep_nov_avg": 62.2,          # 평균 62.2%
            "dec_feb_avg": 98.5,          # Dec-Feb 평균 +98.5% (2020:+215.8%, 2023:+20.45%)
            "sep_apr_avg": 210.0,         # 평균 210% (역사적 최대)
            "confidence": 0.9,            # 가장 강한 패턴
            "note": "반감기 다음 해: 역사적 최대 강세 구간 (평균 +210%)",
            # 반감기 후 피크 타이밍 (검증된 패턴):
            # 2017: 530일 후 피크 (12월), 2021: 550일 후 피크 (11월), 2025: 540일 후 피크 (10월)
            "peak_days_avg": 540,         # 반감기 후 평균 540일에 피크
            "peak_days_range": (500, 580), # 500~580일 사이에 피크 형성
        },
        "post_halving_2": {
            "sep_nov_up_ratio": 0.333,    # 33% 상승 (2018, 2022 중 1개만 상승)
            "sep_nov_avg": -19.76,        # 평균 -19.76% (2018:-44.6%, 2022:-14.9%)
            "dec_feb_avg": 40.58,         # Dec-Feb 평균 +40.58% (반등은 항상 있음)
            "sep_apr_avg": -1.6,          # Sep-Apr 전체 약간 하락
            "confidence": 0.5,            # 약한 신호
            "note": "반감기 2년 후: 베어마켓, 하지만 Dec-Feb 반등은 반드시 발생 (+40%)",
        },
        "pre_halving": {
            "sep_nov_up_ratio": 0.667,    # 67% 상승 (2019 하락, 2023 상승)
            "sep_nov_avg": 14.64,         # 평균 14.64%
            "dec_feb_avg": 38.89,         # Dec-Feb 평균 +38.89%
            "sep_apr_avg": 68.32,         # 평균 68.32%
            "confidence": 0.6,
            "note": "반감기 전 해: 회복/축적 구간",
        },
    }

    # 사용자 데이터 (연도별 매핑)
    HISTORICAL_DATA = [
        # 2016년 반감기 (7월 9일) 기준
        {
            "year": 2017,
            "halving_days": 247,  # 2016.7.9 ~ 2017.9.1 약 450일, 그래서 post_halving_1 (365~730)
            "phase": "post_halving_1",
            "sep_nov": 113.25,
            "dec_low_day": 17,
            "dec_feb_return": -69.57,  # 2017은 ATH 고점 후 크래시 (예외)
            "is_exception": True,
        },
        # 2016년 반감기 (7월 9일) 기준
        {
            "year": 2018,
            "halving_days": 730 + 53,  # post_halving_2 시작 (730~1095)
            "phase": "post_halving_2",
            "sep_nov": -44.59,
            "dec_low_day": 15,
            "dec_feb_return": 28.92,
        },
        {
            "year": 2019,
            "halving_days": 1095 + 54,  # pre_halving (1095~1460)
            "phase": "pre_halving",
            "sep_nov": -19.95,
            "dec_low_day": 18,
            "dec_feb_return": 57.33,
        },
        # 2020년 반감기 (5월 11일) 기준
        {
            "year": 2020,
            "halving_days": 113,  # halving_year (0~365)
            "phase": "halving_year",
            "sep_nov": 66.61,
            "dec_low_day": 11,
            "dec_feb_return": 215.78,
        },
        {
            "year": 2021,
            "halving_days": 478,  # post_halving_1 (365~730)
            "phase": "post_halving_1",
            "sep_nov": 19.83,
            "dec_low_day": 4,
            "dec_feb_return": 6.0,  # 약한 반등
        },
        {
            "year": 2022,
            "halving_days": 843,  # post_halving_2 (730~1095)
            "phase": "post_halving_2",
            "sep_nov": -14.93,
            "dec_low_day": 19,
            "dec_feb_return": 52.23,
        },
        # 2024년 반감기 (4월 19일) 기준
        {
            "year": 2023,
            "halving_days": 1341,  # pre_halving (1095~1460, 실제로는 2024.4.19 306일 전)
            "phase": "pre_halving",
            "sep_nov": 49.23,
            "dec_low_day": 11,
            "dec_feb_return": 20.45,
        },
    ]

    def __init__(self):
        self.current_signal: dict = {}
        self._last_update: datetime | None = None

    def get_halving_phase(self, dt: date | None = None) -> dict:
        """현재 반감기 사이클 위치 계산 (정확함)"""
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
        cycle_length = (next_halving - last_halving).days if next_halving else 1461  # 4년

        # 사이클 내 위치 (0~1)
        progress = days_since / cycle_length

        # 반감기 사이클 단계 판별 (정확한 기준)
        if days_since < 365:
            phase = "halving_year"
        elif days_since < 730:
            phase = "post_halving_1"
        elif days_since < 1095:
            phase = "post_halving_2"
        else:
            phase = "pre_halving"

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
        pattern = self.VALIDATED_PATTERNS.get(phase, {})

        signal = {
            "halving_phase": phase,
            "halving_info": halving,
            "season": self._get_season_name(month),
            "pattern_confidence": pattern.get("confidence", 0),
            "confidence": 0.0,
            "direction": "neutral",
            "score": 0.0,
            "reason": "",
            "patterns": [],
        }

        patterns = []

        # === 패턴 1: December Low → February Bounce (100% 발생 패턴) ===
        if month == 12 and day >= 1:
            # 12월: 저점 매수 구간 진입
            dec_feb_return = pattern.get("dec_feb_avg", 40)

            if day < 20:
                # 12월 초~중순: 저점 형성 중
                confidence = pattern.get("confidence", 0.7) * 0.9
                patterns.append({
                    "name": "december_low_forming",
                    "confidence": confidence,
                    "expected_return": dec_feb_return,
                    "direction": "bullish_setup",
                    "message": f"12월 저점 형성 구간 ({phase}단계, 역사적 평균 +{dec_feb_return:.0f}%)",
                })
                signal["direction"] = "bullish"
                signal["score"] = 0.3
                signal["confidence"] = confidence

            else:
                # 12월 하순: 저점 통과 확정
                confidence = pattern.get("confidence", 0.7)
                patterns.append({
                    "name": "december_low_confirmed",
                    "confidence": confidence,
                    "expected_return": dec_feb_return,
                    "direction": "bullish",
                    "message": f"12월 저점 확정 → 2월까지 +{dec_feb_return:.0f}% 반등 기대 ({phase}단계)",
                })
                signal["direction"] = "bullish"
                signal["score"] = 0.5
                signal["confidence"] = confidence

        elif month == 1:
            # 1월: 반등 진행 중
            dec_feb_return = pattern.get("dec_feb_avg", 40)
            confidence = pattern.get("confidence", 0.7)

            patterns.append({
                "name": "january_bounce_active",
                "confidence": confidence,
                "expected_return": dec_feb_return,
                "direction": "bullish",
                "message": f"1월 반등 구간 ({phase}단계, 기대수익 +{dec_feb_return:.0f}%)",
            })
            signal["direction"] = "bullish"
            signal["score"] = 0.4
            signal["confidence"] = confidence

        elif month == 2 and day <= 20:
            # 2월 초~중순: 반등 고점 접근
            confidence = pattern.get("confidence", 0.7) * 0.7
            patterns.append({
                "name": "february_peak_approaching",
                "confidence": confidence,
                "direction": "cautious_bullish",
                "message": f"2월 반등 고점 접근 → 익절 준비 ({phase}단계)",
            })
            signal["direction"] = "bullish"
            signal["score"] = 0.2
            signal["confidence"] = confidence

        elif month == 2 and day > 20:
            # 2월 하순: 반등 고점 도달
            patterns.append({
                "name": "february_peak_zone",
                "confidence": 0.4,
                "direction": "neutral",
                "message": f"2월 반등 고점 구간 → 신규 롱 자제 ({phase}단계)",
            })
            signal["direction"] = "neutral"
            signal["score"] = 0.0
            signal["confidence"] = 0.4

        # === 패턴 2: Sep-Nov 90일 구간 (단계별 다름) ===
        elif month >= 9 and month <= 11:
            sep_nov_up_ratio = pattern.get("sep_nov_up_ratio", 0.5)
            sep_nov_avg = pattern.get("sep_nov_avg", 0)
            confidence = pattern.get("confidence", 0.5)

            if sep_nov_up_ratio > 0.55:
                signal["direction"] = "bullish"
                signal["score"] = sep_nov_up_ratio * 0.3
                signal["confidence"] = confidence
                patterns.append({
                    "name": f"sep_nov_{phase}_bullish",
                    "confidence": confidence,
                    "avg_pct": sep_nov_avg,
                    "direction": "bullish",
                    "message": f"Sep-Nov {phase} 단계 역사적 상승 확률 {sep_nov_up_ratio:.0%} (평균 {sep_nov_avg:+.0f}%)",
                })
            else:
                signal["direction"] = "bearish"
                signal["score"] = -((1 - sep_nov_up_ratio) * 0.3)
                signal["confidence"] = confidence
                patterns.append({
                    "name": f"sep_nov_{phase}_bearish",
                    "confidence": confidence,
                    "avg_pct": sep_nov_avg,
                    "direction": "bearish",
                    "message": f"Sep-Nov {phase} 단계 역사적 하락 확률 {1-sep_nov_up_ratio:.0%} (평균 {sep_nov_avg:+.0f}%)",
                })

        # === 패턴 3: 단계별 특징 (post_halving_1은 sub-phase 적용) ===
        days_since = halving.get("days_since_halving", 0)

        phase_insights = {
            "halving_year": {
                "bias": "bullish_setup",
                "strength": 0.2,
                "desc": "반감기 연도: 공급 감소 시작 → 강세 준비, Sep-Nov 약 50/50",
            },
            "post_halving_2": {
                "bias": "bearish",
                "strength": -0.4,
                "desc": "반감기 2~3년 후: 베어마켓 구간 (Sep-Nov 67% 하락), 하지만 Dec-Feb 반등은 반드시 +41%",
            },
            "pre_halving": {
                "bias": "moderate_bullish",
                "strength": 0.3,
                "desc": "반감기 전 해: 회복/축적, Sep-Nov 67% 상승, Sep-Apr +68%",
            },
        }

        # post_halving_1 sub-phase (검증된 피크 타이밍: 500~580일)
        if phase == "post_halving_1":
            peak_range = pattern.get("peak_days_range", (500, 580))
            if days_since < peak_range[0]:
                # 상승 구간 (365~500일): 강세 진행 중
                phase_insights["post_halving_1"] = {
                    "bias": "strong_bullish",
                    "strength": 0.8,
                    "desc": f"post_halving_1 상승 구간 (피크까지 ~{peak_range[0]-days_since}일): 역사적 최대 강세",
                }
            elif days_since <= peak_range[1]:
                # 피크 구간 (500~580일): 고점 경고
                phase_insights["post_halving_1"] = {
                    "bias": "peak_warning",
                    "strength": 0.1,
                    "desc": f"post_halving_1 피크 구간 (반감기 후 {days_since}일): 고점 형성 중! 신규 롱 자제, 익절 준비",
                }
            else:
                # 하락 구간 (580~730일): 피크 후 조정
                decline_days = days_since - peak_range[1]
                phase_insights["post_halving_1"] = {
                    "bias": "post_peak_bearish",
                    "strength": -0.5,
                    "desc": f"post_halving_1 피크 후 조정 ({decline_days}일 경과): 고점 대비 하락 중, 반등 매도 or 관망",
                }
        else:
            if "post_halving_1" not in phase_insights:
                phase_insights["post_halving_1"] = {
                    "bias": "strong_bullish",
                    "strength": 0.8,
                    "desc": "반감기 다음 해: 역사적 최대 강세 구간",
                }

        if phase in phase_insights:
            insight = phase_insights[phase]
            patterns.append({
                "name": f"halving_phase_{phase}",
                "insight": insight["desc"],
                "bias": insight["bias"],
                "strength": insight["strength"],
            })
            # 최종 스코어에 단계 강도 반영
            if signal["score"] == 0:
                signal["score"] = insight["strength"]
            else:
                signal["score"] = signal["score"] * 0.7 + insight["strength"] * 0.3

        signal["patterns"] = patterns
        signal["reason"] = " | ".join(p["message"] for p in patterns[:2] if "message" in p)

        self.current_signal = signal
        self._last_update = dt if isinstance(dt, datetime) else datetime.combine(dt, datetime.min.time())

        return signal

    def get_features(self) -> dict:
        """ML 모델에 입력할 계절 피처"""
        if not self.current_signal:
            self.get_seasonal_signal()

        s = self.current_signal
        halving = s.get("halving_info", {})
        phase = halving.get("phase", "unknown")

        features = {
            "seasonal_score": s.get("score", 0),
            "seasonal_confidence": s.get("confidence", 0),
            "halving_progress": halving.get("cycle_progress", 0.5),
            "halving_days_since": halving.get("days_since_halving", 0) / 1461,
            "is_halving_year": 1.0 if phase == "halving_year" else 0.0,
            "is_post_halving_1": 1.0 if phase == "post_halving_1" else 0.0,
            "is_post_halving_2": 1.0 if phase == "post_halving_2" else 0.0,
            "is_pre_halving": 1.0 if phase == "pre_halving" else 0.0,
            "is_dec_feb_zone": 1.0 if datetime.utcnow().month in [12, 1, 2] else 0.0,
            "month_sin": self._month_cyclical()[0],
            "month_cos": self._month_cyclical()[1],
        }
        return features

    def _month_cyclical(self) -> tuple[float, float]:
        """월을 cyclical encoding (sin/cos)"""
        month = datetime.utcnow().month
        sin_m = math.sin(2 * math.pi * month / 12)
        cos_m = math.cos(2 * math.pi * month / 12)
        return round(sin_m, 4), round(cos_m, 4)

    def _get_season_name(self, month: int) -> str:
        if month in [12, 1, 2]:
            return "dec_feb_bounce"
        elif month in [3, 4, 5]:
            return "spring_transition"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall_setup"

    def get_report(self) -> dict:
        """대시보드용 리포트"""
        if not self.current_signal:
            self.get_seasonal_signal()

        s = self.current_signal
        halving = s.get("halving_info", {})
        phase = halving.get("phase", "unknown")
        days_since = halving.get("days_since_halving", 0)

        # 다음 단계 예상
        if phase == "halving_year":
            next_phase = f"post_halving_1까지 {365 - days_since}일"
        elif phase == "post_halving_1":
            next_phase = f"post_halving_2까지 {730 - days_since}일"
        elif phase == "post_halving_2":
            next_phase = f"pre_halving까지 {1095 - days_since}일"
        else:
            next_phase = f"다음 halving까지 {1460 - days_since}일"

        return {
            "season": s.get("season", "unknown"),
            "halving_phase": phase,
            "days_since_last_halving": days_since,
            "next_phase_in": next_phase,
            "direction": s.get("direction", "neutral"),
            "score": s.get("score", 0),
            "confidence": s.get("confidence", 0),
            "patterns": [{"name": p.get("name"), "message": p.get("message", p.get("insight"))}
                        for p in s.get("patterns", [])],
            "halving_info": halving,
        }
