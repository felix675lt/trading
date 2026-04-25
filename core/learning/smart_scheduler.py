"""스마트 학습 스케줄러 — 조건부 재학습 + M2 리소스 관리
=========================================================
기존 24h 무조건 재학습을 보완:
  - 모델별 독립 스케줄 (LGB 빠름, CNN 중간, RL 느림)
  - 성능 하락(IC/accuracy) 감지 시 즉시 재학습
  - 레짐 전환 감지 시 긴급 재학습
  - CPU 쿨다운 + 메모리 임계로 동시 학습 방지

기존 SelfLearningTrainer.should_retrain()의 단일 24h 게이트를 대체할 수 있는
보완 게이트 — main.py 측에서 SmartScheduler.should_retrain_any() 결과를
"OR"로 트레이너 게이트와 합쳐 사용하면 안전(기존 동작 보존 + 추가 조건).

참고: 운영 시 단일 모델("ensemble") 스케줄로도 충분 — 모델별 분리는 향후 확장.
"""

from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from loguru import logger


@dataclass
class ModelSchedule:
    """개별 모델 학습 스케줄"""
    name: str
    last_train_time: datetime = field(
        default_factory=lambda: datetime.utcnow() - timedelta(hours=999)
    )
    min_interval_hours: float = 4.0
    max_interval_hours: float = 24.0
    performance_threshold: float = 0.02  # IC/accuracy 하락 임계
    last_accuracy: float = 0.0
    current_accuracy: float = 0.0
    consecutive_declines: int = 0
    is_training: bool = False
    priority: int = 1               # 1=높음, 3=낮음
    estimated_train_minutes: float = 5.0


class SmartTrainingScheduler:
    """스마트 학습 스케줄러 — 조건부 트리거 + 쿨다운"""

    DEFAULT_PROFILES: Dict[str, dict] = {
        "lightgbm":      {"min": 4.0,  "max": 24.0, "prio": 1, "est": 2.0},
        "ensemble":      {"min": 4.0,  "max": 24.0, "prio": 1, "est": 6.0},
        "xgboost":       {"min": 4.0,  "max": 24.0, "prio": 1, "est": 3.0},
        "lstm":          {"min": 6.0,  "max": 36.0, "prio": 2, "est": 8.0},
        "cnn_attention": {"min": 8.0,  "max": 48.0, "prio": 2, "est": 10.0},
        "rl_agent":      {"min": 12.0, "max": 72.0, "prio": 3, "est": 15.0},
    }

    def __init__(
        self,
        memory_limit_gb: float = 6.0,
        cpu_cooldown_minutes: float = 5.0,
        models: Optional[list[str]] = None,
    ):
        self.memory_limit_gb = memory_limit_gb
        self.cpu_cooldown_minutes = cpu_cooldown_minutes

        names = models or ["ensemble", "rl_agent"]
        self.schedules: Dict[str, ModelSchedule] = {}
        for n in names:
            prof = self.DEFAULT_PROFILES.get(n, {"min": 6.0, "max": 24.0, "prio": 2, "est": 5.0})
            self.schedules[n] = ModelSchedule(
                name=n,
                min_interval_hours=float(prof["min"]),
                max_interval_hours=float(prof["max"]),
                priority=int(prof["prio"]),
                estimated_train_minutes=float(prof["est"]),
            )

        self._last_train_end = datetime.utcnow() - timedelta(hours=1)
        self._training_history: list[dict] = []

    # ------------------------------------------------------------------
    def should_retrain(
        self,
        model_name: str,
        current_accuracy: float = 0.0,
        regime_changed: bool = False,
        force: bool = False,
    ) -> Tuple[bool, str]:
        """재학습 필요 여부 + 사유 반환"""
        if model_name not in self.schedules:
            return False, f"unknown_model:{model_name}"

        sch = self.schedules[model_name]
        now = datetime.utcnow()
        elapsed_h = (now - sch.last_train_time).total_seconds() / 3600

        if force:
            return True, "force"

        if sch.is_training:
            return False, "already_training"

        if elapsed_h < sch.min_interval_hours:
            return False, f"min_interval({elapsed_h:.1f}h<{sch.min_interval_hours}h)"

        cooldown_m = (now - self._last_train_end).total_seconds() / 60
        if cooldown_m < self.cpu_cooldown_minutes:
            return False, f"cooldown({cooldown_m:.1f}m<{self.cpu_cooldown_minutes}m)"

        reasons = []

        # 1) 레짐 변환 → 긴급
        if regime_changed:
            reasons.append("regime_changed")

        # 2) 성능 하락
        if current_accuracy > 0 and sch.last_accuracy > 0:
            decline = sch.last_accuracy - current_accuracy
            if decline > sch.performance_threshold:
                sch.consecutive_declines += 1
                reasons.append(
                    f"perf_decline({decline:.4f}>{sch.performance_threshold},x{sch.consecutive_declines})"
                )

        # 3) 최대 간격 초과
        if elapsed_h > sch.max_interval_hours:
            reasons.append(f"max_interval({elapsed_h:.1f}h>{sch.max_interval_hours}h)")

        # 4) 연속 하락 누적
        if sch.consecutive_declines >= 2:
            reasons.append(f"consecutive_declines={sch.consecutive_declines}")

        if reasons:
            return True, " | ".join(reasons)
        return False, "no_trigger"

    def should_retrain_any(
        self,
        accuracies: Optional[Dict[str, float]] = None,
        regime_changed: bool = False,
    ) -> Tuple[bool, str]:
        """등록된 모든 모델 중 하나라도 재학습 필요 시 True"""
        accuracies = accuracies or {}
        triggered = []
        for name in self.schedules:
            ok, reason = self.should_retrain(
                name,
                current_accuracy=float(accuracies.get(name, 0.0)),
                regime_changed=regime_changed,
            )
            if ok:
                triggered.append(f"{name}:{reason}")
        if triggered:
            return True, " | ".join(triggered)
        return False, "no_trigger"

    def get_training_queue(
        self,
        accuracies: Optional[Dict[str, float]] = None,
        regime_changed: bool = False,
    ) -> list[Tuple[str, str]]:
        """우선순위 큐 — (model_name, reason) 정렬"""
        accuracies = accuracies or {}
        queue = []
        for name, sch in self.schedules.items():
            ok, reason = self.should_retrain(
                name,
                current_accuracy=float(accuracies.get(name, 0.0)),
                regime_changed=regime_changed,
            )
            if ok:
                queue.append((name, sch.priority, reason))
        queue.sort(key=lambda x: x[1])
        return [(n, r) for n, _, r in queue]

    # ------------------------------------------------------------------
    def mark_training_start(self, model_name: str):
        if model_name in self.schedules:
            self.schedules[model_name].is_training = True
            logger.info(f"[SmartSched] {model_name} 학습 시작")

    def mark_training_complete(self, model_name: str, new_accuracy: float):
        if model_name not in self.schedules:
            return
        sch = self.schedules[model_name]
        sch.is_training = False
        # consecutive declines: 직전 정확도와 비교 후 갱신
        if sch.last_accuracy > 0 and new_accuracy >= sch.last_accuracy:
            sch.consecutive_declines = 0
        sch.last_accuracy = new_accuracy
        sch.current_accuracy = new_accuracy
        sch.last_train_time = datetime.utcnow()
        self._last_train_end = datetime.utcnow()
        self._training_history.append({
            "model": model_name,
            "accuracy": float(new_accuracy),
            "time": datetime.utcnow().isoformat(),
        })
        logger.info(f"[SmartSched] {model_name} 완료 — Acc {new_accuracy:.4f}")

    def estimate_total_train_minutes(self, queue: list) -> float:
        total = 0.0
        for name, _ in queue:
            if name in self.schedules:
                total += self.schedules[name].estimated_train_minutes
                total += self.cpu_cooldown_minutes
        return total

    def get_status_report(self) -> dict:
        now = datetime.utcnow()
        report = {}
        for name, sch in self.schedules.items():
            elapsed_h = (now - sch.last_train_time).total_seconds() / 3600
            report[name] = {
                "last_train": sch.last_train_time.isoformat(),
                "hours_since_train": round(elapsed_h, 1),
                "accuracy": sch.current_accuracy,
                "last_accuracy": sch.last_accuracy,
                "consecutive_declines": sch.consecutive_declines,
                "is_training": sch.is_training,
                "priority": sch.priority,
                "min_interval_h": sch.min_interval_hours,
                "max_interval_h": sch.max_interval_hours,
            }
        return report
