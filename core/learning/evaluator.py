"""모델 성능 평가 및 버전 관리"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger


class ModelEvaluator:
    """모델 성능 평가 및 자동 교체 판단"""

    def __init__(self, model_dir: str = "models_saved"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.performance_log: list[dict] = []
        self._load_log()

    def _load_log(self):
        log_path = self.model_dir / "performance_log.json"
        if log_path.exists():
            with open(log_path) as f:
                self.performance_log = json.load(f)

    def _save_log(self):
        log_path = self.model_dir / "performance_log.json"
        with open(log_path, "w") as f:
            json.dump(self.performance_log[-1000:], f, indent=2)

    def record(self, model_name: str, metrics: dict):
        entry = {
            "timestamp": str(datetime.utcnow()),
            "model": model_name,
            **metrics,
        }
        self.performance_log.append(entry)
        self._save_log()

    def should_replace_model(self, current_metrics: dict, new_metrics: dict) -> bool:
        """신규 모델이 기존 모델보다 나은지 판단"""
        # 여러 지표 가중 비교
        weights = {"sharpe_ratio": 0.3, "win_rate": 0.3, "total_return": 0.2, "max_drawdown": -0.2}
        current_score = 0
        new_score = 0

        for metric, weight in weights.items():
            c_val = current_metrics.get(metric, 0)
            n_val = new_metrics.get(metric, 0)
            current_score += c_val * weight
            new_score += n_val * weight

        improvement = (new_score - current_score) / (abs(current_score) + 1e-8)
        should_replace = improvement > 0.05  # 5% 이상 개선 시 교체

        logger.info(f"모델 비교 - 현재: {current_score:.4f}, 신규: {new_score:.4f}, 개선율: {improvement:.2%}")
        return should_replace

    def get_recent_performance(self, model_name: str, n: int = 10) -> dict:
        recent = [p for p in self.performance_log if p["model"] == model_name][-n:]
        if not recent:
            return {}
        return {
            "avg_sharpe": np.mean([p.get("sharpe_ratio", 0) for p in recent]),
            "avg_win_rate": np.mean([p.get("win_rate", 0) for p in recent]),
            "trend": "improving" if len(recent) > 1 and recent[-1].get("sharpe_ratio", 0) > recent[0].get("sharpe_ratio", 0) else "declining",
        }
