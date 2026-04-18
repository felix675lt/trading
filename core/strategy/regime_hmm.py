"""HMM-based Market Regime Classifier.

전통적 룰 기반 regime (변동성/추세/거래량) 대비 장점:
- 상태 간 확률적 전이 → 갑작스런 레짐 교체 완화 (persistence 보장)
- latent state가 관측 피처(return, volatility) 분포를 학습 → 숨겨진 regime 발견
- posterior probability → 부분적 regime mixing에 대응

구현: hmmlearn GaussianHMM (3상태: bear / neutral / bull 기대)
     feature = [log_return, realized_vol_20]
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from loguru import logger

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False


class HMMRegimeClassifier:
    """Gaussian HMM으로 시장 레짐 분류.

    상태 수 기본 3개 — 학습 후 각 상태의 평균 수익률로 bear/neutral/bull 라벨링.

    사용:
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(close_series)  # 1D 가격 시리즈
        regime = clf.predict(close_series[-200:])  # "bull" / "neutral" / "bear"
        prob = clf.predict_proba(close_series[-200:])  # {"bear": 0.1, ...}
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        random_state: int = 42,
        model_dir: str | Path = "models_saved",
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.state_labels: dict[int, str] = {}  # state_id → "bear"/"neutral"/"bull"
        self.fitted = False

        if not HAS_HMM:
            logger.warning("[HMM] hmmlearn 미설치 — pip install hmmlearn 필요")

    # ------------------------------------------------------------------
    # 피처 생성
    # ------------------------------------------------------------------

    def _build_features(self, prices: np.ndarray) -> np.ndarray:
        """[log_return, 20-bar realized vol] 형태로 변환."""
        prices = np.asarray(prices).astype(np.float64).ravel()
        log_ret = np.diff(np.log(np.maximum(prices, 1e-12)))
        # 20-bar rolling std of returns
        s = pd.Series(log_ret)
        vol = s.rolling(20, min_periods=5).std().fillna(s.std()).values
        X = np.column_stack([log_ret, vol])
        # 이상치 제거
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def fit(self, prices: np.ndarray | pd.Series) -> "HMMRegimeClassifier":
        if not HAS_HMM:
            self.fitted = False
            return self

        X = self._build_features(np.asarray(prices))
        if len(X) < 200:
            logger.warning(f"[HMM] 학습 데이터 부족 ({len(X)}) → fit 생략")
            self.fitted = False
            return self

        try:
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            self.model.fit(X)
            self._label_states()
            self.fitted = True
            logger.info(
                f"[HMM] 학습 완료 | states={self.n_states} samples={len(X)} | "
                f"labels={self.state_labels}"
            )
        except Exception as e:
            logger.error(f"[HMM] 학습 실패: {e}")
            self.fitted = False
        return self

    def _label_states(self):
        """각 상태의 emission mean을 평균 수익률 기준으로 bear/neutral/bull 매핑."""
        if self.model is None:
            return
        means = self.model.means_[:, 0]  # log_return mean (첫 번째 feature)
        order = np.argsort(means)  # 낮은 → 높은
        # n_states=3이면 bear/neutral/bull, 아니면 숫자 붙이기
        if self.n_states == 3:
            labels = ["bear", "neutral", "bull"]
        elif self.n_states == 2:
            labels = ["bear", "bull"]
        elif self.n_states == 4:
            labels = ["crash", "bear", "bull", "euphoria"]
        else:
            labels = [f"regime_{i}" for i in range(self.n_states)]
        self.state_labels = {int(order[i]): labels[i] for i in range(self.n_states)}

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------

    def predict(self, prices: np.ndarray | pd.Series) -> str:
        """최신 시점의 regime 라벨 반환."""
        if not self.fitted or self.model is None:
            return "normal"
        X = self._build_features(np.asarray(prices))
        if len(X) < 5:
            return "normal"
        try:
            states = self.model.predict(X)
            latest = int(states[-1])
            return self.state_labels.get(latest, "normal")
        except Exception as e:
            logger.debug(f"[HMM] predict 실패: {e}")
            return "normal"

    def predict_proba(self, prices: np.ndarray | pd.Series) -> dict[str, float]:
        """최신 시점의 각 regime 확률."""
        if not self.fitted or self.model is None:
            return {}
        X = self._build_features(np.asarray(prices))
        if len(X) < 5:
            return {}
        try:
            proba = self.model.predict_proba(X)[-1]
            return {
                self.state_labels.get(i, f"s{i}"): float(proba[i])
                for i in range(self.n_states)
            }
        except Exception as e:
            logger.debug(f"[HMM] predict_proba 실패: {e}")
            return {}

    # ------------------------------------------------------------------
    # 파라미터 매핑 — adaptive.py 규칙기반과 호환
    # ------------------------------------------------------------------

    def regime_to_adaptive(self, regime: str) -> str:
        """HMM 라벨 → adaptive.MarketRegimeDetector 라벨."""
        mapping = {
            "bull": "strong_uptrend",
            "bear": "strong_downtrend",
            "neutral": "ranging",
            "crash": "extreme_volatility",
            "euphoria": "high_volume_breakout",
        }
        return mapping.get(regime, "normal")

    # ------------------------------------------------------------------
    # 영속화
    # ------------------------------------------------------------------

    def save(self, name: str = "hmm_regime"):
        if not self.fitted:
            return
        path = self.model_dir / f"{name}.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "state_labels": self.state_labels,
                    "n_states": self.n_states,
                }, f)
            logger.info(f"[HMM] 저장: {path}")
        except Exception as e:
            logger.warning(f"[HMM] 저장 실패: {e}")

    def load(self, name: str = "hmm_regime") -> bool:
        path = self.model_dir / f"{name}.pkl"
        if not path.exists() or not HAS_HMM:
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.state_labels = data["state_labels"]
            self.n_states = data["n_states"]
            self.fitted = True
            logger.info(f"[HMM] 로드: {path} ({self.state_labels})")
            return True
        except Exception as e:
            logger.warning(f"[HMM] 로드 실패: {e}")
            return False
