"""피처 엔지니어링 - 기술적 지표 및 시장 피처 생성"""

import numpy as np
import pandas as pd
import ta


class FeatureEngineer:
    """OHLCV 데이터로부터 ML/RL에 사용할 피처를 생성"""

    def __init__(self, feature_list: list[str] | None = None):
        self.feature_list = feature_list or [
            "rsi", "macd", "bbands", "atr", "volume_profile",
            "ema", "stoch", "adx", "obv", "vwap",
        ]
        self.external_features: dict = {}  # 외부 요인 피처 저장

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """전체 피처 생성 파이프라인"""
        result = df.copy()

        if "rsi" in self.feature_list:
            result = self._add_rsi(result)
        if "macd" in self.feature_list:
            result = self._add_macd(result)
        if "bbands" in self.feature_list:
            result = self._add_bbands(result)
        if "atr" in self.feature_list:
            result = self._add_atr(result)
        if "ema" in self.feature_list:
            result = self._add_ema(result)
        if "stoch" in self.feature_list:
            result = self._add_stochastic(result)
        if "adx" in self.feature_list:
            result = self._add_adx(result)
        if "obv" in self.feature_list:
            result = self._add_obv(result)
        if "volume_profile" in self.feature_list:
            result = self._add_volume_profile(result)

        result = self._add_price_features(result)

        # 외부 요인 피처 추가 (있으면)
        if self.external_features:
            result = self._add_external_features(result)

        result = self._add_labels(result)

        return result.dropna()

    def _add_rsi(self, df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
        for period in (periods or [7, 14, 21]):
            df[f"rsi_{period}"] = ta.momentum.RSIIndicator(df["close"], window=period).rsi()
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
        return df

    def _add_bbands(self, df: pd.DataFrame) -> pd.DataFrame:
        bb = ta.volatility.BollingerBands(df["close"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        df["atr_14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
        df["atr_pct"] = df["atr_14"] / df["close"]
        return df

    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in [9, 21, 50, 200]:
            df[f"ema_{period}"] = ta.trend.EMAIndicator(df["close"], window=period).ema_indicator()
        df["ema_cross_short"] = (df["ema_9"] - df["ema_21"]) / df["close"]
        df["ema_cross_long"] = (df["ema_50"] - df["ema_200"]) / df["close"]
        return df

    def _add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        return df

    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx.adx()
        df["di_plus"] = adx.adx_pos()
        df["di_minus"] = adx.adx_neg()
        return df

    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        df["obv_ema"] = df["obv"].ewm(span=20).mean()
        return df

    def _add_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        df["vol_sma_20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma_20"].replace(0, np.nan)
        df["vol_std"] = df["volume"].rolling(20).std()
        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["returns_1"] = df["close"].pct_change(1)
        df["returns_5"] = df["close"].pct_change(5)
        df["returns_20"] = df["close"].pct_change(20)
        df["volatility_20"] = df["returns_1"].rolling(20).std()
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
        df["close_open_range"] = (df["close"] - df["open"]) / df["open"]

        # 상위/하위 그림자 비율
        body = abs(df["close"] - df["open"])
        upper_shadow = df["high"] - df[["close", "open"]].max(axis=1)
        lower_shadow = df[["close", "open"]].min(axis=1) - df["low"]
        total_range = df["high"] - df["low"]
        df["upper_shadow_pct"] = upper_shadow / total_range.replace(0, np.nan)
        df["lower_shadow_pct"] = lower_shadow / total_range.replace(0, np.nan)
        df["body_pct"] = body / total_range.replace(0, np.nan)

        return df

    def _add_labels(self, df: pd.DataFrame, forward_periods: int = 5, threshold: float = 0.002) -> pd.DataFrame:
        """미래 수익률 기반 레이블 생성 (ML 학습용)"""
        future_return = df["close"].pct_change(forward_periods).shift(-forward_periods)
        df["label"] = 1  # 횡보
        df.loc[future_return > threshold, "label"] = 2   # 상승
        df.loc[future_return < -threshold, "label"] = 0  # 하락
        df["future_return"] = future_return
        return df

    def set_external_features(self, features: dict):
        """외부 데이터 매니저에서 받은 피처 설정"""
        self.external_features = features

    def _add_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """외부 요인 피처를 DataFrame에 추가 (모든 행에 동일한 값 적용)"""
        for key, value in self.external_features.items():
            if isinstance(value, (int, float)):
                df[f"ext_{key}"] = float(value)
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """ML 모델에 입력할 피처 컬럼만 반환"""
        exclude = {"open", "high", "low", "close", "volume", "label", "future_return"}
        return [c for c in df.columns if c not in exclude]
