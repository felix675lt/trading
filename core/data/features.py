"""피처 엔지니어링 - 기술적 지표 및 시장 피처 생성"""

import numpy as np
import pandas as pd
import ta

from core.data.labeling import triple_barrier_labels


class FeatureEngineer:
    """OHLCV 데이터로부터 ML/RL에 사용할 피처를 생성"""

    def __init__(self, feature_list: list[str] | None = None,
                 use_triple_barrier: bool = True,
                 tb_pt_mult: float = 2.0,
                 tb_sl_mult: float = 1.0,
                 tb_max_hold: int = 24):
        self.feature_list = feature_list or [
            "rsi", "macd", "bbands", "atr", "volume_profile",
            "ema", "stoch", "adx", "obv", "vwap",
        ]
        self.external_features: dict = {}  # 외부 요인 피처 저장
        # Triple-Barrier 라벨링 (Lopez de Prado) — 전통 forward-return 대비 정보량↑
        self.use_triple_barrier = use_triple_barrier
        self.tb_pt_mult = tb_pt_mult
        self.tb_sl_mult = tb_sl_mult
        self.tb_max_hold = tb_max_hold
        # Cross-Asset BTC Reference (2026-04-20 추가 — 통찰 #2)
        # BTC 5m 캔들이 ETH/SOL/DOGE보다 30s~2min 선행하는 구조 활용
        # set_btc_reference()로 주입 → generate()가 호출되면 자동 피처 추가
        self.btc_reference: pd.DataFrame | None = None

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

        # Cross-Asset BTC 선행 피처 (있으면) — 통찰 #2
        if self.btc_reference is not None:
            result = self._add_btc_features(result)

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

    def _add_labels(self, df: pd.DataFrame, forward_periods: int = 12, threshold: float = 0.005) -> pd.DataFrame:
        """레이블 생성 — Triple-Barrier (Lopez de Prado) 또는 전통 forward-return.

        Triple-Barrier (use_triple_barrier=True, 기본):
          - TP 배리어 = ATR × pt_mult
          - SL 배리어 = ATR × sl_mult
          - 시간 배리어 = max_hold bars
          - 먼저 닿은 배리어로 라벨링 → 경로 기반, 노이즈 강건

        Forward-return fallback (use_triple_barrier=False):
          - 단순 N캔들 뒤 수익률 ±threshold
        """
        # 1) Triple-Barrier 우선
        if self.use_triple_barrier:
            df = triple_barrier_labels(
                df,
                pt_mult=self.tb_pt_mult,
                sl_mult=self.tb_sl_mult,
                max_hold=self.tb_max_hold,
                atr_col="atr_14",
                min_ret=threshold / 2,
            )
            # 기존 학습 코드 호환 — label 컬럼을 tb_label로 덮어씀
            df["label"] = df["tb_label"]
            df["future_return"] = df["tb_ret"]
            return df

        # 2) Forward-return fallback (기존 방식)
        future_return = df["close"].pct_change(forward_periods).shift(-forward_periods)
        df["label"] = 1  # 횡보
        df.loc[future_return > threshold, "label"] = 2   # 상승
        df.loc[future_return < -threshold, "label"] = 0  # 하락
        df["future_return"] = future_return
        return df

    def set_external_features(self, features: dict):
        """외부 데이터 매니저에서 받은 피처 설정"""
        self.external_features = features

    def set_btc_reference(self, btc_df: pd.DataFrame | None):
        """BTC 기준 캔들 주입 (cross-asset lead-lag 피처 생성용)

        통찰 #2: BTC 5m 봉이 alt들보다 30s~2min 선행한다는 학계 보고(Bouri 2018, Katsiampa 2019).
        ETH/SOL/DOGE 모델에 BTC 선행 피처를 주입하면 R² +3~5%p 개선 기대.

        Args:
            btc_df: BTC OHLCV + 피처가 포함된 DataFrame. None이면 주입 해제.
                    인덱스는 alt 심볼 DF와 동일 타임존/주기여야 함.
        """
        self.btc_reference = btc_df

    def _add_btc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """BTC 선행 피처를 alt 심볼 DF에 조인 — ffill로 시간 정렬

        주입 피처:
          - btc_returns_1:  BTC 1캔들 수익률 (약 5분)
          - btc_returns_5:  BTC 5캔들 수익률 (약 25분)
          - btc_returns_20: BTC 20캔들 수익률 (약 1.7시간)
          - btc_rsi_14:     BTC 14봉 RSI (상대강도)
          - btc_volatility: BTC 20봉 변동성 (리스크 환경 지시)
        """
        btc = self.btc_reference
        if btc is None or len(btc) == 0:
            return df

        # BTC에서 필요한 컬럼만 선별 (있는 것만)
        want = ["returns_1", "returns_5", "returns_20", "rsi_14", "volatility_20"]
        btc_cols = [c for c in want if c in btc.columns]
        if not btc_cols:
            return df

        btc_subset = btc[btc_cols].copy()
        # 컬럼명 접두사 변경 (btc_*)
        btc_subset.columns = [f"btc_{c.replace('_20','').replace('volatility','volatility_20')}"
                              if c == "volatility_20" else f"btc_{c}"
                              for c in btc_subset.columns]

        # 시간축 정렬: df.index 기준으로 BTC 최근 값을 ffill — 미래 정보 유입 차단
        # 정확한 시간 정렬을 위해 reindex 후 ffill
        btc_aligned = btc_subset.reindex(df.index, method="ffill")
        result = pd.concat([df, btc_aligned], axis=1)
        return result

    def _add_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """외부 요인 피처를 DataFrame에 추가 (모든 행에 동일한 값 적용)"""
        ext_cols = {}
        for key, value in self.external_features.items():
            if isinstance(value, (int, float)):
                ext_cols[f"ext_{key}"] = float(value)
        if ext_cols:
            import pandas as pd
            ext_df = pd.DataFrame(ext_cols, index=df.index)
            df = pd.concat([df, ext_df], axis=1)
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """ML 모델에 입력할 피처 컬럼만 반환 (외부 피처 포함)"""
        exclude = {"open", "high", "low", "close", "volume", "label", "future_return",
                   "tb_label", "tb_ret", "tb_hit", "tb_t1"}
        # tb_hit은 string이라 제외 — 혹시 남아있을 수 있으니 type 체크도
        cols = [c for c in df.columns if c not in exclude]
        # numeric만 (object dtype 제외)
        return [c for c in cols if df[c].dtype != object]

    def get_base_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """RL 모델용 기본 피처만 반환 (외부 피처 제외 - 차원 고정)"""
        exclude = {"open", "high", "low", "close", "volume", "label", "future_return",
                   "tb_label", "tb_ret", "tb_hit", "tb_t1"}
        cols = [c for c in df.columns if c not in exclude and not c.startswith("ext_")]
        return [c for c in cols if df[c].dtype != object]
