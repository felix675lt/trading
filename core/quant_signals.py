"""
Quant-firm-inspired signal generators for crypto scalping.

Each method returns a score in [-1.0, +1.0] (bearish to bullish)
plus a confidence value, designed to run every ~30 seconds.
"""

import numpy as np
import pandas as pd
from loguru import logger

MAX_HISTORY = 200


class QuantSignals:
    def __init__(self):
        self._ob_history: list[float] = []
        self._vpin_buckets: list[float] = []
        self._basis_history: list[float] = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_mean(arr, default=0.0):
        if len(arr) == 0:
            return default
        return float(np.nanmean(arr))

    @staticmethod
    def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    # ------------------------------------------------------------------
    # 1. Orderbook Imbalance
    # ------------------------------------------------------------------

    def calc_orderbook_imbalance(self, orderbook: dict) -> dict:
        imbalance = orderbook.get("imbalance", 0.0)
        if np.isnan(imbalance):
            imbalance = 0.0

        self._ob_history.append(imbalance)
        if len(self._ob_history) > MAX_HISTORY:
            self._ob_history = self._ob_history[-MAX_HISTORY:]

        window = self._ob_history[-60:]

        imbalance_ma = self._safe_mean(window[-10:]) if len(window) >= 10 else imbalance
        prev_ma = self._safe_mean(window[-20:-10]) if len(window) >= 20 else imbalance_ma
        imbalance_roc = imbalance_ma - prev_ma

        # Score: blend current value with trend
        score = 0.6 * imbalance + 0.3 * imbalance_ma + 0.1 * imbalance_roc * 5.0
        score = self._clamp(score)

        # Extreme imbalance boost
        if abs(imbalance) > 0.5:
            score = self._clamp(score * 1.5)

        # Confidence based on consistency
        if len(window) >= 10:
            same_sign = sum(1 for v in window[-10:] if np.sign(v) == np.sign(imbalance))
            confidence = same_sign / 10.0
        else:
            confidence = 0.3

        return {
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "imbalance": round(imbalance, 6),
            "imbalance_ma": round(imbalance_ma, 6),
            "imbalance_roc": round(imbalance_roc, 6),
        }

    # ------------------------------------------------------------------
    # 2. VPIN
    # ------------------------------------------------------------------

    def calc_vpin(
        self, trades_volume: float, buy_volume: float, sell_volume: float
    ) -> dict:
        total = trades_volume if trades_volume > 0 else 1e-10
        bucket = abs(buy_volume - sell_volume) / total

        self._vpin_buckets.append(bucket)
        if len(self._vpin_buckets) > MAX_HISTORY:
            self._vpin_buckets = self._vpin_buckets[-MAX_HISTORY:]

        recent = self._vpin_buckets[-50:]
        vpin = self._safe_mean(recent)

        if vpin > 0.7:
            risk_level = "high"
            position_scale = 0.3
        elif vpin > 0.4:
            risk_level = "elevated"
            position_scale = 0.7
        else:
            risk_level = "normal"
            position_scale = 1.0

        return {
            "vpin": round(vpin, 4),
            "risk_level": risk_level,
            "position_scale": round(position_scale, 2),
        }

    # ------------------------------------------------------------------
    # 3. Basis Spread
    # ------------------------------------------------------------------

    def calc_basis_spread(self, spot_price: float, futures_price: float) -> dict:
        if spot_price <= 0:
            return {
                "score": 0.0,
                "confidence": 0.0,
                "basis_pct": 0.0,
                "basis_ma": 0.0,
                "signal_type": "neutral",
            }

        basis_pct = (futures_price - spot_price) / spot_price * 100.0

        self._basis_history.append(basis_pct)
        if len(self._basis_history) > MAX_HISTORY:
            self._basis_history = self._basis_history[-MAX_HISTORY:]

        window = self._basis_history[-60:]
        basis_ma = self._safe_mean(window[-20:]) if len(window) >= 20 else basis_pct

        # Rate of change
        if len(window) >= 10:
            prev_ma = self._safe_mean(window[-20:-10])
            basis_roc = basis_ma - prev_ma
        else:
            basis_roc = 0.0

        # Determine signal type
        abs_basis = abs(basis_pct)
        if abs_basis > 0.3:
            # Extreme basis -> contrarian reversal signal
            signal_type = "reversal"
            score = -np.sign(basis_pct) * min(abs_basis / 0.6, 1.0)
        else:
            # Normal basis -> momentum signal
            signal_type = "momentum"
            score = basis_pct / 0.3  # normalise to [-1, 1]
            score += basis_roc * 2.0  # add roc component

        score = self._clamp(score)

        # Confidence: higher when basis is consistent
        if len(window) >= 10:
            std = float(np.std(window[-10:]))
            confidence = max(0.2, 1.0 - std * 5.0)
        else:
            confidence = 0.3

        return {
            "score": round(score, 4),
            "confidence": round(self._clamp(confidence, 0.0, 1.0), 4),
            "basis_pct": round(basis_pct, 6),
            "basis_ma": round(basis_ma, 6),
            "signal_type": signal_type,
        }

    # ------------------------------------------------------------------
    # 4. Momentum Crash Protection
    # ------------------------------------------------------------------

    def calc_momentum_crash_risk(
        self,
        returns: list[float],
        current_volatility: float,
        avg_volatility: float,
    ) -> dict:
        if avg_volatility <= 0:
            avg_volatility = 1e-10

        vol_ratio = current_volatility / avg_volatility

        # Reversal detection: compare last 3 vs last 20 returns
        if len(returns) >= 20:
            recent_mean = float(np.mean(returns[-3:]))
            longer_mean = float(np.mean(returns[-20:]))
            # Reversal = sign flip + magnitude
            if longer_mean != 0:
                reversal_score = -recent_mean / (abs(longer_mean) + 1e-10)
                reversal_score = self._clamp(reversal_score)
            else:
                reversal_score = 0.0
            reversal_detected = (
                np.sign(recent_mean) != np.sign(longer_mean)
                and abs(recent_mean) > abs(longer_mean) * 0.5
            )
        elif len(returns) >= 3:
            recent_mean = float(np.mean(returns[-3:]))
            reversal_score = 0.0
            reversal_detected = False
        else:
            recent_mean = 0.0
            reversal_score = 0.0
            reversal_detected = False

        # Crash probability combines vol and reversal
        crash_prob = 0.0
        if vol_ratio > 2.0:
            crash_prob += 0.4
        elif vol_ratio > 1.5:
            crash_prob += 0.2

        if reversal_detected:
            crash_prob += 0.3

        crash_prob += max(0.0, (vol_ratio - 1.0) * 0.15)
        crash_prob = min(crash_prob, 1.0)

        # Position scale: inverse of vol ratio, floored at 0.3
        position_scale = max(0.3, min(1.0, 1.0 / max(1.0, vol_ratio)))

        return {
            "crash_risk": round(crash_prob, 4),
            "position_scale": round(position_scale, 4),
            "vol_ratio": round(vol_ratio, 4),
            "reversal_detected": bool(reversal_detected),
        }

    # ------------------------------------------------------------------
    # 5. Formula Alpha Signals (WQ-101 inspired)
    # ------------------------------------------------------------------

    def calc_formula_alphas(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty or len(df) < 20:
            return {
                "score": 0.0,
                "confidence": 0.0,
                "alphas": {},
                "active_signals": 0,
            }

        close = df["close"].values.astype(float)
        opn = df["open"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)

        alphas: dict[str, float] = {}
        active = 0

        # Alpha#1: VWAP deviation
        try:
            cv = close[-20:] * volume[-20:]
            vwap = np.sum(cv) / (np.sum(volume[-20:]) + 1e-10)
            a1 = (close[-1] - vwap) / (vwap + 1e-10)
            a1 = self._clamp(a1 * 10.0)  # scale up small deviations
            alphas["vwap_dev"] = round(a1, 4)
            if abs(a1) > 0.1:
                active += 1
        except Exception:
            alphas["vwap_dev"] = 0.0

        # Alpha#2: OBV rate of change
        try:
            sign = np.sign(np.diff(close))
            obv = np.cumsum(np.concatenate([[0], sign * volume[1:]]))
            if len(obv) >= 6 and abs(obv[-6]) > 0:
                a2 = (obv[-1] - obv[-6]) / (abs(obv[-6]) + 1e-10)
                a2 = self._clamp(a2)
            else:
                a2 = 0.0
            alphas["obv_roc"] = round(a2, 4)
            if abs(a2) > 0.1:
                active += 1
        except Exception:
            alphas["obv_roc"] = 0.0

        # Alpha#3: Price-volume divergence
        try:
            n = min(len(close), 20)
            price_trend = (close[-1] - close[-n]) / (close[-n] + 1e-10)
            vol_trend = (
                np.mean(volume[-5:]) - np.mean(volume[-n:])
            ) / (np.mean(volume[-n:]) + 1e-10)
            # Bearish divergence: price up, volume down
            # Bullish divergence: price down, volume up
            if price_trend > 0 and vol_trend < 0:
                a3 = -min(abs(vol_trend), 1.0)  # bearish
            elif price_trend < 0 and vol_trend > 0:
                a3 = min(abs(vol_trend), 1.0)  # bullish
            else:
                a3 = 0.0
            alphas["pv_divergence"] = round(a3, 4)
            if abs(a3) > 0.1:
                active += 1
        except Exception:
            alphas["pv_divergence"] = 0.0

        # Alpha#4: Intrabar momentum
        try:
            hl_range = high[-10:] - low[-10:] + 1e-10
            intrabar = (close[-10:] - opn[-10:]) / hl_range
            a4 = float(np.mean(intrabar))
            a4 = self._clamp(a4)
            alphas["intrabar_mom"] = round(a4, 4)
            if abs(a4) > 0.1:
                active += 1
        except Exception:
            alphas["intrabar_mom"] = 0.0

        # Alpha#5: Volume-weighted RSI
        try:
            deltas = np.diff(close)
            n_rsi = min(14, len(deltas))
            if n_rsi >= 2:
                gains = np.where(deltas > 0, deltas, 0.0)[-n_rsi:]
                losses = np.where(deltas < 0, -deltas, 0.0)[-n_rsi:]
                vol_weights = volume[-n_rsi:] / (np.mean(volume[-n_rsi:]) + 1e-10)
                avg_gain = np.sum(gains * vol_weights) / n_rsi
                avg_loss = np.sum(losses * vol_weights) / n_rsi
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - 100.0 / (1.0 + rs)
                a5 = (rsi - 50.0) / 50.0  # map 0-100 -> -1 to +1
                a5 = self._clamp(a5)
            else:
                a5 = 0.0
            alphas["vw_rsi"] = round(a5, 4)
            if abs(a5) > 0.1:
                active += 1
        except Exception:
            alphas["vw_rsi"] = 0.0

        score = self._safe_mean(list(alphas.values()))
        score = self._clamp(score)
        confidence = active / 5.0

        return {
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "alphas": alphas,
            "active_signals": active,
        }

    # ------------------------------------------------------------------
    # 6. Enhanced Regime Detection
    # ------------------------------------------------------------------

    def calc_regime(self, df: pd.DataFrame) -> dict:
        default = {
            "regime": "quiet",
            "confidence": 0.0,
            "params": {"position_scale": 1.0, "tighten_sl": True},
            "adx": 0.0,
            "bb_width": 0.0,
            "vol_ratio": 1.0,
        }
        if df is None or df.empty or len(df) < 50:
            return default

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)

        # --- ADX calculation (14-period) ---
        try:
            n = 14
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1]),
                ),
            )
            plus_dm = np.where(
                (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                np.maximum(high[1:] - high[:-1], 0),
                0.0,
            )
            minus_dm = np.where(
                (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                np.maximum(low[:-1] - low[1:], 0),
                0.0,
            )

            def _ema(arr, period):
                out = np.empty_like(arr)
                out[0] = arr[0]
                alpha = 1.0 / period
                for i in range(1, len(arr)):
                    out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
                return out

            atr = _ema(tr, n)
            plus_di = 100.0 * _ema(plus_dm, n) / (atr + 1e-10)
            minus_di = 100.0 * _ema(minus_dm, n) / (atr + 1e-10)
            dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = float(_ema(dx, n)[-1])
            trend_dir = 1.0 if float(plus_di[-1]) > float(minus_di[-1]) else -1.0
        except Exception:
            adx = 0.0
            trend_dir = 0.0

        # --- Bollinger Band width ---
        try:
            sma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            bb_width = (2.0 * std20) / (sma20 + 1e-10)
        except Exception:
            bb_width = 0.0

        # --- Volume ratio ---
        try:
            vol_recent = np.mean(volume[-5:])
            vol_avg = np.mean(volume[-50:])
            vol_ratio = vol_recent / (vol_avg + 1e-10)
        except Exception:
            vol_ratio = 1.0

        # --- Classify regime ---
        if adx > 25 and bb_width < 0.06:
            regime = "strong_trend"
            confidence = min(1.0, adx / 50.0)
            params = {"position_scale": 1.2, "entry_bias": trend_dir}
        elif adx > 25 and bb_width >= 0.06 and vol_ratio > 1.5:
            regime = "volatile_breakout"
            confidence = min(1.0, vol_ratio / 3.0)
            params = {"position_scale": 0.8, "widen_sl": True}
        elif adx < 20 and bb_width < 0.03:
            regime = "quiet"
            confidence = 0.6
            params = {"position_scale": 1.0, "tighten_sl": True}
        elif adx < 20:
            regime = "ranging"
            confidence = max(0.4, 1.0 - adx / 20.0)
            params = {"position_scale": 0.5, "prefer_no_entry": True}
        else:
            regime = "weak_trend"
            confidence = 0.5
            params = {"position_scale": 0.7}

        return {
            "regime": regime,
            "confidence": round(confidence, 4),
            "params": params,
            "adx": round(adx, 4),
            "bb_width": round(bb_width, 6),
            "vol_ratio": round(vol_ratio, 4),
        }

    # ------------------------------------------------------------------
    # Combined signal
    # ------------------------------------------------------------------

    def get_all_signals(
        self,
        orderbook: dict,
        spot_price: float,
        futures_price: float,
        trades_volume: float,
        buy_volume: float,
        sell_volume: float,
        returns: list[float],
        current_vol: float,
        avg_vol: float,
        df: pd.DataFrame,
    ) -> dict:
        ob = self.calc_orderbook_imbalance(orderbook)
        vpin = self.calc_vpin(trades_volume, buy_volume, sell_volume)
        basis = self.calc_basis_spread(spot_price, futures_price)
        crash = self.calc_momentum_crash_risk(returns, current_vol, avg_vol)
        alpha = self.calc_formula_alphas(df)
        regime = self.calc_regime(df)

        # Weighted combined score (directional signals only)
        combined_score = (
            ob["score"] * 0.25
            + regime.get("params", {}).get("entry_bias", 0.0)
            * 0.20
            * regime["confidence"]
            + alpha["score"] * 0.20
            + basis["score"] * 0.15
        )
        # Risk-based scaling from VPIN and crash
        risk_scale = min(vpin["position_scale"], crash["position_scale"])
        combined_score *= risk_scale
        combined_score = self._clamp(combined_score)

        # Overall confidence
        avg_conf = np.mean(
            [ob["confidence"], basis["confidence"], alpha["confidence"], regime["confidence"]]
        )

        result = {
            "combined_score": round(combined_score, 4),
            "combined_confidence": round(float(avg_conf), 4),
            "risk_scale": round(risk_scale, 4),
            "orderbook": ob,
            "vpin": vpin,
            "basis": basis,
            "crash": crash,
            "alpha": alpha,
            "regime": regime,
        }

        logger.debug(
            f"QuantSignals | score={combined_score:.4f} conf={avg_conf:.3f} "
            f"risk_scale={risk_scale:.2f} regime={regime['regime']}"
        )

        return result
