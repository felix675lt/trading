"""적응형 파라미터 최적화 - 시장 상태에 따라 파라미터 자동 조정"""

import hashlib
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
from loguru import logger


class MarketRegimeDetector:
    """시장 레짐 감지 (추세/횡보/고변동성)"""

    def detect(self, prices: np.ndarray, volumes: np.ndarray) -> str:
        if len(prices) < 50:
            return "normal"

        # 변동성 측정
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-20:])
        avg_volatility = np.std(returns[-100:]) if len(returns) >= 100 else volatility

        # 추세 강도 (선형 회귀 R²)
        x = np.arange(min(50, len(prices)))
        y = prices[-len(x):]
        correlation = np.corrcoef(x, y)[0, 1]
        trend_strength = abs(correlation)

        # 거래량 이상 감지
        vol_ratio = np.mean(volumes[-5:]) / (np.mean(volumes[-50:]) + 1e-8)

        # 레짐 판별
        if volatility > avg_volatility * 2:
            return "extreme_volatility"
        elif trend_strength > 0.7:
            if correlation > 0:
                return "strong_uptrend"
            else:
                return "strong_downtrend"
        elif trend_strength < 0.3:
            return "ranging"
        elif vol_ratio > 2.0:
            return "high_volume_breakout"
        else:
            return "normal"


class AdaptiveOptimizer:
    """시장 레짐에 따라 전략 파라미터를 자동 조정"""

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.current_regime = "normal"
        self.regime_params = {
            "strong_uptrend": {
                "signal_threshold": 0.1,
                "min_confidence": 0.5,
                # [2026-04-23] 1.2 → 0.5 — 상승 LONG 표본 n=3 소규모, fade 교정 초기단계라 축소
                "position_scale": 0.5,
                "stop_loss_mult": 1.5,
                "prefer_direction": "long",
            },
            "strong_downtrend": {
                "signal_threshold": 0.1,
                "min_confidence": 0.5,
                "position_scale": 1.2,
                "stop_loss_mult": 1.5,
                "prefer_direction": "short",
            },
            "ranging": {
                "signal_threshold": 0.25,
                "min_confidence": 0.65,
                "position_scale": 0.6,
                "stop_loss_mult": 0.8,
                "prefer_direction": "neutral",
            },
            "extreme_volatility": {
                "signal_threshold": 0.3,
                "min_confidence": 0.75,
                "position_scale": 0.3,
                "stop_loss_mult": 2.0,
                "prefer_direction": "neutral",
            },
            "high_volume_breakout": {
                "signal_threshold": 0.12,
                "min_confidence": 0.55,
                "position_scale": 1.0,
                "stop_loss_mult": 1.2,
                "prefer_direction": "neutral",
            },
            "normal": {
                "signal_threshold": 0.15,
                "min_confidence": 0.55,
                "position_scale": 1.0,
                "stop_loss_mult": 1.0,
                "prefer_direction": "neutral",
            },
        }

    def update(self, prices: np.ndarray, volumes: np.ndarray) -> dict:
        """시장 상태 업데이트 및 최적 파라미터 반환"""
        new_regime = self.regime_detector.detect(prices, volumes)

        if new_regime != self.current_regime:
            logger.info(f"시장 레짐 전환: {self.current_regime} → {new_regime}")
            self.current_regime = new_regime

        return self.get_params()

    def get_params(self) -> dict:
        params = self.regime_params.get(self.current_regime, self.regime_params["normal"])
        params["regime"] = self.current_regime
        return params


class StrategyOptimizer:
    """자동 전략 최적화 시스템 — Paper/Live 독립 추적

    거래 결과를 config 해시별로 기록하고,
    최적 파라미터 조합을 자동으로 탐색합니다.
    """

    def __init__(self):
        self.current_config: dict = {}
        self.config_performance: dict[str, dict] = defaultdict(
            lambda: {"trades": [], "total_pnl": 0.0, "wins": 0, "losses": 0}
        )
        # performance_history: config_hash → [trade_list] (호환용 alias)
        self.performance_history: dict[str, list] = defaultdict(list)
        # adjustments: optimize_daily에서 산출한 실제 파라미터 조정값
        self.adjustments: dict = {}
        # DSR(Deflated Sharpe Ratio) 기반 오버핏 차단 목록 (2026-04-24)
        # {config_hash: {"dsr": float, "reason": str, "blocked_at": iso}}
        self.blocked_configs: dict[str, dict] = {}
        # 마지막 DSR 검증 시각 (주기 제어용)
        self._last_dsr_validation: datetime | None = None
        logger.info("[StrategyOptimizer] 자동 전략 최적화 시스템 초기화")

    def _config_to_hash(self, config: dict) -> str:
        """설정 dict를 고유 해시로 변환"""
        serialized = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()[:12]

    def record_trade(self, config_hash: str, trade: dict):
        """거래 결과를 config 해시에 연결하여 기록"""
        perf = self.config_performance[config_hash]
        pnl = trade.get("pnl", 0)
        trade_record = {
            "pnl": pnl,
            "timestamp": str(trade.get("timestamp", datetime.utcnow())),
            "symbol": trade.get("symbol", ""),
            "hour": trade.get("hour", 0),
        }
        perf["trades"].append(trade_record)
        perf["total_pnl"] += pnl
        if pnl > 0:
            perf["wins"] += 1
        else:
            perf["losses"] += 1

        # performance_history도 동기화 (호환용)
        self.performance_history[config_hash].append(trade_record)

        # 최근 100건만 유지
        if len(perf["trades"]) > 100:
            perf["trades"] = perf["trades"][-100:]
        if len(self.performance_history[config_hash]) > 100:
            self.performance_history[config_hash] = self.performance_history[config_hash][-100:]

    def get_best_config(self) -> tuple[str, dict]:
        """가장 성과 좋은 config 해시 반환 (DSR 차단된 config는 자동 제외).

        selection bias 방어: `validate_configs_dsr()`가 Deflated Sharpe Ratio로
        우연에 의한 상위 성과를 걸러내고 blocked_configs에 등록. 여기서는
        그 차단 목록을 자동으로 skip해서 "통계적 유의성 확인된" config만 승격.
        """
        best_hash = ""
        best_pnl = float("-inf")
        for h, perf in self.config_performance.items():
            if h in self.blocked_configs:
                continue  # DSR 오버핏 차단
            if len(perf["trades"]) >= 3 and perf["total_pnl"] > best_pnl:
                best_pnl = perf["total_pnl"]
                best_hash = h
        return best_hash, self.config_performance.get(best_hash, {})

    # ------------------------------------------------------------------
    # DSR/CPCV 오버핏 자동 검증 (2026-04-24)
    # ------------------------------------------------------------------
    def validate_configs_dsr(
        self,
        min_samples: int = 30,
        dsr_threshold: float = 0.50,
        force: bool = False,
    ) -> dict:
        """전체 config별 Deflated Sharpe Ratio를 계산해 오버핏을 자동 차단.

        Bailey & Lopez de Prado (2014) — N번 시도한 백테스트에서 나온 최상
        Sharpe는 selection bias로 부풀어져 있다. DSR은 "이 성과가 우연이
        아닐 확률"을 돌려주며, DSR<0.50은 동전던지기에 가깝다는 의미.

        본 메서드는 live 거래 결과(paper 시뮬레이션 or 실전)에 대해 돌려서,
        우연적으로 좋아 보이는 config를 blocked_configs에 등록한다. 이후
        get_best_config()는 자동으로 이들을 스킵.

        주기 제어: 직전 실행 후 20분 이상 경과해야 재실행 (force=True 예외).

        Args:
            min_samples: DSR 계산을 위한 최소 거래 수 (기본 30)
            dsr_threshold: 이 값 미만이면 차단 (기본 0.50 — 보수적)
            force: 주기 체크 건너뛰기

        Returns:
            {total_configs, validated, newly_blocked, currently_blocked}
        """
        # 주기 제어
        now = datetime.utcnow()
        if not force and self._last_dsr_validation is not None:
            if (now - self._last_dsr_validation).total_seconds() < 1200:
                return {"skipped": "rate_limited"}

        try:
            from core.backtest.dsr_cpcv import DeflatedSharpe
        except Exception as e:
            logger.debug(f"[DSR] 모듈 로드 실패: {e}")
            return {"error": str(e)}

        total = len(self.config_performance)
        n_trials = max(total, 1)
        newly_blocked = 0
        validated = 0
        for h, perf in list(self.config_performance.items()):
            trades = perf.get("trades", [])
            if len(trades) < min_samples:
                continue
            # 거래 수익률 (pnl)을 DSR에 넣기 전에 dollar-normalize는 생략 —
            # Sharpe는 scale-invariant이므로 pnl 그대로 써도 결과 동일.
            rets = np.array([float(t.get("pnl", 0.0)) for t in trades], dtype=float)
            rets = rets[np.isfinite(rets)]
            if len(rets) < min_samples:
                continue
            try:
                result = DeflatedSharpe.compute(rets, n_trials=n_trials, annualization=1)
            except Exception as e:
                logger.debug(f"[DSR] {h} 계산 실패: {e}")
                continue
            validated += 1
            is_blocked = result.dsr < dsr_threshold
            already_blocked = h in self.blocked_configs
            if is_blocked and not already_blocked:
                self.blocked_configs[h] = {
                    "dsr": round(result.dsr, 4),
                    "sharpe": round(result.sharpe, 4),
                    "n_samples": result.n_samples,
                    "reason": f"DSR({result.dsr:.3f})<{dsr_threshold} — 오버핏 의심",
                    "blocked_at": now.isoformat(),
                }
                newly_blocked += 1
                logger.warning(
                    f"[DSR] ❌ config {h} 차단 — "
                    f"SR={result.sharpe:+.2f} DSR={result.dsr:.3f} "
                    f"n={result.n_samples} n_trials={n_trials}"
                )
            elif not is_blocked and already_blocked:
                # 추가 데이터로 유의성 회복됐으면 차단 해제
                logger.info(
                    f"[DSR] ✅ config {h} 차단해제 — "
                    f"DSR={result.dsr:.3f} ≥ {dsr_threshold}"
                )
                self.blocked_configs.pop(h, None)
        self._last_dsr_validation = now
        report = {
            "total_configs": total,
            "validated": validated,
            "newly_blocked": newly_blocked,
            "currently_blocked": len(self.blocked_configs),
            "n_trials": n_trials,
            "threshold": dsr_threshold,
        }
        if validated > 0:
            logger.info(
                f"[DSR] 검증: {validated}/{total} config, "
                f"신규차단 {newly_blocked}, 총차단 {len(self.blocked_configs)}"
            )
        return report

    def get_blocked_report(self) -> dict:
        """차단된 config 목록 — 대시보드/로그용."""
        return dict(self.blocked_configs)

    def get_report(self) -> dict:
        """성과 리포트"""
        configs = {}
        for h, perf in self.config_performance.items():
            total = perf["wins"] + perf["losses"]
            configs[h] = {
                "trades": total,
                "win_rate": perf["wins"] / total if total > 0 else 0,
                "total_pnl": round(perf["total_pnl"], 2),
            }
        total_trades = sum(c["trades"] for c in configs.values())
        total_wins = sum(perf["wins"] for perf in self.config_performance.values())
        return {
            "total_configs": len(configs),
            "total_trades": total_trades,
            "current_win_rate": total_wins / total_trades if total_trades > 0 else 0,
            "configs": configs,
        }

    def get_adjustments(self) -> dict:
        """현재 적용 중인 자동 조정값 반환"""
        return self.adjustments

    def get_position_scale(self, symbol: str = "", hour: int = -1) -> float:
        """종목/시간대별 포지션 크기 조정 배율"""
        scale = self.adjustments.get("global_scale", 1.0)
        if symbol and f"scale_{symbol}" in self.adjustments:
            scale *= self.adjustments[f"scale_{symbol}"]
        if hour >= 0 and self.adjustments.get(f"avoid_hour_{hour}"):
            scale *= 0.0  # 해당 시간대 거래 차단
        if hour >= 0 and f"boost_hour_{hour}" in self.adjustments:
            scale *= self.adjustments[f"boost_hour_{hour}"]
        return max(0.0, min(2.0, scale))

    def optimize_daily(self, all_trades: list[dict]) -> dict:
        """일일 최적화 — 거래 결과 기반 파라미터 자동 조정

        분석 후 self.adjustments에 실제 파라미터 변경을 저장하여
        다음 거래에 반영합니다.
        """
        if len(all_trades) < 10:
            return {}

        wins = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
        losses = len(all_trades) - wins
        total_pnl = sum(t.get("pnl", 0) for t in all_trades)
        win_rate = wins / len(all_trades) if all_trades else 0
        changes = []

        # === 1. 시간대별 성과 분석 → 거래 회피/선호 시간대 ===
        hour_pnl = defaultdict(list)
        for t in all_trades:
            hour_pnl[t.get("hour", 0)].append(t.get("pnl", 0))

        best_hours = sorted(hour_pnl.keys(), key=lambda h: sum(hour_pnl[h]), reverse=True)[:3]
        worst_hours = sorted(hour_pnl.keys(), key=lambda h: sum(hour_pnl[h]))[:3]

        for h in worst_hours:
            trades_h = hour_pnl[h]
            if len(trades_h) >= 3:
                h_wr = sum(1 for p in trades_h if p > 0) / len(trades_h)
                if h_wr < 0.25 and sum(trades_h) < 0:
                    self.adjustments[f"avoid_hour_{h}"] = True
                    changes.append(f"UTC {h}시 회피 (승률 {h_wr:.0%})")

        for h in best_hours:
            trades_h = hour_pnl[h]
            if len(trades_h) >= 3:
                h_wr = sum(1 for p in trades_h if p > 0) / len(trades_h)
                if h_wr > 0.65:
                    self.adjustments[f"boost_hour_{h}"] = 1.2
                    changes.append(f"UTC {h}시 확신도 ×1.2 (승률 {h_wr:.0%})")

        # === 2. 종목별 성과 → 종목 우선순위 자동 조정 ===
        symbol_pnl = defaultdict(list)
        for t in all_trades:
            symbol_pnl[t.get("symbol", "")].append(t.get("pnl", 0))

        for sym, pnls in symbol_pnl.items():
            if len(pnls) >= 5:
                sym_wr = sum(1 for p in pnls if p > 0) / len(pnls)
                sym_total = sum(pnls)
                if sym_wr < 0.25 and sym_total < 0:
                    self.adjustments[f"scale_{sym}"] = 0.5
                    changes.append(f"{sym} 포지션 50% 축소 (승률 {sym_wr:.0%})")
                elif sym_wr > 0.6 and sym_total > 0:
                    self.adjustments[f"scale_{sym}"] = 1.3
                    changes.append(f"{sym} 포지션 30% 확대 (승률 {sym_wr:.0%})")

        # === 3. 승률 기반 전체 포지션 스케일 조정 ===
        if win_rate < 0.3:
            self.adjustments["global_scale"] = max(0.3, self.adjustments.get("global_scale", 1.0) - 0.1)
            changes.append(f"전체 포지션 축소 → {self.adjustments['global_scale']:.1f}x (승률 {win_rate:.0%})")
        elif win_rate > 0.55 and total_pnl > 0:
            self.adjustments["global_scale"] = min(1.5, self.adjustments.get("global_scale", 1.0) + 0.1)
            changes.append(f"전체 포지션 확대 → {self.adjustments['global_scale']:.1f}x (승률 {win_rate:.0%})")

        # === 4. 평균 손실 대비 평균 이익 비율 (RR비) → TP/SL 조정 제안 ===
        win_pnls = [t.get("pnl", 0) for t in all_trades if t.get("pnl", 0) > 0]
        loss_pnls = [t.get("pnl", 0) for t in all_trades if t.get("pnl", 0) <= 0]
        if win_pnls and loss_pnls:
            avg_win = sum(win_pnls) / len(win_pnls)
            avg_loss = abs(sum(loss_pnls) / len(loss_pnls))
            rr_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
            self.adjustments["rr_ratio"] = round(rr_ratio, 2)
            if rr_ratio < 1.0:
                changes.append(f"RR비 {rr_ratio:.2f} (손실>이익) → TP 확대 or SL 축소 필요")
            elif rr_ratio > 2.0:
                changes.append(f"RR비 {rr_ratio:.2f} (양호)")

        logger.info(
            f"[StrategyOptimizer] 일일 최적화: {len(all_trades)}건 | "
            f"승률 {win_rate:.0%} | PnL: ${total_pnl:.2f} | "
            f"베스트시간: {best_hours} | 워스트시간: {worst_hours}"
        )
        if changes:
            for c in changes:
                logger.info(f"[StrategyOptimizer] 📐 조정: {c}")

        return {"win_rate": win_rate, "total_pnl": total_pnl, "changes": changes}
