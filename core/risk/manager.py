"""리스크 매니저 - 포지션 크기, 드로다운, 일일 손실, 동적 레버리지, 상관관계 관리

v5 — Capital Tier 연동:
- Kelly fractional sizing (tier=mid+ 활성화)
- CVaR tail risk cap (tier=mid+ 활성화)
- PnL 히스토리 덱 (CVaR/분석용)
"""

from collections import deque
from datetime import datetime, timedelta

import numpy as np
from loguru import logger


class RiskManager:
    """거래 리스크 관리 시스템"""

    def __init__(self, config: dict):
        self.max_position_pct = config.get("max_position_pct", 0.1)
        self.max_daily_loss_pct = config.get("max_daily_loss_pct", 0.05)
        self.max_drawdown_pct = config.get("max_drawdown_pct", 0.15)
        self.max_open_positions = config.get("max_open_positions", 3)
        self.stop_loss_pct = config.get("stop_loss_pct", 0.02)

        # 동적 레버리지 설정
        lev_cfg = config.get("dynamic_leverage", {})
        self.dynamic_leverage_enabled = lev_cfg.get("enabled", True)
        self.base_leverage = lev_cfg.get("base", 5)
        self.min_leverage = lev_cfg.get("min", 2)
        self.max_leverage = lev_cfg.get("max", 10)

        # 쿨다운 설정
        self.cooldown_after_losses = config.get("cooldown_after_losses", 3)
        self.cooldown_minutes = config.get("cooldown_minutes", 30)
        self._cooldown_until: datetime | None = None

        # 상관관계 관리
        self._price_histories: dict[str, list[float]] = {}
        self._position_sides: dict[str, str] = {}  # symbol → "long"/"short"

        self.initial_equity = 0.0
        self.peak_equity = 0.0
        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.utcnow()
        self.is_trading_halted = False
        self.halt_reason = ""

        # 연패 추적 (쿨다운용)
        self.consecutive_losses = 0
        self._last_leverage: float = self.base_leverage

        # PnL 히스토리 (CVaR/분석용) — 최근 200건
        self._pnl_history: deque = deque(maxlen=200)
        # 마지막 Kelly/CVaR 상태 (디버깅/리포트용)
        self._last_kelly: dict = {"used": False, "fraction": 0.0, "size": 0.0}
        self._last_cvar: dict = {"checked": False, "cvar_pct": 0.0, "threshold_pct": 0.0, "passed": True}

    def initialize(self, equity: float):
        self.initial_equity = equity
        self.peak_equity = equity

    def check_can_trade(self, equity: float, num_positions: int) -> tuple[bool, str]:
        """거래 가능 여부 확인 (리스크 관리 해제 모드)"""
        self._check_daily_reset()

        # [해제됨] 쿨다운 체크 — 로그만 남김
        if self._cooldown_until and datetime.utcnow() < self._cooldown_until:
            remaining = (self._cooldown_until - datetime.utcnow()).seconds // 60
            logger.info(f"[리스크해제] 쿨다운 무시: {remaining}분 남았으나 통과 (연패 {self.consecutive_losses}회)")

        # [해제됨] 드로다운 체크 — 로그만 남김
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown > self.max_drawdown_pct:
            logger.info(f"[리스크해제] 드로다운 무시: {drawdown:.2%} > {self.max_drawdown_pct:.2%}")

        # [해제됨] 일일 손실 체크 — 로그만 남김
        daily_loss = -self.daily_pnl / self.initial_equity if self.initial_equity > 0 else 0
        if daily_loss > self.max_daily_loss_pct:
            logger.info(f"[리스크해제] 일일 손실 무시: {daily_loss:.2%}")

        # 최대 포지션 수 체크 (이것만 유지 — 과다 포지션 방지)
        if num_positions >= self.max_open_positions:
            return False, f"최대 포지션 수 도달: {num_positions}/{self.max_open_positions}"

        # 거래 중지 상태 해제
        self.is_trading_halted = False
        self.halt_reason = ""

        return True, "OK"

    def calculate_dynamic_leverage(
        self,
        confidence: float,
        volatility: float,
        regime: str = "normal",
        external_agreement: bool = True,
    ) -> int:
        """동적 레버리지 계산

        규칙:
        - 확신도 높음 + 변동성 낮음 → 레버리지 상향
        - 확신도 낮음 + 변동성 높음 → 레버리지 하향
        - 극심한 변동성 → 최소 레버리지
        - 외부 요인 불일치 → 레버리지 감소
        """
        if not self.dynamic_leverage_enabled:
            return self.base_leverage

        leverage = float(self.base_leverage)

        # 확신도 기반 (0.5~1.0 → 0.6x~1.4x)
        conf_factor = 0.6 + (confidence - 0.5) * 1.6
        conf_factor = max(0.6, min(1.4, conf_factor))
        leverage *= conf_factor

        # 변동성 기반 (낮을수록 레버리지 높임)
        if volatility > 0:
            # 정상 변동성 ~0.01 (1%), 높으면 축소
            vol_factor = min(1.3, 0.01 / (volatility + 1e-8))
            vol_factor = max(0.4, vol_factor)
            leverage *= vol_factor

        # 시장 레짐 기반
        regime_factors = {
            "extreme_volatility": 0.4,
            "high_volume_breakout": 0.8,
            "ranging": 0.7,
            "strong_uptrend": 1.2,
            "strong_downtrend": 1.2,
            "normal": 1.0,
        }
        leverage *= regime_factors.get(regime, 1.0)

        # 외부 요인 불일치 시 감소
        if not external_agreement:
            leverage *= 0.7

        # 연패 중이면 감소
        if self.consecutive_losses >= 2:
            leverage *= max(0.5, 1.0 - self.consecutive_losses * 0.1)

        # 범위 제한 및 정수화
        leverage = max(self.min_leverage, min(self.max_leverage, leverage))
        leverage = int(round(leverage))

        self._last_leverage = leverage
        return leverage

    def cap_leverage_by_risk(
        self,
        leverage: int,
        sl_pct: float,
        max_risk_pct: float = 0.05,
    ) -> int:
        """SL% × leverage ≤ max_risk_pct 되도록 레버리지 캡핑

        예) SL 4%, max_risk 5% → max_leverage = 1x
            SL 1.2%, max_risk 3% → max_leverage = 2x
        """
        if sl_pct <= 0:
            return leverage
        max_lev = max_risk_pct / sl_pct
        capped = max(1, int(max_lev))  # 최소 1x
        if capped < leverage:
            logger.info(
                f"[리스크] 레버리지 캡핑: {leverage}x → {capped}x "
                f"(SL {sl_pct*100:.1f}% × {leverage}x = {sl_pct*leverage*100:.1f}% 손실 > "
                f"허용 {max_risk_pct*100:.1f}%)"
            )
        return min(leverage, capped)

    def check_correlation(
        self,
        symbol: str,
        side: str,
        current_positions: dict,
    ) -> tuple[bool, str, float]:
        """포지션 간 상관관계 체크

        BTC와 ETH가 같은 방향 포지션이면 사실상 2배 리스크.
        상관계수 > 0.7이면 포지션 크기를 절반으로.

        Returns:
            (can_trade, reason, size_multiplier)
        """
        if not current_positions or len(self._price_histories) < 2:
            return True, "OK", 1.0

        # 현재 포지션의 심볼들과 방향
        active_symbols = []
        for pos_symbol, pos_info in current_positions.items():
            pos_side = pos_info.get("side", "") if isinstance(pos_info, dict) else ""
            if pos_side:
                active_symbols.append((pos_symbol, pos_side))

        if not active_symbols:
            return True, "OK", 1.0

        # 상관계수 계산
        for pos_symbol, pos_side in active_symbols:
            # 같은 심볼은 skip
            base_new = symbol.split("/")[0] if "/" in symbol else symbol.replace("USDT", "")
            base_pos = pos_symbol.split("/")[0] if "/" in pos_symbol else pos_symbol.replace("USDT", "")

            if base_new == base_pos:
                continue

            correlation = self._calculate_correlation(symbol, pos_symbol)

            if correlation > 0.7 and side == pos_side:
                # 높은 상관관계 + 같은 방향 = 중복 리스크
                logger.info(
                    f"[상관관계] {symbol}({side}) ↔ {pos_symbol}({pos_side}) "
                    f"상관계수={correlation:.2f} → 포지션 50% 축소"
                )
                return True, f"상관관계 {correlation:.2f} → 크기 축소", 0.5

            elif correlation > 0.7 and side != pos_side:
                # 높은 상관관계 + 반대 방향 = 헤지 (OK)
                logger.info(
                    f"[상관관계] {symbol}({side}) ↔ {pos_symbol}({pos_side}) "
                    f"상관계수={correlation:.2f} → 헤지 포지션 허용"
                )

        return True, "OK", 1.0

    def update_price_history(self, symbol: str, price: float):
        """가격 히스토리 업데이트 (상관관계 계산용)"""
        if symbol not in self._price_histories:
            self._price_histories[symbol] = []
        self._price_histories[symbol].append(price)
        if len(self._price_histories[symbol]) > 500:
            self._price_histories[symbol] = self._price_histories[symbol][-500:]

    def _calculate_correlation(self, sym1: str, sym2: str) -> float:
        """두 심볼 간 수익률 상관계수"""
        h1 = self._price_histories.get(sym1, [])
        h2 = self._price_histories.get(sym2, [])

        min_len = min(len(h1), len(h2))
        if min_len < 30:
            return 0.0  # 데이터 부족

        # 수익률 계산
        p1 = np.array(h1[-min_len:])
        p2 = np.array(h2[-min_len:])
        r1 = np.diff(p1) / p1[:-1]
        r2 = np.diff(p2) / p2[:-1]

        if len(r1) < 20:
            return 0.0

        correlation = np.corrcoef(r1, r2)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    def calculate_position_size(
        self,
        equity: float,
        confidence: float,
        volatility: float,
        adaptive_scale: float = 1.0,
        kelly_enabled: bool = False,
        kelly_fraction: float = 0.25,
        kelly_stats: dict | None = None,
    ) -> float:
        """포지션 크기 계산 — confidence/volatility 기반, tier에 따라 Kelly 혼합

        Args:
            kelly_enabled: Capital Tier가 kelly_enabled=True인 경우 활성화
            kelly_fraction: Fractional Kelly 비율 (0.25 = 1/4 Kelly, 안전)
            kelly_stats: TradeFeedbackAnalyzer.get_kelly_stats() 반환값
                         {win_rate, avg_win, avg_loss, sample_size, kelly_fraction_raw}

        Kelly 전략:
        - 샘플 ≥ 10이고 kelly_fraction_raw > 0일 때만 Kelly 적용
        - Kelly 원시 비율에 fractional 계수 곱해서 안전성 확보
        - Kelly와 기존 confidence 기반 크기를 min() 처리 — 더 보수적인 쪽 채택
        - 샘플 부족 시 기존 confidence 기반 크기로 fallback
        """
        # === 기존 confidence + volatility 기반 사이즈 ===
        base_size = equity * self.max_position_pct

        confidence_factor = max(0.3, min(1.0, confidence))
        sized = base_size * confidence_factor

        if volatility > 0:
            vol_factor = min(1.0, 0.02 / (volatility + 1e-8))
            sized *= vol_factor

        sized *= adaptive_scale

        min_size = equity * 0.01
        max_size = equity * self.max_position_pct
        sized = max(min_size, min(max_size, sized))

        # === Kelly 혼합 (tier=mid+ 활성화) ===
        if kelly_enabled and kelly_stats:
            kelly_raw = kelly_stats.get("kelly_fraction_raw", 0.0)
            sample_size = kelly_stats.get("sample_size", 0)
            if sample_size >= 10 and kelly_raw > 0:
                # Fractional Kelly: f = f* × fraction (예: f* × 0.25)
                fractional = max(0.0, min(1.0, kelly_raw * kelly_fraction))
                kelly_size = equity * fractional
                # 더 보수적인 쪽 채택 — 폭주 방지
                final_size = min(sized, kelly_size)
                # 최소 equity × 1% 보장
                final_size = max(min_size, final_size)
                self._last_kelly = {
                    "used": True,
                    "fraction": round(fractional, 4),
                    "kelly_raw": round(kelly_raw, 4),
                    "win_rate": kelly_stats.get("win_rate", 0),
                    "payoff_ratio": kelly_stats.get("payoff_ratio", 0),
                    "sample_size": sample_size,
                    "size": round(final_size, 2),
                    "confidence_based_size": round(sized, 2),
                }
                logger.info(
                    f"[Kelly] WR={kelly_stats.get('win_rate'):.2f} "
                    f"payoff={kelly_stats.get('payoff_ratio'):.2f} "
                    f"f*={kelly_raw:.3f} → f={fractional:.3f} "
                    f"size=${final_size:.2f} (conf=${sized:.2f}, 채택={min(sized, kelly_size)/equity:.1%})"
                )
                return final_size
            else:
                self._last_kelly = {
                    "used": False,
                    "reason": f"sample={sample_size}<10 or kelly_raw={kelly_raw:.3f}<=0",
                    "size": round(sized, 2),
                }
        else:
            self._last_kelly = {"used": False, "reason": "disabled", "size": round(sized, 2)}

        return sized

    def check_cvar_limit(
        self,
        proposed_notional: float,
        equity: float,
        threshold_pct: float = 0.05,
        alpha: float = 0.95,
    ) -> tuple[bool, float, str]:
        """CVaR (Expected Shortfall) tail risk 체크

        최근 PnL 히스토리에서 worst 5% (α=0.95) 평균 손실이
        제안된 notional의 % 대비 threshold 초과하면 차단.

        Returns:
            (passed, cvar_pct, reason)
            - passed: False면 거래 거부
            - cvar_pct: 관찰된 CVaR (절댓값) / equity 비율
            - reason: 차단 사유 (passed=True일 때는 "OK")
        """
        if len(self._pnl_history) < 20:
            # 데이터 부족 → 통과 (fallback)
            self._last_cvar = {"checked": False, "reason": "sample<20", "passed": True}
            return True, 0.0, "샘플 부족"

        pnls = np.array(list(self._pnl_history))
        losses = -pnls[pnls < 0]  # 손실만 추출 (양수로 변환)

        if len(losses) < 5:
            self._last_cvar = {"checked": False, "reason": "losses<5", "passed": True}
            return True, 0.0, "손실 샘플 부족"

        # VaR(α): 손실 분포의 α-quantile
        # CVaR(α) = E[L | L ≥ VaR(α)] — worst (1-α)% 손실의 평균
        var_threshold = np.quantile(losses, alpha)
        tail_losses = losses[losses >= var_threshold]
        cvar = float(np.mean(tail_losses)) if len(tail_losses) > 0 else float(np.max(losses))

        # equity 대비 CVaR 비율
        cvar_pct_of_equity = cvar / max(equity, 1.0)

        # 제안 notional의 손실 잠재력 ~ notional × SL% ≈ CVaR 히스토릭 평균
        # → equity 대비 cvar_pct가 threshold보다 크면 차단
        self._last_cvar = {
            "checked": True,
            "cvar_usd": round(cvar, 2),
            "cvar_pct": round(cvar_pct_of_equity, 4),
            "threshold_pct": threshold_pct,
            "sample_size": len(losses),
            "passed": cvar_pct_of_equity <= threshold_pct,
        }

        if cvar_pct_of_equity > threshold_pct:
            reason = (
                f"CVaR(95%)=${cvar:.2f} ({cvar_pct_of_equity:.1%}/equity) > "
                f"허용 {threshold_pct:.1%} — 과거 tail 손실 과다"
            )
            logger.warning(f"[CVaR] 거래 차단: {reason}")
            return False, cvar_pct_of_equity, reason

        return True, cvar_pct_of_equity, "OK"

    def record_trade_result(self, pnl: float):
        """거래 결과 기록 (PnL + 연패/연승 추적)"""
        self.daily_pnl += pnl
        self._pnl_history.append(float(pnl))

        if pnl < 0:
            self.consecutive_losses += 1
            # 연패 쿨다운
            if self.consecutive_losses >= self.cooldown_after_losses:
                self._cooldown_until = datetime.utcnow() + timedelta(minutes=self.cooldown_minutes)
                logger.warning(
                    f"[쿨다운] {self.consecutive_losses}연패 → "
                    f"{self.cooldown_minutes}분 거래 중단 (~{self._cooldown_until.strftime('%H:%M')})"
                )
        else:
            self.consecutive_losses = 0
            self._cooldown_until = None

    def record_pnl(self, pnl: float):
        """PnL 기록 (기존 호환성 유지)"""
        self.record_trade_result(pnl)

    def _check_daily_reset(self):
        now = datetime.utcnow()
        if now - self.daily_reset_time > timedelta(hours=24):
            self.daily_pnl = 0.0
            self.daily_reset_time = now
            if self.is_trading_halted and "일일" in self.halt_reason:
                self.is_trading_halted = False
                self.halt_reason = ""
                logger.info("일일 거래 제한 해제")

    def force_resume(self):
        """수동 거래 재개"""
        self.is_trading_halted = False
        self.halt_reason = ""
        self._cooldown_until = None
        self.consecutive_losses = 0
        logger.info("거래 수동 재개")

    def get_status(self) -> dict:
        drawdown = (self.peak_equity - self.initial_equity) / self.peak_equity if self.peak_equity > 0 else 0
        return {
            "is_halted": self.is_trading_halted,
            "halt_reason": self.halt_reason,
            "daily_pnl": self.daily_pnl,
            "peak_equity": self.peak_equity,
            "current_drawdown": drawdown,
            "max_drawdown_limit": self.max_drawdown_pct,
            "max_daily_loss_limit": self.max_daily_loss_pct,
            "consecutive_losses": self.consecutive_losses,
            "current_leverage": self._last_leverage,
            "cooldown_active": self._cooldown_until is not None and datetime.utcnow() < self._cooldown_until,
            "pnl_history_size": len(self._pnl_history),
            "last_kelly": self._last_kelly,
            "last_cvar": self._last_cvar,
            "correlations": {
                f"{s1}-{s2}": round(self._calculate_correlation(s1, s2), 2)
                for i, s1 in enumerate(self._price_histories)
                for s2 in list(self._price_histories.keys())[i+1:]
                if len(self._price_histories.get(s1, [])) > 30 and len(self._price_histories.get(s2, [])) > 30
            },
        }
