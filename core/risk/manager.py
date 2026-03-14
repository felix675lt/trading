"""리스크 매니저 - 포지션 크기, 드로다운, 일일 손실 관리"""

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

        self.initial_equity = 0.0
        self.peak_equity = 0.0
        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.utcnow()
        self.is_trading_halted = False
        self.halt_reason = ""

    def initialize(self, equity: float):
        self.initial_equity = equity
        self.peak_equity = equity

    def check_can_trade(self, equity: float, num_positions: int) -> tuple[bool, str]:
        """거래 가능 여부 확인"""
        self._check_daily_reset()

        # 드로다운 체크
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown > self.max_drawdown_pct:
            self.is_trading_halted = True
            self.halt_reason = f"최대 드로다운 초과: {drawdown:.2%} > {self.max_drawdown_pct:.2%}"
            logger.warning(f"거래 중지: {self.halt_reason}")
            return False, self.halt_reason

        # 일일 손실 체크
        daily_loss = -self.daily_pnl / self.initial_equity if self.initial_equity > 0 else 0
        if daily_loss > self.max_daily_loss_pct:
            self.is_trading_halted = True
            self.halt_reason = f"일일 최대 손실 초과: {daily_loss:.2%}"
            logger.warning(f"거래 중지: {self.halt_reason}")
            return False, self.halt_reason

        # 최대 포지션 수 체크
        if num_positions >= self.max_open_positions:
            return False, f"최대 포지션 수 도달: {num_positions}/{self.max_open_positions}"

        if self.is_trading_halted:
            return False, self.halt_reason

        return True, "OK"

    def calculate_position_size(
        self,
        equity: float,
        confidence: float,
        volatility: float,
        adaptive_scale: float = 1.0,
    ) -> float:
        """켈리 기준 + 변동성 기반 포지션 크기 계산"""
        base_size = equity * self.max_position_pct

        # 확신도 비례 조정
        confidence_factor = max(0.3, min(1.0, confidence))
        sized = base_size * confidence_factor

        # 변동성 역비례 조정 (변동성 높으면 포지션 줄임)
        if volatility > 0:
            vol_factor = min(1.0, 0.02 / (volatility + 1e-8))
            sized *= vol_factor

        # 적응형 스케일
        sized *= adaptive_scale

        # 최소/최대 제한
        min_size = equity * 0.01
        max_size = equity * self.max_position_pct
        sized = max(min_size, min(max_size, sized))

        return sized

    def record_pnl(self, pnl: float):
        """PnL 기록"""
        self.daily_pnl += pnl

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
        }
