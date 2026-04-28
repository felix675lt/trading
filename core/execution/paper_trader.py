"""페이퍼 트레이딩 엔진 — LIVE와 동일한 체결 모델로 시뮬레이션

현실화된 체결 모델 (2026-04-18 개정):
1. 슬리피지 — 진입/청산 모두 방향 불리하게 기본 5bp + ATR 가변분
2. 펀딩비 — 8h 경계 자동 감지 후 차감 (롱 편향시 체계적 비용)
3. Maker 체결률 피드백 — Live 실측으로 업데이트 가능 (기본 0.45)
4. SL 체결가 통일 — Live와 동일하게 "현재가 + 슬리피지" (SL 레벨 고정 X)
5. 레이턴시 시뮬 — SL 감지 ~ 실제 체결 간 지연을 슬리피지에 반영
6. Liquidation 체크 — 거래소 강제청산 근사 (maint margin 0.5% + 수수료)

모든 수치는 LIVE 실측 데이터를 기반으로 재보정 가능.
"""

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from loguru import logger


@dataclass
class PaperPosition:
    symbol: str
    side: str
    size: float
    entry_price: float
    leverage: int = 5
    stop_loss: float = 0.0
    take_profit: float = 0.0
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0   # 진입 후 최고가 (롱)
    lowest_price: float = 0.0    # 진입 후 최저가 (숏)
    trailing_activated: bool = False  # 트레일링 스탑 활성화 여부
    trade_type: str = "scalp"    # "scalp" or "swing"
    entry_time: datetime = None   # 개시 시각 (펀딩/stale-age 공통)
    last_funded_ts: int = 0      # 마지막으로 펀딩 차감한 8h 경계 (unix_h // 8)
    funding_rate: float = 0.0    # 심볼별 funding rate (8h당, 롱=+이면 지불)


class PaperTrader:
    """LIVE와 동일한 체결 모델로 시뮬레이션하는 페이퍼 트레이딩 엔진"""

    SL_COOLDOWN_SECONDS = 300  # SL 청산 후 5분간 같은 심볼 재진입 차단

    # === 현실성 파라미터 (LIVE 피드백으로 덮어쓰기 가능) ===
    # 슬리피지: 거래소 mid-price 대비 체결가 편차
    DEFAULT_SLIPPAGE_BPS = 5         # 기본 5bp (=0.05%) — Binance Futures 평균
    SLIPPAGE_ATR_COEF = 0.15         # ATR 대비 추가 슬리피지 (변동성↑ → 슬리피지↑)
    SL_SLIPPAGE_EXTRA_BPS = 8        # SL 히트 시 추가 슬리피지 (stop market 급체결)
    MARKET_FALLBACK_EXTRA_BPS = 3    # limit→market fallback 시 추가

    # Maker 체결률 — LIVE 실측 없을 때 fallback (보수적 45%)
    DEFAULT_MAKER_FILL_RATE = 0.45

    # 레이턴시 시뮬 (ms) → 슬리피지로 변환
    LATENCY_MS = 500
    LATENCY_SLIPPAGE_BPS = 2         # 500ms 대기 시 추가 2bp

    # Liquidation 근사 (Binance Futures maintenance margin)
    MAINT_MARGIN_RATE = 0.005
    LIQ_FEE_BPS = 15                 # 강제청산 수수료 근사

    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.0004,
                 trailing_config: dict | None = None,
                 trade_profiles: dict | None = None,
                 variant: str = "PAPER"):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.commission = commission                       # taker fee (기본 0.04%)
        self.maker_commission = commission * 0.5           # maker fee 추정 (0.02%)
        self.trade_profiles = trade_profiles or {}
        self.positions: dict[str, PaperPosition] = {}
        self.trade_history: list[dict] = []
        self.equity_history: list[dict] = []
        self._sl_cooldown: dict[str, datetime] = {}  # symbol → SL 청산 시각
        # A/B 테스트 variant 태그 (2026-04-21) — storage/feedback 에서 격리용
        # 값: "PAPER" (legacy) / "PAPER_MACRO_ON" / "PAPER_MACRO_OFF"
        self.variant = variant

        # 트레일링 스탑 기본값 (fallback)
        tc = trailing_config or {}
        self.trailing_activate_pct = tc.get("activate_pct", 0.02)
        self.trailing_distance_pct = tc.get("distance_pct", 0.015)
        self.trailing_step_pct = tc.get("step_pct", 0.005)

        # 자동 청산 콜백 (SL/TP/트레일링으로 청산 시 호출)
        self._on_auto_close_callback = None

        # [Patch K, 2026-04-28] 활성 포지션 디스크 persist —
        # launchd 재기동 시 in-memory 포지션 손실 방지 (SL/TP 정보까지 보존).
        # variant별 분리: PAPER / PAPER_MACRO_ON / PAPER_MACRO_OFF
        safe_var = (variant or "PAPER").lower().replace("/", "_")
        self._positions_path = Path(f"data/paper_positions_{safe_var}.json")
        self._load_positions()

        # Limit-first routing (tier=small+ 활성화)
        self.limit_first_enabled = False
        # 지정가 체결률 통계 — maker fee 적용률 추적
        self.limit_fill_stats = {"attempts": 0, "filled": 0, "fallback_market": 0}
        # LIVE 피드백으로 덮어쓰기 가능한 maker fill rate
        self.maker_fill_rate = self.DEFAULT_MAKER_FILL_RATE

        # === 인스턴스-레벨 슬리피지 파라미터 (2026-04-24 A: LIVE 피드백 가능) ===
        # 클래스 상수 DEFAULT_*는 초기값만 제공. 이하 4개는 sync_from_live_execution()
        # 으로 LIVE 실측값에 맞춰 동적 갱신됨 — 이렇게 해야 Paper의 슬리피지가
        # 실전과 괴리되지 않음.
        self._slip_entry_bps = float(self.DEFAULT_SLIPPAGE_BPS)
        self._slip_sl_extra_bps = float(self.SL_SLIPPAGE_EXTRA_BPS)
        self._slip_market_fallback_bps = float(self.MARKET_FALLBACK_EXTRA_BPS)
        self._slip_latency_bps = float(self.LATENCY_SLIPPAGE_BPS)
        # 슬리피지 동기화 이력 (대시보드/디버깅용)
        self._slip_sync_history: list[dict] = []

        # 심볼별 funding rate 캐시 (외부에서 주입)
        # { "BTC/USDT:USDT": 0.0001 }  — 8h당 rate (롱이 숏에게 지불하면 +)
        self._funding_rates: dict[str, float] = {}

        # 심볼별 ATR 캐시 (슬리피지 동적 계산용)
        self._atr_cache: dict[str, float] = {}

    # ---------------------------------------------------------------------
    # 외부 주입 — LIVE 피드백으로 현실 반영
    # ---------------------------------------------------------------------

    def set_maker_fill_rate(self, rate: float):
        """LIVE 실측 maker 체결률로 덮어쓰기 (0.0~1.0)"""
        rate = max(0.0, min(1.0, float(rate)))
        if abs(rate - self.maker_fill_rate) > 0.02:
            logger.info(f"[Paper-Feedback] maker fill rate 업데이트: {self.maker_fill_rate:.2%} → {rate:.2%}")
        self.maker_fill_rate = rate

    def set_funding_rate(self, symbol: str, rate: float):
        """심볼별 funding rate 주입 (8h당, 소수점 — 0.0001 = 0.01%)"""
        self._funding_rates[symbol] = float(rate)

    def set_atr(self, symbol: str, atr_pct: float):
        """심볼별 ATR(%) 주입 — 슬리피지 동적 계산용"""
        self._atr_cache[symbol] = max(0.0, float(atr_pct))

    # ------------------------------------------------------------------
    # [Patch K, 2026-04-28] 활성 포지션 디스크 persist
    # 목적: launchd 재기동 시 in-memory 포지션 손실 방지.
    # 무인 1개월 운영 중 launchd 한 번이라도 재기동되면 SL/TP 정보 + 진입가
    # 모두 사라져 청산 못 함 → 학습 데이터 손실 + 가상 자본 정확도 손상.
    # 해결: open/close/SL-TP 자동청산 시점마다 atomic write.
    # ------------------------------------------------------------------
    def _save_positions(self):
        try:
            data = {
                "saved_at": datetime.utcnow().isoformat(),
                "variant": self.variant,
                "equity": float(self.equity),
                "positions": {},
                "sl_cooldown": {
                    s: t.isoformat() for s, t in self._sl_cooldown.items()
                },
            }
            for sym, pos in self.positions.items():
                d = asdict(pos)
                # entry_time은 datetime → ISO 문자열로
                if d.get("entry_time"):
                    d["entry_time"] = d["entry_time"].isoformat() if hasattr(d["entry_time"], "isoformat") else str(d["entry_time"])
                data["positions"][sym] = d
            tmp = self._positions_path.with_suffix(".tmp")
            self._positions_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, indent=2, default=str))
            tmp.rename(self._positions_path)
        except Exception as e:
            logger.warning(f"[Paper-Persist] 포지션 저장 실패: {e}")

    def _load_positions(self):
        if not self._positions_path.exists():
            return
        try:
            data = json.loads(self._positions_path.read_text())
            saved_at = data.get("saved_at", "?")
            self.equity = float(data.get("equity", self.equity))
            for sym, d in (data.get("positions") or {}).items():
                # entry_time 복원
                et = d.get("entry_time")
                if et and isinstance(et, str):
                    try:
                        d["entry_time"] = datetime.fromisoformat(et)
                    except Exception:
                        d["entry_time"] = datetime.utcnow()
                self.positions[sym] = PaperPosition(**d)
            for s, ts in (data.get("sl_cooldown") or {}).items():
                try:
                    self._sl_cooldown[s] = datetime.fromisoformat(ts)
                except Exception:
                    continue
            n = len(self.positions)
            if n > 0:
                logger.warning(
                    f"[Paper-Persist] 디스크 복원 완료 — {self.variant} {n}개 활성 포지션 "
                    f"(저장 시각 {saved_at}) — SL/TP 정보 보존, 즉시 update_prices에서 자동 청산 평가"
                )
        except Exception as e:
            logger.warning(f"[Paper-Persist] 포지션 복원 실패: {e}")

    def set_routing(self, limit_first: bool):
        """Capital Tier 기반 라우팅 설정"""
        self.limit_first_enabled = limit_first
        if limit_first:
            logger.info(
                f"[Paper-Routing] Limit-first 시뮬레이션 활성화 "
                f"(maker fee {self.maker_commission*100:.3f}% × fill_rate {self.maker_fill_rate:.0%})"
            )

    def set_auto_close_callback(self, callback):
        """SL/TP/트레일링 자동 청산 시 호출할 콜백 등록
        callback(trade: dict) — trade는 close_position 반환값과 동일
        """
        self._on_auto_close_callback = callback

    def set_profit_callback(self, callback):
        """포지션 청산 시 항상 호출될 추가 콜백 (BTC Reserve 적립 등)

        모든 close_position 경로(수동/SL/TP/liquidation)에서 PnL과 관계없이 호출됨.
        콜백 내부에서 PnL>0 조건 등을 자체 판단해야 함.
        callback(trade: dict)
        """
        self._profit_callback = callback

    # ---------------------------------------------------------------------
    # 슬리피지 모델
    # ---------------------------------------------------------------------

    def _slippage_bps(self, symbol: str, is_sl: bool = False, is_market_fallback: bool = False) -> float:
        """방향 불리한 슬리피지를 bps로 반환.

        인스턴스-레벨 파라미터 사용 — `sync_from_live_execution()`이 LIVE 실측
        중앙값으로 이들을 동적 갱신. 따라서 PAPER 슬리피지가 실전과 자동으로 수렴.

        - 기본 entry: self._slip_entry_bps (기본 5bp, LIVE 실측으로 덮어씀)
        - ATR × 0.15 추가 (변동성 가변, symbol별)
        - SL 히트면 +self._slip_sl_extra_bps (기본 8bp)
        - market fallback이면 +self._slip_market_fallback_bps (기본 3bp)
        - 레이턴시 +self._slip_latency_bps (기본 2bp)
        """
        bps = self._slip_entry_bps
        atr_pct = self._atr_cache.get(symbol, 0.0)
        bps += atr_pct * 10000 * self.SLIPPAGE_ATR_COEF  # atr_pct 0.01 → +15bp
        if is_sl:
            bps += self._slip_sl_extra_bps
        if is_market_fallback:
            bps += self._slip_market_fallback_bps
        bps += self._slip_latency_bps
        return bps

    # ------------------------------------------------------------------
    # LIVE 실측 체결 통계로부터 슬리피지 / 체결률 자동 동기화 (2026-04-24)
    # ------------------------------------------------------------------
    def sync_from_live_execution(self, live_stats: dict, smoothing: float = 0.30) -> dict:
        """LIVE OrderManager.get_execution_stats() 결과를 Paper에 반영.

        핵심 철학: PAPER가 LIVE와 괴리되면 학습된 모델이 실전에서 기대 대비
        수익률이 꺾인다. 따라서 LIVE의 실측 체결 모델(슬리피지, maker rate)을
        주기적으로 Paper에 지수스무딩으로 주입 → 두 환경이 통계적으로 수렴.

        Args:
            live_stats: OrderManager.get_execution_stats() 반환 dict.
                       기대 키: entry_slip_bps_med, exit_slip_bps_med,
                                sl_slip_bps_med, maker_fill_rate, n_samples
            smoothing: (1-s)*current + s*observed. 기본 0.30 (3~5회 관측으로 수렴).

        Returns:
            적용 전/후 delta 요약 — 대시보드/로그용.
        """
        if not live_stats or not isinstance(live_stats, dict):
            return {"skipped": "no_stats"}
        n = int(live_stats.get("n_samples", 0) or 0)
        if n < 5:
            return {"skipped": f"n={n}<5"}

        s = max(0.0, min(1.0, float(smoothing)))
        before = {
            "entry": self._slip_entry_bps,
            "sl_extra": self._slip_sl_extra_bps,
            "maker_fill": self.maker_fill_rate,
        }

        # 엔트리 슬리피지 — LIVE 실측 중앙값으로 수렴
        obs_entry = live_stats.get("entry_slip_bps_med")
        if obs_entry is not None and obs_entry >= 0:
            self._slip_entry_bps = (1 - s) * self._slip_entry_bps + s * float(obs_entry)
            # 클리핑 (outlier 방어): [1bp, 50bp]
            self._slip_entry_bps = max(1.0, min(50.0, self._slip_entry_bps))

        # SL 추가 슬리피지 — (exit_sl - exit_normal) 로 분해
        obs_exit = live_stats.get("exit_slip_bps_med")
        obs_sl = live_stats.get("sl_slip_bps_med")
        if obs_sl is not None and obs_exit is not None and obs_sl > obs_exit:
            extra = float(obs_sl - obs_exit)
            self._slip_sl_extra_bps = (1 - s) * self._slip_sl_extra_bps + s * extra
            # 클리핑: [0bp, 40bp]
            self._slip_sl_extra_bps = max(0.0, min(40.0, self._slip_sl_extra_bps))

        # Maker 체결률
        obs_maker = live_stats.get("maker_fill_rate")
        if obs_maker is not None and 0.0 <= obs_maker <= 1.0:
            self.maker_fill_rate = (1 - s) * self.maker_fill_rate + s * float(obs_maker)

        after = {
            "entry": self._slip_entry_bps,
            "sl_extra": self._slip_sl_extra_bps,
            "maker_fill": self.maker_fill_rate,
        }
        delta = {
            "n_samples": n,
            "entry_before": round(before["entry"], 2),
            "entry_after": round(after["entry"], 2),
            "sl_extra_before": round(before["sl_extra"], 2),
            "sl_extra_after": round(after["sl_extra"], 2),
            "maker_fill_before": round(before["maker_fill"], 3),
            "maker_fill_after": round(after["maker_fill"], 3),
            "smoothing": s,
        }
        # 변경이 의미있으면 로그
        if (
            abs(after["entry"] - before["entry"]) > 0.3
            or abs(after["sl_extra"] - before["sl_extra"]) > 0.3
            or abs(after["maker_fill"] - before["maker_fill"]) > 0.01
        ):
            logger.info(
                f"[Paper-LiveSync] n={n} | entry {before['entry']:.1f}→{after['entry']:.1f}bp | "
                f"SL+ {before['sl_extra']:.1f}→{after['sl_extra']:.1f}bp | "
                f"maker {before['maker_fill']:.2f}→{after['maker_fill']:.2f}"
            )
        # 이력 기록 (최근 20개)
        self._slip_sync_history.append({
            "ts": str(datetime.utcnow()), **delta,
        })
        if len(self._slip_sync_history) > 20:
            self._slip_sync_history = self._slip_sync_history[-20:]
        return delta

    def get_execution_profile(self) -> dict:
        """현재 Paper 체결 모델 상태 — 대시보드/디버깅용."""
        return {
            "entry_slip_bps": round(self._slip_entry_bps, 2),
            "sl_extra_bps": round(self._slip_sl_extra_bps, 2),
            "market_fallback_bps": round(self._slip_market_fallback_bps, 2),
            "latency_bps": round(self._slip_latency_bps, 2),
            "maker_fill_rate": round(self.maker_fill_rate, 3),
            "sync_events": len(self._slip_sync_history),
        }

    def _apply_slippage(self, side: str, price: float, bps: float, is_entry: bool) -> float:
        """슬리피지 적용 — 항상 방향 불리하게.

        진입:
          - long: 더 비싸게 매수 (+bps)
          - short: 더 싸게 매도 (-bps)
        청산(반대 방향):
          - long 청산 = 매도: 더 싸게 (-bps)
          - short 청산 = 매수: 더 비싸게 (+bps)
        """
        mult = bps / 10000.0
        if is_entry:
            return price * (1 + mult) if side == "long" else price * (1 - mult)
        else:
            return price * (1 - mult) if side == "long" else price * (1 + mult)

    def _get_trailing_params(self, trade_type: str) -> tuple[float, float, float]:
        """trade_type별 트레일링 파라미터"""
        profile = self.trade_profiles.get(trade_type, {})
        tc = profile.get("trailing", {})
        return (
            tc.get("activate_pct", self.trailing_activate_pct),
            tc.get("distance_pct", self.trailing_distance_pct),
            tc.get("step_pct", self.trailing_step_pct),
        )

    # ---------------------------------------------------------------------
    # Open
    # ---------------------------------------------------------------------

    def open_position(self, symbol: str, side: str, size_usdt: float,
                      price: float, leverage: int = 5, sl_pct: float = 0.02,
                      tp_pct: float = 0.04, atr_pct: float = 0.0,
                      trade_type: str = "scalp") -> PaperPosition | None:
        if symbol in self.positions:
            logger.warning(f"[Paper] {symbol} 이미 포지션 보유")
            return None

        # SL 쿨다운 체크
        sl_time = self._sl_cooldown.get(symbol)
        if sl_time:
            elapsed = (datetime.utcnow() - sl_time).total_seconds()
            if elapsed < self.SL_COOLDOWN_SECONDS:
                logger.info(f"[Paper-쿨다운] {symbol} SL 후 재진입 차단 ({elapsed:.0f}s/{self.SL_COOLDOWN_SECONDS}s)")
                return None
            else:
                del self._sl_cooldown[symbol]

        # ATR 캐시 업데이트 (슬리피지용)
        if atr_pct and atr_pct > 0 and atr_pct == atr_pct:
            self._atr_cache[symbol] = atr_pct

        # === Limit-first 시뮬레이션 (tier=small+ 활성화) ===
        # maker_fill_rate 확률로 maker 체결, 나머지는 market fallback
        is_maker = False
        is_market_fallback = False
        if self.limit_first_enabled:
            self.limit_fill_stats["attempts"] += 1
            is_maker = random.random() < self.maker_fill_rate
            if is_maker:
                self.limit_fill_stats["filled"] += 1
            else:
                self.limit_fill_stats["fallback_market"] += 1
                is_market_fallback = True

        # === 진입 슬리피지 적용 ===
        # maker 체결이면 슬리피지 0 (limit price로 정확 체결 가정)
        # market (taker)면 방향 불리하게 슬리피지
        if is_maker:
            fill_price = price
        else:
            slip_bps = self._slippage_bps(
                symbol, is_sl=False,
                is_market_fallback=is_market_fallback,
            )
            fill_price = self._apply_slippage(side, price, slip_bps, is_entry=True)

        amount = (size_usdt * leverage) / fill_price
        effective_commission = self.maker_commission if is_maker else self.commission
        fee = size_usdt * leverage * effective_commission  # notional 기준 수수료
        self.equity -= fee

        # trade_type 프로파일에서 기본값 가져오기
        profile = self.trade_profiles.get(trade_type, {})

        # ATR 기반 동적 SL/TP (atr_pct가 유효하면 사용)
        if atr_pct and atr_pct > 0 and atr_pct == atr_pct:  # NaN 체크
            atr_sl_mult = profile.get("atr_sl_multiplier", 2.0)
            atr_tp_mult = profile.get("atr_tp_multiplier", 3.5)
            sl_floor = profile.get("sl_floor_pct", 0.005)
            sl_cap = profile.get("sl_cap_pct", 0.030)
            final_sl = max(sl_floor, min(sl_cap, atr_pct * atr_sl_mult))
            final_tp = max(final_sl * 1.5, atr_pct * atr_tp_mult)
        else:
            final_sl = sl_pct if sl_pct != 0.02 else profile.get("sl_pct", sl_pct)
            final_tp = tp_pct if tp_pct != 0.04 else profile.get("tp_pct", tp_pct)

        # SL/TP는 fill_price 기준으로 계산 (진입가 슬리피지 반영)
        if side == "long":
            sl = fill_price * (1 - final_sl)
            tp = fill_price * (1 + final_tp)
        else:
            sl = fill_price * (1 + final_sl)
            tp = fill_price * (1 - final_tp)

        now = datetime.utcnow()
        funding_rate = self._funding_rates.get(symbol, 0.0)
        pos = PaperPosition(
            symbol=symbol, side=side, size=amount,
            entry_price=fill_price, leverage=leverage,
            stop_loss=sl, take_profit=tp,
            highest_price=fill_price, lowest_price=fill_price,
            trade_type=trade_type,
            entry_time=now,
            last_funded_ts=int(now.timestamp()) // (8 * 3600),
            funding_rate=funding_rate,
        )
        self.positions[symbol] = pos
        self._save_positions()  # [Patch K] 진입 즉시 디스크 동기화
        fill_tag = "MAKER" if is_maker else ("MKT-FALLBACK" if is_market_fallback else "MARKET")
        slip_pct = (fill_price - price) / price * 100 if side == "long" else (price - fill_price) / price * 100
        logger.info(
            f"[Paper] 포지션 개시: {side} {amount:.6f} {symbol} @ {fill_price:.4f} "
            f"(req {price:.4f}, slip {slip_pct:+.3f}%, {fill_tag}) | "
            f"타입: {trade_type} | SL: {final_sl*100:.1f}% TP: {final_tp*100:.1f}% | fee ${fee:.3f}"
        )
        return pos

    # ---------------------------------------------------------------------
    # Close — 슬리피지 적용 (SL 히트 시 추가 슬리피지)
    # ---------------------------------------------------------------------

    def close_position(self, symbol: str, price: float, reason: str = "") -> dict:
        if symbol not in self.positions:
            return {}

        pos = self.positions[symbol]

        # === 청산 슬리피지 적용 ===
        # SL/liquidation 히트면 추가 슬리피지 (stop market 급체결)
        is_sl_hit = ("SL" in reason.upper()) or ("liq" in reason.lower())
        slip_bps = self._slippage_bps(symbol, is_sl=is_sl_hit, is_market_fallback=False)
        fill_price = self._apply_slippage(pos.side, price, slip_bps, is_entry=False)

        if pos.side == "long":
            pnl = (fill_price - pos.entry_price) * pos.size * pos.leverage
        else:
            pnl = (pos.entry_price - fill_price) * pos.size * pos.leverage

        # 수수료 — taker (청산은 대부분 market)
        notional = pos.size * fill_price
        fee = notional * self.commission

        # Liquidation이면 강제청산 수수료 추가
        if "liq" in reason.lower():
            fee += notional * (self.LIQ_FEE_BPS / 10000.0)

        net_pnl = pnl - fee
        self.equity += net_pnl

        duration_minutes = 0
        if pos.entry_time:
            duration_minutes = (datetime.utcnow() - pos.entry_time).total_seconds() / 60

        trade = {
            "timestamp": str(datetime.utcnow()),
            "symbol": symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": fill_price,
            "requested_price": price,
            "slippage_bps": slip_bps,
            "size": pos.size,
            "notional": notional,
            "pnl": net_pnl,
            "fee": fee,
            "reason": reason,
            "duration_minutes": duration_minutes,
            "leverage": pos.leverage,
            "variant": self.variant,  # A/B 테스트용 태그 (2026-04-21)
        }
        self.trade_history.append(trade)
        del self.positions[symbol]
        self._save_positions()  # [Patch K] 청산 즉시 디스크 동기화

        # SL 청산이면 쿨다운 등록
        if "SL" in reason.upper() or "sl" in reason:
            self._sl_cooldown[symbol] = datetime.utcnow()
            logger.info(f"[Paper-쿨다운] {symbol} SL 청산 → {self.SL_COOLDOWN_SECONDS}초 재진입 차단")

        slip_pct = (price - fill_price) / price * 100 if pos.side == "long" else (fill_price - price) / price * 100
        logger.info(
            f"[Paper] 포지션 청산: {symbol} {pos.side} @ {fill_price:.4f} "
            f"(req {price:.4f}, slip {slip_pct:+.3f}% [{slip_bps:.1f}bp]) | "
            f"PnL: {net_pnl:+.2f} | fee ${fee:.3f} | 사유: {reason}"
        )

        # === BTC Reserve 적립 콜백 (수익 시 가상 BTC 누적) ===
        cb = getattr(self, "_profit_callback", None)
        if cb is not None:
            try:
                cb(trade)
            except Exception as e:
                logger.debug(f"[Paper] profit_callback 실패 (무시): {e}")

        return trade

    # ---------------------------------------------------------------------
    # Update — 가격 업데이트 + 펀딩 + Liquidation + SL/TP
    # ---------------------------------------------------------------------

    def update_prices(self, prices: dict[str, float]):
        """현재가 업데이트 + 펀딩비 차감 + 강제청산 + 트레일링 스탑 + SL/TP"""
        auto_closed = []  # (symbol, close_price, reason) — 루프 후 처리
        now = datetime.utcnow()
        now_8h = int(now.timestamp()) // (8 * 3600)

        for symbol, price in prices.items():
            if symbol not in self.positions:
                continue
            pos = self.positions[symbol]
            t_activate, t_distance, t_step = self._get_trailing_params(pos.trade_type)

            # === 1. 펀딩비 차감 — 8h 경계 넘었으면 ===
            # Binance Futures: 00/08/16 UTC에 funding 지불/수취
            if now_8h > pos.last_funded_ts and pos.funding_rate:
                fr = self._funding_rates.get(symbol, pos.funding_rate)
                notional = pos.size * price
                # 롱은 +fr일 때 지불, 숏은 +fr일 때 수취 (기본 방향)
                if pos.side == "long":
                    funding_pnl = -notional * fr
                else:
                    funding_pnl = notional * fr
                self.equity += funding_pnl
                pos.last_funded_ts = now_8h
                logger.info(
                    f"[Paper-Funding] {symbol} {pos.side} notional ${notional:.2f} × rate {fr*100:.4f}% "
                    f"= {funding_pnl:+.4f}"
                )

            # === 2. Liquidation 체크 (maintenance margin) ===
            # unrealized PnL이 -(1/leverage - maint_margin) 넘어서면 강제청산
            # 예: 5x 레버리지, maint 0.5% → -(0.20 - 0.005) = -19.5% 손실 시 liq
            if pos.side == "long":
                loss_pct = (pos.entry_price - price) / pos.entry_price
            else:
                loss_pct = (price - pos.entry_price) / pos.entry_price

            liq_threshold = (1.0 / pos.leverage) - self.MAINT_MARGIN_RATE
            if loss_pct > liq_threshold:
                logger.warning(
                    f"[Paper-LIQ] {symbol} {pos.side} 강제청산! "
                    f"손실 {loss_pct*100:.2f}% > liq {liq_threshold*100:.2f}% (lev {pos.leverage}x)"
                )
                auto_closed.append((symbol, price, "liquidation"))
                continue

            # === 3. SL/TP/트레일링 ===
            if pos.side == "long":
                pos.unrealized_pnl = (price - pos.entry_price) / pos.entry_price * pos.leverage
                pos.highest_price = max(pos.highest_price, price)
                profit_pct = (price - pos.entry_price) / pos.entry_price

                # 트레일링 스탑 로직
                if profit_pct >= t_activate and not pos.trailing_activated:
                    pos.trailing_activated = True
                    new_sl = pos.highest_price * (1 - t_distance)
                    pos.stop_loss = max(pos.stop_loss, new_sl)
                    logger.info(f"[Trailing] {symbol}({pos.trade_type}) 트레일링 활성화 | 수익 {profit_pct:.2%} | SL → {pos.stop_loss:.2f}")

                if pos.trailing_activated:
                    new_sl = pos.highest_price * (1 - t_distance)
                    if new_sl > pos.stop_loss + (pos.entry_price * t_step):
                        old_sl = pos.stop_loss
                        pos.stop_loss = new_sl
                        logger.info(f"[Trailing] {symbol}({pos.trade_type}) SL 상향 | {old_sl:.2f} → {pos.stop_loss:.2f}")

                # SL/TP 체크 — Live와 동일하게 현재가로 청산 (SL 레벨 X)
                if price <= pos.stop_loss:
                    reason = "트레일링 SL 도달" if pos.trailing_activated else "SL 도달"
                    auto_closed.append((symbol, price, reason))
                elif price >= pos.take_profit and not pos.trailing_activated:
                    pos.trailing_activated = True
                    pos.stop_loss = pos.entry_price * (1 + t_activate)
                    pos.take_profit = float("inf")
                    logger.info(f"[Trailing] {symbol}({pos.trade_type}) TP 도달 → 트레일링 전환 | 수익 확보선: {pos.stop_loss:.2f}")

            else:  # short
                pos.unrealized_pnl = (pos.entry_price - price) / pos.entry_price * pos.leverage
                pos.lowest_price = min(pos.lowest_price, price) if pos.lowest_price > 0 else price
                profit_pct = (pos.entry_price - price) / pos.entry_price

                if profit_pct >= t_activate and not pos.trailing_activated:
                    pos.trailing_activated = True
                    new_sl = pos.lowest_price * (1 + t_distance)
                    pos.stop_loss = min(pos.stop_loss, new_sl)
                    logger.info(f"[Trailing] {symbol}({pos.trade_type}) 숏 트레일링 활성화 | 수익 {profit_pct:.2%} | SL → {pos.stop_loss:.2f}")

                if pos.trailing_activated:
                    new_sl = pos.lowest_price * (1 + t_distance)
                    if new_sl < pos.stop_loss - (pos.entry_price * t_step):
                        old_sl = pos.stop_loss
                        pos.stop_loss = new_sl
                        logger.info(f"[Trailing] {symbol}({pos.trade_type}) 숏 SL 하향 | {old_sl:.2f} → {pos.stop_loss:.2f}")

                # SL/TP 체크 — Live와 동일하게 현재가
                if price >= pos.stop_loss:
                    reason = "트레일링 SL 도달" if pos.trailing_activated else "SL 도달"
                    auto_closed.append((symbol, price, reason))
                elif price <= pos.take_profit and not pos.trailing_activated:
                    pos.trailing_activated = True
                    pos.stop_loss = pos.entry_price * (1 - t_activate)
                    pos.take_profit = 0.0
                    logger.info(f"[Trailing] {symbol}({pos.trade_type}) 숏 TP 도달 → 트레일링 전환 | 수익 확보선: {pos.stop_loss:.2f}")

        # 루프 밖에서 청산 처리 (dict 변경 안전)
        for symbol, close_price, reason in auto_closed:
            trade = self.close_position(symbol, close_price, reason)
            if trade and self._on_auto_close_callback:
                try:
                    self._on_auto_close_callback(trade)
                except Exception as e:
                    logger.error(f"[Paper] 자동청산 콜백 실패 {symbol}: {e}")

        self.equity_history.append({
            "timestamp": str(datetime.utcnow()),
            "equity": self.equity,
        })

    def get_stats(self) -> dict:
        if not self.trade_history:
            return {"total_trades": 0, "equity": self.equity}
        pnls = [t["pnl"] for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        return {
            "total_trades": len(self.trade_history),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / len(pnls),
            "max_win": max(pnls) if pnls else 0,
            "max_loss": min(pnls) if pnls else 0,
            "equity": self.equity,
            "return_pct": (self.equity - self.initial_capital) / self.initial_capital,
        }
