"""Capital Tier System — 시드 구간별 기능 자동 활성화.

핵심 아이디어:
- LIVE 잔고가 커지면 상위 티어 기능이 자동으로 켜진다 (코드 수정 없이).
- PAPER는 paper_virtual_seed(예: $10K)로 상위 티어 기능을 실시간 시장에서
  선행 검증한다 — 실 자본 투입 전에 Kelly/HMM/메타라벨링을 paper로 증명.

사용:
    tm = CapitalTierManager(config)
    tm.update(live_equity=100, paper_equity=10500)
    if tm.feature_enabled("kelly_enabled", mode="paper"):
        # paper는 Kelly, live는 confidence 방식 병행 가능
        ...
    symbols = tm.get_feature("symbols", mode="live", default=[])
"""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class CapitalTier:
    """단일 티어 정의"""
    name: str
    min_equity: float
    max_equity: float  # float('inf') for top tier
    description: str
    features: dict[str, Any] = field(default_factory=dict)

    def contains(self, equity: float) -> bool:
        return self.min_equity <= equity < self.max_equity


class CapitalTierManager:
    """시드 구간별 기능 자동 활성화.

    - LIVE: 실제 잔고(self.live_equity)로 티어 결정
    - PAPER: paper_virtual_seed (config 지정) 또는 paper_equity 실값으로 결정
    - 각 루프에서 update() 호출 → 티어 변경 시 로그
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 최상위 config dict (`capital_tiers` 키 사용)
        """
        cfg = config.get("capital_tiers", {})
        self.paper_virtual_seed = float(cfg.get("paper_virtual_seed", 10000))
        self.paper_use_virtual = cfg.get("paper_use_virtual", True)
        raw_tiers = cfg.get("tiers") or self._default_tiers()
        self.tiers: list[CapitalTier] = self._parse_tiers(raw_tiers)

        # === Mode별 심볼 오버라이드 (2026-04-18) ===
        # trading.paper_symbols_override / trading.live_symbols_override
        # 설정되면 해당 mode 심볼을 완전히 교체 (티어 심볼 무시)
        trading_cfg = config.get("trading", {}) or {}
        self.symbol_overrides: dict[str, list[str]] = {}
        po = trading_cfg.get("paper_symbols_override")
        lo = trading_cfg.get("live_symbols_override")
        if po:
            self.symbol_overrides["paper"] = list(po)
        if lo:
            self.symbol_overrides["live"] = list(lo)

        # 상태
        self.live_equity: float = 0.0
        self.paper_equity: float = 0.0
        self._last_live_tier: str | None = None
        self._last_paper_tier: str | None = None

        logger.info(
            f"[CapitalTier] 초기화 | 티어 {len(self.tiers)}개 | "
            f"PAPER 가상시드 ${self.paper_virtual_seed:,.0f} (use={self.paper_use_virtual})"
        )
        if self.symbol_overrides:
            logger.warning(
                f"[CapitalTier] 심볼 오버라이드 활성: "
                f"PAPER={self.symbol_overrides.get('paper', '티어기본')} | "
                f"LIVE={self.symbol_overrides.get('live', '티어기본')}"
            )

    # ---------------------------------------------------------------------
    # 티어 정의
    # ---------------------------------------------------------------------

    def _default_tiers(self) -> list[dict]:
        """config 누락 시 fallback 기본 티어"""
        return [
            {
                "name": "micro",
                "min": 0,
                "max": 500,
                "description": "소시드 집중매매 ($0~$500) — 1포지션 풀시드",
                "features": {
                    "symbols": ["ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT"],
                    "max_positions": 1,
                    "concentration_mode": True,
                    "sizing_method": "confidence",
                    "order_routing": "market_only",
                    "kelly_enabled": False,
                    "limit_order_pref": False,
                    "hmm_regime": False,
                    "meta_labeling": False,
                    "walk_forward_cv": False,
                    "cvar_risk": False,
                    "max_leverage": 5,
                },
            },
            {
                "name": "small",
                "min": 500,
                "max": 2000,
                "description": "소형 시드 ($500~$2K) — 2포지션 + 제한주문",
                "features": {
                    "symbols": ["ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT"],
                    "max_positions": 2,
                    "concentration_mode": False,
                    "sizing_method": "confidence",
                    "order_routing": "limit_first",
                    "kelly_enabled": False,
                    "limit_order_pref": True,
                    "hmm_regime": False,
                    "meta_labeling": False,
                    "walk_forward_cv": True,
                    "cvar_risk": False,
                    "max_leverage": 5,
                },
            },
            {
                "name": "mid",
                "min": 2000,
                "max": 10000,
                "description": "중형 시드 ($2K~$10K) — Kelly + BTC",
                "features": {
                    "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT"],
                    "max_positions": 3,
                    "concentration_mode": False,
                    "sizing_method": "kelly_fractional",
                    "order_routing": "limit_first",
                    "kelly_enabled": True,
                    "kelly_fraction": 0.25,
                    "limit_order_pref": True,
                    "hmm_regime": False,
                    "meta_labeling": False,
                    "walk_forward_cv": True,
                    "cvar_risk": True,
                    "max_leverage": 4,
                },
            },
            {
                "name": "large",
                "min": 10000,
                "max": 50000,
                "description": "대형 시드 ($10K~$50K) — HMM+메타라벨링+TWAP",
                "features": {
                    "symbols": [
                        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                        "DOGE/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
                    ],
                    "max_positions": 5,
                    "concentration_mode": False,
                    "sizing_method": "kelly_fractional",
                    "order_routing": "twap",
                    "kelly_enabled": True,
                    "kelly_fraction": 0.30,
                    "limit_order_pref": True,
                    "hmm_regime": True,
                    "meta_labeling": True,
                    "walk_forward_cv": True,
                    "cvar_risk": True,
                    "max_leverage": 3,
                },
            },
            {
                "name": "pro",
                "min": 50000,
                "max": float("inf"),
                "description": "프로 ($50K+) — 풀 포트폴리오 + 통계차익",
                "features": {
                    "symbols": [
                        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                        "DOGE/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
                        "AVAX/USDT:USDT", "ARB/USDT:USDT",
                    ],
                    "max_positions": 8,
                    "concentration_mode": False,
                    "sizing_method": "hrp",
                    "order_routing": "smart",
                    "kelly_enabled": True,
                    "kelly_fraction": 0.35,
                    "limit_order_pref": True,
                    "hmm_regime": True,
                    "meta_labeling": True,
                    "walk_forward_cv": True,
                    "cvar_risk": True,
                    "stat_arb_pairs": True,
                    "max_leverage": 3,
                },
            },
        ]

    def _parse_tiers(self, raw: list[dict]) -> list[CapitalTier]:
        """dict 리스트 → CapitalTier 객체 리스트 (min_equity 오름차순)"""
        tiers = []
        for r in raw:
            max_raw = r.get("max", float("inf"))
            # YAML의 .inf 또는 "inf" 문자열 처리
            if isinstance(max_raw, str) and max_raw.lower() in ("inf", "infinity"):
                max_val = float("inf")
            else:
                max_val = float(max_raw)
            tiers.append(CapitalTier(
                name=r["name"],
                min_equity=float(r.get("min", 0)),
                max_equity=max_val,
                description=r.get("description", ""),
                features=r.get("features", {}),
            ))
        tiers.sort(key=lambda t: t.min_equity)
        return tiers

    # ---------------------------------------------------------------------
    # 상태 업데이트
    # ---------------------------------------------------------------------

    def _tier_for(self, equity: float) -> CapitalTier:
        """equity 값에 해당하는 티어 반환"""
        for t in self.tiers:
            if t.contains(equity):
                return t
        return self.tiers[-1]

    def update(self, live_equity: float, paper_equity: float) -> dict:
        """매 루프 호출 — 티어 변경 감지 + 로그.

        Returns:
            변경 dict: {"live": (old, new), "paper": (old, new)} — 변경 없으면 빈 dict
        """
        self.live_equity = max(0.0, float(live_equity))
        self.paper_equity = max(0.0, float(paper_equity))

        live_tier = self._tier_for(self.live_equity).name
        paper_ref = (
            self.paper_virtual_seed if self.paper_use_virtual else self.paper_equity
        )
        paper_tier = self._tier_for(paper_ref).name

        changes: dict = {}

        if live_tier != self._last_live_tier:
            if self._last_live_tier is not None:
                logger.warning(
                    f"[CapitalTier] 🎯 LIVE 티어 변경: {self._last_live_tier} → {live_tier} "
                    f"(equity=${self.live_equity:,.2f})"
                )
            changes["live"] = (self._last_live_tier, live_tier)
            self._last_live_tier = live_tier

        if paper_tier != self._last_paper_tier:
            if self._last_paper_tier is not None:
                logger.warning(
                    f"[CapitalTier] 📊 PAPER 티어 변경: {self._last_paper_tier} → {paper_tier} "
                    f"(ref=${paper_ref:,.2f}, virtual={self.paper_use_virtual})"
                )
            changes["paper"] = (self._last_paper_tier, paper_tier)
            self._last_paper_tier = paper_tier

        return changes

    # ---------------------------------------------------------------------
    # 조회 API
    # ---------------------------------------------------------------------

    def get_tier(self, mode: str = "live") -> CapitalTier:
        """mode별 현재 티어 반환 (mode in {'live', 'paper'})"""
        if mode == "live":
            return self._tier_for(self.live_equity)
        # paper
        eq = self.paper_virtual_seed if self.paper_use_virtual else self.paper_equity
        return self._tier_for(eq)

    def feature_enabled(self, feature: str, mode: str = "live") -> bool:
        """기능 on/off — 현재 mode 티어에서 해당 기능이 활성화됐는가?"""
        tier = self.get_tier(mode)
        return bool(tier.features.get(feature, False))

    def get_feature(self, feature: str, mode: str = "live", default: Any = None) -> Any:
        """기능 값 조회 (bool/list/dict/float 모두 지원)"""
        tier = self.get_tier(mode)
        return tier.features.get(feature, default)

    def get_symbols(self, mode: str = "live") -> list[str]:
        """mode 티어가 허용하는 심볼 목록.

        오버라이드가 설정되어 있으면 티어 심볼을 무시하고 오버라이드 사용.
        (mode별로 집중 유니버스를 강제하고 싶을 때 — PAPER=BTC/ETH, LIVE=알트 등)
        """
        if mode in self.symbol_overrides:
            return list(self.symbol_overrides[mode])
        return list(self.get_feature("symbols", mode, []))

    def allowed_symbol(self, symbol: str, mode: str = "live") -> bool:
        """symbol이 mode 티어에서 거래 가능한가?"""
        return symbol in self.get_symbols(mode)

    def union_symbols(self) -> list[str]:
        """LIVE + PAPER 티어 심볼의 합집합 (데이터 수집용)"""
        return sorted(set(self.get_symbols("live")) | set(self.get_symbols("paper")))

    def next_tier(self, mode: str = "live") -> CapitalTier | None:
        """다음 티어 (없으면 None)"""
        current = self.get_tier(mode)
        for i, t in enumerate(self.tiers):
            if t.name == current.name and i + 1 < len(self.tiers):
                return self.tiers[i + 1]
        return None

    # ---------------------------------------------------------------------
    # 대시보드 리포트
    # ---------------------------------------------------------------------

    def status_report(self) -> dict:
        """대시보드/API용 상태 요약"""
        live_tier = self.get_tier("live")
        paper_tier = self.get_tier("paper")
        live_next = self.next_tier("live")

        def _progress(tier: CapitalTier, equity: float) -> float:
            if tier.max_equity == float("inf"):
                return 100.0
            span = tier.max_equity - tier.min_equity
            if span <= 0:
                return 100.0
            return max(0.0, min(100.0, (equity - tier.min_equity) / span * 100))

        return {
            "live": {
                "equity": round(self.live_equity, 2),
                "tier": live_tier.name,
                "description": live_tier.description,
                "symbols": live_tier.features.get("symbols", []),
                "features": live_tier.features,
                "next_tier": live_next.name if live_next else None,
                "next_threshold": (
                    live_next.min_equity if live_next else None
                ),
                "next_needed": (
                    round(live_next.min_equity - self.live_equity, 2)
                    if live_next else 0.0
                ),
                "progress_pct": round(_progress(live_tier, self.live_equity), 1),
            },
            "paper": {
                "equity": round(self.paper_equity, 2),
                "virtual_seed": self.paper_virtual_seed,
                "use_virtual": self.paper_use_virtual,
                "tier": paper_tier.name,
                "description": paper_tier.description,
                "symbols": paper_tier.features.get("symbols", []),
                "features": paper_tier.features,
            },
            "all_tiers": [
                {
                    "name": t.name,
                    "min": t.min_equity,
                    "max": (t.max_equity if t.max_equity != float("inf") else None),
                    "description": t.description,
                }
                for t in self.tiers
            ],
        }
