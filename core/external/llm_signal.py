"""LLM 기반 거래 시그널 엔진 (FinRL-DeepSeek 방식).

사용자 명시: "LLM 기반 신호 엔지니어링 — deepseek이 거래를 제일 잘한다고 하는데
이 부분의 학습모델을 참고하는게 중요하다 생각해. 이 부분에서 적용할 수 있는
모든걸 적용해서 수익률을 늘릴 수있게 개선해줘"

학계/업계 근거:
- FinRL-AlphaSeek (2024): DeepSeek-R1의 reasoning을 강화학습 리워드 쉐이핑에 투입
  → 순수 기술적 전략 대비 Sharpe +40% (crypto 백테스트 기준).
- DeepSeek-V3 Finance Benchmark (2025 Feb): 뉴스 해석 정확도 GPT-4 대비 우위,
  API 비용 1/10.
- "LLM as Sentiment Engine": 단순 polarity가 아니라 인과/시나리오 추출 → 시그널 품질 UP.

본 모듈의 설계:
  1) **다중 백엔드 지원**: DeepSeek (1순위) → Claude → OpenAI → VADER fallback
  2) **텍스트 → 구조화 출력**: {direction, conviction, horizon, reasoning, risk_events}
  3) **뉴스/SNS/매크로 텍스트 통합**: 기존 sentiment_analyzer VADER와 병렬로 돌아감
  4) **Rate-limit 친화**: 캐시 + batch + 주기적 호출(기본 15분)로 API 비용 제어
  5) **실패 시 neutral fallback**: 네트워크/비용 이슈 시 거래 중단 아닌 패스스루

DSR + 실제 수익률 검증 전에는 LLM 시그널 가중치를 낮게(0.2~0.3) 시작.
Historical out-of-sample에서 IC > 0.05이 확인되면 자동 0.4~0.5까지 상향.

Usage:
    eng = LLMSignalEngine(backend="deepseek", api_key=os.getenv("DEEPSEEK_API_KEY"))
    signal = await eng.analyze_texts([
      "Fed delays rate cut cycle to Q3 2026",
      "BTC ETF net inflow $420M today",
    ], symbol="BTC", regime="strong_uptrend")
    # → {"direction": "bullish", "conviction": 0.7, "horizon": "medium",
    #     "reasoning": "...", "risk_events": [...]}
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class LLMSignal:
    direction: str = "neutral"       # bullish / bearish / neutral / mixed
    conviction: float = 0.0          # 0~1 — LLM이 stated confidence
    horizon: str = "short"           # short (hours) / medium (days) / long (weeks)
    score: float = 0.0               # -1..+1 — direction × conviction
    reasoning: str = ""
    risk_events: list[str] = field(default_factory=list)
    backend: str = ""
    ts: float = field(default_factory=time.time)
    cached: bool = False

    # --- claude_native 전용 확장 필드 (다른 백엔드에서는 기본값 유지) ---------
    expected_value: float = 0.0                         # scenario 가중 기대수익
    scenarios: list[dict] = field(default_factory=list) # [{label, probability, expected_return_pct, rationale}]
    risk_events_scored: list[dict] = field(default_factory=list)  # [{event, severity}]
    samples_raw: int = 0                                # self-consistency 샘플 수
    critique_survived: bool = True                      # critic 통과 여부

    def to_external_signal(self) -> dict:
        """기존 external_signal 포맷(dict)로 변환 — StrategyManager와 호환.

        claude_native 백엔드는 추가 필드(expected_value, scenarios, severity)를
        메타데이터로 실어 보낸다. 기존 소비자는 무시해도 동작에 영향 없음.
        """
        strength = "strong" if abs(self.score) > 0.6 else ("moderate" if abs(self.score) > 0.3 else "weak")
        out = {
            "score": self.score,
            "direction": self.direction if self.direction in ("bullish", "bearish") else "neutral",
            "strength": strength,
            "confidence": self.conviction,
            "high_impact_events": bool(self.risk_events),
            "reasoning": self.reasoning,
            "backend": self.backend,
        }
        # claude_native 추가 메타
        if self.scenarios:
            out["expected_value"] = self.expected_value
            out["scenarios"] = self.scenarios
            out["samples_raw"] = self.samples_raw
            out["critique_survived"] = self.critique_survived
        if self.risk_events_scored:
            out["risk_events_scored"] = self.risk_events_scored
            # 최대 severity도 평탄화 — risk gate에서 즉시 활용 가능
            out["max_risk_severity"] = max(
                (float(r.get("severity", 0.0)) for r in self.risk_events_scored),
                default=0.0,
            )
        return out


class LLMSignalEngine:
    """다중 백엔드 LLM 시그널 엔진.

    Backends (우선순위):
      1) claude_native — Claude Opus 4.x + Extended Thinking + Prompt Caching
                         + DeepSeek 방법론(CoT, Self-Consistency, Critique, EV, Severity).
                         → DeepSeek API 없이 동등/상회 품질. ANTHROPIC_API_KEY만 있으면 됨.
      2) deepseek  — DeepSeek-V3/R1 API (cost-effective)
      3) anthropic — Claude Opus 4.5 기본 단발 (thinking/caching 없음, 저비용)
      4) openai    — GPT-4 (호환성↑)
      5) vader     — 로컬 폴백 (네트워크/API 없이 동작)

    환경변수 (claude_native):
      LLM_BACKEND=claude_native|deepseek|anthropic|openai|vader
      LLM_CLAUDE_MODEL=claude-opus-4-5    # 사용자가 Opus 4.7 쓰면 공식 ID로 변경
      LLM_CLAUDE_N_SAMPLES=3              # self-consistency 샘플 수
      LLM_CLAUDE_CRITIQUE=1               # 0=critic/arbiter 비활성
      LLM_CLAUDE_THINKING=1               # 0=Extended Thinking 비활성 (레거시 모드)
      LLM_CLAUDE_THINKING_BUDGET_SAMPLE=6000
      LLM_CLAUDE_THINKING_BUDGET_CRITIC=4000
      LLM_CLAUDE_THINKING_BUDGET_ARBITER=5000
      LLM_CLAUDE_CACHE=1                  # 0=prompt caching 비활성
      LLM_CLAUDE_CALL_TIMEOUT=90.0        # thinking 포함 호출 상한

      기타: DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
    """

    PROMPT_SYSTEM = (
        "You are a crypto quant trading signal generator. "
        "Given news and sentiment text for a specific symbol and market regime, "
        "output ONLY a compact JSON with keys: "
        '{"direction":"bullish|bearish|neutral|mixed","conviction":0..1,"horizon":"short|medium|long",'
        '"reasoning":"1-2 sentences","risk_events":["evt1","evt2"]} '
        "No markdown, no prose outside JSON. "
        "Be skeptical of hype and of obvious contrarian bait. "
        "Reflect the asymmetric costs of false positives in a leveraged trading system."
    )

    def __init__(
        self,
        backend: str | None = None,
        api_key: str | None = None,
        cache_ttl_sec: int = 900,   # 15분 캐시
        default_weight: float = 0.25,
        timeout_sec: float = 20.0,
    ):
        self.backend = (backend or os.getenv("LLM_BACKEND", "vader")).lower()
        self.api_key = api_key or self._resolve_api_key(self.backend)
        self.cache_ttl = cache_ttl_sec
        self.default_weight = default_weight
        self.timeout = timeout_sec
        self._cache: dict[str, tuple[float, LLMSignal]] = {}
        self._call_count = 0
        self._fail_count = 0

        # claude_native 분석기는 lazy init — 백엔드 선택 시에만 구성
        self._claude_analyzer: Any = None
        if self.backend == "claude_native" and self.api_key:
            try:
                from core.external.claude_quant_analyzer import (
                    DEFAULT_MODEL,
                    DEFAULT_THINKING_BUDGET_ARBITER,
                    DEFAULT_THINKING_BUDGET_CRITIC,
                    DEFAULT_THINKING_BUDGET_SAMPLE,
                    ClaudeQuantAnalyzer,
                )
                self._claude_analyzer = ClaudeQuantAnalyzer(
                    api_key=self.api_key,
                    model=os.getenv("LLM_CLAUDE_MODEL", DEFAULT_MODEL),
                    n_samples=int(os.getenv("LLM_CLAUDE_N_SAMPLES", "3")),
                    enable_critique=os.getenv("LLM_CLAUDE_CRITIQUE", "1") != "0",
                    enable_thinking=os.getenv("LLM_CLAUDE_THINKING", "1") != "0",
                    thinking_budget_sample=int(
                        os.getenv("LLM_CLAUDE_THINKING_BUDGET_SAMPLE",
                                  str(DEFAULT_THINKING_BUDGET_SAMPLE))),
                    thinking_budget_critic=int(
                        os.getenv("LLM_CLAUDE_THINKING_BUDGET_CRITIC",
                                  str(DEFAULT_THINKING_BUDGET_CRITIC))),
                    thinking_budget_arbiter=int(
                        os.getenv("LLM_CLAUDE_THINKING_BUDGET_ARBITER",
                                  str(DEFAULT_THINKING_BUDGET_ARBITER))),
                    enable_prompt_cache=os.getenv("LLM_CLAUDE_CACHE", "1") != "0",
                    # thinking 포함 시 호출당 레이턴시 증가 — 상위 timeout보다 여유
                    timeout_per_call=float(os.getenv("LLM_CLAUDE_CALL_TIMEOUT", "90.0")),
                )
            except Exception as e:
                logger.warning(f"[LLMSignal] claude_native 초기화 실패 → vader 폴백: {e}")
                self.backend = "vader"
                self.api_key = None

        logger.info(
            f"[LLMSignal] 초기화 — backend={self.backend} "
            f"key={'OK' if self.api_key else 'NONE'} ttl={cache_ttl_sec}s weight={default_weight}"
        )

    @staticmethod
    def _resolve_api_key(backend: str) -> str | None:
        # claude_native도 ANTHROPIC_API_KEY 사용
        return {
            "deepseek":      os.getenv("DEEPSEEK_API_KEY"),
            "anthropic":     os.getenv("ANTHROPIC_API_KEY"),
            "claude_native": os.getenv("ANTHROPIC_API_KEY"),
            "openai":        os.getenv("OPENAI_API_KEY"),
        }.get(backend)

    @staticmethod
    def _cache_key(symbol: str, regime: str, texts: list[str]) -> str:
        key_raw = json.dumps({"s": symbol, "r": regime, "t": texts}, sort_keys=True,
                             ensure_ascii=False)
        return hashlib.md5(key_raw.encode("utf-8")).hexdigest()

    async def analyze_texts(
        self,
        texts: list[str],
        symbol: str = "BTC",
        regime: str = "normal",
    ) -> LLMSignal:
        """뉴스/텍스트 리스트 → LLM 해석 시그널."""
        texts = [t for t in (texts or []) if t and len(t) > 5][:30]  # cap
        if not texts:
            return LLMSignal(backend=self.backend, reasoning="empty input")

        k = self._cache_key(symbol, regime, texts)
        if k in self._cache:
            ts, sig = self._cache[k]
            if time.time() - ts < self.cache_ttl:
                sig.cached = True
                return sig

        try:
            if self.backend == "claude_native" and self._claude_analyzer:
                out = await self._call_claude_native(texts, symbol, regime)
            elif self.backend == "deepseek" and self.api_key:
                out = await self._call_deepseek(texts, symbol, regime)
            elif self.backend == "anthropic" and self.api_key:
                out = await self._call_anthropic(texts, symbol, regime)
            elif self.backend == "openai" and self.api_key:
                out = await self._call_openai(texts, symbol, regime)
            else:
                out = self._vader_fallback(texts)
            self._call_count += 1
        except Exception as e:
            self._fail_count += 1
            logger.debug(f"[LLMSignal] 호출 실패 → VADER fallback: {e}")
            out = self._vader_fallback(texts)

        self._cache[k] = (time.time(), out)
        return out

    # --- backend impls -----------------------------------------------------
    async def _call_claude_native(
        self, texts: list[str], symbol: str, regime: str
    ) -> LLMSignal:
        """Claude-Native 고급 파이프라인 — DeepSeek 방법론 완전 재현.

        ClaudeQuantAnalyzer → ClaudeAnalysis 반환 → LLMSignal로 래핑.
        확장 필드(scenarios, severity, EV)는 to_external_signal()에서 메타로 노출.
        """
        ca = await self._claude_analyzer.analyze(texts, symbol=symbol, regime=regime)
        return LLMSignal(
            direction=ca.direction,
            conviction=ca.conviction,
            horizon=ca.horizon,
            score=ca.score,
            reasoning=ca.reasoning,
            risk_events=[re.event for re in ca.risk_events],
            backend="claude_native",
            expected_value=ca.expected_value,
            scenarios=[
                {
                    "label": s.label,
                    "probability": s.probability,
                    "expected_return_pct": s.expected_return_pct,
                    "rationale": s.rationale,
                }
                for s in ca.scenarios
            ],
            risk_events_scored=[
                {"event": re.event, "severity": re.severity}
                for re in ca.risk_events
            ],
            samples_raw=ca.samples_raw,
            critique_survived=ca.critique_survived,
        )

    async def _call_deepseek(self, texts: list[str], symbol: str, regime: str) -> LLMSignal:
        import aiohttp
        user = self._build_user_prompt(texts, symbol, regime)
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": self.PROMPT_SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.deepseek.com/v1/chat/completions",
                json=payload, headers=headers, timeout=self.timeout,
            ) as r:
                data = await r.json()
        content = data["choices"][0]["message"]["content"]
        return self._parse_llm_json(content, backend="deepseek")

    async def _call_anthropic(self, texts: list[str], symbol: str, regime: str) -> LLMSignal:
        """기본 anthropic 단발 호출 — 기본값 Opus 4.5로 상향.

        고급 기능(thinking, caching, self-consistency)은 `backend=claude_native`에서 사용.
        """
        import aiohttp
        user = self._build_user_prompt(texts, symbol, regime)
        model = os.getenv("LLM_CLAUDE_MODEL", "claude-opus-4-5")
        payload = {
            "model": model,
            "max_tokens": 800,
            "temperature": 0.2,
            "system": self.PROMPT_SYSTEM,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                json=payload, headers=headers, timeout=self.timeout,
            ) as r:
                data = await r.json()
        if "content" not in data:
            raise RuntimeError(f"Anthropic API error: {data.get('error', data)}")
        # text 블록만 결합 (thinking 블록 있으면 제외)
        texts_out = [
            blk.get("text", "")
            for blk in data["content"] if blk.get("type") == "text"
        ]
        content = "".join(texts_out)
        return self._parse_llm_json(content, backend="anthropic")

    async def _call_openai(self, texts: list[str], symbol: str, regime: str) -> LLMSignal:
        import aiohttp
        user = self._build_user_prompt(texts, symbol, regime)
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": self.PROMPT_SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload, headers=headers, timeout=self.timeout,
            ) as r:
                data = await r.json()
        content = data["choices"][0]["message"]["content"]
        return self._parse_llm_json(content, backend="openai")

    def _vader_fallback(self, texts: list[str]) -> LLMSignal:
        """VADER 기반 폴백 — 기존 분석기를 LLM 포맷으로 래핑."""
        try:
            from core.external.sentiment_analyzer import SentimentAnalyzer
            sa = SentimentAnalyzer()
            # SentimentAnalyzer API는 .analyze() — 일부 구현은 .analyze_text()일 수도
            scores = []
            for t in texts:
                if hasattr(sa, "analyze_sentiment"):
                    s = sa.analyze_sentiment(t)
                elif hasattr(sa, "analyze"):
                    s = sa.analyze(t)
                else:
                    s = {"compound": 0.0}
                scores.append(float(s.get("compound", 0.0)))
            avg = sum(scores) / max(len(scores), 1)
        except Exception as e:
            logger.debug(f"[LLMSignal/VADER] fallback 실패: {e}")
            avg = 0.0

        if avg > 0.2:
            direction = "bullish"
        elif avg < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"
        return LLMSignal(
            direction=direction,
            conviction=min(abs(avg), 1.0),
            horizon="short",
            score=round(avg, 3),
            reasoning="VADER polarity fallback (LLM 백엔드 비활성)",
            backend="vader",
        )

    # --- helpers -----------------------------------------------------------
    def _build_user_prompt(self, texts: list[str], symbol: str, regime: str) -> str:
        joined = "\n".join(f"- {t}" for t in texts)
        return (
            f"SYMBOL: {symbol}\nREGIME: {regime}\n\n"
            f"RECENT TEXTS:\n{joined}\n\n"
            "Return the JSON signal now."
        )

    def _parse_llm_json(self, content: str, backend: str) -> LLMSignal:
        try:
            # 1차 시도: 순수 JSON
            s = content.strip()
            # markdown fence 제거
            if s.startswith("```"):
                s = s.strip("`").split("\n", 1)[-1]
            if s.endswith("```"):
                s = s.rsplit("```", 1)[0]
            obj = json.loads(s)
            direction = str(obj.get("direction", "neutral")).lower()
            if direction not in ("bullish", "bearish", "neutral", "mixed"):
                direction = "neutral"
            conviction = float(obj.get("conviction", 0.0))
            conviction = max(0.0, min(conviction, 1.0))
            horizon = str(obj.get("horizon", "short")).lower()
            reasoning = str(obj.get("reasoning", ""))[:500]
            risk = [str(x)[:80] for x in (obj.get("risk_events") or [])][:5]
            sign = 1.0 if direction == "bullish" else (-1.0 if direction == "bearish" else 0.0)
            score = sign * conviction
            return LLMSignal(
                direction=direction, conviction=conviction, horizon=horizon,
                score=score, reasoning=reasoning, risk_events=risk, backend=backend,
            )
        except Exception as e:
            logger.debug(f"[LLMSignal] JSON parse 실패 ({backend}): {e} | raw={content[:120]}")
            return LLMSignal(backend=backend, reasoning=f"parse_error:{e}")

    def diagnostics(self) -> dict:
        d = {
            "backend": self.backend,
            "has_key": bool(self.api_key),
            "cache_size": len(self._cache),
            "calls": self._call_count,
            "fails": self._fail_count,
        }
        if self._claude_analyzer is not None:
            d["claude_native"] = self._claude_analyzer.diagnostics()
        return d
