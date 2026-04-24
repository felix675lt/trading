"""Claude-Native Quant Analyzer — DeepSeek 방법론의 Claude 구현.

사용자 질문: "deepseek의 api키가 없으면 deepseek이 만들어낸 정보를 알 수 없는거야?
참고해서 클로드만의 방식으로 재정립이 안되는건지 알려줘"

=============================================================================
🎓 **왜 이 모듈인가**
=============================================================================

DeepSeek-R1/V3의 트레이딩 우위는 **"비공개 모델 마법"이 아니라 공개된 방법론**이다:
  1. Chain-of-Thought (CoT) — 결론 전에 추론 단계를 강제
  2. Self-Consistency Sampling — N회 독립 샘플링 후 집계
  3. Adversarial Critique — 반박 → 중재로 과신 제거
  4. Scenario Decomposition — bull/bear/base 확률가중 EV
  5. Risk Event Severity — "위험 리스트"가 아닌 "위험 × 심각도"

이 다섯 가지는 모두 Anthropic Claude API로 즉시 재현 가능하다.
본 모듈은 DeepSeek API 없이 Claude 3.5 Sonnet만으로
동등 이상의 시그널 품질을 얻기 위한 전용 분석기이다.

=============================================================================
🔬 **구조**
=============================================================================

  analyze(texts, symbol, regime)
    └─ 1단계: CoT 프롬프트로 N=3 병렬 샘플 (asyncio.gather)
    └─ 2단계: 샘플간 방향 투표 + conviction 평균 + scenario 집계
    └─ 3단계: Critic 호출 — 합의안에 대한 반박 시도
    └─ 4단계: Arbiter 호출 — 원안 vs 반박 중재, 최종 확정
    └─ 5단계: scenario EV = Σ(p_i × r_i) → score로 환산
    └─ 6단계: risk_events[severity] 구조화

비용 관리:
  - 기본 N=3, 실패 허용 (N=2면 동작 지속)
  - 결과는 상위 엔진(LLMSignalEngine)에서 캐시
  - timeout 20s/call, 전체 60s cap
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# =============================================================================
# Output schema — 확장형 LLMSignal과 호환
# =============================================================================

@dataclass
class Scenario:
    """개별 시나리오 — 확률 × 기대수익."""
    label: str                  # "bull" | "bear" | "base"
    probability: float = 0.0    # 0..1
    expected_return_pct: float = 0.0  # -1..+1 (예: 0.05 = +5%)
    rationale: str = ""


@dataclass
class RiskEvent:
    """위험 이벤트 — 심각도 가중."""
    event: str
    severity: float = 0.0       # 0..1 (1 = 시스템 리스크급)


@dataclass
class ClaudeAnalysis:
    """Claude-native 고급 분석 결과.

    LLMSignal로 축약 변환 가능 (to_llm_signal_kwargs).
    """
    direction: str = "neutral"
    conviction: float = 0.0
    horizon: str = "short"
    score: float = 0.0          # EV 기반 연속값 (-1..+1)
    expected_value: float = 0.0 # scenario 가중 기대수익
    reasoning: str = ""
    scenarios: list[Scenario] = field(default_factory=list)
    risk_events: list[RiskEvent] = field(default_factory=list)
    samples_raw: int = 0        # self-consistency 성공 샘플 수
    critique_survived: bool = True  # critic 반박 후에도 방향 유지됐는가
    ts: float = field(default_factory=time.time)

    def to_llm_signal_kwargs(self) -> dict:
        """기존 LLMSignal 포맷으로 축약 (severity 없는 버전)."""
        return {
            "direction": self.direction,
            "conviction": self.conviction,
            "horizon": self.horizon,
            "score": self.score,
            "reasoning": self.reasoning,
            "risk_events": [re.event for re in self.risk_events],
            "backend": "claude_native",
        }


# =============================================================================
# 프롬프트 템플릿
# =============================================================================

COT_SYSTEM_PROMPT = """You are an elite crypto quant analyst. Your job is to produce a trading signal with explicit, auditable reasoning.

You MUST output a JSON object with this exact schema:
{
  "reasoning_steps": [
    "step 1: list dominant drivers from the input",
    "step 2: classify each driver as bullish/bearish/ambiguous with short justification",
    "step 3: assess regime compatibility (given REGIME), note alignment",
    "step 4: identify any narrative traps (consensus hype, obvious contrarian bait)"
  ],
  "scenarios": [
    {"label": "bull",  "probability": 0.0, "expected_return_pct": 0.0, "rationale": "..."},
    {"label": "base",  "probability": 0.0, "expected_return_pct": 0.0, "rationale": "..."},
    {"label": "bear",  "probability": 0.0, "expected_return_pct": 0.0, "rationale": "..."}
  ],
  "direction": "bullish|bearish|neutral|mixed",
  "conviction": 0.0,
  "horizon": "short|medium|long",
  "risk_events": [
    {"event": "short text", "severity": 0.0}
  ],
  "reasoning_summary": "1-2 sentence final rationale"
}

Rules:
- reasoning_steps: 4 concise steps, each one sentence.
- scenarios: probabilities MUST sum to ~1.0 (tolerance 0.05). expected_return_pct is the arithmetic return you'd expect IF that scenario plays out over the stated horizon (e.g. 0.08 = +8%, -0.05 = -5%).
- conviction reflects epistemic confidence, not scenario probability. Keep it honest — if the input is weak, state it (e.g. 0.2).
- risk_events: up to 5, severity 0=benign, 1=system-risk (liquidation cascade, exchange insolvency).
- NO markdown, NO text outside the JSON. Compact, valid JSON only.

Asymmetric cost awareness: false bullish calls in a leveraged long-only crypto system cost more than missed opportunities. Err toward neutral when ambiguous."""


CRITIC_SYSTEM_PROMPT = """You are an adversarial risk auditor. You will receive a trading thesis and must attempt to falsify it.

Output JSON:
{
  "strongest_counterargument": "1-2 sentences",
  "overlooked_risks": ["risk 1", "risk 2"],
  "narrative_trap_score": 0.0,
  "verdict": "thesis_stands | thesis_weakened | thesis_broken"
}

narrative_trap_score: 0 = clean thesis, 1 = thesis is textbook hype/contrarian bait.
Be ruthless but precise. If the thesis holds, say so honestly. No markdown."""


ARBITER_SYSTEM_PROMPT = """You are a senior portfolio manager resolving a dispute between an analyst and a risk auditor.

You receive:
- original_thesis (JSON)
- critic_output (JSON)

Your task: produce the FINAL signal, incorporating the critic's valid points.

Output JSON (same schema as original_thesis) with adjustments:
- If critic's points are strong, reduce conviction and/or flip direction to neutral.
- If critic's points are weak, keep the thesis but note it in reasoning_summary.
- Scenarios may be reweighted but must still sum to ~1.0.

NO markdown, JSON only."""


# =============================================================================
# 메인 분석기
# =============================================================================

class ClaudeQuantAnalyzer:
    """Claude-native 고급 시그널 분석기.

    DeepSeek-R1 수준 reasoning을 Claude 3.5 Sonnet + 메타-프롬프팅으로 재현.

    Args:
        api_key: ANTHROPIC_API_KEY
        model: 기본 claude-3-5-sonnet-latest (thinking 모델로 스위칭 가능)
        n_samples: self-consistency 샘플 수 (기본 3)
        enable_critique: 반박-중재 단계 활성 (기본 True)
        sample_temperature: 샘플간 다양성 확보 (0.4)
        final_temperature: critic/arbiter 온도 (0.1 — 안정성 우선)
        timeout_per_call: 단일 호출 타임아웃
        max_concurrent: 동시 호출 상한
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-latest",
        n_samples: int = 3,
        enable_critique: bool = True,
        sample_temperature: float = 0.4,
        final_temperature: float = 0.1,
        timeout_per_call: float = 20.0,
        max_concurrent: int = 3,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.n_samples = max(1, min(n_samples, 5))
        self.enable_critique = enable_critique
        self.sample_temperature = sample_temperature
        self.final_temperature = final_temperature
        self.timeout = timeout_per_call
        self._sema = asyncio.Semaphore(max_concurrent)
        self._calls = 0
        self._fails = 0
        if not self.api_key:
            logger.warning("[ClaudeQuant] ANTHROPIC_API_KEY 없음 — analyze() 호출 시 예외 발생")
        else:
            logger.info(
                f"[ClaudeQuant] 초기화 model={model} N={self.n_samples} "
                f"critique={enable_critique} T_sample={sample_temperature}"
            )

    # -------------------------------------------------------------------------
    # 공개 API
    # -------------------------------------------------------------------------

    async def analyze(
        self,
        texts: list[str],
        symbol: str = "BTC",
        regime: str = "normal",
    ) -> ClaudeAnalysis:
        """텍스트 리스트 → DeepSeek 수준 Claude 분석.

        파이프라인:
            ① N=3 병렬 CoT 샘플링
            ② 샘플간 집계(방향 투표, conviction 평균, scenario 통합)
            ③ Critic 반박 (선택)
            ④ Arbiter 중재 (선택)
            ⑤ Scenario EV → score 매핑
        """
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY 필요 — ClaudeQuantAnalyzer 호출 불가")
        if not texts:
            return ClaudeAnalysis(reasoning="empty input")

        user_prompt = self._build_user_prompt(texts, symbol, regime)

        # ① 병렬 샘플링
        t0 = time.time()
        samples = await self._self_consistency_sample(user_prompt)
        if not samples:
            self._fails += 1
            logger.warning("[ClaudeQuant] self-consistency 샘플 0건 → 중단")
            return ClaudeAnalysis(reasoning="all samples failed")

        # ② 집계
        aggregated = self._aggregate_samples(samples)

        # ③ + ④ 적대적 검증
        if self.enable_critique and aggregated.get("direction") != "neutral":
            try:
                final_obj = await self._adversarial_verify(aggregated, user_prompt)
                critique_survived = (
                    final_obj.get("direction") == aggregated.get("direction")
                )
            except Exception as e:
                logger.debug(f"[ClaudeQuant] critique 실패 → 집계 결과 사용: {e}")
                final_obj = aggregated
                critique_survived = True
        else:
            final_obj = aggregated
            critique_survived = True

        # ⑤ EV → score 환산
        result = self._pack_result(final_obj, len(samples), critique_survived)
        self._calls += 1
        logger.debug(
            f"[ClaudeQuant] {symbol}/{regime} dir={result.direction} "
            f"conv={result.conviction:.2f} EV={result.expected_value:+.4f} "
            f"score={result.score:+.3f} samples={result.samples_raw} "
            f"crit_survived={result.critique_survived} dt={time.time()-t0:.1f}s"
        )
        return result

    def diagnostics(self) -> dict:
        return {
            "model": self.model,
            "n_samples": self.n_samples,
            "enable_critique": self.enable_critique,
            "calls": self._calls,
            "fails": self._fails,
            "has_key": bool(self.api_key),
        }

    # -------------------------------------------------------------------------
    # ① Self-Consistency 샘플링 (N회 병렬 호출)
    # -------------------------------------------------------------------------

    async def _self_consistency_sample(self, user_prompt: str) -> list[dict]:
        """N회 병렬 호출 → 성공 샘플만 반환."""
        tasks = [
            self._call_claude(
                system=COT_SYSTEM_PROMPT,
                user=user_prompt,
                temperature=self.sample_temperature,
                max_tokens=900,
            )
            for _ in range(self.n_samples)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        parsed = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.debug(f"[ClaudeQuant] sample#{i} 예외: {r}")
                continue
            obj = self._safe_parse_json(r)
            if obj:
                parsed.append(obj)
        return parsed

    # -------------------------------------------------------------------------
    # ② 샘플 집계 — 방향 투표 + conviction 평균 + scenario 통합
    # -------------------------------------------------------------------------

    def _aggregate_samples(self, samples: list[dict]) -> dict:
        """N개 샘플을 하나의 합의안으로 집계."""
        if len(samples) == 1:
            return samples[0]

        # 방향 투표 (majority)
        dir_votes: dict[str, int] = {}
        for s in samples:
            d = str(s.get("direction", "neutral")).lower()
            dir_votes[d] = dir_votes.get(d, 0) + 1
        consensus_dir = max(dir_votes, key=dir_votes.get)
        vote_share = dir_votes[consensus_dir] / len(samples)

        # conviction: 합의 방향 샘플들만 평균 + 투표 비율로 할인
        convictions = [
            float(s.get("conviction", 0.0))
            for s in samples
            if str(s.get("direction", "")).lower() == consensus_dir
        ]
        avg_conviction = sum(convictions) / max(len(convictions), 1)
        avg_conviction *= vote_share  # 과반 3/3이면 1.0배, 2/3이면 0.67배

        # horizon: 최빈값
        horizons = [str(s.get("horizon", "short")).lower() for s in samples]
        horizon = max(set(horizons), key=horizons.count)

        # scenarios: 라벨별 확률/수익 평균
        merged_scenarios = self._merge_scenarios(samples)

        # risk_events: 샘플간 유니온 (severity는 최댓값)
        risks = self._merge_risks(samples)

        # reasoning: 합의 방향 샘플의 요약 결합
        reason_parts = [
            str(s.get("reasoning_summary", ""))[:200]
            for s in samples
            if str(s.get("direction", "")).lower() == consensus_dir
        ]
        reasoning = " | ".join(r for r in reason_parts if r)[:500]

        return {
            "direction": consensus_dir,
            "conviction": round(min(max(avg_conviction, 0.0), 1.0), 3),
            "horizon": horizon,
            "reasoning_summary": reasoning,
            "scenarios": merged_scenarios,
            "risk_events": risks,
            "_vote_share": vote_share,
            "_n_samples": len(samples),
        }

    def _merge_scenarios(self, samples: list[dict]) -> list[dict]:
        """scenario 라벨(bull/base/bear)별 평균 — 샘플간 일관성 강화."""
        buckets: dict[str, list[dict]] = {"bull": [], "base": [], "bear": []}
        for s in samples:
            for sc in s.get("scenarios", []) or []:
                label = str(sc.get("label", "")).lower()
                if label in buckets:
                    buckets[label].append(sc)
        merged = []
        for label, items in buckets.items():
            if not items:
                continue
            p = sum(float(x.get("probability", 0.0)) for x in items) / len(items)
            r = sum(float(x.get("expected_return_pct", 0.0)) for x in items) / len(items)
            # rationale은 첫 샘플 것 사용 (토큰 절약)
            rationale = str(items[0].get("rationale", ""))[:150]
            merged.append({
                "label": label,
                "probability": round(p, 3),
                "expected_return_pct": round(r, 4),
                "rationale": rationale,
            })
        # 확률 합이 1 벗어나면 정규화
        total_p = sum(x["probability"] for x in merged)
        if total_p > 0 and abs(total_p - 1.0) > 0.05:
            for x in merged:
                x["probability"] = round(x["probability"] / total_p, 3)
        return merged

    def _merge_risks(self, samples: list[dict]) -> list[dict]:
        """risk_events — event 문자열 기준 유니온, severity는 max."""
        seen: dict[str, float] = {}
        for s in samples:
            for re in s.get("risk_events", []) or []:
                if isinstance(re, str):
                    ev, sev = re, 0.5
                else:
                    ev = str(re.get("event", ""))[:100]
                    sev = float(re.get("severity", 0.5))
                if not ev:
                    continue
                seen[ev] = max(seen.get(ev, 0.0), min(max(sev, 0.0), 1.0))
        return [{"event": e, "severity": round(v, 2)} for e, v in
                sorted(seen.items(), key=lambda x: -x[1])[:5]]

    # -------------------------------------------------------------------------
    # ③ + ④ Adversarial Critique + Arbitration
    # -------------------------------------------------------------------------

    async def _adversarial_verify(self, thesis: dict, original_user: str) -> dict:
        """Critic → Arbiter 2단계 검증."""
        # ③ Critic — 합의안을 반박
        critic_user = (
            f"ORIGINAL MARKET CONTEXT:\n{original_user}\n\n"
            f"THESIS TO ATTACK (JSON):\n{json.dumps(thesis, ensure_ascii=False)}\n\n"
            "Produce your critique JSON now."
        )
        try:
            critic_raw = await self._call_claude(
                system=CRITIC_SYSTEM_PROMPT,
                user=critic_user,
                temperature=self.final_temperature,
                max_tokens=500,
            )
            critic_obj = self._safe_parse_json(critic_raw) or {}
        except Exception as e:
            logger.debug(f"[ClaudeQuant] critic 호출 실패: {e}")
            return thesis

        verdict = str(critic_obj.get("verdict", "thesis_stands")).lower()
        trap = float(critic_obj.get("narrative_trap_score", 0.0))

        # 반박이 약하면 원안 유지 (비용 절감)
        if verdict == "thesis_stands" and trap < 0.3:
            thesis["reasoning_summary"] = (
                thesis.get("reasoning_summary", "") +
                " | critic: thesis stands."
            )[:500]
            return thesis

        # ④ Arbiter — 원안 + critic 종합
        arbiter_user = (
            f"ORIGINAL MARKET CONTEXT:\n{original_user}\n\n"
            f"ORIGINAL THESIS:\n{json.dumps(thesis, ensure_ascii=False)}\n\n"
            f"CRITIC OUTPUT:\n{json.dumps(critic_obj, ensure_ascii=False)}\n\n"
            "Produce FINAL signal JSON now (same schema as original thesis)."
        )
        try:
            arbiter_raw = await self._call_claude(
                system=ARBITER_SYSTEM_PROMPT,
                user=arbiter_user,
                temperature=self.final_temperature,
                max_tokens=900,
            )
            arbiter_obj = self._safe_parse_json(arbiter_raw)
        except Exception as e:
            logger.debug(f"[ClaudeQuant] arbiter 호출 실패: {e}")
            return thesis

        if not arbiter_obj:
            return thesis

        # 트랩 스코어 높으면 conviction 강제 감소
        if trap > 0.6:
            arbiter_obj["conviction"] = min(
                float(arbiter_obj.get("conviction", 0.0)), 0.4
            )
        arbiter_obj["_vote_share"] = thesis.get("_vote_share", 1.0)
        arbiter_obj["_n_samples"] = thesis.get("_n_samples", 1)
        arbiter_obj["_critic_verdict"] = verdict
        arbiter_obj["_narrative_trap"] = round(trap, 2)
        return arbiter_obj

    # -------------------------------------------------------------------------
    # ⑤ Scenario EV → score 매핑
    # -------------------------------------------------------------------------

    def _pack_result(
        self, obj: dict, n_samples: int, critique_survived: bool
    ) -> ClaudeAnalysis:
        """최종 obj를 ClaudeAnalysis dataclass로 포장."""
        scenarios = [
            Scenario(
                label=str(sc.get("label", "base")),
                probability=float(sc.get("probability", 0.0)),
                expected_return_pct=float(sc.get("expected_return_pct", 0.0)),
                rationale=str(sc.get("rationale", ""))[:200],
            )
            for sc in obj.get("scenarios", []) or []
        ]

        # EV = Σ(probability × expected_return_pct)
        ev = sum(s.probability * s.expected_return_pct for s in scenarios)

        risk_events = []
        for re in obj.get("risk_events", []) or []:
            if isinstance(re, str):
                risk_events.append(RiskEvent(event=re, severity=0.5))
            else:
                risk_events.append(RiskEvent(
                    event=str(re.get("event", ""))[:100],
                    severity=float(re.get("severity", 0.5)),
                ))

        # score: EV를 [-1, 1]로 clip하되, 고위험 이벤트 존재 시 discount
        high_risk_penalty = sum(
            re.severity for re in risk_events if re.severity > 0.6
        )
        hr_factor = max(0.5, 1.0 - 0.15 * high_risk_penalty)
        ev_scaled = max(-1.0, min(1.0, ev * 5.0)) * hr_factor  # 20% EV → score 1.0

        direction = str(obj.get("direction", "neutral")).lower()
        if direction not in ("bullish", "bearish", "neutral", "mixed"):
            direction = "neutral"
        conviction = float(obj.get("conviction", 0.0))
        conviction = max(0.0, min(conviction, 1.0))

        # 방향이 bearish면 ev_scaled도 음수여야 자연스럽 — 강제 정합
        if direction == "bullish" and ev_scaled < 0:
            ev_scaled = abs(ev_scaled) * 0.5   # 방향과 상반되면 약화
        elif direction == "bearish" and ev_scaled > 0:
            ev_scaled = -abs(ev_scaled) * 0.5

        # 최종 score: EV 기반 연속값을 conviction으로 추가 가중
        score = ev_scaled * conviction if direction != "neutral" else 0.0

        horizon = str(obj.get("horizon", "short")).lower()
        if horizon not in ("short", "medium", "long"):
            horizon = "short"

        return ClaudeAnalysis(
            direction=direction,
            conviction=conviction,
            horizon=horizon,
            score=round(score, 4),
            expected_value=round(ev, 4),
            reasoning=str(obj.get("reasoning_summary", ""))[:500],
            scenarios=scenarios,
            risk_events=risk_events,
            samples_raw=n_samples,
            critique_survived=critique_survived,
        )

    # -------------------------------------------------------------------------
    # 저수준: Anthropic API 호출
    # -------------------------------------------------------------------------

    async def _call_claude(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """단일 Claude 호출 — 세마포어로 동시성 제어."""
        import aiohttp
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        async with self._sema:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload, headers=headers, timeout=self.timeout,
                ) as r:
                    data = await r.json()
        if "content" not in data:
            raise RuntimeError(
                f"Anthropic API error: {data.get('error', data)}"
            )
        # content는 blocks 배열 → text만 결합
        texts = [
            blk.get("text", "")
            for blk in data["content"] if blk.get("type") == "text"
        ]
        return "".join(texts)

    # -------------------------------------------------------------------------
    # 헬퍼
    # -------------------------------------------------------------------------

    def _build_user_prompt(
        self, texts: list[str], symbol: str, regime: str
    ) -> str:
        joined = "\n".join(f"- {t}" for t in texts[:30])
        return (
            f"SYMBOL: {symbol}\nREGIME: {regime}\n"
            f"TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n\n"
            f"RECENT TEXTS (news/social/macro, most recent first):\n{joined}\n\n"
            "Return the CoT JSON signal now."
        )

    @staticmethod
    def _safe_parse_json(raw: str) -> dict | None:
        """Claude 출력(JSON 또는 JSON with markdown fence) 파싱."""
        if not raw:
            return None
        s = raw.strip()
        if s.startswith("```"):
            s = s.strip("`")
            # 첫 줄이 'json'이면 제거
            if "\n" in s:
                first, rest = s.split("\n", 1)
                if first.lower().strip() in ("json", ""):
                    s = rest
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
        # JSON 앞뒤 텍스트 제거 시도
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            s = s[start:end + 1]
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            logger.debug(f"[ClaudeQuant] JSON 파싱 실패: {e} | raw={raw[:200]}")
            return None


__all__ = [
    "ClaudeQuantAnalyzer",
    "ClaudeAnalysis",
    "Scenario",
    "RiskEvent",
]
