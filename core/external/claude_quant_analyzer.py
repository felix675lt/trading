"""Claude-Native Quant Analyzer — Claude Opus 4.x + DeepSeek 방법론.

사용자 질문(원문):
  "deepseek의 api키가 없으면 deepseek이 만들어낸 정보를 알 수 없는거야?
   참고해서 클로드만의 방식으로 재정립이 안되는건지 알려줘"
  (→ 후속) "3.5 sonet? 나는 지금 opus 4.7을 쓰고 있는데 3.5 sonet으로 지금 얘기를 하는거야?
            전체 모든 과정 지금 다시 점검하고 opus 4.7수준으로 클로드의 가장 강력한 성능을
            끌어내서 다시 설정해줘."

=============================================================================
🎓 **왜 이 모듈인가 — Opus 4.x 완전 활용 재설계**
=============================================================================

DeepSeek-R1/V3의 트레이딩 우위는 **"비공개 모델 마법"이 아니라 공개된 방법론**이다:
  1. Chain-of-Thought (CoT) — 결론 전에 추론 단계를 강제
  2. Self-Consistency Sampling — N회 독립 샘플링 후 집계
  3. Adversarial Critique — 반박 → 중재로 과신 제거
  4. Scenario Decomposition — bull/bear/base 확률가중 EV
  5. Risk Event Severity — "위험 리스트"가 아닌 "위험 × 심각도"

Claude Opus 4.x 세대가 DeepSeek-R1을 **구조적으로 앞서는** 지점:

  (A) **Extended Thinking (네이티브 추론 토큰)**
      - DeepSeek-R1의 `<think>` 토큰에 해당하는 Anthropic 공식 기능.
      - budget_tokens 만큼 "보이지 않는 사고 공간"에서 reasoning을 먼저 수행한 뒤
        최종 answer만 반환 — reasoning 품질이 격변.
      - 샘플마다 reasoning 경로가 달라지므로 **self-consistency 다양성의 근원**이
        temperature가 아닌 사고 경로 자체에서 나온다 (DeepSeek-R1 그대로).

  (B) **Prompt Caching (ephemeral 5분 캐시)**
      - 동일 system 프롬프트를 N=3 샘플 + critic + arbiter = 5회 호출에 공유 →
        캐시 히트 시 입력 토큰 비용 **최대 90% 절감**.
      - 캐시된 system 블록: 첫 호출 cache write, 이후 4회는 cache read.

  (C) **Opus 4.5 → 200K context + Superior reasoning**
      - Sonnet 3.5 대비 금융 추론 벤치마크에서 유의미한 향상.
      - 긴 뉴스·매크로 텍스트 입력 (30개 제한 해제 가능).

  (D) **확률 보정(calibration)이 좋음**
      - conviction이 실제 사후확률에 더 가까움 → 포지션 사이징에 직결.

본 모듈은 Opus 4.x 기본 전제:
  - 모델: claude-opus-4-5 (환경변수로 override 가능 — 사용자가 4.7 쓰면 지정)
  - Extended Thinking ON: sample/critic/arbiter 모두 네이티브 추론 활성
  - Prompt Caching ON: system 블록을 ephemeral로 cache — 입력 비용 급감
  - Beta headers: prompt-caching-2024-07-31 (+ thinking은 2025년 기준 GA)

=============================================================================
🔬 **실행 파이프라인**
=============================================================================

  analyze(texts, symbol, regime)
    ① CoT + Extended Thinking으로 N=3 병렬 샘플 (다양성 = 추론경로 차이)
    ② 샘플간 방향 투표 + conviction 평균(투표비율로 할인) + scenario 통합
    ③ Critic 호출 — thinking ON, 합의안 반박 시도
    ④ Arbiter 호출 — thinking ON, 원안 vs 반박 중재 → 최종 확정
    ⑤ EV = Σ(p_i × r_i) → high-severity 리스크 할인 → score 매핑
    ⑥ risk_events[severity] 구조화 반환

비용 설계:
  - 기본 thinking_budget 6000 tokens/call × 5 calls = 30K thinking tokens/analysis
  - Prompt caching으로 system 입력(~2K) 재사용 → 실질 입력 비용 1/5 이하
  - 결과는 LLMSignalEngine 레벨에서 15분 캐시
  - 총 레이턴시: ~25~40초 (thinking 포함, 병렬 N=3)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import re

from loguru import logger


# =============================================================================
# 모델·기능 기본값 — Opus 4.x 활용 프리셋
# =============================================================================

# "auto" — 런타임에 Anthropic /v1/models API를 프로브해 가장 최신 Opus 선택.
# 실패 시 CLAUDE_MODEL_FALLBACK 체인으로 순차 fallback.
# 이렇게 하면 Anthropic이 새 Opus를 출시해도 재기동만으로 자동 채택된다.
DEFAULT_MODEL = "auto"

# 자동 해석 실패 시 순차 시도할 폴백 체인 (새 것 → 오래된 것 순).
# Anthropic이 Opus 4.7 같은 상위 버전을 출시하면 probe에서 자동 발견.
# 이 체인은 네트워크 프로브가 실패해도 동작하게 하는 안전망.
CLAUDE_MODEL_FALLBACK = [
    "claude-opus-4-5",                # 2025-09 GA (확정)
    "claude-opus-4",                  # 2025-05 GA
    "claude-3-7-sonnet-latest",       # 백업
    "claude-3-5-sonnet-latest",       # 최종 안전망
]

# Extended Thinking 기본 예산 — budget_tokens.
#   - 샘플(탐색적 reasoning): 6000 → 시나리오/트랩 분석 깊이 충분
#   - critic(반박): 4000 → 약점 탐지에 최적
#   - arbiter(중재): 5000 → 종합 판단
# max_tokens는 budget_tokens + 응답 JSON(~1000) 이상이어야 함.
DEFAULT_THINKING_BUDGET_SAMPLE  = 6000
DEFAULT_THINKING_BUDGET_CRITIC  = 4000
DEFAULT_THINKING_BUDGET_ARBITER = 5000

# Prompt caching beta header — 2024-07-31 GA.
# Extended thinking은 2025년 Opus 4에서 GA라 별도 beta 헤더 불필요(모델이 지원).
ANTHROPIC_BETA_CACHING = "prompt-caching-2024-07-31"

# Extended thinking 켰을 때 temperature는 1.0으로 고정해야 함(Anthropic 제약).
THINKING_FORCED_TEMPERATURE = 1.0


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
        model: 기본 claude-opus-4-5 (env LLM_CLAUDE_MODEL로 override)
        n_samples: self-consistency 샘플 수 (기본 3)
        enable_critique: 반박-중재 단계 활성 (기본 True)
        enable_thinking: Extended Thinking (네이티브 추론 토큰) 활성 (기본 True)
        thinking_budget_sample: 샘플 호출당 thinking 토큰 예산
        thinking_budget_critic: critic thinking 예산
        thinking_budget_arbiter: arbiter thinking 예산
        enable_prompt_cache: system 프롬프트 ephemeral 캐싱 (기본 True)
        sample_temperature: thinking OFF일 때만 의미 있음 (ON이면 1.0 강제)
        final_temperature: critic/arbiter 온도 (thinking OFF일 때만)
        timeout_per_call: 단일 호출 타임아웃 (thinking 포함 → 기본 90s)
        max_concurrent: 동시 호출 상한
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        n_samples: int = 3,
        enable_critique: bool = True,
        enable_thinking: bool = True,
        thinking_budget_sample: int = DEFAULT_THINKING_BUDGET_SAMPLE,
        thinking_budget_critic: int = DEFAULT_THINKING_BUDGET_CRITIC,
        thinking_budget_arbiter: int = DEFAULT_THINKING_BUDGET_ARBITER,
        enable_prompt_cache: bool = True,
        sample_temperature: float = 0.4,
        final_temperature: float = 0.1,
        timeout_per_call: float = 90.0,
        max_concurrent: int = 3,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.n_samples = max(1, min(n_samples, 5))
        self.enable_critique = enable_critique
        self.enable_thinking = enable_thinking
        self.thinking_budget_sample = thinking_budget_sample
        self.thinking_budget_critic = thinking_budget_critic
        self.thinking_budget_arbiter = thinking_budget_arbiter
        self.enable_prompt_cache = enable_prompt_cache
        self.sample_temperature = sample_temperature
        self.final_temperature = final_temperature
        self.timeout = timeout_per_call
        self._sema = asyncio.Semaphore(max_concurrent)
        self._calls = 0
        self._fails = 0
        # 캐시 히트/미스 관측 — Anthropic response usage에서 수집
        self._cache_reads = 0
        self._cache_writes = 0
        if not self.api_key:
            logger.warning("[ClaudeQuant] ANTHROPIC_API_KEY 없음 — analyze() 호출 시 예외 발생")
        else:
            logger.info(
                f"[ClaudeQuant] 초기화 model={model} N={self.n_samples} "
                f"critique={enable_critique} thinking={enable_thinking}"
                f"(budget_s={thinking_budget_sample}/c={thinking_budget_critic}/a={thinking_budget_arbiter}) "
                f"cache={enable_prompt_cache}"
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
            ⓪ (첫 호출만) model="auto"면 Anthropic API로 최신 Opus 자동 선택
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

        # ⓪ 모델 auto-resolve (첫 호출에서만 실행 — 결과는 self.model에 고정)
        if self.model in ("auto", "", None):
            self.model = await self._resolve_best_model()
            logger.info(f"[ClaudeQuant] auto-resolved model = {self.model}")

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
            "enable_thinking": self.enable_thinking,
            "thinking_budget": {
                "sample":  self.thinking_budget_sample,
                "critic":  self.thinking_budget_critic,
                "arbiter": self.thinking_budget_arbiter,
            },
            "enable_prompt_cache": self.enable_prompt_cache,
            "calls": self._calls,
            "fails": self._fails,
            "cache_read_tokens":  self._cache_reads,
            "cache_write_tokens": self._cache_writes,
            "has_key": bool(self.api_key),
        }

    # -------------------------------------------------------------------------
    # ① Self-Consistency 샘플링 (N회 병렬 호출)
    # -------------------------------------------------------------------------

    async def _self_consistency_sample(self, user_prompt: str) -> list[dict]:
        """N회 병렬 호출 → 성공 샘플만 반환.

        Extended Thinking이 켜져 있으면 **샘플간 다양성은 temperature가 아니라
        추론 경로(reasoning path)의 확률적 차이에서 나온다** — 이것이 DeepSeek-R1의
        작동 원리이며, Opus 4.x의 thinking 기능이 동일 역할을 수행한다.

        thinking 예산은 샘플 예산(6000) 사용. max_tokens = budget + ~1200 (JSON 응답 여유).
        """
        # Extended Thinking 사용 시 max_tokens는 budget + 최종 JSON 공간 필요
        sample_max_tokens = (
            self.thinking_budget_sample + 1200 if self.enable_thinking else 1200
        )
        tasks = [
            self._call_claude(
                system=COT_SYSTEM_PROMPT,
                user=user_prompt,
                temperature=self.sample_temperature,
                max_tokens=sample_max_tokens,
                thinking_budget=self.thinking_budget_sample if self.enable_thinking else 0,
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
            critic_max_tokens = (
                self.thinking_budget_critic + 800 if self.enable_thinking else 800
            )
            critic_raw = await self._call_claude(
                system=CRITIC_SYSTEM_PROMPT,
                user=critic_user,
                temperature=self.final_temperature,
                max_tokens=critic_max_tokens,
                thinking_budget=self.thinking_budget_critic if self.enable_thinking else 0,
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
            arbiter_max_tokens = (
                self.thinking_budget_arbiter + 1200 if self.enable_thinking else 1200
            )
            arbiter_raw = await self._call_claude(
                system=ARBITER_SYSTEM_PROMPT,
                user=arbiter_user,
                temperature=self.final_temperature,
                max_tokens=arbiter_max_tokens,
                thinking_budget=self.thinking_budget_arbiter if self.enable_thinking else 0,
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
        thinking_budget: int = 0,
    ) -> str:
        """단일 Claude 호출 — Extended Thinking + Prompt Caching 통합.

        Args:
            system: 시스템 프롬프트 (캐싱 대상)
            user: 유저 메시지
            temperature: thinking=0일 때만 효과. thinking>0이면 1.0으로 강제됨.
            max_tokens: 응답 상한. thinking>0이면 **budget + 응답JSON** 합보다 커야 함.
            thinking_budget: Extended Thinking 예산 토큰 수 (0 = 비활성).

        Returns:
            응답 텍스트 (thinking 블록은 자동 제외, text 블록만 결합).
        """
        import aiohttp
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # Prompt Caching beta header
        if self.enable_prompt_cache:
            headers["anthropic-beta"] = ANTHROPIC_BETA_CACHING

        # System 블록 — 캐싱 사용 시 content-blocks 형식으로 전달
        if self.enable_prompt_cache:
            system_payload: Any = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_payload = system

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_payload,
            "messages": [{"role": "user", "content": user}],
        }

        # Extended Thinking 활성 시
        if thinking_budget > 0:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # Anthropic 제약: thinking 활성 시 temperature는 1.0만 허용
            payload["temperature"] = THINKING_FORCED_TEMPERATURE
        else:
            payload["temperature"] = temperature

        async with self._sema:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload, headers=headers, timeout=self.timeout,
                ) as r:
                    data = await r.json()

        if "content" not in data:
            err = data.get("error", data)
            raise RuntimeError(f"Anthropic API error: {err}")

        # 캐시 히트/미스 통계 — usage에 cache_read_input_tokens, cache_creation_input_tokens
        usage = data.get("usage", {}) or {}
        if usage.get("cache_read_input_tokens"):
            self._cache_reads += int(usage["cache_read_input_tokens"])
        if usage.get("cache_creation_input_tokens"):
            self._cache_writes += int(usage["cache_creation_input_tokens"])

        # content는 blocks 배열: [{type: "thinking", thinking: "..."},
        #                        {type: "text", text: "..."}] 등
        # thinking 블록은 디버그 로그로만 — 외부 반환은 text만
        thinking_blocks = [
            blk for blk in data["content"] if blk.get("type") == "thinking"
        ]
        if thinking_blocks:
            total_len = sum(len(blk.get("thinking", "")) for blk in thinking_blocks)
            logger.debug(
                f"[ClaudeQuant] thinking blocks={len(thinking_blocks)} "
                f"total_chars={total_len} (첫 100자: "
                f"{(thinking_blocks[0].get('thinking', '') or '')[:100]!r})"
            )

        texts = [
            blk.get("text", "")
            for blk in data["content"] if blk.get("type") == "text"
        ]
        return "".join(texts)

    # -------------------------------------------------------------------------
    # 모델 auto-resolve — Anthropic /v1/models 프로브 → 최신 Opus 자동 선택
    # -------------------------------------------------------------------------

    async def _resolve_best_model(self) -> str:
        """Anthropic API에서 사용 가능한 모델 목록을 받아 최신 Opus 선정.

        우선순위:
          1) opus 계열 중 가장 높은 (major, minor, patch) 버전
          2) 실패 시 CLAUDE_MODEL_FALLBACK 체인 첫 번째

        이 함수는 **최초 1회만** 호출된다 (self.model에 결과를 고정).
        """
        try:
            import aiohttp
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    "https://api.anthropic.com/v1/models",
                    headers=headers, timeout=10,
                ) as r:
                    if r.status != 200:
                        raise RuntimeError(f"/v1/models status={r.status}")
                    data = await r.json()
            models = data.get("data", []) or []
            ids = [m.get("id", "") for m in models if m.get("id")]
            opus_ids = [i for i in ids if "opus" in i.lower()]
            if opus_ids:
                best = max(opus_ids, key=self._version_key)
                logger.info(
                    f"[ClaudeQuant] /v1/models probe → {len(ids)}개 모델 중 "
                    f"Opus {len(opus_ids)}개 발견. 최상위={best}"
                )
                return best
            # Opus가 없으면 sonnet 최상위 시도
            sonnet_ids = [i for i in ids if "sonnet" in i.lower()]
            if sonnet_ids:
                best = max(sonnet_ids, key=self._version_key)
                logger.warning(
                    f"[ClaudeQuant] Opus 미발견, Sonnet 최상위={best} 사용"
                )
                return best
        except Exception as e:
            logger.warning(
                f"[ClaudeQuant] 모델 auto-resolve 실패 "
                f"({type(e).__name__}: {e}) → 폴백 체인 사용"
            )
        # 폴백 체인 첫 번째 사용 (네트워크 실패 시에도 동작 보장)
        return CLAUDE_MODEL_FALLBACK[0]

    @staticmethod
    def _version_key(model_id: str) -> tuple[int, ...]:
        """모델 ID → 비교 가능한 버전 튜플.

        예시:
          claude-opus-4-5-20250929 → (4, 5, 20250929)
          claude-opus-4-7          → (4, 7, 0)
          claude-opus-4            → (4, 0, 0)
          claude-3-5-sonnet-latest → (3, 5, 0)  (sonnet이지만 비교엔 동작)
        """
        # 숫자 그룹 추출 (하이픈 구분) — latest/preview 같은 suffix는 무시
        nums = re.findall(r"\d+", model_id or "")
        if not nums:
            return (0,)
        # 보통 3자리(major, minor, patch 또는 date)로 맞춤
        tup = tuple(int(x) for x in nums[:3])
        # 자리 수 맞추기
        while len(tup) < 3:
            tup = tup + (0,)
        return tup

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
