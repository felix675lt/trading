"""[Patch W, 2026-06-19] 가동 감시 — dead-man switch + 다운타임 감지 + 생애주기 알림.

배경: 6/18 22:46 맥 재부팅 후 봇이 ~20시간 정지했으나 아무 알림도 없었음.
원인: (1) 알림 메커니즘 부재, (2) 로컬 프로세스는 맥이 꺼지면 같이 죽어 자기 죽음을
      알릴 수 없음 → 외부 감시자 필요.

이 모듈:
  1) HEALTHCHECK_URL(.env)로 매 루프 ping → 외부 서비스(healthchecks.io 등)가
     ping 끊김을 감지해 사용자에게 알림. 정전·크래시·네트워크단절 전부 커버.
     URL 미설정 시 무동작(안전) — 사용자가 무료가입 후 URL만 넣으면 활성.
  2) data/last_alive.json에 매 루프 타임스탬프 기록 → 재기동 시 직전 기록과 비교해
     '봇이 N시간 죽어 있었다'를 부팅 텔레그램으로 통지(외부 의존 없이 즉시 동작).
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

ALIVE_MARKER = Path("data/last_alive.json")
# 이 간격(초)보다 큰 공백이면 '다운'으로 간주
DOWNTIME_THRESHOLD_SEC = 300  # 5분


def _now() -> datetime:
    return datetime.now(timezone.utc)


def write_alive_marker(extra: Optional[dict] = None) -> None:
    """매 루프 호출 — 현재 살아있음을 디스크에 원자적으로 기록."""
    try:
        ALIVE_MARKER.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ts": _now().isoformat(), "pid": os.getpid()}
        if extra:
            payload.update(extra)
        tmp = ALIVE_MARKER.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(ALIVE_MARKER)  # atomic
    except Exception as e:
        logger.debug(f"[Heartbeat] alive marker 기록 실패(무시): {e}")


def read_downtime_seconds() -> Optional[float]:
    """부팅 시 호출 — 직전 alive 기록과 현재의 공백(초). 기록 없으면 None."""
    try:
        if not ALIVE_MARKER.exists():
            return None
        data = json.loads(ALIVE_MARKER.read_text())
        last = datetime.fromisoformat(data["ts"])
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return max(0.0, (_now() - last).total_seconds())
    except Exception as e:
        logger.debug(f"[Heartbeat] downtime 계산 실패(무시): {e}")
        return None


def ping_healthcheck(suffix: str = "") -> None:
    """HEALTHCHECK_URL로 fire-and-forget ping (dead-man switch).

    suffix: '/fail' 또는 '/start' 등 healthchecks.io 신호. 기본은 OK ping.
    URL 미설정 시 조용히 무동작.
    """
    base = os.getenv("HEALTHCHECK_URL", "").strip()
    if not base:
        return
    url = base.rstrip("/") + suffix if suffix else base
    try:
        urllib.request.urlopen(url, timeout=10)
    except Exception as e:
        # ping 실패는 거래에 영향 없음 — 디버그만
        logger.debug(f"[Heartbeat] healthcheck ping 실패(무시): {e}")


def format_downtime(seconds: float) -> str:
    """초 → '2시간 13분' 형식."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, _ = divmod(rem, 60)
    if h > 0:
        return f"{h}시간 {m}분"
    return f"{m}분"
