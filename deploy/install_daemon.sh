#!/bin/bash
# [Patch W, 2026-06-19] LaunchAgent → LaunchDaemon 전환 (재부팅 생존)
#
# 실행: sudo bash /Users/d/trading/deploy/install_daemon.sh
#
# 안전장치: LaunchAgent를 먼저 내린 뒤 Daemon을 올림 → 봇 이중실행(이중주문) 방지.
# 멱등: 여러 번 실행해도 안전.
set -euo pipefail

AGENT_PLIST="/Users/d/Library/LaunchAgents/com.autotrader.ai.plist"
DAEMON_SRC="/Users/d/trading/deploy/com.autotrader.ai.daemon.plist"
DAEMON_DST="/Library/LaunchDaemons/com.autotrader.ai.plist"
UID_D=$(id -u d)

if [ "$(id -u)" -ne 0 ]; then
  echo "❌ sudo 로 실행하세요: sudo bash $0"
  exit 1
fi

echo "1) 기존 LaunchAgent 정지/해제 (이중실행 방지)…"
launchctl bootout "gui/${UID_D}/com.autotrader.ai" 2>/dev/null || true
# 자동 재기동 방지를 위해 Agent plist를 비활성 위치로 이동(삭제 아님 — 롤백 가능)
if [ -f "$AGENT_PLIST" ]; then
  mv "$AGENT_PLIST" "${AGENT_PLIST}.disabled"
  echo "   → ${AGENT_PLIST}.disabled 로 이동(보관)"
fi

echo "2) LaunchDaemon 설치…"
cp "$DAEMON_SRC" "$DAEMON_DST"
chown root:wheel "$DAEMON_DST"
chmod 644 "$DAEMON_DST"

echo "3) Daemon 로드/기동…"
launchctl bootout system/com.autotrader.ai 2>/dev/null || true
launchctl bootstrap system "$DAEMON_DST"
launchctl enable system/com.autotrader.ai

echo ""
echo "✅ 완료 — 이제 로그인 없이 재부팅해도 봇이 자동 기동합니다."
echo "   상태 확인:  sudo launchctl print system/com.autotrader.ai | head -20"
echo "   롤백:       sudo bash /Users/d/trading/deploy/uninstall_daemon.sh"
