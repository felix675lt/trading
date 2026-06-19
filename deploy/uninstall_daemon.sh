#!/bin/bash
# [Patch W] LaunchDaemon → LaunchAgent 롤백
# 실행: sudo bash /Users/d/trading/deploy/uninstall_daemon.sh
set -euo pipefail

AGENT_PLIST="/Users/d/Library/LaunchAgents/com.autotrader.ai.plist"
DAEMON_DST="/Library/LaunchDaemons/com.autotrader.ai.plist"
UID_D=$(id -u d)

if [ "$(id -u)" -ne 0 ]; then
  echo "❌ sudo 로 실행하세요: sudo bash $0"
  exit 1
fi

echo "1) Daemon 정지/제거…"
launchctl bootout system/com.autotrader.ai 2>/dev/null || true
rm -f "$DAEMON_DST"

echo "2) LaunchAgent 복원…"
if [ -f "${AGENT_PLIST}.disabled" ]; then
  mv "${AGENT_PLIST}.disabled" "$AGENT_PLIST"
fi
launchctl bootstrap "gui/${UID_D}" "$AGENT_PLIST" 2>/dev/null || true

echo "✅ 롤백 완료 — 다시 LaunchAgent(로그인 시 기동) 방식입니다."
