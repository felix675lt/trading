#!/bin/bash
# AutoTrader AI 서비스 제거

PLIST_PATH="$HOME/Library/LaunchAgents/com.autotrader.ai.plist"

launchctl unload "$PLIST_PATH" 2>/dev/null
rm -f "$PLIST_PATH"

echo "AutoTrader AI 서비스가 제거되었습니다."
