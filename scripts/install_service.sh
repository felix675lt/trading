#!/bin/bash
# AutoTrader AI - macOS 백그라운드 서비스 설치 스크립트
# 터미널 없이 자동 실행, 맥 재부팅 시에도 자동 시작

PLIST_NAME="com.autotrader.ai"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"
TRADING_DIR="$HOME/trading"
VENV_PYTHON="$TRADING_DIR/venv/bin/python3"
LOG_DIR="$TRADING_DIR/logs"

mkdir -p "$LOG_DIR"

cat > "$PLIST_PATH" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${VENV_PYTHON}</string>
        <string>${TRADING_DIR}/main.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${TRADING_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/service_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/service_stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:${TRADING_DIR}/venv/bin</string>
    </dict>
</dict>
</plist>
PLIST

echo "=== AutoTrader AI 서비스 설치 ==="
echo ""
echo "서비스 파일 생성 완료: $PLIST_PATH"
echo ""
echo "사용법:"
echo "  시작:  launchctl load $PLIST_PATH"
echo "  중지:  launchctl unload $PLIST_PATH"
echo "  상태:  launchctl list | grep autotrader"
echo "  로그:  tail -f $LOG_DIR/service_stdout.log"
echo ""
echo "맥 재부팅 시 자동으로 시작됩니다."
echo ""

# 기존 서비스 중지 (있으면)
launchctl unload "$PLIST_PATH" 2>/dev/null

# 서비스 시작
launchctl load "$PLIST_PATH"
echo "서비스 시작됨!"
echo "대시보드: http://localhost:8888"
echo "로그 확인: tail -f $LOG_DIR/service_stdout.log"
