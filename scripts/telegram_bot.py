"""
AutoTrader Telegram Notification Module
- 거래 알림 (진입/청산/PnL)
- 일간/주간 리포트
- 에러/이상 감지 알림
"""

import json
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

# .env 파일에서 환경변수 로드 (dotenv 없이 직접)
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"


def send_message(text: str, parse_mode: str = "HTML", silent: bool = False):
    """텔레그램 메시지 전송"""
    if not BOT_TOKEN or not CHAT_ID:
        print("[Telegram] BOT_TOKEN or CHAT_ID not set")
        return

    try:
        url = f"{API_BASE}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": silent,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read())
        if not result.get("ok"):
            print(f"[Telegram] API error: {result.get('description', '')}")
    except Exception as e:
        print(f"[Telegram] Send failed: {e}")


def format_trade_open(mode: str, symbol: str, action: str, price: float,
                      notional: float, leverage: int, reason: str) -> str:
    """거래 진입 알림"""
    emoji = "🟢" if action == "long" else "🔴"
    dir_text = "LONG" if action == "long" else "SHORT"
    return (
        f"{emoji} <b>{mode} {dir_text}</b>\n"
        f"━━━━━━━━━━━━━\n"
        f"📌 {symbol}\n"
        f"💰 ${price:,.2f}\n"
        f"📊 ${notional:,.2f} ({leverage}x)\n"
        f"📝 {reason}"
    )


def format_trade_close(mode: str, symbol: str, pnl: float, reason: str,
                       duration_min: float = 0) -> str:
    """거래 청산 알림"""
    emoji = "🟢" if pnl > 0 else "🔴"
    return (
        f"{emoji} <b>{mode} CLOSE</b>\n"
        f"━━━━━━━━━━━━━\n"
        f"📌 {symbol}\n"
        f"💰 PnL: <b>${pnl:+,.4f}</b>\n"
        f"📝 {reason}"
    )


def format_daily_report(data: dict) -> str:
    """일간 리포트"""
    trades = data.get("trades", {})
    equity = data.get("equity", {})
    models = data.get("models", {})
    optimizer = data.get("optimizer", {})

    count = trades.get("count", 0)
    win_rate = trades.get("win_rate", 0)
    total_pnl = trades.get("total_pnl", 0)
    best = trades.get("best_trade", 0)
    worst = trades.get("worst_trade", 0)

    acc = models.get("latest_accuracy", 0)
    trend = models.get("accuracy_trend", "")
    trend_emoji = "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️"

    opt_trades = optimizer.get("total_trades", 0)
    opt_wr = optimizer.get("win_rate", 0)

    msg = (
        f"📊 <b>AutoTrader 일간 리포트</b>\n"
        f"━━━━━━━━━━━━━━━━━\n\n"
        f"💼 <b>거래 성과</b>\n"
        f"  거래: {count}건 | 승률: {win_rate:.0%}\n"
        f"  PnL: <b>${total_pnl:+.4f}</b>\n"
    )
    if best:
        msg += f"  최고: +${best:,.4f} | 최악: ${worst:,.4f}\n"
    msg += (
        f"\n💰 <b>자산</b>\n"
        f"  현재: ${equity.get('current', 0):,.2f}\n"
        f"  수익률: {equity.get('return_pct', 0):+.2%}\n"
        f"\n🤖 <b>ML 모델</b>\n"
        f"  정확도: {acc:.2%} {trend_emoji}\n"
    )
    if opt_trades > 0:
        msg += (
            f"\n📐 <b>전략 최적화</b>\n"
            f"  추적거래: {opt_trades}건 | 승률: {opt_wr:.0%}\n"
        )
    return msg


def format_weekly_report(data: dict) -> str:
    """주간 리포트"""
    return (
        f"📊 <b>AutoTrader 주간 리포트</b>\n"
        f"━━━━━━━━━━━━━━━━━\n"
        f"{json.dumps(data, indent=2, default=str)[:1000]}"
    )


def format_system_alert(msg: str) -> str:
    """시스템 알림"""
    return f"⚠️ <b>시스템 알림</b>\n━━━━━━━━━━━━━\n{msg}"


def format_health_alert(data: dict) -> str:
    """자가진단 알림"""
    status = data.get("status", "unknown")
    issues = data.get("issues", [])
    fixes = data.get("fixes", [])
    msg = f"🔍 <b>자가진단</b> {status}\n━━━━━━━━━━━━━\n"
    if issues:
        msg += "\n".join(f"⚠️ {i}" for i in issues) + "\n"
    if fixes:
        msg += "\n".join(f"🔧 {f}" for f in fixes) + "\n"
    return msg


if __name__ == "__main__":
    if len(sys.argv) > 1:
        send_message(" ".join(sys.argv[1:]))
    else:
        send_message("🤖 AutoTrader Telegram Bot 테스트 메시지")
    sys.exit(0)
