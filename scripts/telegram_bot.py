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


def format_external_alert(alert_type: str, data: dict) -> str:
    """외부 요인 변동 알림 (뉴스/매크로/지정학/센티먼트)"""
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    header = ""
    body = ""

    if alert_type == "macro_shift":
        # 매크로 지표 급변
        header = "🌍 <b>매크로 시장 변동 감지</b>"
        changes = data.get("changes", [])
        score = data.get("score", 0)
        direction = "🟢 Bullish" if score > 0 else "🔴 Bearish"
        body = f"📊 매크로 점수: <b>{score:+.2f}</b> ({direction})\n\n"
        for c in changes:
            body += f"  {c}\n"

    elif alert_type == "breaking_news":
        # 크립토 영향 뉴스
        header = "📰 <b>크립토 영향 뉴스 감지</b>"
        headlines = data.get("headlines", [])
        impact = data.get("impact", "medium")
        emoji = "🔴" if impact == "high" else "🟡"
        body = f"{emoji} 임팩트: <b>{impact.upper()}</b>\n\n"
        for h in headlines[:5]:
            body += f"  • {h[:120]}\n"

    elif alert_type == "geopolitical":
        # 지정학 이벤트
        header = "⚔️ <b>지정학 이벤트 감지</b>"
        events = data.get("events", [])
        geo_risk = data.get("geo_risk", 0)
        body = f"🎯 지정학 리스크: <b>{geo_risk:.0%}</b>\n\n"
        for e in events[:5]:
            body += f"  • {e[:120]}\n"

    elif alert_type == "sentiment_shift":
        # 센티먼트 급변
        header = "📊 <b>시장 센티먼트 급변</b>"
        old_score = data.get("old_score", 0)
        new_score = data.get("new_score", 0)
        direction = "🟢 긍정 전환" if new_score > old_score else "🔴 부정 전환"
        body = (
            f"{direction}\n"
            f"  이전: {old_score:+.2f} → 현재: <b>{new_score:+.2f}</b>\n"
            f"  변화폭: {abs(new_score - old_score):.2f}\n"
        )
        if data.get("details"):
            body += f"\n  📝 {data['details']}"

    elif alert_type == "oil_move":
        # 유가 급변
        header = "🛢️ <b>유가 급변 감지</b>"
        price = data.get("price", 0)
        change = data.get("change", 0)
        emoji = "📈" if change > 0 else "📉"
        body = (
            f"{emoji} WTI: <b>${price:.1f}</b> ({change:+.1%})\n"
            f"  Brent: ${data.get('brent', 0):.1f}\n"
        )
        if data.get("implication"):
            body += f"\n  💡 <i>{data['implication']}</i>"

    elif alert_type == "dxy_move":
        # 달러 인덱스 급변
        header = "💵 <b>달러 인덱스(DXY) 급변</b>"
        dxy = data.get("dxy", 0)
        change = data.get("change", 0)
        direction = "약세 📉" if change < 0 else "강세 📈"
        crypto_impact = "🟢 크립토 호재" if change < 0 else "🔴 크립토 악재"
        body = (
            f"  DXY: <b>{dxy:.1f}</b> ({change:+.2%})\n"
            f"  달러 {direction} → {crypto_impact}\n"
        )

    elif alert_type == "fear_greed_extreme":
        # 공포탐욕 극단값
        header = "😱 <b>공포탐욕 지수 극단값</b>"
        value = data.get("value", 50)
        label = data.get("label", "")
        body = (
            f"  지수: <b>{value}</b> ({label})\n"
            f"  💡 <i>극단적 공포는 역발상 매수 시그널 가능</i>\n"
        )

    elif alert_type == "polymarket":
        # 폴리마켓 급변
        header = "🔮 <b>예측 시장(Polymarket) 변동</b>"
        events = data.get("events", [])
        score = data.get("score", 0)
        body = f"📊 PM 점수: <b>{score:+.2f}</b>\n\n"
        for e in events[:5]:
            body += f"  • {e[:120]}\n"

    elif alert_type == "composite_shift":
        # 종합 신호 큰 변화
        header = "🔄 <b>종합 외부 신호 전환</b>"
        old = data.get("old_score", 0)
        new = data.get("new_score", 0)
        components = data.get("components", {})
        emoji = "🟢" if new > old else "🔴"
        body = f"{emoji} {old:+.2f} → <b>{new:+.2f}</b>\n\n"
        if components:
            body += "<b>구성요소:</b>\n"
            for k, v in components.items():
                body += f"  {k}: {v:+.3f}\n"

    else:
        header = "📢 <b>외부 요인 알림</b>"
        body = json.dumps(data, ensure_ascii=False, default=str)[:500]

    return (
        f"{header}\n"
        f"━━━━━━━━━━━━━━━━━\n"
        f"{body}\n"
        f"\n🕐 {now}"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        send_message(" ".join(sys.argv[1:]))
    else:
        send_message("🤖 AutoTrader Telegram Bot 테스트 메시지")
    sys.exit(0)
