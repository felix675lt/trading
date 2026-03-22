"""FastAPI 대시보드 서버"""

import collections
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AutoTrader AI Dashboard")

# 실시간 학습 로그 버퍼 (최근 200개)
_live_logs: collections.deque = collections.deque(maxlen=200)


def add_live_log(entry: dict):
    """실시간 로그 추가 (main.py에서 호출)"""
    _live_logs.append(entry)

# 전역 상태 (main.py에서 주입)
_state = {
    "trader": None,
    "storage": None,
}


def set_state(trader, storage):
    _state["trader"] = trader
    _state["storage"] = storage


@app.get("/health")
async def health_check():
    """Health check endpoint (from ralph-wiggum pattern)"""
    trader = _state.get("trader")
    return JSONResponse({
        "success": True,
        "status": "running" if trader else "no_trader",
        "mode": getattr(trader, "mode", None) if trader else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/status")
async def get_status():
    trader = _state.get("trader")
    if not trader:
        return {"success": False, "status": "not_running"}
    return {
        "success": True,
        "status": "running",
        "mode": getattr(trader, "mode", "unknown"),
        "uptime": str(getattr(trader, "uptime", "N/A")),
    }


@app.get("/api/positions")
async def get_positions():
    trader = _state.get("trader")
    if not trader:
        return {"positions": []}
    positions = getattr(trader, "get_positions", lambda: [])()
    return {"positions": positions}


@app.get("/api/trades")
async def get_trades():
    storage = _state.get("storage")
    if not storage:
        return {"trades": []}
    return {"trades": storage.get_trades(limit=50)}


@app.get("/api/equity")
async def get_equity():
    trader = _state.get("trader")
    if not trader:
        return {"equity": 0, "history": []}
    return {
        "equity": getattr(trader, "equity", 0),
        "initial": getattr(trader, "initial_capital", 10000),
        "pnl": getattr(trader, "total_pnl", 0),
    }


@app.get("/api/signals")
async def get_signals():
    trader = _state.get("trader")
    if not trader:
        return {"signals": {}}
    return {"signals": getattr(trader, "last_signals", {})}


@app.get("/api/risk")
async def get_risk():
    trader = _state.get("trader")
    if not trader:
        return {}
    risk_mgr = getattr(trader, "risk_manager", None)
    if risk_mgr:
        return risk_mgr.get_status()
    return {}


@app.get("/api/feedback")
async def get_feedback():
    trader = _state.get("trader")
    if not trader:
        return {}
    fb = getattr(trader, "feedback", None)
    if fb:
        return fb.get_report()
    return {}


@app.get("/api/external")
async def get_external():
    """외부 데이터 분석 정보"""
    trader = _state.get("trader")
    if not trader:
        return {}
    ext = getattr(trader, "external_manager", None)
    if ext:
        return ext.get_report()
    return {}


@app.get("/api/live_logs")
async def get_live_logs():
    """실시간 학습/거래 로그"""
    return {"logs": list(_live_logs)}


@app.get("/api/strategy-optimization")
async def get_strategy_optimization():
    """전략 최적화 상태 (Paper + Live 각각)"""
    trader = _state.get("trader")
    empty = {"config": {}, "win_rate": 0, "total_trades": 0, "best_hour": None}
    if not trader:
        return {"paper": empty, "live": empty, "mode": "offline"}

    result = {"mode": getattr(trader, "mode", "paper")}
    for label, attr in [("paper", "strategy_optimizer_paper"),
                         ("live", "strategy_optimizer_live")]:
        opt = getattr(trader, attr, None)
        if opt:
            report = opt.get_report()
            result[label] = {
                "config": report.get("current_config", {}),
                "win_rate": report.get("current_win_rate", 0) or 0,
                "total_trades": report.get("total_trades", 0),
                "best_hour": report.get("best_hour"),
            }
        else:
            result[label] = empty
    return result
