"""FastAPI 대시보드 서버"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AutoTrader AI Dashboard")

# 전역 상태 (main.py에서 주입)
_state = {
    "trader": None,
    "storage": None,
}


def set_state(trader, storage):
    _state["trader"] = trader
    _state["storage"] = storage


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/status")
async def get_status():
    trader = _state.get("trader")
    if not trader:
        return {"status": "not_running"}
    return {
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
