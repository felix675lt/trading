#!/usr/bin/env python3
"""[Patch H, 2026-04-27] 1개월 무인 운영 후 사용자 리뷰용 종합 큐레이션 스크립트

목적:
  사용자가 한 달 후 돌아왔을 때 한 눈에 읽을 수 있는 정리된 리포트 출력.
  - 트레이딩 성과 (PAPER vs LIVE)
  - BTC Reserve 누적 (paper 시뮬 vs live 실매수)
  - ML 학습 사이클 통계 (성공/실패 횟수, 평균 acc)
  - 시간별/요일별/심볼별 EV
  - Patch별 활성화 여부 확인

실행: ./venv/bin/python3 scripts/monthly_review.py
출력: reports/monthly_review_<TS>.json + 콘솔 요약
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path("data/autotrader.db")
RESERVE_PATH = Path("data/btc_reserve.json")
LOG_PATH = Path("logs/autotrader.log")
OUT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 30


def _q(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchall()


def trade_summary(conn) -> dict:
    cutoff = (datetime.utcnow() - timedelta(days=WINDOW_DAYS)).isoformat()
    rows = _q(
        conn,
        """SELECT mode, COUNT(*) AS n,
                  ROUND(SUM(pnl),2) AS sum_pnl,
                  ROUND(AVG(pnl),3) AS avg_pnl,
                  SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) AS wins,
                  SUM(CASE WHEN pnl<0 THEN 1 ELSE 0 END) AS losses,
                  ROUND(AVG(CASE WHEN pnl>0 THEN pnl END),2) AS avg_win,
                  ROUND(AVG(CASE WHEN pnl<0 THEN pnl END),2) AS avg_loss,
                  MIN(timestamp), MAX(timestamp)
           FROM trades WHERE timestamp > ? GROUP BY mode""",
        (cutoff,),
    )
    out = {}
    for r in rows:
        mode, n, sum_pnl, avg_pnl, wins, losses, avg_win, avg_loss, first, last = r
        wr = wins / n if n > 0 else 0.0
        out[mode] = {
            "n_trades": n,
            "sum_pnl": sum_pnl,
            "avg_pnl": avg_pnl,
            "win_rate": round(wr, 3),
            "wins": wins,
            "losses": losses,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "first": first,
            "last": last,
        }
    return out


def hourly_ev(conn) -> dict:
    cutoff = (datetime.utcnow() - timedelta(days=WINDOW_DAYS)).isoformat()
    rows = _q(
        conn,
        """SELECT mode, CAST(strftime('%H', timestamp) AS INTEGER) AS hr,
                  COUNT(*) AS n,
                  ROUND(AVG(pnl),3) AS avg_pnl,
                  ROUND(SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END)*1.0/COUNT(*), 3) AS wr
           FROM trades WHERE timestamp > ?
           GROUP BY mode, hr ORDER BY mode, hr""",
        (cutoff,),
    )
    out: dict = {}
    for mode, hr, n, avg_pnl, wr in rows:
        out.setdefault(mode, []).append({"hour_utc": hr, "n": n, "avg_pnl": avg_pnl, "wr": wr})
    return out


def symbol_breakdown(conn) -> dict:
    cutoff = (datetime.utcnow() - timedelta(days=WINDOW_DAYS)).isoformat()
    rows = _q(
        conn,
        """SELECT mode, symbol, COUNT(*) AS n, ROUND(SUM(pnl),2) AS sum_pnl,
                  ROUND(AVG(pnl),3) AS avg_pnl
           FROM trades WHERE timestamp > ?
           GROUP BY mode, symbol ORDER BY mode, sum_pnl DESC""",
        (cutoff,),
    )
    out: dict = {}
    for mode, sym, n, sum_pnl, avg_pnl in rows:
        out.setdefault(mode, []).append({"symbol": sym, "n": n, "sum_pnl": sum_pnl, "avg_pnl": avg_pnl})
    return out


def reserve_status() -> dict:
    if not RESERVE_PATH.exists():
        return {"exists": False, "note": "BTC Reserve 파일 없음 — 아직 적립 트리거 안 됨"}
    try:
        d = json.loads(RESERVE_PATH.read_text())
        return {"exists": True, **d}
    except Exception as e:
        return {"exists": True, "error": str(e)}


def learning_health() -> dict:
    """logs/autotrader.log에서 최근 학습 사이클 통계 추출."""
    if not LOG_PATH.exists():
        return {"error": "log file not found"}
    try:
        # 큰 로그를 라인 카운트로 빠르게 처리
        cutoff = datetime.utcnow() - timedelta(days=WINDOW_DAYS)
        n_train_start = 0
        n_train_done = 0
        n_meta_ok = 0
        n_meta_fail = 0
        n_cnn_oom = 0
        n_xgb_rollback = 0
        last_xgb_acc = None
        last_lstm_acc = None
        last_lgb_acc = None
        with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "자기학습 사이클 시작" in line:
                    n_train_start += 1
                elif "자기학습 사이클 완료" in line:
                    n_train_done += 1
                elif "[Meta] 학습 완료" in line:
                    n_meta_ok += 1
                elif "[Meta] 학습 실패" in line:
                    n_meta_fail += 1
                elif "MPS backend out of memory" in line:
                    n_cnn_oom += 1
                elif "[WalkForward-XGB] OOS" in line and "롤백" in line:
                    n_xgb_rollback += 1
                elif "XGBoost 모델 로드: 정확도" in line:
                    try:
                        last_xgb_acc = float(line.split("정확도")[-1].strip())
                    except Exception:
                        pass
                elif "LSTM 모델 로드: 정확도" in line:
                    try:
                        last_lstm_acc = float(line.split("정확도")[-1].strip())
                    except Exception:
                        pass
                elif "LightGBM 로드:" in line and "Acc:" in line:
                    try:
                        last_lgb_acc = float(line.split("Acc:")[-1].split(")")[0].strip())
                    except Exception:
                        pass
        return {
            "train_cycles_started": n_train_start,
            "train_cycles_completed": n_train_done,
            "meta_train_success": n_meta_ok,
            "meta_train_failed": n_meta_fail,
            "cnn_mps_oom_count": n_cnn_oom,
            "xgb_rollbacks": n_xgb_rollback,
            "last_loaded": {
                "xgb_acc": last_xgb_acc,
                "lstm_acc": last_lstm_acc,
                "lgb_acc": last_lgb_acc,
            },
        }
    except Exception as e:
        return {"error": str(e)}


def patch_status() -> dict:
    """주요 패치 활성화 흔적을 로그에서 확인."""
    if not LOG_PATH.exists():
        return {"error": "log file not found"}
    flags = {
        "patch_a_time_features": "input_dim 변경 45→54",  # CNN-Attn에 9개 추가됨
        "patch_b_smart_short_live_only": "LIVE_LONG_ONLY",
        "patch_c_d_schema_health": "[Schema-Boot]",
        "patch_f_time_blacklist": "LIVE Time Blacklist",
        "patch_g_meta_train": "Meta",
        "btc_reserve_active": "[BTCReserve]",
        "smart_scheduler": "[SmartSched]",
        "live_ev_monitor": "[EV모니터]",
    }
    found = {k: False for k in flags}
    try:
        with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                for k, needle in flags.items():
                    if not found[k] and needle in line:
                        found[k] = True
                if all(found.values()):
                    break
        return found
    except Exception as e:
        return {"error": str(e)}


def main() -> int:
    if not DB_PATH.exists():
        print(f"[ERROR] {DB_PATH} 없음")
        return 1
    conn = sqlite3.connect(DB_PATH)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "window_days": WINDOW_DAYS,
        "trade_summary": trade_summary(conn),
        "hourly_ev": hourly_ev(conn),
        "symbol_breakdown": symbol_breakdown(conn),
        "btc_reserve": reserve_status(),
        "learning_health": learning_health(),
        "patch_status": patch_status(),
    }
    conn.close()

    out = OUT_DIR / f"monthly_review_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    # ===== 콘솔 요약 =====
    print("=" * 70)
    print(f"📊 1개월 운영 리뷰 — {datetime.utcnow().isoformat()}Z")
    print("=" * 70)

    ts = report["trade_summary"]
    for mode, d in ts.items():
        print(f"\n[{mode}] {d['n_trades']}건 | WR {d['win_rate']*100:.1f}% | "
              f"sum {d['sum_pnl']:+.2f} | avg {d['avg_pnl']:+.3f} | "
              f"win/loss {d['avg_win']}/{d['avg_loss']}")

    rs = report["btc_reserve"]
    if rs.get("exists"):
        print("\n🏛️ BTC Reserve")
        # btc_reserve.json 구조에 따라 다름
        try:
            for src in ("live", "paper"):
                node = rs.get(src) or {}
                if node:
                    print(f"  [{src}] BTC={node.get('total_btc', 0):.6f} | "
                          f"USDT={node.get('total_usdt_equivalent', 0):.2f} | "
                          f"trades={node.get('n_acquisitions', 0)}")
        except Exception:
            pass
    else:
        print(f"\n🏛️ BTC Reserve: {rs.get('note')}")

    lh = report["learning_health"]
    if "error" not in lh:
        print(f"\n🧠 학습: 시작 {lh['train_cycles_started']}회 / 완료 {lh['train_cycles_completed']}회 | "
              f"Meta {lh['meta_train_success']}성공 {lh['meta_train_failed']}실패 | "
              f"CNN OOM {lh['cnn_mps_oom_count']}회 | XGB 롤백 {lh['xgb_rollbacks']}회")
        ll = lh["last_loaded"]
        print(f"  최근 로드 정확도 — XGB {ll['xgb_acc']} | LSTM {ll['lstm_acc']} | LGB {ll['lgb_acc']}")

    ps = report["patch_status"]
    if "error" not in ps:
        ok = sum(1 for v in ps.values() if v)
        print(f"\n🩹 패치 활성: {ok}/{len(ps)}")
        for k, v in ps.items():
            print(f"  {'✅' if v else '❌'} {k}")

    print(f"\n[Done] 리포트 저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
