#!/usr/bin/env python3
"""[Patch J, 2026-04-28] 인퍼런스 경로 smoke test

목적:
  매 코드 변경 후 PAPER 거래 결정 경로가 죽지 않았는지 5초 안에 검증.
  - features.generate(df) 호출
  - ensemble.predict(df) 호출 — 모든 5개 ML 모델 (XGB/LSTM/LGB/CNN/Meta)
  - rl_agent.predict(obs) 호출
  - strategy_manager.decide() 호출
  - 결과: dict 정상 반환 → exit 0, 예외 → exit 1

사용:
  ./venv/bin/python3 scripts/smoke_test_inference.py
  echo $?  # 0 = 정상, 1 = 인퍼런스 경로 깨짐

매 코드 변경 후 commit 전 반드시 실행.
ext_llm_*, ext_deriv_* 같은 학습-추론 컬럼 불일치를 즉시 검출.
"""
from __future__ import annotations

import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    print("=" * 60)
    print("🔬 인퍼런스 경로 SMOKE TEST")
    print("=" * 60)

    failed = []
    try:
        # 1) Storage + Features
        print("[1/4] FeatureEngineer 로드 중...", end=" ", flush=True)
        from core.data.features import FeatureEngineer

        fe = FeatureEngineer()
        # 최근 캔들 200개 가져오기 (DB에서)
        conn = sqlite3.connect("data/autotrader.db")
        df = pd.read_sql_query(
            "SELECT timestamp, open, high, low, close, volume FROM candles "
            "WHERE symbol='BTC/USDT:USDT' AND timeframe='5m' "
            "ORDER BY timestamp DESC LIMIT 300",
            conn,
        )
        conn.close()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        df = fe.generate(df)
        feature_cols = fe.get_feature_columns(df)
        print(f"OK | n_rows={len(df)} n_features={len(feature_cols)}")
    except Exception as e:
        print(f"FAIL: {e}")
        failed.append(("FeatureEngineer", str(e)))
        return 1

    # 2) Ensemble (XGB/LSTM/LGB/CNN)
    try:
        print("[2/4] Ensemble.predict 호출 중...", end=" ", flush=True)
        from core.models.ensemble import EnsembleSignalGenerator

        ens = EnsembleSignalGenerator()
        ens.load_all()
        result = ens.predict(df, regime="neutral")
        sig = result.get("signal", 0.0)
        conf = result.get("confidence", 0.0)
        print(f"OK | signal={sig:+.3f} conf={conf:.3f}")
    except Exception as e:
        print(f"FAIL: {e}")
        failed.append(("Ensemble.predict", str(e)))

    # 3) RL Agent
    try:
        print("[3/4] RL Agent.predict 호출 중...", end=" ", flush=True)
        from core.rl.agent import RLAgent

        rl = RLAgent(config={})
        rl.load()
        base_cols = fe.get_base_feature_columns(df)
        rl_obs = df[base_cols].values[-1].astype(np.float32)
        rl_obs = np.nan_to_num(rl_obs, nan=0.0)
        position_info = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        obs = np.concatenate([rl_obs, position_info])
        action, conf = rl.predict(obs)
        print(f"OK | action={action} conf={conf:.3f}")
    except Exception as e:
        print(f"FAIL: {e}")
        failed.append(("RL Agent.predict", str(e)))

    # 4) StrategyManager.decide (이걸 통과해야 거래 결정이 됨)
    try:
        print("[4/4] StrategyManager.decide 호출 중...", end=" ", flush=True)
        from core.strategy.manager import StrategyManager

        sm = StrategyManager(config={})
        decision = sm.decide(
            ml_signal=result,
            rl_action=int(action),
            rl_confidence=float(conf),
            current_position=0.0,
            regime="neutral",
            external_signal={"signal": 0.0, "confidence": 0.0},
            momentum=0.0,
            feedback_blacklist=set(),
            funding_rate=0.0,
            mode="paper",
            ohlcv_df=df,
        )
        print(f"OK | action={decision.action} conf={decision.confidence:.3f} reason={decision.reason[:60]}")
    except Exception as e:
        print(f"FAIL: {e}")
        failed.append(("StrategyManager.decide", str(e)))

    print("=" * 60)
    if failed:
        print(f"❌ SMOKE TEST 실패 — {len(failed)}건")
        for stage, err in failed:
            print(f"  - {stage}: {err}")
        print("=" * 60)
        return 1
    print("✅ SMOKE TEST 통과 — 인퍼런스 경로 정상")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
