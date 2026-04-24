"""Meta-Labeler 부트스트랩 학습 스크립트.

사용:
    python3 -m scripts.train_meta_labeler --symbol BTC/USDT:USDT --days 180
    python3 -m scripts.train_meta_labeler  # 기본: BTC 180일

Meta-labeler(Lopez de Prado, AFML Ch.3)는 1차 시그널(XGB) 위에
"이 트레이드 진입해야 하는가(1) vs 스킵(0)"를 학습하는 2차 분류기다.

이 스크립트는 메인 트레이너가 돌기 전 오프라인에서 Meta-Labeler를
부트스트랩 학습시켜 초기 진입 필터링을 즉시 활성화한다.

Pipeline:
  1. Binance에서 최근 N일 OHLCV 수집
  2. Feature engineering + triple-barrier labeling
  3. XGBoost 1차 학습 → primary_signal/confidence 컬럼 주입
  4. Meta-Labeler 학습 → models_saved/meta_labeler.pkl 저장
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml
from loguru import logger


async def main(symbol: str, days: int, timeframe: str):
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from core.data.collector import DataCollector
    from core.data.features import FeatureEngineer
    from core.data.labeling import triple_barrier_labels
    from core.learning.meta_labeler import MetaLabeler
    from core.models.xgboost_model import XGBoostPredictor

    cfg = yaml.safe_load(open("config/default.yaml"))

    logger.info(f"[MetaTrain] 수집 시작 — {symbol} {timeframe} {days}일")
    collector = DataCollector(cfg["exchanges"])
    df = await collector.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=min(days * 96, 5000),  # 15m 기준 하루 96봉, exchange 상한 5000
    )
    if df is None or len(df) < 500:
        logger.error(f"[MetaTrain] 데이터 부족: len={0 if df is None else len(df)}")
        return

    logger.info(f"[MetaTrain] 원본 {len(df)}봉 → 피처 엔지니어링")
    fe = FeatureEngineer(cfg.get("ml", {}).get("features"))
    df = fe.add_all_features(df)
    feature_cols = fe.get_feature_columns(df)

    # Triple-barrier labels (tb_label: 0=SL, 1=time-exp, 2=TP)
    df = triple_barrier_labels(df, pt_mult=2.0, sl_mult=1.0, max_hold=24)
    df = df.dropna(subset=feature_cols + ["tb_label"]).copy()
    if len(df) < 300:
        logger.error(f"[MetaTrain] 레이블 후 샘플 부족: {len(df)}")
        return

    logger.info(f"[MetaTrain] 라벨링 후 {len(df)}샘플 | feature수={len(feature_cols)}")

    # === 1차 XGB 학습 → primary_signal / primary_confidence 주입 ===
    logger.info("[MetaTrain] 1차 XGBoost 학습 중...")
    xgb = XGBoostPredictor(model_dir="models_saved")
    xgb.train(df, feature_cols)
    # 전체 in-sample 예측 (meta 학습용 1차 시그널 주입)
    proba = xgb.model.predict_proba(df[feature_cols])
    # 3-class → signal = P(TP) - P(SL)
    if proba.shape[1] == 3:
        df["primary_signal"] = proba[:, 2] - proba[:, 0]
    else:
        df["primary_signal"] = proba[:, -1] - proba[:, 0]
    df["primary_confidence"] = proba.max(axis=1)

    # === 2차 Meta-Labeler 학습 ===
    logger.info("[MetaTrain] 2차 Meta-Labeler 학습 중...")
    meta = MetaLabeler(threshold=0.55, model_dir="models_saved")
    meta.train(
        df,
        primary_signal_col="primary_signal",
        primary_confidence_col="primary_confidence",
        outcome_col="tb_label",
        feature_cols=feature_cols,
    )
    meta.save()
    logger.info(
        f"[MetaTrain] 완료 ✅ | acc={meta.accuracy:.3f} precision={meta.precision:.3f} "
        f"| models_saved/meta_labeler.pkl"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--timeframe", default="15m")
    args = p.parse_args()
    asyncio.run(main(args.symbol, args.days, args.timeframe))
