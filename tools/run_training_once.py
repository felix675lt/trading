"""단독 학습 스크립트 — 서비스 OFF 상태에서 모델을 한 번 완주시킴

사용:
    venv/bin/python3 tools/run_training_once.py

전제조건:
    - launchctl com.autotrader.ai 중단 상태
    - models_saved/ 비어있거나 비대한 xgboost.pkl 백업됨
    - data/autotrader.db 에 과거 캔들 이미 저장되어 있음

동작:
    - 트레이딩/외부데이터/대시보드 없이 순수 학습만 실행
    - BTC/USDT:USDT 5m 기준으로 train_cycle 1회 완주
    - 완료 시 models_saved/ 에 새 모델 + .last_train_time 저장

메모리 기대치:
    - 기존 4.5GB (서비스+학습) → 1.5~2GB (학습만)
    - Mac mini에서 OOM 없이 완주 가능
"""

import asyncio
import os
import re
import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv
from loguru import logger

# .env 로드
load_dotenv(Path(__file__).parent.parent / ".env")

from core.data.collector import DataCollector
from core.data.storage import Storage
from core.learning.trainer import SelfLearningTrainer
from core.models.ensemble import EnsembleSignalGenerator
from core.rl.agent import RLAgent


def _resolve_env(value):
    """${VAR} 형식을 환경변수로 치환"""
    if isinstance(value, str):
        m = re.match(r"^\$\{(\w+)\}$", value)
        if m:
            return os.environ.get(m.group(1), "")
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


async def main():
    # 1. config 로드 + env 치환
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = _resolve_env(config)

    logger.info("=" * 60)
    logger.info("단독 학습 스크립트 시작")
    logger.info(f"거래소: {list(config['exchanges'].keys())}")
    logger.info("=" * 60)

    # 2. 거래소 클라이언트 (데이터 수집용)
    collector = DataCollector(config["exchanges"])
    await collector.initialize()

    # 3. 학습 관련 객체만 생성 (트레이더/외부데이터/대시보드 없음)
    storage = Storage()
    ensemble = EnsembleSignalGenerator()
    rl_agent = RLAgent(config.get("rl", {}))

    trainer = SelfLearningTrainer(
        collector=collector,
        storage=storage,
        ensemble=ensemble,
        rl_agent=rl_agent,
        config=config,
        tier_manager=None,   # Walk-Forward CV 비활성 (메모리 절약)
        meta_labeler=None,
        hmm_regime=None,
    )

    # 4. 학습 대상 심볼 (PAPER/LIVE 합집합)
    symbols = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
        "DOGE/USDT:USDT",
    ]
    primary_tf = config["trading"]["timeframes"][0]
    exchange_name = "binance"

    # 5. 순차 학습 (메모리 안전)
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n{'█' * 60}")
        logger.info(f"[{i}/{len(symbols)}] {symbol} 학습 시작")
        logger.info(f"{'█' * 60}")
        try:
            await trainer.train_cycle(exchange_name, symbol, primary_tf)
            logger.success(f"[{i}/{len(symbols)}] {symbol} 학습 완주 ✓")
        except Exception as e:
            logger.exception(f"[{i}/{len(symbols)}] {symbol} 학습 실패: {e}")

    # 6. last_train_time 기록 (서비스 재가동 시 should_retrain=False 유도)
    trainer.last_train_time = __import__("datetime").datetime.utcnow()
    trainer._save_last_train_time()

    # 7. 정리
    await collector.close()

    # 8. 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("학습 완료 — 저장된 모델:")
    models_dir = Path(__file__).parent.parent / "models_saved"
    for f in sorted(models_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  {f.name}: {size_mb:.2f} MB")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
