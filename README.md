# AutoTrader AI - 자기학습 선물 트레이딩 시스템

모든 시장에 대응 가능한 하이브리드 자기학습 선물 자동매매 시스템

## 특징

- **하이브리드 AI**: 강화학습(PPO) + ML 앙상블(XGBoost, LSTM) 결합
- **자기학습**: 시장 데이터로 자동 재학습, 성능 기반 모델 교체
- **적응형 전략**: 시장 레짐(추세/횡보/고변동성) 자동 감지 및 파라미터 조정
- **리스크 관리**: 드로다운 차단, 일일 손실 한도, 동적 포지션 사이징
- **멀티 거래소**: Binance Futures, Bybit 지원 (ccxt 기반)
- **3단계 검증**: 백테스트 → 페이퍼 트레이딩 → 실거래
- **웹 대시보드**: 실시간 모니터링 (FastAPI)

## 아키텍처

```
[거래소 API] → [데이터 수집] → [피처 엔지니어링] → [ML 앙상블 시그널]
                                                        ↓
[자기학습 루프] ← [성능 평가] ← [거래 이력] ← [주문 실행] ← [RL 에이전트]
                                                  ↑
                                           [리스크 매니저]
```

## 설치

```bash
pip install -r requirements.txt
```

## 설정

1. `.env` 파일 생성 (`.env.example` 참조):
```
BINANCE_API_KEY=your_key
BINANCE_SECRET=your_secret
BYBIT_API_KEY=your_key
BYBIT_SECRET=your_secret
```

2. `config/default.yaml`에서 설정 조정

## 실행

### 백테스트
```bash
# config/default.yaml에서 mode: "backtest" 설정 후
python main.py
```

### 페이퍼 트레이딩
```bash
# config/default.yaml에서 mode: "paper" 설정 후
python main.py
```

### 실거래 (주의!)
```bash
# config/default.yaml에서 mode: "live" 설정 후
# testnet: false 설정 후
python main.py
```

## 대시보드

실행 후 http://localhost:8888 에서 모니터링 가능

## 구조

```
autotrader/
├── config/default.yaml          # 설정
├── core/
│   ├── data/                    # 데이터 수집/저장/피처
│   ├── models/                  # XGBoost, LSTM, 앙상블
│   ├── rl/                      # PPO 강화학습 환경/에이전트
│   ├── strategy/                # 전략 매니저, 적응형 최적화
│   ├── execution/               # 거래소, 주문, 페이퍼트레이딩
│   ├── risk/                    # 리스크 관리
│   └── learning/                # 자기학습 트레이너
├── backtest/                    # 백테스트 엔진
├── dashboard/                   # 웹 대시보드
└── main.py                      # 메인 실행
```

## 안전장치

- 기본값은 **테스트넷 + 페이퍼 트레이딩** (실제 자금 사용 안 함)
- 드로다운 15% 초과 시 자동 거래 중지
- 일일 손실 5% 초과 시 당일 거래 중지
- 모델 정확도 55% 미만 시 시그널 무시
- API 키는 환경변수로만 관리

## 주의사항

- 이 프로그램은 교육/연구 목적입니다
- 실거래 사용 시 반드시 테스트넷에서 충분히 검증하세요
- 투자 손실에 대한 책임은 사용자에게 있습니다
