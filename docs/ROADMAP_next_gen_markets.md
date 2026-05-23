# 차세대 거래 로드맵 — 탈중앙 + Pre-IPO

**작성일**: 2026-05-22
**상태**: 참고용 메모 (즉시 구현 X, 단계적 검토)

## 사용자 통찰 (2026-05-22)

> "차세대 거래는 탈중앙도 필요한 거 같고, 코인 시장에서 IPO 해서 SpaceX도
> 하는 거 보면 참고해놔야 할 것 같아."

핵심 포인트:
1. **탈중앙 거래소(DEX)** — 차세대 거래 표준
2. **Pre-IPO 토큰화 자산** — SpaceX, OpenAI 등 비상장 기업 지분의 토큰화
3. **거래량 좋은 시장에서 데이터 확보** (직전 사용자 통찰의 연장선)

---

## ① 현황 정리 (2026 기준)

### 탈중앙 영구선물 (DEX Perp)
| 플랫폼 | TVL/거래량 | 특징 |
|---|---|---|
| **Hyperliquid** | 일 거래대금 수십억$ — 일부 시점 Binance 추격 | 자체 L1 체인, HYPE 토큰 (오늘 추가) |
| dYdX v4 | 중급 | Cosmos 기반 |
| GMX (Arbitrum) | 중급 | LP 풀 모델 |
| Vertex / Aevo | 소중급 | hybrid orderbook |

### Pre-IPO 토큰화 자산
| 플랫폼 | 상장 자산 | 특징 |
|---|---|---|
| Robinhood EU (2025년 6월~) | OpenAI, SpaceX, Anthropic, Stripe tokenized | EU 사용자만, 24/7 거래 |
| Backed Finance (xStocks) | AAPLx, TSLAx, NVDAx + 일부 pre-IPO | Solana 기반, Kraken/Bybit 상장 |
| Republic, Linqto | 제한적 pre-IPO 직접 투자 | KYC 무거움, 유동성 낮음 |

---

## ② 데이터 관점 — 가치 분석

| 시장 | OHLCV 품질 | 거래량 | 24/7 | 시스템 통합 난이도 |
|---|---|---|---|---|
| Binance Perp (현재) | 🟢 최고 | 🟢 최대 | ✅ | ✅ 완료 |
| **Hyperliquid HYPE** | 🟢 우수 | 🟢 큼 | ✅ | ✅ **오늘 추가** (HYPE/USDT Binance 선물) |
| Hyperliquid 네이티브 DEX | 🟢 우수 | 🟢 큼 | ✅ | 🟡 별도 API |
| xStocks (Bybit) | 🟡 중간 | 🟡 작음 | ✅ | 🟡 ccxt 부분 지원 |
| Robinhood pre-IPO | 🔴 데이터 API 미공개 | 🟡 작음 | ✅ | 🔴 매우 어려움 |
| Backed Finance 토큰 | 🟡 중간 | 🔴 작음 | ✅ | 🟡 Solana RPC |

---

## ③ 단계적 통합 계획

### Phase 0 (오늘 완료) — HYPE 추가
- **상태**: ✅ Binance Futures HYPE/USDT:USDT 추가
- PAPER 트래킹 시작
- Pattern Bank 102,888 패턴 빌드 (6.9MB)
- 1주일 검증 후 LIVE 검토

### Phase A (1주일 후 점검 시) — 주식 OHLCV
- yfinance로 SPY/QQQ/AAPL/NVDA/TSLA 일봉 수집
- Pattern Bank에 주식 인덱스 추가
- 코인-주식 cross-asset 상관 학습
- 거래 X, 데이터만

### Phase B (1개월 후) — DEX 가격 데이터
- Hyperliquid 네이티브 API로 raw orderbook + funding 수집
- DEX vs CEX 가격 괴리(arbitrage 시그널) 추적
- 거래 X, 데이터만

### Phase C (검증 후, 선택) — xStocks
- Bybit ccxt로 AAPLx/TSLAx 5분봉
- Pattern Bank 추가
- 유동성 낮으므로 PAPER 시뮬만 검토

### Phase D (장기, 신중) — Pre-IPO 토큰
- Robinhood/Backed Finance pre-IPO 자산
- 데이터 API 가용 시점에 재검토
- 현재는 데이터 접근 자체가 제한적

---

## ④ 즉시 적용된 변경 (Patch P)

### 코드 변경
- `config/default.yaml`: paper_symbols_override에 HYPE 추가
- `scripts/fetch_hype.py`: 신규 — HYPE 1년치 캔들 수집 (103,139개)
- `scripts/build_pattern_banks.py`: SYMBOLS에 HYPE 추가
- `data/pattern_bank/HYPE_USDT_USDT_5m.npz`: 102,888 패턴

### 즉시 효과
- PAPER 분석 4 → 5 심볼
- Pattern Memory Bank 총 2,567,254 → 2,670,142 패턴
- HYPE도 Patch O Fusion 게이트 적용 받음

---

## ⑤ 위험 / 주의사항

| 위험 | 완화책 |
|---|---|
| HYPE 변동성 큼 (신생 토큰) | PAPER 트래킹만 — LIVE 즉시 투입 X |
| HYPE Binance 상장 불안정 가능 | 데이터 끊기면 Hyperliquid native API fallback 검토 |
| Pre-IPO 자산 규제 변동 | 미국/한국 거주자 접근 제한 가능 |
| DEX API 통합 비용 | 한 번 만들면 다중 DEX 지원 가능 |

---

## ⑥ 1주일 점검 시 확인 항목 (Patch P 추가)

- HYPE PAPER 진입 횟수 / WR
- HYPE Pattern Bank fusion veto/확증 빈도
- HYPE 변동성이 다른 4종목 대비 어떤지 (ATR/슬리피지)
- HYPE 데이터 수집 안정성 (Binance API 끊김 없음)

좋으면 Phase A (주식 OHLCV) 진행. 1주일 후 결정.
