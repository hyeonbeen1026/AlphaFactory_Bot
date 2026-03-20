# Alpha Factory: Autonomous Quant Trading System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Alpaca](https://img.shields.io/badge/Alpaca-API-yellow?style=flat-square)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Automated-2088FF?style=flat-square&logo=github-actions)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Alpha Factory는 유전 알고리즘(Genetic Algorithm)을 활용하여 S&P 500 시장에서 유효한 팩터 및 전략을 발굴하고, 검증된 전략들을 앙상블(Ensemble)하여 리밸런싱을 수행하는 자동화 퀀트 트레이딩 파이프라인입니다.

---

## Key Features

### 1. 진화 알고리즘 기반 리서치 엔진 (Continuous Learning)
- **Genetic Algorithm:** 무작위 수식으로 이루어진 초기 전략 풀(Population)을 생성하고, 교배 및 돌연변이 연산을 통해 시장 대비 초과 수익(Alpha)을 달성하는 전략으로 진화시킵니다.
- **Walk-Forward Validation:** 과적합(Overfitting) 방지를 위해 시계열 데이터를 In-Sample(훈련)과 2개의 Out-of-Sample(검증)로 분리하여 다단계 교차 검증을 수행합니다.

### 2. 앙상블 기반 동적 리밸런싱 (Ensemble Live Trading)
- **Top N Voting System:** 누적된 전략 DB 중 OOS 기준 상위 N개의 뷰(View)를 종합합니다. 이를 통해 개별 전략의 노이즈를 상쇄하고 최적의 포트폴리오 타겟 비중을 산출합니다.
- **Delta Rebalancing:** 현재 계좌의 포지션과 타겟 비중의 오차(Delta)를 계산하여, 불필요한 잦은 매매를 줄이고 슬리피지를 최소화하는 방식으로 주문을 집행합니다.

### 3. 클라우드 자동화 (CI/CD Pipeline)
- **Weekend Miner (`weekend_factory.yml`):** 매주 주말 GitHub Actions 클라우드 환경에서 백그라운드로 실행되며, 최신 시장 데이터를 반영하여 엘리트 전략 CSV 파일을 갱신합니다.
- **Daily Trader (`daily_trading.yml`):** 평일 미국 주식 시장 마감 전 스케줄러가 작동하여 당일 리밸런싱 주문(Alpaca API)을 전송합니다.
- **Telegram Notification:** 매매 체결 내역 및 계좌 잔고 현황을 실시간으로 전송받아 시스템의 정상 동작 여부를 모니터링합니다.

---

## System Architecture

본 시스템은 리서치 환경과 실전 트레이딩 환경이 분리된 구조로 설계되었습니다.

1. **`data_pipeline.py`:** S&P 500 종목 데이터 수집, 결측치 처리, 유동성 필터링 및 횡단면/시계열 팩터 전처리
2. **`main_factory.py`:** Joblib을 활용한 병렬 백테스트 및 진화 알고리즘 기반 신규 전략 스크리닝
3. **`ensemble_bot.py`:** 전략 DB 로드 및 앙상블 투표 로직을 통한 타겟 비중 산출
4. **`live_trader.py`:** Alpaca API 연동을 통한 실계좌(또는 모의계좌) 주문 집행

---

## Engineering Challenges & Solutions

* **병렬 연산 시 메모리 초과(OOM) 문제 해결**
  * **Issue:** 다수의 전략을 병렬 평가(Multiprocessing)할 때 발생하는 데이터 Pickling 오버헤드로 인해 클라우드 서버(7GB RAM) 인스턴스가 강제 종료되는 현상 발생.
  * **Solution:** `joblib`의 `backend='threading'`을 적용하여 메모리 복사 없이 공유 메모리 환경을 구축하고, 데이터프레임의 float64 타입을 float32로 변환하여 메모리 점유율을 최적화함.

* **Lookahead Bias (미래 참조 편향) 통제**
  * **Issue:** 시그널 생성 시 당일 종가 데이터를 참조하여 다음 날의 결과를 예측하는 논리적 오류 발생 가능성.
  * **Solution:** 전처리 파이프라인 단계에서 타겟 변수(Target)를 제외한 모든 피처(Momentum, Volatility 등)에 `shift(1)`을 일괄 적용하여 데이터의 무결성을 확보함.

---

## Disclaimer
본 레포지토리의 코드는 퀀트 알고리즘 및 시스템 트레이딩 연구 목적으로 작성되었습니다. 실제 투자에 대한 책임은 전적으로 사용자 본인에게 있으며, 발생한 재정적 손실에 대해 책임지지 않습니다.
