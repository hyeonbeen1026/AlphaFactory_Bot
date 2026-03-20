# 🏭 Alpha Factory: Autonomous Quant Trading System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Alpaca](https://img.shields.io/badge/Alpaca-API-yellow?style=flat-square)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Automated-2088FF?style=flat-square&logo=github-actions)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Alpha Factory는 유전 알고리즘(Genetic Algorithm)을 활용하여 S&P 500 시장에서 유효한 알파(Alpha) 전략을 스스로 발굴하고, 검증된 전략들을 앙상블(Ensemble)하여 **100% 무인으로 자동 매매를 수행하는 퀀트 트레이딩 시스템**입니다.

---

## 🚀 Key Features

### 1. 🧬 지속 학습형 진화 엔진 (Continuous Learning)
- **Genetic Algorithm:** 무작위 수식으로 이루어진 수천 개의 초기 전략 풀(Population)을 생성하고, 교배 및 돌연변이를 통해 시장을 이기는 전략으로 진화시킵니다.
- **Walk-Forward Validation:** 과적합(Overfitting)을 방지하기 위해 데이터를 In-Sample(훈련)과 2개의 Out-of-Sample(검증)로 엄격하게 분리하여 실전 투입 전 2단계 심층 면접을 진행합니다.

### 2. ⚖️ 앙상블 기반 동적 리밸런싱 (Ensemble Live Trading)
- **Top N Voting System:** 발굴된 수많은 엘리트 전략 중 상위 10개의 뷰(View)를 종합하여, 중복 시그널을 필터링하고 최적의 포트폴리오 비중을 산출합니다.
- **Delta Rebalancing:** 현재 계좌의 포지션과 타겟 비중의 오차(Delta)를 정밀하게 계산하여, 최소한의 슬리피지와 수수료로 포트폴리오를 조정합니다.

### 3. ☁️ 100% 클라우드 완전 자동화 (CI/CD Pipeline)
- **Weekend Miner (`weekend_factory.yml`):** 매주 주말 깃허브 클라우드 서버가 가동되어 새로운 시장 데이터를 학습하고 엘리트 전략 CSV 파일을 자동 업데이트합니다.
- **Daily Trader (`daily_trading.yml`):** 미국 주식 시장 마감 15분 전, 봇이 스스로 깨어나 리밸런싱 주문(Alpaca API)을 전송합니다.
- **Telegram Notification:** 매매가 완료되면 즉시 체결 내역과 총자산을 스마트폰으로 전송합니다.

---

## 🛠️ System Architecture

이 시스템은 주말 리
