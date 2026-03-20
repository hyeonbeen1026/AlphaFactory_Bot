import yfinance as yf
import pandas as pd
import numpy as np

class DataPipeline:
    def __init__(self, universe, lookbacks=[5, 10, 20, 60, 120, 252]):
        self.universe = universe
        self.lookbacks = lookbacks
        self.df = pd.DataFrame()

    def fetch_data(self, period="5y"):
        print(f"📥 유니버스 {len(self.universe)}개 종목 OHLCV 다운로드 중...")
        df = yf.download(self.universe, period=period, group_by='ticker', auto_adjust=True, progress=False)
        
        df = df.stack(level=0).rename_axis(['date', 'ticker']).reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        # 기본 거래량 필터
        df = df[df['volume'] > 0].copy()
        self.df = df.sort_values(['ticker', 'date']).set_index(['date', 'ticker'])
        print(f"✅ 기초 데이터 로드 완료: {self.df.shape[0]} rows")

    def build_factors(self):
        print("🏭 시계열 및 횡단면 팩터 고속 계산 중...")
        df = self.df.copy()
        ticker_group = df.groupby(level='ticker')
        
        # 0. 핵심 정답지(Target) 및 기초 데이터
        df['target_1d'] = ticker_group['close'].shift(-1) / df['close'] - 1
        df['return_1'] = ticker_group['close'].pct_change()
        
        prev_close = ticker_group['close'].shift(1)
        df['gap_return'] = df['open'] / prev_close - 1
        df['overnight_return'] = df['close'] / df['open'] - 1
        df['dollar_volume'] = df['close'] * df['volume']
        
        # ==========================================
        # [1] & [3] Liquidity & Micro-cap Filter (실전 필수 방어막)
        # ==========================================
        # 주가 $5 이상, 거래대금 $5M 이상인 튼튼한 종목만 필터링하여 가짜 알파 원천 차단
        df = df[(df['close'] > 5) & (df['dollar_volume'] > 5_000_000)].copy()
        
        # 필터링 후 끊어진 그룹을 다시 묶어줌
        ticker_group = df.groupby(level='ticker')
        prev_close = ticker_group['close'].shift(1)

        # ATR 계산
        true_range = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - prev_close),
                abs(df['low'] - prev_close)
            )
        )
        df['true_range'] = true_range
        df['atr_20'] = ticker_group['true_range'].rolling(20).mean().reset_index(level=0, drop=True)
        
        # 동적 팩터 생성
        for lb in self.lookbacks:
            df[f'return_{lb}'] = ticker_group['close'].pct_change(lb)
            df[f'momentum_{lb}'] = df[f'return_{lb}'] 
            df[f'volatility_{lb}'] = ticker_group['return_1'].rolling(lb).std().reset_index(level=0, drop=True) * np.sqrt(252)
            
            adv = ticker_group['dollar_volume'].rolling(lb).mean().reset_index(level=0, drop=True)
            df[f'turnover_{lb}'] = df['dollar_volume'] / (adv + 1e-8) 
            
            roll_mean = ticker_group['close'].rolling(lb).mean().reset_index(level=0, drop=True)
            roll_std = ticker_group['close'].rolling(lb).std().reset_index(level=0, drop=True)
            df[f'price_zscore_{lb}'] = (df['close'] - roll_mean) / (roll_std + 1e-8)

        # 횡단면 랭킹 연산
        date_group = df.groupby(level='date')
        for lb in self.lookbacks:
            df[f'momentum_{lb}_rank'] = date_group[f'momentum_{lb}'].rank(pct=True)
            df[f'volatility_{lb}_rank'] = date_group[f'volatility_{lb}'].rank(pct=True)
            df[f'price_zscore_{lb}_rank'] = date_group[f'price_zscore_{lb}'].rank(pct=True) 

        # 백테스터 필수 파라미터 매핑
        df['volatility'] = df['volatility_20'] 
        df['tie_breaker_score'] = date_group['dollar_volume'].rank(pct=True) + date_group['momentum_20'].rank(pct=True)
        
        # ==========================================
        # [2] Future Leak (Lookahead Bias) 원천 방지
        # ==========================================
        # 시그널 생성에 사용될 모든 팩터를 하루(1)씩 밀어서(Shift), "오늘 시그널"이 완벽히 "과거 데이터"로만 만들어지게 함
        factor_cols = [c for c in df.columns if any(x in c for x in ['momentum', 'volatility', 'rank', 'zscore', 'turnover', 'atr', 'return_'])]
        # target_1d, return_1 등은 제외
        factor_cols = [c for c in factor_cols if c not in ['target_1d', 'return_1']] 
        
        df[factor_cols] = df.groupby(level='ticker')[factor_cols].shift(1)

        # 결측치 정돈 (Shift 및 랭킹 연산으로 발생한 초기 NaN 제거)
        self.df = df.dropna(subset=['target_1d', 'volatility', 'tie_breaker_score'] + factor_cols).copy()
        print(f"🧹 전처리 및 필터링 완료: 최종 데이터 {self.df.shape[0]} rows")

    def get_dataframe(self):
        return self.df

    def save_pipeline(self, filename="alpha_factory_data.parquet"):
        self.df.reset_index().to_parquet(filename, index=False)
        print(f"💾 데이터 파이프라인 저장 완료: {filename}")