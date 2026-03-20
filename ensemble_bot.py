import pandas as pd
import re
from collections import Counter

class EnsembleBot:
    def __init__(self, top_n=10, min_votes=2, max_positions=15, max_weight_per_stock=0.20, strategies_csv="alpha_factory_top_strategies.csv"):
        self.top_n = top_n
        self.min_votes = min_votes           
        self.max_positions = max_positions   
        self.max_weight_per_stock = max_weight_per_stock 
        
        try:
            df_strats = pd.read_csv(strategies_csv)
            
            required_cols = ["entry_rule", "OOS_avg_sharpe"]
            for col in required_cols:
                if col not in df_strats.columns:
                    raise ValueError(f"🚨 치명적 오류: DB 파일에 필수 컬럼 '{col}'이 없습니다!")
                    
            # OOS 평균 샤프 지수 기준으로 내림차순 정렬 후 Top N 추출
            self.elite_strats = df_strats.sort_values(by='OOS_avg_sharpe', ascending=False).head(self.top_n)
            print(f"🔥 앙상블 가동: 상위 {self.top_n}개 엘리트 전략 로드 완료!")
            
        except Exception as e:
            print(f"⚠️ 전략 DB 로드 실패: {e}")
            self.elite_strats = pd.DataFrame()

    def validate_rule_columns(self, rule_str, df_columns):
        """룰에 사용된 팩터(변수)들이 오늘 데이터(DataFrame)에 존재하는지 사전 검증"""
        # 정규식: 영문자로 시작하는 단어 추출 (변수명)
        words = re.findall(r'[A-Za-z_]\w*', rule_str)
        # Pandas query 예약어 제외
        reserved = {'and', 'or', 'not', 'True', 'False', 'in', 'is'}
        
        for w in words:
            if w not in reserved and w not in df_columns:
                return False, w # 누락된 컬럼 반환
        return True, None

    def generate_target_portfolio(self, today_data_df):
        if self.elite_strats.empty:
            return {}

        master_buy_list = []
        unique_signal_sets = [] 
        
        active_strategies_count = 0 

        # 1. 엘리트 전략별 시그널 추출
        for _, strat in self.elite_strats.iterrows():
            entry_rule = strat['entry_rule']
            
            if not isinstance(entry_rule, str) or not entry_rule.strip():
                print("⚠️ 빈 문자열 룰 감지. 스킵합니다.")
                continue
                
            is_valid, missing_col = self.validate_rule_columns(entry_rule, today_data_df.columns)
            if not is_valid:
                print(f"🚨 팩터 누락 경고: '{missing_col}' 데이터가 없어 전략을 스킵합니다. -> [{entry_rule}]")
                continue

            try:
                # 검증을 통과한 안전한 룰만 query 실행
                selected_tickers = today_data_df.query(entry_rule).index.tolist()
                
                if not selected_tickers:
                    active_strategies_count += 1
                    continue
                
                ticker_set = frozenset(selected_tickers) 
                
                # 중복 시그널(클론) 방어
                if ticker_set in unique_signal_sets:
                    print(f"🕵️ 중복 시그널 감지 (투표 기권): {entry_rule}")
                    continue
                
                unique_signal_sets.append(ticker_set)
                master_buy_list.extend(selected_tickers)
                active_strategies_count += 1
                
            except Exception as e:
                print(f"⚠️ 룰 평가 에러 ({entry_rule}): {e}")
                continue
                
        print(f"📊 총 {len(self.elite_strats)}개 전략 중 {active_strategies_count}개 정상 작동 완료")

        # 2. 앙상블 투표 (Voting) 집계
        vote_counts = Counter(master_buy_list)
        filtered_votes = {ticker: votes for ticker, votes in vote_counts.items() if votes >= self.min_votes}
        
        if not filtered_votes:
            print(f"📉 조건(최소 {self.min_votes}표)을 만족하는 확신(Conviction) 종목이 없습니다. (현금 관망)")
            return {}

        sorted_tickers = sorted(filtered_votes.items(), key=lambda x: x[1], reverse=True)[:self.max_positions]
        
        # 3. 비중(Weight) 할당
        valid_total_votes = sum(votes for _, votes in sorted_tickers)
        target_weights = {}
        
        for ticker, votes in sorted_tickers:
            raw_weight = votes / valid_total_votes
            target_weights[ticker] = round(min(raw_weight, self.max_weight_per_stock), 4)
            
        total_weight = sum(target_weights.values())
        if total_weight > 1.0:
            print(f"⚠️ 비중 총합({total_weight:.2f})이 1.0을 초과했습니다. 강제 정규화를 실행합니다.")
            # 초과분을 비율대로 깎아서 정확히 1.0 이하로 맞춤
            target_weights = {k: round(v / total_weight, 4) for k, v in target_weights.items()}
            total_weight = sum(target_weights.values()) # 재계산
            
        print(f"💡 총 자산의 {total_weight*100:.1f}% 자금 배분 완료 (나머지 현금 대기)")
            
        return target_weights

# ==========================================
# [실행 예시]
# ==========================================
if __name__ == "__main__":
    bot = EnsembleBot(top_n=10, min_votes=2, max_positions=15, max_weight_per_stock=0.20)
    
    dummy_data = pd.DataFrame({
        'momentum_20_rank': [0.95, 0.8, 0.99, 0.4, 0.91],
        'volatility_10_rank': [0.1, 0.2, 0.15, 0.9, 0.12],
        'price_zscore_20': [-1.5, -2.1, 0.5, 3.0, -1.0]
    }, index=['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN'])
    
    target_portfolio = bot.generate_target_portfolio(dummy_data)
    
    print("\n🎯 [오늘의 앙상블 타겟 포트폴리오 비중]")
    if target_portfolio:
        for ticker, weight in target_portfolio.items():
            print(f" - {ticker}: {weight*100:.1f}%")