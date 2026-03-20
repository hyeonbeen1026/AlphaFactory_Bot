import os
import time
import requests
import io
import pandas as pd
from joblib import Parallel, delayed
import gc

# 우리가 만든 3개의 독립 모듈 (같은 폴더에 있어야 함)
from data_pipeline import DataPipeline
from generator import AlphaEngine
from backtester import VectorizedBacktester

# ==========================================
# [1] 시스템 설정 및 하이퍼파라미터
# ==========================================
UNIVERSE_SIZE = 500
POPULATION_SIZE = 900 
MAX_GENERATIONS = 40   
TARGET_SHARPE_IS = 1.2  
TARGET_SHARPE_OOS = 0.7 
DATA_FILE = "alpha_factory_data.parquet"
RESULTS_FILE = "alpha_factory_top_strategies.csv"

DATE_SPLIT_1 = "2024-01-01" 
DATE_SPLIT_2 = "2025-01-01"

def get_sp500_universe():
    print(" S&P 500 유니버스 실시간 스크래핑 중...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        df = pd.read_html(io.StringIO(response.text))[0]
        tickers = df['Symbol'].astype(str).str.replace('.', '-').tolist()
        print(f"✅ 유니버스 로드 완료: {len(tickers)} 종목")
        return tickers
    except Exception as e:
        print(f"⚠️ 위키피디아 스크래핑 실패: {e}")
        return ['SPY', 'AAPL', 'MSFT'] 

def load_or_build_data():
    print("🌐 [클라우드 모드] 실시간으로 S&P 500 데이터를 다운로드하고 전처리합니다. (약 5~10분 소요)")
    full_universe = get_sp500_universe() 
    
    # 파이프라인 가동 (파일 저장 로직 삭제)
    pipeline = DataPipeline(universe=full_universe)
    pipeline.fetch_data(period="5y")
    pipeline.build_factors()
    return pipeline.get_dataframe()

# ==========================================
# [2] Joblib 병렬 처리를 위한 래퍼(Wrapper) 함수
# ==========================================
def evaluate_strat_parallel(strat, backtester):
    try:
        # 정상적인 경우 백테스트 진행
        sharpe, cagr, mdd, trades = backtester.run_backtest(strat)
        return (strat, sharpe, cagr, mdd, trades)
    except Exception as e:
        return (strat, -999.0, -1.0, -1.0, 0)

def run_alpha_factory():
    print("\n" + "="*60)
    print("🚀 [Alpha Factory] 자동화 퀀트 리서치 (메모리 최적화본)")
    print("="*60)

    df_factors = load_or_build_data()
    print("🗜️ DataFrame 메모리 압축 중 (float64 -> float32)...")
    for col in df_factors.columns:
        if df_factors[col].dtype == "float64":
            df_factors[col] = df_factors[col].astype("float32")
    
    dates = df_factors.index.get_level_values('date')
    
    # ==========================================
    # [3] Walk-Forward 데이터 분할 (IS, OOS1, OOS2)
    # ==========================================
    df_is = df_factors[dates < DATE_SPLIT_1].copy()                               
    df_oos1 = df_factors[(dates >= DATE_SPLIT_1) & (dates < DATE_SPLIT_2)].copy() 
    df_oos2 = df_factors[dates >= DATE_SPLIT_2].copy()                            
    del df_factors
    gc.collect()
    
    print(f"📊 Walk-Forward 데이터 세팅 완료!")
    print(f"   - In-Sample (~{DATE_SPLIT_1}): {len(df_is)} rows")
    print(f"   - OOS Fold 1 (2024년): {len(df_oos1)} rows")
    print(f"   - OOS Fold 2 (2025년~): {len(df_oos2)} rows")

    bt_is = VectorizedBacktester(data_df=df_is, max_positions=20, transaction_cost=0.0025)
    bt_oos1 = VectorizedBacktester(data_df=df_oos1, max_positions=20, transaction_cost=0.0025)
    bt_oos2 = VectorizedBacktester(data_df=df_oos2, max_positions=20, transaction_cost=0.0025)
    
    engine = AlphaEngine(population_size=POPULATION_SIZE)
    engine.generate_initial_population()

    if os.path.exists(RESULTS_FILE):
        top_db = pd.read_csv(RESULTS_FILE).to_dict('records')
    else:
        top_db = []

    for generation in range(1, MAX_GENERATIONS + 1):
        print(f"\n🧬 [Gen {generation}/{MAX_GENERATIONS}] IS 평가 중 (Threading 병렬 처리)...")
        start_time = time.time()
        
        # ==========================================
        # [4] Joblib 초고속/저메모리 병렬 백테스트
        # ==========================================
        results = Parallel(
            n_jobs=2,              
            batch_size='auto', 
            backend='threading' 
        )(
            delayed(evaluate_strat_parallel)(strat, bt_is) for strat in engine.population
        )
            
        updated_population = []
        for strat, sharpe, cagr, mdd, trades in results:
            strat.evaluate_fitness(sharpe, cagr, mdd, trades)
            updated_population.append(strat)
            
        engine.population = updated_population 
        
        engine.population.sort(key=lambda x: x.fitness_score, reverse=True)
            
        best_strat = engine.population[0]
        gen_time = time.time() - start_time
        
        print(f"⚡ 소요 시간: {gen_time:.1f}초 (전략당 {gen_time/POPULATION_SIZE:.5f}초)")
        print(f"🏆 [Gen {generation} Top IS] Sharpe: {best_strat.backtest_metrics.get('sharpe', 0):.2f} | "
              f"CAGR: {best_strat.backtest_metrics.get('cagr', 0)*100:.1f}% | MDD: {best_strat.backtest_metrics.get('mdd', 0)*100:.1f}%")

        # ==========================================
        # [5] Walk-Forward OOS 검증
        # ==========================================
        new_discoveries = 0
        existing_rules = set([str(s.get('entry_rule', '')) + "|" + str(s.get('exit_rule', '')) for s in top_db])
        
        OOS_CANDIDATE_LIMIT = 30 # 상위 30개만
        candidates = engine.population[:OOS_CANDIDATE_LIMIT] 
        
        for strat in candidates:
            if strat.fitness_score > 0 and strat.backtest_metrics.get('sharpe', 0) >= TARGET_SHARPE_IS:
                rules = strat.get_eval_strings()
                rule_signature = rules['entry'] + "|" + rules['exit']
                
                if rule_signature not in existing_rules:
                    sharpe_oos1, cagr_oos1, mdd_oos1, trades_oos1 = bt_oos1.run_backtest(strat)
                    sharpe_oos2, cagr_oos2, mdd_oos2, trades_oos2 = bt_oos2.run_backtest(strat)
                    
                    avg_oos_sharpe = (sharpe_oos1 + sharpe_oos2) / 2.0
                    
                    if sharpe_oos1 > 0 and sharpe_oos2 > 0 and avg_oos_sharpe >= TARGET_SHARPE_OOS:
                        if trades_oos1 > 5 and trades_oos2 > 5:
                            top_db.append({
                                'strategy_id': strat.strategy_id,
                                'generation': generation,
                                'IS_sharpe': strat.backtest_metrics['sharpe'],
                                'IS_cagr': strat.backtest_metrics['cagr'],
                                'OOS1_sharpe': sharpe_oos1,
                                'OOS2_sharpe': sharpe_oos2,
                                'OOS_avg_sharpe': avg_oos_sharpe,
                                'entry_rule': rules['entry'],
                                'exit_rule': rules['exit'],
                                'holding_days': rules['holding_days']
                            })
                            existing_rules.add(rule_signature) 
                            new_discoveries += 1
                            
                            print(f"   🔥 [WFV 통과!] 실전 투입 알파 발굴 -> IS: {strat.backtest_metrics['sharpe']:.2f} | OOS1: {sharpe_oos1:.2f} | OOS2: {sharpe_oos2:.2f}")
        
        if new_discoveries > 0:
            pd.DataFrame(top_db).to_csv(RESULTS_FILE, index=False)
            print(f"💾 고유 실전용 전략 {new_discoveries}개 DB 저장 완료! (총 누적: {len(top_db)}개)")

        if generation < MAX_GENERATIONS:
            engine.evolve_population()

        # 메모리 누수 방지용 가비지 컬렉터 강제 실행
        gc.collect()

    print("\n🎉 모든 진화 사이클 종료. Alpha Factory 가동을 마칩니다.")

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None

    run_alpha_factory()




