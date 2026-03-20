import random
import uuid

# ==========================================
# [1] 팩터 유니버스 및 파라미터 풀 정의 (Price/Volume & 통계적 팩터 중심)
# ==========================================
LOOKBACKS = [5, 10, 20, 60, 120, 252]


THRESHOLD_MAP = {
    'zscore': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    'return': [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2],
    'volatility': [0.1, 0.15, 0.2, 0.3, 0.4],
    'rank': [0.05, 0.10, 0.20, 0.80, 0.90, 0.95], 
    'turnover': [0.5, 1.0, 1.5, 2.0, 3.0]
}

# 각 팩터가 어떤 Threshold 풀을 참조할지 정의 메타데이터
FACTOR_META = {
    'return': 'return',
    'momentum': 'return',
    'gap_return': 'return',        # 전일 종가 vs 당일 시가
    'overnight_return': 'return',  # 전일 시가 vs 당일 종가
    'volatility': 'volatility',
    'volatility_change': 'return', # 변동성 증감률
    'volume_zscore': 'zscore',
    'dollar_volume_zscore': 'zscore',
    'price_zscore': 'zscore',
    'turnover': 'turnover',
    'beta': 'zscore',
    'skewness': 'zscore'
}

OPERATORS_IND = ['>', '<', '>=', '<=']
OPERATORS_VAL = ['>', '<', '==']

# ==========================================
# [2] 전략 유전자 (Strategy DNA) 클래스
# ==========================================
class StrategyDNA:
    def __init__(self, num_entry=2, num_exit=1):
        self.strategy_id = str(uuid.uuid4())
        
        # 진입(Entry)과 청산(Exit) 로직 분리
        self.entry_genes = [self._generate_random_gene() for _ in range(num_entry)]
        self.exit_genes = [self._generate_random_gene() for _ in range(num_exit)]
        self.holding_days = random.choice([0, 5, 10, 20]) # 0이면 조건부 청산만 사용
        
        self.trade_count = 0 
        self.fitness_score = -999.0 # 초기값은 최하점
        self.backtest_metrics = {}

    def _generate_random_gene(self):
        """ind_vs_ind, ind_vs_val, rank_vs_val 3가지 타입의 논리 생성"""
        gene_type = random.choice(['ind_vs_ind', 'ind_vs_val', 'rank_vs_val'])
        ind1 = random.choice(list(FACTOR_META.keys()))
        lb1 = random.choice(LOOKBACKS)

        if gene_type == 'ind_vs_ind':
            return {
                'type': 'ind_vs_ind', 
                'ind1': ind1, 'lb1': lb1, 
                'op': random.choice(OPERATORS_IND), 
                'ind2': random.choice(list(FACTOR_META.keys())), 
                'lb2': random.choice(LOOKBACKS)
            }
        elif gene_type == 'ind_vs_val':
            thresh_type = FACTOR_META[ind1]
            return {
                'type': 'ind_vs_val', 
                'ind1': ind1, 'lb1': lb1, 
                'op': random.choice(OPERATORS_VAL), 
                'threshold': random.choice(THRESHOLD_MAP[thresh_type])
            }
        else: # rank_vs_val (Cross-sectional 랭킹)
            return {
                'type': 'rank_vs_val', 
                'ind1': ind1, 'lb1': lb1, 
                'op': random.choice(['<', '<=', '>', '>=']), 
                'threshold': random.choice(THRESHOLD_MAP['rank'])
            }

    def get_eval_strings(self):
        """Pandas Vectorized 연산을 위한 문자열 쿼리 반환"""
        def build_string(genes):
            if not genes: return "True"
            rules = []
            for g in genes:
                if g['type'] == 'ind_vs_ind': 
                    rules.append(f"({g['ind1']}_{g['lb1']} {g['op']} {g['ind2']}_{g['lb2']})")
                elif g['type'] == 'ind_vs_val': 
                    rules.append(f"({g['ind1']}_{g['lb1']} {g['op']} {g['threshold']})")
                elif g['type'] == 'rank_vs_val': 
                    rules.append(f"({g['ind1']}_{g['lb1']}_rank {g['op']} {g['threshold']})")
            return " & ".join(rules)
            
        return {
            'entry': build_string(self.entry_genes), 
            'exit': build_string(self.exit_genes), 
            'holding_days': self.holding_days
        }

    def evaluate_fitness(self, sharpe, cagr, mdd, trade_count):
        """통계적 유의성(Trade Count)을 포함한 엄격한 Fitness 평가"""
        self.trade_count = trade_count
        self.backtest_metrics = {'sharpe': sharpe, 'cagr': cagr, 'mdd': mdd}
        # 필터링 1: 거래 횟수 부족(우연) 또는 과다(수수료 폭탄)
        if self.trade_count < 30 or self.trade_count > 1000:
            self.fitness_score = -999.0
            return self.fitness_score
            
        # 필터링 2: 기본 수익성 미달
        if sharpe < 0 or cagr < 0:
            self.fitness_score = -999.0
            return self.fitness_score
            
        # 블렌디드 스코어: Sharpe(40%) + CAGR(40%) - MDD패널티(20%)
        self.fitness_score = (0.4 * sharpe) + (0.4 * cagr) - (0.2 * abs(mdd))
        return self.fitness_score

# ==========================================
# [3] 알파 팩토리 진화 엔진 (Alpha Engine)
# ==========================================
class AlphaEngine:
    def __init__(self, population_size=1000):
        self.population_size = population_size
        self.population = []
        
    def generate_initial_population(self):
        print(f"🧬 [Generation 0] 초기 전략 {self.population_size}개 무작위 생성 중...")
        self.population = [StrategyDNA(num_entry=random.randint(1, 3), num_exit=random.randint(0, 2)) 
                           for _ in range(self.population_size)]
        
    def selection(self, retain_ratio=0.2):
        """Fitness 기준 상위 부모 선별 (엘리트 보존)"""
        # 평가를 통과한(-900점 이상) 전략만 필터링
        valid_pop = [p for p in self.population if p.fitness_score > -900]
        valid_pop.sort(key=lambda x: x.fitness_score, reverse=True)
        return valid_pop[:max(2, int(len(valid_pop) * retain_ratio))]

    def crossover(self, p1, p2):
        """교차: 두 우수 전략의 진입/청산 룰 조합"""
        child = StrategyDNA(num_entry=0, num_exit=0)
        
        # 부모의 유전자를 반씩 가져와 결합
        child.entry_genes = random.sample(p1.entry_genes, len(p1.entry_genes)//2 + (len(p1.entry_genes)%2)) + \
                            random.sample(p2.entry_genes, len(p2.entry_genes)//2)
        child.exit_genes = random.sample(p1.exit_genes, len(p1.exit_genes)//2 + (len(p1.exit_genes)%2)) + \
                           random.sample(p2.exit_genes, len(p2.exit_genes)//2)
        child.holding_days = random.choice([p1.holding_days, p2.holding_days])
        
        return child

    def mutate(self, child_strat, mutation_rate=0.15):
        """강력한 다차원 돌연변이 엔진 (과최적화 방지 및 다양성 확보)"""
        
        # 1. 파라미터 변이 (Point Mutation)
        for target_genes in [child_strat.entry_genes, child_strat.exit_genes]:
            for gene in target_genes:
                if random.random() < mutation_rate:
                    mut_target = random.choice(['lb1', 'op', 'threshold', 'ind2', 'lb2'])
                    
                    if mut_target == 'lb1':
                        gene['lb1'] = random.choice(LOOKBACKS)
                    elif mut_target == 'op':
                        gene['op'] = random.choice(OPERATORS_IND if gene['type'] == 'ind_vs_ind' else OPERATORS_VAL)
                    elif mut_target == 'threshold' and gene['type'] in ['ind_vs_val', 'rank_vs_val']:
                        thresh_type = 'rank' if gene['type'] == 'rank_vs_val' else FACTOR_META[gene['ind1']]
                        gene['threshold'] = random.choice(THRESHOLD_MAP[thresh_type])
                    elif gene['type'] == 'ind_vs_ind':
                        if mut_target == 'ind2': gene['ind2'] = random.choice(list(FACTOR_META.keys()))
                        elif mut_target == 'lb2': gene['lb2'] = random.choice(LOOKBACKS)

        # 2. 구조적 변이 (Insertion / Deletion)
        if random.random() < mutation_rate:
            if random.random() < 0.5 and len(child_strat.entry_genes) > 1:
                child_strat.entry_genes.pop(random.randrange(len(child_strat.entry_genes)))
            else:
                child_strat.entry_genes.append(child_strat._generate_random_gene())

        if random.random() < mutation_rate:
            if random.random() < 0.5 and len(child_strat.exit_genes) > 0:
                child_strat.exit_genes.pop(random.randrange(len(child_strat.exit_genes)))
            else:
                child_strat.exit_genes.append(child_strat._generate_random_gene())

        # 3. 타임 엑시트 변이
        if random.random() < mutation_rate:
            child_strat.holding_days = random.choice([0, 5, 10, 20])

        return child_strat

    def evolve_population(self):
        """다음 세대(Next Generation) 생성 사이클"""
        parents = self.selection()
        
        if len(parents) < 2:
            print(f"⚠️ 대멸종 발생: 생존한 부모가 {len(parents)}명뿐입니다. 신규 무작위 유전자로 다음 세대를 보충합니다.")
            next_gen = parents[:] # 살아남은 1명이 있다면 일단 다음 세대로 보존
            
            # 부족한 인원수만큼 완전히 새로운 무작위 전략을 공장에서 찍어내어 수혈
            while len(next_gen) < self.population_size:
                next_gen.append(StrategyDNA(num_entry=random.randint(1, 3), num_exit=random.randint(0, 2)))
                
            self.population = next_gen
            print(f"🔄 진화(및 신규 수혈) 완료: 새로운 세대 {len(self.population)}개 전략 세팅 됨.")
            return

        next_gen = parents[:] # 엘리트 부모 세대 보존 (Elitism)
        
        while len(next_gen) < self.population_size:
            p1, p2 = random.sample(parents, 2)
            child = self.crossover(p1, p2)
            child = self.mutate(child, mutation_rate=0.15) # 돌연변이 주입
            next_gen.append(child)
            
        self.population = next_gen
        print(f"🔄 진화 사이클 완료: 새로운 세대 {len(self.population)}개 전략 세팅 됨.")

# ==========================================
# [4] 실행 및 테스트
# ==========================================
if __name__ == "__main__":
    engine = AlphaEngine(population_size=3)
    engine.generate_initial_population()
    
    print("\n" + "="*50)
    print("🏆 생성된 샘플 전략 규칙 (Pandas Vectorized 연산용)")
    print("="*50)
    
    for i, strat in enumerate(engine.population):
        rules = strat.get_eval_strings()
        print(f"\n[Strategy {i+1} | ID: {strat.strategy_id.split('-')[0]}]")
        print(f"🟢 진입(Entry): {rules['entry']}")
        print(f"🔴 청산(Exit):  {rules['exit']}")
        print(f"⏱️ 타임 엑시트: {rules['holding_days']}일")