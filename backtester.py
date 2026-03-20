import numpy as np
import pandas as pd

class VectorizedBacktester:
    def __init__(self, data_df, max_positions=20, transaction_cost=0.0025):
        """
        data_df에는 'target_1d', 'tie_breaker_score' 외에도 
        역변동성 가중치를 위한 'volatility' 컬럼이 반드시 포함되어야 함.
        """
        self.df = data_df
        self.max_positions = max_positions
        self.tc = transaction_cost
        
        self.returns_2d = self.df['target_1d'].unstack(level='ticker').fillna(0.0)
        self.score_2d = self.df['tie_breaker_score'].unstack(level='ticker').fillna(-np.inf)
        
        # [1] 역변동성 비중을 위한 변동성 2D 매트릭스 (결측치는 무한대로 처리해 비중을 0으로 만듦)
        self.vol_2d = self.df['volatility'].unstack(level='ticker').fillna(np.inf)

    def run_backtest(self, strategy_dna):
        rules = strategy_dna.get_eval_strings()
        entry_rule = rules['entry']
        exit_rule = rules['exit']
        holding_days = rules['holding_days']

        try:
            raw_entries_1d = self.df.eval(entry_rule)
            raw_entries_2d = raw_entries_1d.unstack(level='ticker').fillna(False)
            
            exits_2d = pd.DataFrame(False, index=raw_entries_2d.index, columns=raw_entries_2d.columns)
            if exit_rule != "True":
                exits_2d = self.df.eval(exit_rule).unstack(level='ticker').fillna(False)

            # 쿨다운 방어
            if holding_days > 1:
                recent_entries = raw_entries_2d.shift(1).rolling(window=holding_days - 1, min_periods=1).max().fillna(0).astype(bool)
                clean_entries_2d = raw_entries_2d & (~recent_entries)
            else:
                clean_entries_2d = raw_entries_2d

            # Max Positions Limit
            signal_scores = self.score_2d.where(clean_entries_2d, -np.inf)
            top_mask = signal_scores.rank(axis=1, method='first', ascending=False) <= self.max_positions
            final_entries_2d = clean_entries_2d & top_mask

            # ==========================================
            # [2] Minimum Trade Filter (Early Exit 최적화)
            # ==========================================
            trade_count = final_entries_2d.sum().sum()
            if trade_count < 50:
                # 거래 횟수 미달 시 무의미한 연산을 스킵하고 즉시 도태 점수 반환
                return -1.0, -1.0, -1.0, trade_count

            # 좀비 포지션 방어 (Time Exit & Signal Exit 병합)
            positions = final_entries_2d.copy()
            if holding_days > 0:
                for i in range(1, holding_days):
                    shifted_entries = final_entries_2d.shift(i).fillna(False)
                    has_exited_since = exits_2d.rolling(window=i, min_periods=1).max().fillna(0).astype(bool)
                    positions = positions | (shifted_entries & (~has_exited_since))
            else:
                positions = final_entries_2d.astype(int)
                positions[exits_2d] = 0
                positions = positions.ffill().fillna(0)

            positions = positions.shift(1).fillna(0.0)

            # ==========================================
            # [1] Inverse Volatility Sizing & ZeroDivision 방어
            # ==========================================
            inv_vol = 1.0 / self.vol_2d
            raw_weights = positions * inv_vol
            # 분모가 0이 되는 것을 방지하기 위해 0을 np.nan으로 치환 후 div 연산
            weight_sum = raw_weights.sum(axis=1).replace(0, np.nan)
            weights = raw_weights.div(weight_sum, axis=0).fillna(0.0)

            # 수익률 계산 및 25bps 비용 차감
            daily_turnover = weights.diff().fillna(0.0).abs().sum(axis=1) / 2.0
            transaction_costs = daily_turnover * self.tc 
            
            port_daily_return = (weights * self.returns_2d).sum(axis=1) - transaction_costs

            # ==========================================
            # [2] CAGR 및 Sharpe 계산 안정성 확보
            # ==========================================
            cum_return = (1 + port_daily_return).cumprod()
            n_days = len(port_daily_return)
            years = n_days / 252.0
            
            cagr = (cum_return.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0.0
            
            std = port_daily_return.std()
            sharpe = 0.0 if std == 0.0 else (port_daily_return.mean() / std) * np.sqrt(252)
            
            drawdown = (cum_return / cum_return.cummax()) - 1
            mdd = drawdown.min()

            return sharpe, cagr, mdd, trade_count

        except Exception as e:
            # 수식 에러 발생 시 도태 점수 반환
            return -1.0, -1.0, -1.0, 0