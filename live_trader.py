import os
import time
import requests
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed 

from ensemble_bot import EnsembleBot

# ==========================================
# [1] API 및 텔레그램 세팅 (환경 변수)
# ==========================================
API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

try:
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
except Exception as e:
    print(f"Alpaca API 클라이언트 초기화 실패: {e}")

# 💡 텔레그램 메시지 전송 함수
def send_telegram_msg(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ 텔레그램 토큰이 설정되지 않아 알림을 생략합니다.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': msg,
        'parse_mode': 'HTML'
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"⚠️ 텔레그램 전송 실패: {e}")

# ==========================================
# [2] S&P 500 유니버스 로드
# ==========================================
def get_sp500_universe():
    print(" S&P 500 유니버스 실시간 스크래핑 중...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        df = pd.read_html(io.StringIO(response.text))[0]
        # Alpaca 호환을 위해 점(.)을 그대로 유지
        tickers = df['Symbol'].astype(str).str.replace('-', '.').tolist()
        return tickers
    except Exception as e:
        print(f"⚠️ 위키피디아 스크래핑 실패: {e}")
        return ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META'] 

# ==========================================
# [3] Alpaca 팩터 계산 파이프라인
# ==========================================
def get_live_factors(universe, lookbacks=[5, 10, 20, 60, 120, 252]):
    print(f"[Alpaca Live Data] {len(universe)}개 종목 실시간 데이터 다운로드 중...")
    
    start_dt = datetime.now() - timedelta(days=730)
    end_dt = datetime.now()
    all_bars = []
    chunk_size = 100 
    
    for i in range(0, len(universe), chunk_size):
        chunk = universe[i:i+chunk_size]
        request_params = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
            feed=DataFeed.IEX 
        )
        try:
            bars = data_client.get_stock_bars(request_params).df
            all_bars.append(bars)
        except Exception as e:
            print(f"⚠️ 데이터 다운로드 실패 (Chunk {i}): {e}")
            
    if not all_bars:
        raise ValueError("⚠️ 다운로드된 데이터가 없습니다!")

    bars_df = pd.concat(all_bars).reset_index()
    bars_df.rename(columns={'symbol': 'ticker', 'timestamp': 'date'}, inplace=True)
    bars_df['date'] = pd.to_datetime(bars_df['date']).dt.tz_localize(None)
    
    df = bars_df.copy()
    ticker_group = df.groupby('ticker')
    df['return_1'] = ticker_group['close'].pct_change()
    prev_close = ticker_group['close'].shift(1)
    df['dollar_volume'] = df['close'] * df['volume']
    
    true_range = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - prev_close), abs(df['low'] - prev_close))
    )
    df['true_range'] = true_range
    df['atr_20'] = ticker_group['true_range'].rolling(20).mean().reset_index(level=0, drop=True)
    
    for lb in lookbacks:
        df[f'return_{lb}'] = ticker_group['close'].pct_change(lb)
        df[f'momentum_{lb}'] = df[f'return_{lb}']
        df[f'volatility_{lb}'] = ticker_group['return_1'].rolling(lb).std().reset_index(level=0, drop=True) * np.sqrt(252)
        
        adv = ticker_group['dollar_volume'].rolling(lb).mean().reset_index(level=0, drop=True)
        df[f'turnover_{lb}'] = df['dollar_volume'] / (adv + 1e-8)
        
        roll_mean = ticker_group['close'].rolling(lb).mean().reset_index(level=0, drop=True)
        roll_std = ticker_group['close'].rolling(lb).std().reset_index(level=0, drop=True)
        df[f'price_zscore_{lb}'] = (df['close'] - roll_mean) / (roll_std + 1e-8)

    date_group = df.groupby('date')
    for lb in lookbacks:
        df[f'momentum_{lb}_rank'] = date_group[f'momentum_{lb}'].rank(pct=True)
        df[f'volatility_{lb}_rank'] = date_group[f'volatility_{lb}'].rank(pct=True)
        df[f'price_zscore_{lb}_rank'] = date_group[f'price_zscore_{lb}'].rank(pct=True)

    df['volatility'] = df['volatility_20']
    
    latest_date = df['date'].max()
    live_df = df[df['date'] == latest_date].copy()
    live_df.set_index('ticker', inplace=True)
    live_df.dropna(inplace=True) 
    
    return live_df

# ==========================================
# [4] 완벽한 리밸런싱 (부분 매도/매수 적용 & 텔레그램 메시지 정리)
# ==========================================
def rebalance_portfolio(target_weights):
    account = trading_client.get_account()
    total_equity = float(account.portfolio_value)
    
    current_positions = {pos.symbol: float(pos.market_value) for pos in trading_client.get_all_positions()}
    all_symbols = set(current_positions.keys()).union(set(target_weights.keys()))
    
    orders_to_submit = []
    
    msg_sells = []
    msg_buys = []
    
    for symbol in all_symbols:
        target_dollar = total_equity * target_weights.get(symbol, 0.0)
        current_dollar = current_positions.get(symbol, 0.0)
        
        delta = target_dollar - current_dollar 
        
        if abs(delta) < 10.0:
            continue
            
        if delta < 0:
            sell_amount = abs(delta)
            if target_weights.get(symbol, 0.0) == 0.0:
                msg_sells.append(f"🔴 <b>{symbol}</b>: 전량 매도")
                trading_client.close_position(symbol)
            else:
                msg_sells.append(f"🔻 <b>{symbol}</b>: -${sell_amount:,.0f} (비중축소)")
                orders_to_submit.append(MarketOrderRequest(
                    symbol=symbol, notional=round(sell_amount, 2), side=OrderSide.SELL, time_in_force=TimeInForce.DAY
                ))
        elif delta > 0:
            msg_buys.append(f"🟢 <b>{symbol}</b>: +${delta:,.0f}")
            orders_to_submit.append(MarketOrderRequest(
                symbol=symbol, notional=round(delta, 2), side=OrderSide.BUY, time_in_force=TimeInForce.DAY
            ))

    sell_orders = [o for o in orders_to_submit if o.side == OrderSide.SELL]
    buy_orders = [o for o in orders_to_submit if o.side == OrderSide.BUY]
    
    for order in sell_orders:
        trading_client.submit_order(order_data=order)
    if sell_orders: time.sleep(3) 
    
    for order in buy_orders:
        trading_client.submit_order(order_data=order)
        
    report = f" <b>[Alpha Factory 리밸런싱 완료]</b>\n\n"
    report += f" <b>총 자산:</b> ${total_equity:,.2f}\n"
    report += f" <b>날짜:</b> {datetime.now().strftime('%Y-%m-%d')}\n\n"
    
    if msg_sells:
        report += "<b>[매도 내역]</b>\n" + "\n".join(msg_sells) + "\n\n"
    if msg_buys:
        report += "<b>[매수 내역]</b>\n" + "\n".join(msg_buys) + "\n"
        
    if not msg_sells and not msg_buys:
        report += " 타겟 비중과 현재 비중이 일치하여 주문을 생략했습니다."

    send_telegram_msg(report)
    print(" 매매 및 텔레그램 알림 전송이 모두 완료되었습니다!")

# ==========================================
# 메인 실행 블록
# ==========================================
if __name__ == "__main__":
    print("[Live Quant Master] 실전 앙상블 매매 봇 가동 시작!")
    universe = get_sp500_universe()
    live_data = get_live_factors(universe=universe)
    
    bot = EnsembleBot(top_n=10, min_votes=2, max_positions=15, max_weight_per_stock=0.20)
    target_weights = bot.generate_target_portfolio(live_data)
    
    if target_weights:
        rebalance_portfolio(target_weights)
    else:
        # 매수할 종목이 없을 때도 텔레그램으로 알려줌
        msg = f" <b>[Alpha Factory]</b>\n\n 조건에 맞는 확신 종목이 없어 오늘은 매매를 쉬어갑니다 (현금 관망)."
        send_telegram_msg(msg)
        print("오늘은 매수 확신(Conviction)을 가진 종목이 없습니다. 관망합니다.")
