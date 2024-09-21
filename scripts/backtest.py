import configparser
import backtrader as bt
import csv
from datetime import datetime
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


# Read configuration and data
config = configparser.ConfigParser()
config.read('D:/code/trading_system/data/config.ini')
file_path = 'D:/code/trading_system/data/predictions/predictions.pkl'
all_data_pandas = pd.read_pickle(file_path)
all_data_pandas['date'] = pd.to_datetime(all_data_pandas['date'])
all_tickers = all_data_pandas['symbol'].unique().tolist()
time_str = datetime.now().strftime("%d-%b-%y %H:%M:%S")

# Define strategy
time = datetime.now().strftime('%d%m%Y')
strategy = f"multiple_models"
buy_price_denominator = config.getint('parameter', 'buy_vol_fraction')
backtest_sell_signal_strength = config.getint('parameter', 'backtest_sell_signal_strength')
backtest_buy_signal_strength = config.getint('parameter', 'backtest_buy_signal_strength')
backtest_yesterday_signal_strength = config.getint('parameter', 'backtest_yesterday_signal_strength')
lookback_days_for_signal = config.getint('parameter', 'lookback_days_for_signal')

tickers = all_tickers
# Calculate index for the last 20% of the list
start_index = len(tickers) - int(0.2 * len(tickers))

# Slice the list to get the last 20%
# tickers = tickers[start_index:]
tickers = ["RAYMOND"]  # Example tickers, replace with your actual tickers
# print("Tickers:", tickers)
# CSV file path
csv_file = 'D:/code/trading_system/data/backtest_results.csv'
def calculate_position_size(trading_capital, sharpe_ratio, price_volatility, current_price, forecast):
    # Avoid division by zero
    if current_price == 0 or price_volatility == 0:
        print(f"Warning: Current price or price volatility is zero. Current Price: {current_price}, Price Volatility: {price_volatility}")
        return 0
    
    annualised_target = trading_capital * (sharpe_ratio / 2)
    daily_target = annualised_target / 16
    instrument_volatility = current_price * price_volatility
    
    if instrument_volatility == 0:
        print(f"Error: Instrument volatility computed as zero. This should not happen with non-zero inputs.")
        return 0
    
    volatility_scalar = daily_target / instrument_volatility
    subsystem_position = (forecast * volatility_scalar) / 10
    
    # Check for valid subsystem position
    if np.isinf(subsystem_position) or np.isnan(subsystem_position):
        print(f"Error: Computed subsystem position is invalid (Inf/Nan). Daily Target: {daily_target}, Instrument Volatility: {instrument_volatility}, Forecast: {forecast}")
        return 0
    
    final_position_size = round(subsystem_position)
    return final_position_size

def calculate_price_volatility(close_prices, lookback_period=config.getint('parameter', 'lookback_period_for_price_volatility')):
    """
    Calculate the price volatility based on close prices using an Exponentially Weighted Moving Average (EWMA) approach.

    Args:
        close_prices (np.array): Array of close prices.
        lookback_period (int): Number of periods to look back for calculating volatility.

    Returns:
        float: Price volatility calculated based on the EWMA approach.
    """
    # Check if close_prices is empty
    if len(close_prices) == 0:
        raise ValueError('close_prices must not be empty.')
    
    close_prices = np.array(close_prices)
    
    # Check if lookback_period is within the range of close_prices
    if lookback_period >= len(close_prices):
        raise ValueError(f'lookback_period ({lookback_period}) must be less than the length of close_prices ({len(close_prices)}).')
    
    # Calculate returns
    returns = close_prices[1:] / close_prices[:-1] - 1
    
    lamda = 2 / (1 + lookback_period)
    
    # Calculate squared returns
    squared_returns = returns ** 2
    ewma_variance = squared_returns[0]
    
    # Calculate EWMA variance
    for i in range(1, len(returns)):
        ewma_variance = lamda * squared_returns[i] + (1 - lamda) * ewma_variance
    
    # Calculate volatility
    volatility = np.sqrt(ewma_variance)
    
    return volatility

class PandasData_Custom(bt.feeds.PandasData):
    lines = ('signal',)
    params = (('signal', 8),)

class ColumnBasedStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        if len(tickers) ==1:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))
    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))
    def next(self):
        if not self.position:
            if self.data.signal[0] > backtest_buy_signal_strength and self.data.signal[lookback_days_for_signal] < backtest_yesterday_signal_strength and len(self.data.close) > config.getint('parameter', 'min_initial_period'):
                forecast = self.data.signal[0]
                price_volatility = calculate_price_volatility(self.data.close.get(size=len(self.data.close)))
                size = calculate_position_size(self.broker.getcash(), 0.6, price_volatility, self.data.close[0], forecast)
                buy_price = self.data.close[0] * (1 - price_volatility/buy_price_denominator)
                self.buy(size=size, exectype=bt.Order.Limit, price=buy_price,valid=0)
                self.log(f'BUY CREATE at {buy_price:.2f}. {self.data.close[0] * (price_volatility/buy_price_denominator)} Lower than close price Close price: {self.data.close[0]}. Volatility: {self.data.close[0] * price_volatility:.2f}. Forecast: {forecast:.2f}. Size: {size:.2f}')
        else:
            if self.data.signal[0] < backtest_sell_signal_strength:
                self.close()

# class ColumnBasedStrategy(bt.Strategy):
#     params = (
#         ('trailing_stop_multiplier', 4),  # Trailing stop multiplier for volatility
#     )

#     def log(self, txt, dt=None):
#         ''' Logging function for this strategy'''
#         if len(tickers) == 1:
#             dt = dt or self.datas[0].datetime.date(0)
#             print('%s, %s' % (dt.isoformat(), txt))

#     def __init__(self):
#         self.dataclose = self.datas[0].close
#         self.signal = self.datas[0].signal  # Assuming you have a signal column
#         self.order = None
#         self.buyprice = None
#         self.buycomm = None
#         self.trailing_stop_price = None  # To keep track of the trailing stop price
#         self.highest_price = None  # To track the highest price reached

#     def notify_order(self, order):
#         if order.status in [order.Submitted, order.Accepted]:
#             return

#         if order.status in [order.Completed]:
#             if order.isbuy():
#                 self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
#                          (order.executed.price,
#                           order.executed.value,
#                           order.executed.comm))
#                 self.buyprice = order.executed.price
#                 self.buycomm = order.executed.comm
#                 self.highest_price = order.executed.price  # Initialize the highest price
#                 self.update_trailing_stop()  # Initialize trailing stop
#             else:
#                 self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
#                          (order.executed.price,
#                           order.executed.value,
#                           order.executed.comm))

#             self.bar_executed = len(self)

#         elif order.status in [order.Canceled, order.Margin, order.Rejected]:
#             self.log('Order Canceled/Margin/Rejected')

#         self.order = None

#     def notify_trade(self, trade):
#         if not trade.isclosed:
#             return

#         self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
#                  (trade.pnl, trade.pnlcomm))

#     def update_trailing_stop(self):
#         # Update the trailing stop price based on the highest price and volatility
#         price_volatility = calculate_price_volatility(self.data.close.get(size=len(self.data.close)))
#         self.trailing_stop_price = self.highest_price - (price_volatility * self.params.trailing_stop_multiplier)

#     def next(self):
#         if not self.position:
#             if self.signal[0] > backtest_buy_signal_strength and self.signal[lookback_days_for_signal] < backtest_yesterday_signal_strength and len(self.data.close) > config.getint('parameter', 'min_initial_period'):
#                 forecast = self.signal[0]
#                 price_volatility = calculate_price_volatility(self.data.close.get(size=len(self.data.close)))
#                 size = calculate_position_size(self.broker.getcash(), 0.6, price_volatility, self.data.close[0], forecast)
#                 buy_price = self.data.close[0] * (1 - price_volatility/buy_price_denominator)
#                 self.buy(size=size, exectype=bt.Order.Limit, price=buy_price, valid=0)
#                 self.log(f'BUY CREATE at {buy_price:.2f}. {self.data.close[0] * (price_volatility/buy_price_denominator)} Lower than close price Close price: {self.data.close[0]}. Volatility: {price_volatility:.2f}. Forecast: {forecast:.2f}. Size: {size:.2f}')
#         else:
#             # Update the highest price reached since the position was opened
#             self.highest_price = max(self.highest_price, self.data.close[0])

#             # Check for trailing stop loss condition
#             self.update_trailing_stop()
#             if self.data.close[0] < self.trailing_stop_price:
#                 self.log(f'SELL CREATE at {self.data.close[0]:.2f} due to trailing stop loss.')
#                 self.close()
#             elif self.signal[0] < backtest_sell_signal_strength:
#                 self.log(f'SELL CREATE at {self.data.close[0]:.2f} due to signal.')
#                 self.close()




# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        if len(tickers) ==1:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] > self.sma[0]:

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.dataclose[0] < self.sma[0]:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

def run_strategy_for_ticker(ticker):
    filtered_df_pandas = all_data_pandas[all_data_pandas['symbol'].isin([ticker])]
    data = PandasData_Custom(dataname=filtered_df_pandas, timeframe=bt.TimeFrame.Days, datetime='date', high=-1, low=-1, open=-1, close=-1, volume=-1, signal=-1)
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(ColumnBasedStrategy)
    # cerebro.addstrategy(SwingTradingStrategy)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    # cerebro.addanalyzer(bt.analyzers.Calmar, _name='calmar_ratio')
    # cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    # cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name='time_drawdown')
    # cerebro.addanalyzer(bt.analyzers.LogReturnsRolling, _name='log_returns_rolling')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe_ratio_a')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    # cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')
    # cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    # cerebro.addanalyzer(bt.analyzers.Transactions, _name='transactions')
    cerebro.broker.set_cash(config.getint('parameter', 'trading_capital'))
    cerebro.broker.setcommission(commission=config.getfloat('parameter', 'commission'))
    results = cerebro.run()
    if len(tickers) ==1:
        cerebro.plot(style='bar')
    sharpe_ratio = results[0].analyzers.sharpe_ratio.get_analysis()
    annual_return = results[0].analyzers.annual_return.get_analysis()
    # calmar_ratio = results[0].analyzers.calmar_ratio.calmar
    # drawdown = results[0].analyzers.drawdown.get_analysis()
    # time_drawdown = results[0].analyzers.time_drawdown.get_analysis()
    # log_returns_rolling = results[0].analyzers.log_returns_rolling.get_analysis()
    sharpe_ratio_a = results[0].analyzers.sharpe_ratio_a.get_analysis()
    sqn = results[0].analyzers.sqn.get_analysis()
    # trade_analyzer = results[0].analyzers.trade_analyzer.get_analysis()
    vwr = results[0].analyzers.vwr.get_analysis()
    return {
        'strategy': strategy,
        'ticker': ticker,
        'start_portfolio_value': cerebro.broker.startingcash,
        'end_portfolio_value': cerebro.broker.getvalue(),
        'sharpe_ratio': sharpe_ratio.get('sharperatio', None),
        'backtest_time': time_str,
        'sharpe_ratio_a': sharpe_ratio_a.get('sharperatio', None),
        'sqn': sqn.get('sqn', None),
        'vwr': vwr.get('vwr', None),
        'annual_return': annual_return,
        'buy_price_denominator': buy_price_denominator,
        'backtest_sell_signal_strength': backtest_sell_signal_strength,
        'backtest_buy_signal_strength': backtest_buy_signal_strength,
        'backtest_yesterday_signal_strength': backtest_yesterday_signal_strength,
        'lookback_days_for_signal': lookback_days_for_signal
    }

def save_results_to_csv(csv_file, results):
    csv_header = [
        'Time', 'Strategy', 'Ticker', 'Start_Portfolio_Value', 'End_Portfolio_Value', 'Sharpe_Ratio', 'Backtest_Time',
        'SharpeRatio_A', 'SQN', 'VWR', 'Annual_Return', 'buy_price_denominator', 'backtest_sell_signal_strength',
        'backtest_buy_signal_strength', 'backtest_yesterday_signal_strength', 'lookback_days_for_signal'
    ]
    file_exists = False
    try:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            if next(reader, None) == csv_header:
                file_exists = True
    except FileNotFoundError:
        pass

    if not file_exists:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        for result in results:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            annual_return_str = ', '.join([f"{year}: {return_value * 100:.2f}%" for year, return_value in result['annual_return'].items()]) if result['annual_return'] else ''
            writer.writerow([
                current_time, result['strategy'], result['ticker'], result['start_portfolio_value'], result['end_portfolio_value'],
                result.get('sharpe_ratio', ''), result['backtest_time'],
                result.get('sharpe_ratio_a', ''), result.get('sqn', ''), result.get('vwr', ''), annual_return_str,
                result['buy_price_denominator'], result['backtest_sell_signal_strength'],
                result['backtest_buy_signal_strength'], result['backtest_yesterday_signal_strength'],
                result['lookback_days_for_signal']
            ])

if __name__ == '__main__':
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(run_strategy_for_ticker, tickers), total=len(tickers), desc="Processing tickers"))

    save_results_to_csv(csv_file, results)
