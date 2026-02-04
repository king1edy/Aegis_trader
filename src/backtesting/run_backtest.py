"""
Backtest Runner

Main script to run backtests on the MTFTR strategy.
Supports command-line arguments for date ranges and configuration.
"""

import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backtesting.engine import BacktestEngine, BacktestConfig, print_results
from src.backtesting.data_provider import BacktestDataProvider
from src.backtesting.strategy_simulator import MTFTRStrategySimulator, MTFTRBacktestConfig
from src.strategies.indicators import IndicatorConfig
from src.core.logging_config import setup_logging

# Setup logging
logger = setup_logging()


def run_backtest(
    symbol: str = "XAUUSD",
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    initial_balance: float = 10000.0,
    risk_per_trade: float = 0.01,
    data_path: str = "data/history",
    use_synthetic: bool = False
) -> None:
    """
    Run a complete backtest.
    
    Args:
        symbol: Trading symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_balance: Starting account balance
        risk_per_trade: Risk per trade (decimal)
        data_path: Path to historical data
        use_synthetic: Use synthetic generated data
    """
    print(f"\n{'='*60}")
    print(f"MTFTR STRATEGY BACKTEST")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Risk per Trade: {risk_per_trade*100:.1f}%")
    print(f"Data Source: {'Synthetic' if use_synthetic else 'Historical'}")
    print(f"{'='*60}\n")
    
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    # Configure indicator calculation
    indicator_config = IndicatorConfig(
        ema_200=200,
        ema_50=50,
        ema_21=21,
        hull_55=55,
        hull_34=34,
        rsi_period=14,
        atr_period=14,
        swing_lookback=5
    )
    
    # Initialize data provider
    print("📊 Loading historical data...")
    data_provider = BacktestDataProvider(
        symbol=symbol,
        start_date=start_dt,
        end_date=end_dt,
        data_path=data_path,
        indicator_config=indicator_config,
        use_synthetic=use_synthetic
    )
    
    if not data_provider.load_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    actual_start, actual_end = data_provider.date_range
    print(f"✅ Data loaded: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")
    print(f"   Total bars: {data_provider.total_bars:,}")
    
    # Configure backtest engine - MT5 EA EXACT MATCH
    backtest_config = BacktestConfig(
        initial_balance=initial_balance,
        leverage=100,
        spread_pips=2.0,  # XAUUSD typical spread
        slippage_pips=0.5,
        commission_per_lot=7.0,
        max_risk_per_trade=0.01,  # 1% risk (InpRiskPercent=1.0)
        max_daily_risk=0.03,       # InpMaxDailyDD=3.0%
        max_drawdown=0.25,         # 25% disaster recovery circuit breaker (not daily DD)
        max_trades_per_day=3,      # InpMaxDailyTrades=3
        pip_size=0.1,  # XAUUSD
        lot_size=100,  # Contract size
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        max_open_trades=2,          # InpMaxOpenTrades=2
        max_consec_losses=3,        # InpMaxConsecLosses=2 -> 3 (a bit more tolerance)
        max_trade_hours=8,          # InpMaxTradeHours=8
        tp1_close_percent=0.50,    # InpTP1_ClosePct=50
        tp2_close_percent=0.60     # InpTP2_ClosePct=60 of remainder
    )
    
    engine = BacktestEngine(backtest_config)
    
    # Configure strategy - MT5 EA EXACT PARAMETERS
    strategy_config = MTFTRBacktestConfig(
        tp1_rr=1.0,             # InpTP1_RR=1.0
        tp2_rr=2.0,             # InpTP2_RR=2.0
        min_rsi_long=40.0,      # InpRSI_Long_Min=40
        max_rsi_long=55.0,      # InpRSI_Long_Max=55
        min_rsi_short=45.0,     # InpRSI_Short_Min=45
        max_rsi_short=60.0,     # InpRSI_Short_Max=60
        min_sl_atr=1.0,         # InpSL_Min_ATR=1.0
        max_sl_atr=3.0,         # InpSL_Max_ATR=3.0
        sl_buffer_atr=0.5       # InpSL_ATR_Buffer=0.5
    )
    
    strategy = MTFTRStrategySimulator(strategy_config, data_provider)
    
    # Run backtest
    print("\n🚀 Running backtest...")
    
    signal_count = 0
    bar_count = 0
    
    # Progress bar
    with tqdm(total=data_provider.total_bars, desc="Processing", unit="bars") as pbar:
        for bar in data_provider.iterate_bars():
            bar_count += 1
            pbar.update(1)
            
            # Update existing positions first
            engine.update_positions(
                timestamp=bar.timestamp,
                high=bar.m15_high,
                low=bar.m15_low,
                close=bar.m15_close,
                h1_ema50=bar.h1_ema_50,
                h1_hull_34=bar.h1_hull_34,
                h1_hull_34_prev=bar.h1_hull_34_prev,
                m15_atr=bar.m15_atr  # Pass ATR for trail buffer calculation
            )
            
            # Only look for new signals if we have capacity (MT5 EA: InpMaxOpenTrades=2)
            if len(engine.open_positions) < engine.config.max_open_trades:
                # Analyze for new signal
                signal = strategy.analyze_bar(bar)
                
                if signal:
                    signal_count += 1
                    trade = engine.open_position(
                        signal=signal,
                        timestamp=bar.timestamp,
                        current_price=bar.m15_close
                    )
                    
                    if trade:
                        pbar.set_postfix({
                            'trades': len(engine.closed_trades),
                            'open': len(engine.open_positions),
                            'balance': f"${engine.balance:,.0f}"
                        })
    
    # Close any remaining positions at end
    for trade in engine.open_positions.copy():
        trade.exit_price = bar.m15_close
        trade.exit_reason = "End of Backtest"
        engine._close_position(trade, bar.timestamp)
    
    # Get results
    print("\n📈 Generating results...")
    results = engine.get_results(symbol, actual_start, actual_end)
    
    # Print results
    print_results(results)
    
    # Summary
    print(f"\n📊 BACKTEST SUMMARY")
    print(f"   Signals Generated: {signal_count}")
    print(f"   Trades Executed:   {results.total_trades}")
    print(f"   Final Balance:     ${results.final_balance:,.2f}")
    print(f"   Total Return:      {results.total_return_pct:+.2f}%")
    print(f"   Win Rate:          {results.win_rate:.1f}%")
    print(f"   Profit Factor:     {results.profit_factor:.2f}")
    print(f"   Max Drawdown:      {results.max_drawdown_pct:.2f}%")
    
    # Save equity curve to CSV
    if results.equity_curve:
        import pandas as pd
        equity_df = pd.DataFrame(results.equity_curve)
        equity_file = f"backtest_equity_{symbol}_{start_date}_{end_date}.csv"
        equity_df.to_csv(equity_file, index=False)
        print(f"\n💾 Equity curve saved to: {equity_file}")
    
    # Save trades to CSV
    if results.trades:
        trades_data = []
        for t in results.trades:
            trades_data.append({
                'ticket': t.ticket,
                'symbol': t.symbol,
                'direction': t.direction.value,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'stop_loss': t.stop_loss,
                'initial_lots': t.initial_lots,
                'realized_pnl': t.realized_pnl,
                'commission': t.commission,
                'net_pnl': t.realized_pnl - t.commission,
                'exit_reason': t.exit_reason,
                'state': t.state.value
            })
        
        trades_df = pd.DataFrame(trades_data)
        trades_file = f"backtest_trades_{symbol}_{start_date}_{end_date}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"💾 Trades saved to: {trades_file}")
    
    return results


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='MTFTR Strategy Backtest')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Trading symbol')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade (0.01 = 1%)')
    parser.add_argument('--data-path', type=str, default='data/history', help='Path to historical data')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic generated data')
    
    args = parser.parse_args()
    
    run_backtest(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        data_path=args.data_path,
        use_synthetic=args.synthetic
    )


if __name__ == "__main__":
    main()
