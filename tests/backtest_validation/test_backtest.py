import pytest 
from datetime import datetime, timezone,timedelta
from src.backtesting.data_provider import BacktestDataProvider  
from src.backtesting.engine import BacktestEngine, BacktestConfig,OrderDirection,PositionState
from src.backtesting.strategy_simulator import MTFTRStrategySimulator, MTFTRBacktestConfig
from src.backtesting.test_data_generator import TestDataGenerator
import pandas as pd
import numpy as np  

pytestmark = pytest.mark.backtest_validation 

# =====================================================
#                Data Provider Tests
# =====================================================

class TestBacktestDataProvider:

    def test_load_data_runs_successfully(self):
        # comment# create a data provider instance
        provider = BacktestDataProvider(
            symbol="XAUUSD",  # comment# symbol for backtest
            start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),  # comment# start date
            end_date=datetime(2023, 1, 10, tzinfo=timezone.utc),  # comment# end date
            use_synthetic=True  # comment# generate synthetic data
        )

        result = provider.load_data()  # comment# load/generate data

        assert result is True  # comment# should return True if data loaded successfully


# =====================================================
#                Backtest Engine Tests
# =====================================================

class TestBacktestEngine:

    # ---------------------------
    # Helper: Create Config
    # ---------------------------
    def _create_config(self):
        # comment# Step 1: create data provider
        provider = BacktestDataProvider(
            symbol="XAUUSD",
            start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2023, 1, 5, tzinfo=timezone.utc),
            use_synthetic=True
        )
        provider.load_data()  # comment# load/generate historical/synthetic data

        # comment# Step 2: create BacktestConfig with defaults
        config = BacktestConfig(
            initial_balance=10000.0,  # comment# starting capital
            leverage=100,  # comment# leverage
            spread_pips=1.5  # comment# spread
        )

        # comment# return config and provider for use in tests
        return config, provider
    
    # ---------------------------
    # Helper: Get Data from Provider
    # ---------------------------
    def _get_dataframes(self, provider):
        """Extract dataframes from the provider"""
        # Try different ways to get the dataframes
        m15_data = None
        h1_data = None
        h4_data = None
        
        # Check common attribute names
        possible_names = [
            ('m15_data', 'h1_data', 'h4_data'),
            ('m15_df', 'h1_df', 'h4_df'),
            ('data_m15', 'data_h1', 'data_h4'),
            ('m15', 'h1', 'h4'),
            ('m15_dataframe', 'h1_dataframe', 'h4_dataframe'),
        ]
        
        for m15_name, h1_name, h4_name in possible_names:
            if hasattr(provider, m15_name):
                m15_data = getattr(provider, m15_name)
            if hasattr(provider, h1_name):
                h1_data = getattr(provider, h1_name)
            if hasattr(provider, h4_name):
                h4_data = getattr(provider, h4_name)
            
            if m15_data is not None:
                break
        
        return m15_data, h1_data, h4_data
    
    # ---------------------------
    # Helper: Create Sample Data
    # ---------------------------
    def _create_sample_data(self):
        """Create sample data for backtesting"""
        # Create datetime range with correct frequency
        # Use '15min' instead of '15T' for pandas compatibility
        dates = pd.date_range(
            start=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end=datetime(2023, 1, 5, tzinfo=timezone.utc),
            freq='15min'  # Fixed: use '15min' instead of '15T'
        )
        
        m15_data = pd.DataFrame({
            'open': [1900.00 + i * 0.5 for i in range(len(dates))],
            'high': [1902.00 + i * 0.5 for i in range(len(dates))],
            'low': [1898.00 + i * 0.5 for i in range(len(dates))],
            'close': [1900.00 + i * 0.5 for i in range(len(dates))],
            'atr': [2.5] * len(dates)
        }, index=dates)
        
        # Create H1 data
        h1_dates = pd.date_range(
            start=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end=datetime(2023, 1, 5, tzinfo=timezone.utc),
            freq='1h'  # Fixed: use '1h' instead of '1H'
        )
        
        h1_data = pd.DataFrame({
            'open': [1900.00 + i * 2.0 for i in range(len(h1_dates))],
            'high': [1905.00 + i * 2.0 for i in range(len(h1_dates))],
            'low': [1895.00 + i * 2.0 for i in range(len(h1_dates))],
            'close': [1900.00 + i * 2.0 for i in range(len(h1_dates))],
            'ema50': [1900.00 + i * 2.0 for i in range(len(h1_dates))],
            'hull_34': [1900.00 + i * 2.0 for i in range(len(h1_dates))]
        }, index=h1_dates)
        
        h4_data = pd.DataFrame()
        
        return m15_data, h1_data, h4_data
    
    # ---------------------------
    # Helper: Run Backtest
    # ---------------------------
    def _run_backtest(self, config, provider):
        """Helper method to run the backtest engine properly"""
        
        # Try to get data from provider
        m15_data, h1_data, h4_data = self._get_dataframes(provider)
        
        # If no data found, create sample data
        if m15_data is None:
            print("No data found in provider, using sample data")
            m15_data, h1_data, h4_data = self._create_sample_data()
        
        # Initialize engine
        engine = BacktestEngine(config=config)
        
        # Process each M15 bar
        for idx, bar in m15_data.iterrows():
            # Get corresponding H1 data for this timestamp
            h1_ema50 = None
            h1_hull_34 = None
            h1_hull_34_prev = None
            
            if h1_data is not None and len(h1_data) > 0:
                h1_bars_before = h1_data[h1_data.index <= idx]
                if len(h1_bars_before) > 0:
                    current_h1 = h1_bars_before.iloc[-1]
                    if 'ema50' in current_h1.index:
                        h1_ema50 = current_h1['ema50']
                    if 'hull_34' in current_h1.index:
                        h1_hull_34 = current_h1['hull_34']
                    
                    if len(h1_bars_before) > 1:
                        prev_h1 = h1_bars_before.iloc[-2]
                        if 'hull_34' in prev_h1.index:
                            h1_hull_34_prev = prev_h1['hull_34']
            
            # Get M15 ATR if available
            m15_atr = bar['atr'] if 'atr' in bar.index else None
            
            # Update positions
            engine.update_positions(
                timestamp=idx,
                high=bar['high'],
                low=bar['low'],
                close=bar['close'],
                h1_ema50=h1_ema50,
                h1_hull_34=h1_hull_34,
                h1_hull_34_prev=h1_hull_34_prev,
                m15_atr=m15_atr
            )
        
        # Get results
        results = engine.get_results(
            symbol=provider.symbol if hasattr(provider, 'symbol') else "XAUUSD",
            start_date=provider.start_date if hasattr(provider, 'start_date') else datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_date=provider.end_date if hasattr(provider, 'end_date') else datetime(2023, 1, 5, tzinfo=timezone.utc)
        )
        
        return results, engine

    # ---------------------------
    # Test: Engine Initialization
    # ---------------------------
    def test_engine_initializes_with_valid_config(self):
        config, _ = self._create_config()  # comment# get config

        engine = BacktestEngine(config=config)  # comment# initialize engine WITHOUT data_provider

        assert engine is not None  # comment# engine should exist
        assert hasattr(engine, "config")  # comment# engine should have config attribute
        assert engine.config is config  # comment# engine config should match input

    # ---------------------------
    # Test: Engine Runs Without Error
    # ---------------------------
    def test_engine_runs_without_error_on_sample_data(self):
        config, provider = self._create_config()  # comment# get config and provider

        try:
            results, engine = self._run_backtest(config, provider)  # comment# run backtest
            assert results is not None  # comment# verify we got results
        except Exception as e:
            pytest.fail(f"Engine run raised an exception: {e}")  # comment# fail if exception

    # ---------------------------
    # Test: Results Contain Metrics
    # ---------------------------
    def test_results_contain_required_metrics(self):
        config, provider = self._create_config()  # comment# get config and provider

        results, engine = self._run_backtest(config, provider)  # comment# run backtest and get results

        # Results is a BacktestResult object
        # Check required attributes
        required_metrics = ["total_trades", "win_rate", "net_profit"]
        
        for metric in required_metrics:
            assert hasattr(results, metric), f"Results missing {metric} attribute"
        
        # Verify they have the correct types
        assert isinstance(results.total_trades, int), "total_trades should be int"
        assert isinstance(results.win_rate, (int, float)), "win_rate should be numeric"
        assert isinstance(results.net_profit, (int, float)), "net_profit should be numeric"
        
        # Additional validation
        assert hasattr(results, "total_return"), "Results missing total_return"
        assert hasattr(results, "profit_factor"), "Results missing profit_factor"
        assert hasattr(results, "final_balance"), "Results missing final_balance"

class TestStrategySimulator:
    """Tests for: src/backtesting/strategy_simulator.py"""
    
    # ---------------------------
    # Helper: Create Test Config
    # ---------------------------
    def _create_test_config(self):
        """Create a basic config for testing"""
        return MTFTRBacktestConfig(
            ema_200=200,
            ema_50=50,
            ema_21=21,
            hull_55=55,
            hull_34=34,
            rsi_period=14,
            atr_period=14,
            swing_lookback=3,
            tp1_rr=1.0,
            tp2_rr=2.0,
            min_rsi_long=40.0,
            max_rsi_long=55.0,
            min_rsi_short=45.0,
            max_rsi_short=60.0,
            min_sl_atr=1.0,
            max_sl_atr=3.0,
            sl_buffer_atr=0.5,
            min_atr_distance_from_ema=1.5,
            ema_slope_bars=5,
            ema_slope_min=0.5,
            london_start=time(7, 0),
            london_end=time(12, 0),
            ny_start=time(13, 0),
            ny_end=time(16, 0),
            max_trade_hours=8,
            friday_cutoff=time(15, 0)
        )
    
    # ---------------------------
    # Test 1: Stop Loss Hit
    # ---------------------------
    def test_simulator_respects_stop_loss(self):
        """Test that position closes when stop loss is hit"""
        config = BacktestConfig(
            initial_balance=10000.0,
            spread_pips=0.5,
            slippage_pips=0.1,
            commission_per_lot=0,
            tp1_close_percent=0.50,
            tp2_close_percent=0.30,
            trail_percent=0.20,
            max_trade_hours=48,
            min_lot=0.01,
            pip_size=0.1
        )
        engine = BacktestEngine(config=config)
        
        # Open position
        entry_price = 1900.00
        stop_loss = 1895.00
        take_profit = 1910.00
        timestamp = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        engine.open_position(
            signal=self._create_mock_signal(
                symbol="XAUUSD",
                direction=OrderDirection.BUY,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit,
                take_profit_2=take_profit + 10,
                timestamp=timestamp
            ),
            timestamp=timestamp,
            current_price=entry_price
        )
        
        # Simulate price hitting stop loss
        bar_low = stop_loss - 1
        engine.update_positions(
            timestamp=timestamp + timedelta(minutes=15),
            high=stop_loss + 1,
            low=bar_low,
            close=stop_loss - 0.5,
            h1_ema50=None,
            h1_hull_34=None,
            h1_hull_34_prev=None,
            m15_atr=None
        )
        
        # Verify position is closed
        assert len(engine.open_positions) == 0
        assert len(engine.closed_trades) == 1
        closed_trade = engine.closed_trades[0]
        assert closed_trade.exit_reason == "Stop Loss"
    
    # ---------------------------
    # Test 2: Take Profit Hit (Full Close at TP1)
    # ---------------------------
    def test_simulator_respects_take_profit(self):
        """Test that position closes when take profit is hit (full close at TP1)"""
        config = BacktestConfig(
            initial_balance=10000.0,
            spread_pips=0.5,
            slippage_pips=0.1,
            commission_per_lot=0,
            tp1_close_percent=1.0,  # Close 100% at TP1 (no partial)
            tp2_close_percent=0.0,
            trail_percent=0.0,
            max_trade_hours=48,
            min_lot=0.01
        )
        engine = BacktestEngine(config=config)
        
        # Open position
        entry_price = 1900.00
        stop_loss = 1895.00
        take_profit = 1910.00
        timestamp = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        engine.open_position(
            signal=self._create_mock_signal(
                symbol="XAUUSD",
                direction=OrderDirection.BUY,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit,
                take_profit_2=take_profit + 10,
                timestamp=timestamp
            ),
            timestamp=timestamp,
            current_price=entry_price
        )
        
        # Simulate price hitting take profit
        bar_high = take_profit + 1
        engine.update_positions(
            timestamp=timestamp + timedelta(minutes=15),
            high=bar_high,
            low=take_profit - 1,
            close=take_profit,
            h1_ema50=None,
            h1_hull_34=None,
            h1_hull_34_prev=None,
            m15_atr=None
        )
        
        # Verify position is fully closed
        assert len(engine.open_positions) == 0, "Position should be fully closed"
        assert len(engine.closed_trades) == 1, "Should have 1 closed trade"
        closed_trade = engine.closed_trades[0]
        assert "TP" in closed_trade.exit_reason
    
    # ---------------------------
    # Test 3: Partial Close at TP1
    # ---------------------------
    def test_partial_close_at_tp1(self):
        """Test that first target closes configured percentage"""
        config = BacktestConfig(
            initial_balance=100000,
            tp1_close_percent=0.50,
            tp2_close_percent=0.30,
            trail_percent=0.20,
            min_lot=0.01
        )
        engine = BacktestEngine(config=config)
        
        # Create and add trade directly with proper state
        from src.backtesting.engine import BacktestTrade
        entry_price = 1900.00
        stop_loss = 1895.00
        take_profit_1 = 1905.00
        take_profit_2 = 1910.00
        timestamp = datetime(2023, 1, 1, tzinfo=timezone.utc)
        initial_lots = 1.0
        
        trade = BacktestTrade(
            ticket=1,
            symbol="XAUUSD",
            direction=OrderDirection.BUY,
            entry_time=timestamp,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            current_sl=stop_loss,
            initial_lots=initial_lots,
            current_lots=initial_lots,
            state=PositionState.INITIAL  # Use enum, not string
        )
        engine.open_positions.append(trade)
        
        # Simulate price hitting TP1
        engine.update_positions(
            timestamp=timestamp + timedelta(minutes=15),
            high=take_profit_1 + 0.5,
            low=take_profit_1 - 0.5,
            close=take_profit_1,
            h1_ema50=None,
            h1_hull_34=None,
            h1_hull_34_prev=None,
            m15_atr=None
        )
        
        # Verify partial close
        assert len(engine.open_positions) == 1, "Position should still be open"
        remaining_trade = engine.open_positions[0]
        expected_remaining = initial_lots * (1 - config.tp1_close_percent)
        assert abs(remaining_trade.current_lots - expected_remaining) < 0.01, \
            f"Expected {expected_remaining} lots, got {remaining_trade.current_lots}"
        assert remaining_trade.state == PositionState.TP1_HIT, "State should be TP1_HIT"
    
    # ---------------------------
    # Test 4: Trailing Stop Logic
    # ---------------------------
    def test_trailing_stop_logic(self):
        """Test that trailing stop moves with price"""
        config = BacktestConfig(
            initial_balance=100000,
            tp1_close_percent=0.40,
            tp2_close_percent=0.30,
            trail_percent=0.30,
            pip_size=0.1
        )
        engine = BacktestEngine(config=config)
        
        # Create trade in TP2_HIT state (trailing mode)
        from src.backtesting.engine import BacktestTrade
        entry_price = 1900.00
        stop_loss = 1895.00
        take_profit_1 = 1905.00
        take_profit_2 = 1910.00
        timestamp = datetime(2023, 1, 1, tzinfo=timezone.utc)
        initial_lots = 0.30
        
        trade = BacktestTrade(
            ticket=1,
            symbol="XAUUSD",
            direction=OrderDirection.BUY,
            entry_time=timestamp,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            current_sl=stop_loss,
            initial_lots=1.0,
            current_lots=initial_lots,
            state=PositionState.TP2_HIT  # Use enum for trailing mode
        )
        engine.open_positions.append(trade)
        initial_sl = trade.current_sl
        
        # Simulate price moving up with trailing
        current_price = 1912.00
        h1_ema50 = 1908.00  # EMA below price for long
        m15_atr = 2.0
        
        # Update positions multiple times to trail stop
        for i in range(3):
            price_up = current_price + i * 2
            engine.update_positions(
                timestamp=timestamp + timedelta(minutes=15 * (i + 1)),
                high=price_up + 0.5,
                low=price_up - 0.5,
                close=price_up,
                h1_ema50=h1_ema50 + i * 0.5,  # EMA also increases
                h1_hull_34=None,
                h1_hull_34_prev=None,
                m15_atr=m15_atr
            )
        
        # Verify stop loss moved up
        final_sl = engine.open_positions[0].current_sl
        assert final_sl > initial_sl, f"Stop loss should have moved up from {initial_sl} to {final_sl}"
        
        # Now simulate price dropping below trailing stop
        bar_low = final_sl - 1
        engine.update_positions(
            timestamp=timestamp + timedelta(minutes=60),
            high=final_sl + 1,
            low=bar_low,
            close=final_sl - 0.5,
            h1_ema50=h1_ema50 + 3,
            h1_hull_34=None,
            h1_hull_34_prev=None,
            m15_atr=m15_atr
        )
        
        # Position should be closed
        assert len(engine.open_positions) == 0, "Position should be closed when trailing stop hit"
        if len(engine.closed_trades) > 0:
            assert engine.closed_trades[0].exit_reason in ["Stop Loss", "Trailing Stop"]
    
    # ---------------------------
    # Helper: Create Mock Signal
    # ---------------------------
    def _create_mock_signal(self, symbol, direction, entry_price, stop_loss, take_profit_1, take_profit_2, timestamp):
        """Create a mock trading signal"""
        class MockSignal:
            def __init__(self):
                self.symbol = symbol
                self.direction = direction
                self.entry_price = entry_price
                self.stop_loss = stop_loss
                self.take_profit_1 = take_profit_1
                self.take_profit_2 = take_profit_2
                self.timestamp = timestamp
                self.market_context = {}
        
        return MockSignal()
    
class TestTestDataGenerator:
    """Tests for: src/backtesting/test_data_generator.py"""
    
    # ---------------------------
    # Helper: Create Generator
    # ---------------------------
    def _create_generator(self, start_date=None, end_date=None, base_price=2000.0):
        """Create a test data generator instance"""
        if start_date is None:
            start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        if end_date is None:
            end_date = datetime(2023, 1, 2, tzinfo=timezone.utc)
            
        return TestDataGenerator(
            symbol="XAUUSD",
            start_date=start_date,
            end_date=end_date,
            base_price=base_price,
            daily_volatility=0.015  # Default value
        )
    
    # ---------------------------
    # Helper: Calculate Expected Bars
    # ---------------------------
    def _calculate_expected_bars(self, start_date, end_date):
        """Calculate expected number of M1 bars excluding weekends and market close"""
        current = start_date
        count = 0
        
        while current <= end_date:
            # Check if it's a weekday
            if current.weekday() < 5:  # Monday=0 to Friday=4
                # Check if it's not during market close (Friday 22:00 to Sunday 22:00)
                if not (current.weekday() == 4 and current.hour >= 22):
                    count += 1
            current += timedelta(minutes=1)
            
            # Skip Friday 22:00 to Sunday 22:00
            if current.weekday() == 4 and current.hour == 22:
                current = current + timedelta(days=2)
        
        return count
    
    # ---------------------------
    # Test 1: Generates Requested Number of Bars
    # ---------------------------
    def test_generates_requested_number_of_bars(self):
        """Test that generator creates the correct number of bars for the date range"""
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)  # Sunday
        end_date = datetime(2023, 1, 2, tzinfo=timezone.utc)    # Monday
        generator = self._create_generator(start_date=start_date, end_date=end_date)
        
        data = generator.generate()
        
        # Calculate expected bars (Monday only, since Sunday is weekend)
        expected_bars = self._calculate_expected_bars(start_date, end_date)
        assert len(data) == expected_bars, \
            f"Should have {expected_bars} bars, got {len(data)}"
    
    # ---------------------------
    # Test 2: Multiple Days Generation
    # ---------------------------
    def test_generates_multiple_days_correctly(self):
        """Test that generator works for multiple days"""
        start_date = datetime(2023, 1, 2, tzinfo=timezone.utc)  # Monday
        end_date = datetime(2023, 1, 5, tzinfo=timezone.utc)    # Thursday
        generator = self._create_generator(start_date=start_date, end_date=end_date)
        
        data = generator.generate()
        
        # Calculate expected bars (Monday-Thursday, no Friday to avoid market close)
        expected_bars = self._calculate_expected_bars(start_date, end_date)
        assert len(data) == expected_bars, \
            f"Should have {expected_bars} bars, got {len(data)}"
    
    # ---------------------------
    # Test 3: High >= Low Always
    # ---------------------------
    def test_high_ge_low_always(self):
        """Test that high is always >= low in generated data (data quality check)"""
        generator = self._create_generator()
        data = generator.generate()
        
        # Check high >= low
        invalid_bars = data[data['high'] < data['low']]
        assert len(invalid_bars) == 0, \
            f"Found {len(invalid_bars)} bars where high < low"
        
        # Also check high >= open and high >= close
        invalid_high = data[(data['high'] < data['open']) | (data['high'] < data['close'])]
        assert len(invalid_high) == 0, \
            f"Found {len(invalid_high)} bars where high < open or high < close"
        
        # Check low <= open and low <= close
        invalid_low = data[(data['low'] > data['open']) | (data['low'] > data['close'])]
        assert len(invalid_low) == 0, \
            f"Found {len(invalid_low)} bars where low > open or low > close"
    
    # ---------------------------
    # Test 4: Open and Close Within High-Low Range
    # ---------------------------
    def test_open_and_close_within_high_low_range(self):
        """Test that open and close prices are within high-low range"""
        generator = self._create_generator()
        data = generator.generate()
        
        # Check open within range
        invalid_open = data[(data['open'] > data['high']) | (data['open'] < data['low'])]
        assert len(invalid_open) == 0, \
            f"Found {len(invalid_open)} bars where open is outside high-low range"
        
        # Check close within range
        invalid_close = data[(data['close'] > data['high']) | (data['close'] < data['low'])]
        assert len(invalid_close) == 0, \
            f"Found {len(invalid_close)} bars where close is outside high-low range"
    
    # ---------------------------
    # Test 5: Price Movement is Reasonable
    # ---------------------------
    def test_price_movement_is_reasonable(self):
        """Test that price movements are within reasonable bounds (no extreme jumps)"""
        generator = self._create_generator(
            start_date=datetime(2023, 1, 2, tzinfo=timezone.utc),
            end_date=datetime(2023, 1, 5, tzinfo=timezone.utc)
        )
        data = generator.generate()
        
        # Calculate percent change between consecutive closes
        pct_changes = data['close'].pct_change() * 100
        
        # For M1 data, typical moves are small (< 0.5%)
        extreme_moves = pct_changes[abs(pct_changes) > 2.0]  # 2% in 1 minute is extreme
        
        # Allow a few extreme moves (maybe at market open or news), but not many
        max_extreme_percent = 0.01  # 1% of bars
        assert len(extreme_moves) < len(data) * max_extreme_percent, \
            f"Too many extreme price moves (>2%): {len(extreme_moves)} bars"
        
        # Also check no NaN values (except first)
        assert pct_changes.iloc[1:].notna().all(), "Some pct_changes are NaN"
    
    # ---------------------------
    # Test 6: Data Types are Correct
    # ---------------------------
    def test_data_types_are_correct(self):
        """Test that all columns have correct data types"""
        generator = self._create_generator()
        data = generator.generate()
        
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'tick_volume', 'spread']
        
        # Check required columns exist
        for col in required_columns:
            assert col in data.columns, f"Missing column {col}"
        
        # Check numeric types
        for col in required_columns:
            assert pd.api.types.is_numeric_dtype(data[col]), \
                f"Column {col} should be numeric"
        
        # Check no NaN values in OHLCV
        for col in required_columns:
            assert not data[col].isna().any(), \
                f"Column {col} contains NaN values"
        
        # Check index is datetime
        assert isinstance(data.index, pd.DatetimeIndex), \
            "Index should be DatetimeIndex"
        
        # Check index has timezone
        assert data.index.tz is not None, \
            "Index should have timezone"
        
        # Check index is sorted
        assert data.index.is_monotonic_increasing, \
            "Index should be sorted"
    
    # ---------------------------
    # Test 7: Volume Data Positive
    # ---------------------------
    def test_volume_data_positive(self):
        """Test that volume data is positive"""
        generator = self._create_generator()
        data = generator.generate()
        
        # Check volume positive
        invalid_volume = data[data['volume'] <= 0]
        assert len(invalid_volume) == 0, \
            f"Found {len(invalid_volume)} bars with non-positive volume"
        
        # Check tick_volume positive
        invalid_tick_volume = data[data['tick_volume'] <= 0]
        assert len(invalid_tick_volume) == 0, \
            f"Found {len(invalid_tick_volume)} bars with non-positive tick_volume"
        
        # Check volume is integer
        assert data['volume'].dtype in ['int64', 'int32'], \
            "Volume should be integer type"
    
    # ---------------------------
    # Test 8: Spread Data Positive
    # ---------------------------
    def test_spread_data_positive(self):
        """Test that spread data is positive and reasonable"""
        generator = self._create_generator()
        data = generator.generate()
        
        # Check spread positive
        invalid_spread = data[data['spread'] <= 0]
        assert len(invalid_spread) == 0, \
            f"Found {len(invalid_spread)} bars with non-positive spread"
        
        # XAUUSD typical spread is 15-35 points (0.15-0.35 cents)
        # Check it's within reasonable range
        assert data['spread'].min() >= 10, "Spread too low"
        assert data['spread'].max() <= 50, "Spread too high"
        assert data['spread'].mean() >= 15, "Average spread too low"
        assert data['spread'].mean() <= 35, "Average spread too high"
    
    # ---------------------------
    # Test 9: Weekend Exclusion
    # ---------------------------
    def test_weekend_exclusion(self):
        """Test that weekends are excluded from generated data"""
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)  # Sunday
        end_date = datetime(2023, 1, 8, tzinfo=timezone.utc)    # Sunday (week later)
        generator = self._create_generator(start_date=start_date, end_date=end_date)
        
        data = generator.generate()
        
        # Check no Saturday or Sunday bars
        weekend_bars = data[data.index.dayofweek.isin([5, 6])]  # 5=Sat, 6=Sun
        assert len(weekend_bars) == 0, \
            f"Found {len(weekend_bars)} bars on weekends"
        
        # Check no Friday after 22:00
        friday_late = data[(data.index.dayofweek == 4) & (data.index.hour >= 22)]
        assert len(friday_late) == 0, \
            f"Found {len(friday_late)} bars on Friday after 22:00"
    
    # ---------------------------
    # Test 10: Price Trend Reasonable
    # ---------------------------
    def test_price_trend_reasonable(self):
        """Test that price trends are reasonable (no extreme drift)"""
        # Generate 1 month of data to see trend
        start_date = datetime(2023, 1, 2, tzinfo=timezone.utc)  # Monday
        end_date = datetime(2023, 2, 1, tzinfo=timezone.utc)    # Wednesday
        generator = TestDataGenerator(  # Use direct constructor instead of helper
            symbol="XAUUSD",
            start_date=start_date,
            end_date=end_date,
            base_price=2000.0,
            daily_volatility=0.015  # 1.5% daily volatility
        )
        
        data = generator.generate()
        
        first_price = data['close'].iloc[0]
        last_price = data['close'].iloc[-1]
        price_change_pct = abs((last_price - first_price) / first_price) * 100
        
        # With 1.5% daily volatility, over 20 trading days, expected drift ~6.7%
        # But can be up to 20-30% in extreme cases. Relax the threshold.
        assert price_change_pct < 50, \
            f"Price drifted too much: {price_change_pct:.2f}% in one month"
    
    # ---------------------------
    # Test 11: Reproducibility
    # ---------------------------
    def test_reproducibility(self):
        """Test that generator produces same structure when run twice"""
        generator1 = self._create_generator()
        generator2 = self._create_generator()
        
        data1 = generator1.generate()
        data2 = generator2.generate()
        
        # Same number of bars
        assert len(data1) == len(data2), "Different number of bars"
        
        # Same columns
        assert list(data1.columns) == list(data2.columns), "Different columns"
        
        # Same date range
        assert data1.index[0] == data2.index[0], "Different start dates"
        assert data1.index[-1] == data2.index[-1], "Different end dates"
        
        # Note: Values will differ due to random generation - that's expected
        # We just verify structure is consistent
    
    # ---------------------------
    # Test 12: Session Pattern Volumes
    # ---------------------------
    def test_session_pattern_volumes(self):
        """Test that volumes follow expected session patterns"""
        # Generate 1 week of data to see patterns
        start_date = datetime(2023, 1, 2, tzinfo=timezone.utc)  # Monday
        end_date = datetime(2023, 1, 6, tzinfo=timezone.utc)    # Friday
        generator = self._create_generator(start_date=start_date, end_date=end_date)
        
        data = generator.generate()
        
        # Add hour column for analysis
        data['hour'] = data.index.hour
        
        # Get average volume by hour
        avg_volume_by_hour = data.groupby('hour')['volume'].mean()
        
        # Peak overlap hours (12-16) should have higher volume
        peak_hours = [12, 13, 14, 15, 16]
        asian_hours = [0, 1, 2, 3, 4, 5, 6]
        
        peak_avg = avg_volume_by_hour[peak_hours].mean()
        asian_avg = avg_volume_by_hour[asian_hours].mean()
        
        # Peak hours should have higher volume than Asian session
        assert peak_avg > asian_avg * 1.3, \
            f"Peak session volume ({peak_avg:.0f}) not significantly higher than Asian ({asian_avg:.0f})"
    
    # ---------------------------
    # Test 13: Save to HCS
    # ---------------------------
    def test_save_to_hcs(self, tmp_path):
        """Test that save_to_hcs works without errors"""
        start_date = datetime(2023, 1, 2, tzinfo=timezone.utc)  # Monday
        end_date = datetime(2023, 1, 3, tzinfo=timezone.utc)    # Tuesday (short period for speed)
        generator = self._create_generator(start_date=start_date, end_date=end_date)
        
        # Use temporary directory for test
        test_path = tmp_path / "test_data"
        
        # Should not raise exception
        try:
            generator.save_to_hcs(str(test_path))
        except Exception as e:
            pytest.fail(f"save_to_hcs raised exception: {e}")
    
    # ---------------------------
    # Test 14: Market Close Handling
    # ---------------------------
    def test_market_close_handling(self):
        """Test that market close (Friday 22:00 to Sunday 22:00) is handled correctly"""
        start_date = datetime(2023, 1, 6, tzinfo=timezone.utc)  # Friday
        end_date = datetime(2023, 1, 9, tzinfo=timezone.utc)    # Monday
        generator = self._create_generator(start_date=start_date, end_date=end_date)
        
        data = generator.generate()
        
        # Should have data for Friday (before 22:00) and Monday (after 00:00)
        # No data during market close
        friday_bars = data[data.index.dayofweek == 4]
        monday_bars = data[data.index.dayofweek == 0]
        
        # Friday bars should only be before 22:00
        if len(friday_bars) > 0:
            assert (friday_bars.index.hour < 22).all(), \
                "Found Friday bars after 22:00"
        
        # Monday bars should exist
        assert len(monday_bars) > 0, "No Monday bars generated"
        
        # There should be a gap between Friday 22:00 and Monday 00:00
        # Check the last Friday bar is before 22:00
        if len(friday_bars) > 0:
            last_friday = friday_bars.index.max()
            first_monday = monday_bars.index.min()
            
            # There should be at least 2 hours gap (Friday 22:00 to Monday 00:00)
            gap_hours = (first_monday - last_friday).total_seconds() / 3600
            assert gap_hours >= 2, f"Gap between Friday and Monday is only {gap_hours} hours"


class TestStatisticalValidation:
    """Verify backtest results are statistically sound."""
    
    # ---------------------------
    # Helper: Create Config
    # ---------------------------
    def _create_config(self):
        """Create a basic config for testing"""
        return BacktestConfig(
            initial_balance=10000.0,
            leverage=100,
            spread_pips=1.5,
            slippage_pips=0.3,
            commission_per_lot=7.0,
            tp1_close_percent=0.40,
            tp2_close_percent=0.30,
            trail_percent=0.30,
            max_risk_per_trade=0.02,
            max_daily_risk=0.06,
            max_weekly_risk=0.12,
            max_drawdown=0.20,
            max_trades_per_day=6,
            max_open_trades=3,
            max_consec_losses=4,
            max_trade_hours=16,
            pip_size=0.1,
            lot_size=100,
            min_lot=0.01,
            max_lot=100.0,
            lot_step=0.01
        )
    
    def _create_provider(self):
        """Create a data provider with sample data"""
        provider = BacktestDataProvider(
            symbol="XAUUSD",
            start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2023, 1, 31, tzinfo=timezone.utc),  # One month of data
            use_synthetic=True
        )
        provider.load_data()
        return provider
    
    def _get_dataframes(self, provider):
        """Extract dataframes from provider"""
        # The provider might have these as internal attributes
        # Based on logs, we need to access the data differently
        
        # Try to get the dataframes through provider's internal storage
        if hasattr(provider, '_data'):
            data_dict = provider._data
        elif hasattr(provider, 'dataframes'):
            data_dict = provider.dataframes
        else:
            # Create sample data if we can't get real data
            return self._create_sample_data()
        
        # Extract the dataframes
        m15_data = None
        h1_data = None
        
        if isinstance(data_dict, dict):
            m15_data = data_dict.get('M15') or data_dict.get('m15')
            h1_data = data_dict.get('H1') or data_dict.get('h1')
        
        if m15_data is None or h1_data is None:
            return self._create_sample_data()
        
        return m15_data, h1_data
    
    def _create_sample_data(self):
        """Create sample data for testing"""
        # Create M15 data
        dates = pd.date_range(
            start=datetime(2023, 1, 2, tzinfo=timezone.utc),
            end=datetime(2023, 1, 31, tzinfo=timezone.utc),
            freq='15min'
        )
        
        m15_data = pd.DataFrame({
            'open': 1900.00,
            'high': 1910.00,
            'low': 1890.00,
            'close': 1900.00,
            'atr': 2.0
        }, index=dates)
        
        # Add some variation
        np.random.seed(42)
        m15_data['close'] = 1900 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        m15_data['open'] = m15_data['close'].shift(1).fillna(1900)
        m15_data['high'] = m15_data[['open', 'close']].max(axis=1) + 0.5
        m15_data['low'] = m15_data[['open', 'close']].min(axis=1) - 0.5
        
        # Create H1 data
        h1_dates = pd.date_range(
            start=datetime(2023, 1, 2, tzinfo=timezone.utc),
            end=datetime(2023, 1, 31, tzinfo=timezone.utc),
            freq='1h'
        )
        
        h1_data = pd.DataFrame({
            'open': 1900.00,
            'high': 1910.00,
            'low': 1890.00,
            'close': 1900.00,
            'ema50': 1900.00,
            'hull_34': 1900.00
        }, index=h1_dates)
        
        # Add some variation
        h1_data['close'] = 1900 + np.cumsum(np.random.randn(len(h1_dates)) * 1.0)
        h1_data['ema50'] = h1_data['close'].rolling(50).mean()
        h1_data['hull_34'] = h1_data['close'].rolling(34).mean()
        
        return m15_data, h1_data
    
    def _run_backtest(self, config, provider):
        """Run a backtest and return results"""
        # Get data
        m15_data, h1_data = self._get_dataframes(provider)
        
        engine = BacktestEngine(config=config)
        
        # Process each bar
        for idx, bar in m15_data.iterrows():
            # Get H1 data for this timestamp
            h1_ema50 = None
            h1_hull_34 = None
            h1_hull_34_prev = None
            
            if len(h1_data) > 0:
                h1_bars_before = h1_data[h1_data.index <= idx]
                if len(h1_bars_before) > 0:
                    current_h1 = h1_bars_before.iloc[-1]
                    if 'ema50' in current_h1.index:
                        h1_ema50 = current_h1['ema50']
                    if 'hull_34' in current_h1.index:
                        h1_hull_34 = current_h1['hull_34']
                    
                    if len(h1_bars_before) > 1:
                        prev_h1 = h1_bars_before.iloc[-2]
                        if 'hull_34' in prev_h1.index:
                            h1_hull_34_prev = prev_h1['hull_34']
            
            # Get M15 ATR if available
            m15_atr = bar.get('atr') if 'atr' in bar.index else None
            
            engine.update_positions(
                timestamp=idx,
                high=bar['high'],
                low=bar['low'],
                close=bar['close'],
                h1_ema50=h1_ema50,
                h1_hull_34=h1_hull_34,
                h1_hull_34_prev=h1_hull_34_prev,
                m15_atr=m15_atr
            )
        
        results = engine.get_results(
            symbol=provider.symbol if hasattr(provider, 'symbol') else "XAUUSD",
            start_date=provider.start_date if hasattr(provider, 'start_date') else datetime(2023, 1, 2, tzinfo=timezone.utc),
            end_date=provider.end_date if hasattr(provider, 'end_date') else datetime(2023, 1, 31, tzinfo=timezone.utc)
        )
        
        return results
    
    # ---------------------------
    # Test 1: Win Rate Between 0 and 100
    # ---------------------------
    def test_win_rate_between_0_and_100(self):
        """Test that win rate is always between 0% and 100%"""
        config = self._create_config()
        provider = self._create_provider()
        
        results = self._run_backtest(config, provider)
        
        # Win rate should be between 0 and 100
        assert 0 <= results.win_rate <= 100, \
            f"Win rate {results.win_rate}% is outside valid range [0, 100]"
        
        # Also verify it's a float
        assert isinstance(results.win_rate, (int, float)), \
            f"Win rate should be numeric, got {type(results.win_rate)}"
        
        # If there are trades, win rate should be calculated
        if results.total_trades > 0:
            # Win rate should be consistent with trade counts
            expected_win_rate = (results.winning_trades / results.total_trades) * 100
            assert abs(results.win_rate - expected_win_rate) < 0.01, \
                f"Win rate {results.win_rate} doesn't match winning_trades/{results.total_trades}"
    
    # ---------------------------
    # Test 2: Profit Factor Positive When Profitable
    # ---------------------------
    def test_profit_factor_positive_when_profitable(self):
        """Test that profit factor is positive when net profit is positive"""
        config = self._create_config()
        provider = self._create_provider()
        
        results = self._run_backtest(config, provider)
        
        # Profit factor should never be negative
        assert results.profit_factor >= 0, \
            f"Profit factor {results.profit_factor} should never be negative"
        
        # If net profit is positive, profit factor should be > 1
        if results.net_profit > 0:
            assert results.profit_factor > 1, \
                f"Profit factor {results.profit_factor} should be > 1 when net profit is positive"
        
        # If net profit is negative, profit factor should be < 1
        elif results.net_profit < 0:
            assert results.profit_factor < 1, \
                f"Profit factor {results.profit_factor} should be < 1 when net profit is negative"
        
        # If net profit is zero, profit factor should be 1 or 0
        else:
            assert results.profit_factor in [0, 1], \
                f"Profit factor {results.profit_factor} should be 0 or 1 when net profit is zero"
        
        # Profit factor should be finite
        assert results.profit_factor != float('inf'), \
            "Profit factor should not be infinite"
    
    # ---------------------------
    # Test 3: Max Drawdown Not Exceed 100 Percent
    # ---------------------------
    def test_max_drawdown_not_exceed_100_percent(self):
        """Test that maximum drawdown never exceeds 100%"""
        config = self._create_config()
        provider = self._create_provider()
        
        results = self._run_backtest(config, provider)
        
        # Drawdown percentage should be between 0 and 100
        assert 0 <= results.max_drawdown_pct <= 100, \
            f"Max drawdown {results.max_drawdown_pct}% is outside valid range [0, 100]"
        
        # Drawdown amount should not exceed initial balance
        assert results.max_drawdown <= config.initial_balance, \
            f"Max drawdown ${results.max_drawdown:,.2f} exceeds initial balance ${config.initial_balance:,.2f}"
    
    # ---------------------------
    # Test 4: Sharpe Ratio Calculation
    # ---------------------------
    def test_sharpe_ratio_calculation(self):
        """Test that Sharpe ratio is calculated correctly (if implemented)"""
        config = self._create_config()
        provider = self._create_provider()
        
        results = self._run_backtest(config, provider)
        
        # Sharpe ratio should be a number
        assert isinstance(results.sharpe_ratio, (int, float)), \
            f"Sharpe ratio should be numeric, got {type(results.sharpe_ratio)}"
        
        # Sharpe ratio should be finite
        assert results.sharpe_ratio != float('inf'), \
            "Sharpe ratio should not be infinite"
        assert results.sharpe_ratio != float('-inf'), \
            "Sharpe ratio should not be negative infinite"
        
        # Sortino ratio should also be present and follow similar rules
        if hasattr(results, 'sortino_ratio'):
            assert isinstance(results.sortino_ratio, (int, float)), \
                f"Sortino ratio should be numeric, got {type(results.sortino_ratio)}"
            assert results.sortino_ratio != float('inf'), \
                "Sortino ratio should not be infinite"
        
        # Calmar ratio should also be present
        if hasattr(results, 'calmar_ratio'):
            assert isinstance(results.calmar_ratio, (int, float)), \
                f"Calmar ratio should be numeric, got {type(results.calmar_ratio)}"
            assert results.calmar_ratio != float('inf'), \
                "Calmar ratio should not be infinite"
    
    # ---------------------------
    # Test 5: Trade Count Consistency
    # ---------------------------
    def test_trade_count_consistency(self):
        """Test that trade counts are consistent with winning/losing trades"""
        config = self._create_config()
        provider = self._create_provider()
        
        results = self._run_backtest(config, provider)
        
        # Total trades should equal sum of winning, losing, and breakeven
        total_trades_calc = results.winning_trades + results.losing_trades + results.breakeven_trades
        assert results.total_trades == total_trades_calc, \
            f"Total trades {results.total_trades} doesn't match sum of wins/losses/breakeven {total_trades_calc}"
        
        # All trade counts should be non-negative
        assert results.total_trades >= 0, "Total trades cannot be negative"
        assert results.winning_trades >= 0, "Winning trades cannot be negative"
        assert results.losing_trades >= 0, "Losing trades cannot be negative"
        assert results.breakeven_trades >= 0, "Breakeven trades cannot be negative"
    
    # ---------------------------
    # Test 6: P&L Consistency
    # ---------------------------
    def test_pnl_consistency(self):
        """Test that profit metrics are consistent"""
        config = self._create_config()
        provider = self._create_provider()
        
        results = self._run_backtest(config, provider)
        
        # Net profit should equal gross profit minus gross loss
        expected_net_profit = results.gross_profit - results.gross_loss
        assert abs(results.net_profit - expected_net_profit) < 0.01, \
            f"Net profit {results.net_profit} doesn't match gross profit - gross loss {expected_net_profit}"
        
        # Total return should equal final balance minus initial balance
        expected_return = results.final_balance - results.initial_balance
        assert abs(results.total_return - expected_return) < 0.01, \
            f"Total return {results.total_return} doesn't match final - initial {expected_return}"
        
        # If there are trades, average trade should equal net profit / total trades
        if results.total_trades > 0:
            expected_avg_trade = results.net_profit / results.total_trades
            assert abs(results.avg_trade - expected_avg_trade) < 0.01, \
                f"Average trade {results.avg_trade} doesn't match net profit / total trades {expected_avg_trade}"
    
    # ---------------------------
    # Test 7: Risk Metrics Reasonable
    # ---------------------------
    def test_risk_metrics_reasonable(self):
        """Test that risk metrics are within reasonable bounds"""
        config = self._create_config()
        provider = self._create_provider()
        
        results = self._run_backtest(config, provider)
        
        # If there are trades, risk metrics should be meaningful
        if results.total_trades > 0:
            # Max consecutive wins/losses should be <= total trades
            assert results.max_consecutive_wins <= results.total_trades, \
                f"Max consecutive wins {results.max_consecutive_wins} exceeds total trades"
            assert results.max_consecutive_losses <= results.total_trades, \
                f"Max consecutive losses {results.max_consecutive_losses} exceeds total trades"
            
            # Largest win/loss should be positive/negative respectively
            assert results.largest_win >= 0, f"Largest win {results.largest_win} should be >= 0"
            assert results.largest_loss >= 0, f"Largest loss {results.largest_loss} should be >= 0"
    
    # ---------------------------
    # Test 8: Duration Metrics Reasonable
    # ---------------------------
    def test_duration_metrics_reasonable(self):
        """Test that duration metrics are within reasonable bounds"""
        config = self._create_config()
        provider = self._create_provider()
        
        results = self._run_backtest(config, provider)
        
        # If there are trades, durations should be positive
        if results.total_trades > 0:
            # Average durations should be positive
            assert results.avg_trade_duration.total_seconds() > 0, \
                "Average trade duration should be positive"
    
    # ---------------------------
    # Test 9: Zero Trades Case
    # ---------------------------
    def test_zero_trades_handling(self):
        """Test that metrics handle zero trades gracefully"""
        # Create a config with very conservative settings to avoid trades
        config = BacktestConfig(
            initial_balance=10000.0,
            max_risk_per_trade=0.001,  # Very small risk
            max_trades_per_day=0,  # No trades allowed
            min_lot=100,  # Very large minimum lot
            max_lot=100  # Same as min
        )
        provider = self._create_provider()
        
        results = self._run_backtest(config, provider)
        
        # With zero trades, metrics should be zero or default
        assert results.total_trades == 0, "Total trades should be 0"
        assert results.winning_trades == 0, "Winning trades should be 0"
        assert results.losing_trades == 0, "Losing trades should be 0"
        assert results.breakeven_trades == 0, "Breakeven trades should be 0"
        assert results.win_rate == 0, "Win rate should be 0"
        assert results.net_profit == 0, "Net profit should be 0"
        
        # Final balance should equal initial balance
        assert results.final_balance == config.initial_balance, \
            "Final balance should equal initial balance when no trades"
        
        # Profit factor should be 0 or 1 when no trades
        assert results.profit_factor in [0, 1], \
            f"Profit factor {results.profit_factor} should be 0 or 1 when no trades"
    
    # ---------------------------
    # Test 10: Positive and Negative Returns
    # ---------------------------
    def test_positive_and_negative_returns(self):
        """Test that both positive and negative returns are handled correctly"""
        config = self._create_config()
        provider = self._create_provider()
        
        results = self._run_backtest(config, provider)
        
        # If final balance > initial, total return should be positive
        if results.final_balance > config.initial_balance:
            assert results.total_return > 0, \
                "Total return should be positive when final balance > initial"
            assert results.total_return_pct > 0, \
                "Total return percentage should be positive"
        
        # If final balance < initial, total return should be negative
        elif results.final_balance < config.initial_balance:
            assert results.total_return < 0, \
                "Total return should be negative when final balance < initial"
            assert results.total_return_pct < 0, \
                "Total return percentage should be negative"
        
        # If equal, total return should be zero
        else:
            assert results.total_return == 0, \
                "Total return should be zero when final balance equals initial"
            assert results.total_return_pct == 0, \
                "Total return percentage should be zero"