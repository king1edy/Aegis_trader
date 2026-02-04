"""
Backtest Data Provider

Provides multi-timeframe historical data for backtesting.
Simulates the real-time data feed by iterating through bars.
"""

import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
from pathlib import Path

from src.data.historical_loader import HistoricalDataLoader
from src.strategies.indicators import IndicatorCalculator, IndicatorConfig
from src.core.logging_config import get_logger

logger = get_logger("backtest_data")


@dataclass
class MultiTimeframeBar:
    """Container for multi-timeframe data at a point in time"""
    timestamp: datetime
    
    # M15 bar (primary)
    m15_open: float
    m15_high: float
    m15_low: float
    m15_close: float
    m15_volume: int
    
    # M15 indicators
    m15_ema_21: float
    m15_rsi: float
    m15_atr: float
    m15_swing_high: Optional[float]
    m15_swing_low: Optional[float]
    
    # H1 data (if new bar)
    h1_new_bar: bool = False
    h1_close: Optional[float] = None
    h1_ema_50: Optional[float] = None
    h1_hull_34: Optional[float] = None
    h1_hull_34_prev: Optional[float] = None
    
    # H4 data (if new bar)
    h4_new_bar: bool = False
    h4_close: Optional[float] = None
    h4_ema_200: Optional[float] = None
    h4_hull_55: Optional[float] = None
    h4_atr: Optional[float] = None


class BacktestDataProvider:
    """
    Provides synchronized multi-timeframe data for backtesting.
    
    Loads historical M1 data and resamples to required timeframes.
    Provides bar-by-bar iteration with proper indicator values.
    
    Supports:
    - Loading from MT5 HCS files
    - Loading from CSV files
    - Generating synthetic test data as fallback
    """
    
    def __init__(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        data_path: str = "data/history",
        indicator_config: Optional[IndicatorConfig] = None,
        use_synthetic: bool = False
    ):
        """
        Initialize data provider.
        
        Args:
            symbol: Trading symbol
            start_date: Start of backtest period
            end_date: End of backtest period
            data_path: Path to historical data
            indicator_config: Indicator configuration
            use_synthetic: If True, generate synthetic data instead of loading
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data_path = data_path
        self.use_synthetic = use_synthetic
        
        self.loader = HistoricalDataLoader(data_path)
        self.indicator_calc = IndicatorCalculator()
        self.indicator_config = indicator_config or IndicatorConfig()
        
        # Cached dataframes
        self._df_m1: Optional[pd.DataFrame] = None
        self._df_m15: Optional[pd.DataFrame] = None
        self._df_h1: Optional[pd.DataFrame] = None
        self._df_h4: Optional[pd.DataFrame] = None
        
        logger.info(
            "Data provider initialized",
            symbol=symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            use_synthetic=use_synthetic
        )
    
    def load_data(self) -> bool:
        """
        Load and prepare all timeframe data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Loading historical data...")
            
            # Option 1: Use synthetic data
            if self.use_synthetic:
                logger.info("Generating synthetic data...")
                from src.backtesting.test_data_generator import TestDataGenerator
                generator = TestDataGenerator(
                    symbol=self.symbol,
                    start_date=self.start_date - timedelta(days=365),  # Extra year for warmup
                    end_date=self.end_date,
                    base_price=1900.0 if "XAU" in self.symbol else 1.1
                )
                self._df_m1 = generator.generate()
            else:
                # Option 2: Try CSV files first
                csv_loaded = self._load_from_csv()
                
                if not csv_loaded:
                    # Option 3: Try HCS files
                    start_year = self.start_date.year
                    end_year = self.end_date.year
                    
                    self._df_m1 = self.loader.load_symbol_data(
                        self.symbol,
                        start_year=start_year - 1,  # Extra year for warmup
                        end_year=end_year
                    )
                
                # Validate data quality
                if self._df_m1 is not None and len(self._df_m1) > 0:
                    # Check for garbage data (unrealistic prices)
                    if self.symbol.startswith("XAU"):
                        valid_prices = (self._df_m1['close'] > 500) & (self._df_m1['close'] < 10000)
                    else:
                        valid_prices = (self._df_m1['close'] > 0) & (self._df_m1['close'] < 1000000)
                    
                    if valid_prices.sum() < len(self._df_m1) * 0.5:
                        logger.warning("Data quality issue detected, falling back to synthetic data")
                        from src.backtesting.test_data_generator import TestDataGenerator
                        generator = TestDataGenerator(
                            symbol=self.symbol,
                            start_date=self.start_date - timedelta(days=365),
                            end_date=self.end_date,
                            base_price=1900.0 if "XAU" in self.symbol else 1.1
                        )
                        self._df_m1 = generator.generate()
            
            if self._df_m1 is None or self._df_m1.empty:
                logger.error("No M1 data loaded")
                return False
            
            logger.info(f"Loaded {len(self._df_m1)} M1 bars")
            
            # Resample to higher timeframes
            self._df_m15 = self._resample(self._df_m1, '15min')
            self._df_h1 = self._resample(self._df_m1, '1h')
            self._df_h4 = self._resample(self._df_m1, '4h')
            
            logger.info(
                "Data resampled",
                m15_bars=len(self._df_m15),
                h1_bars=len(self._df_h1),
                h4_bars=len(self._df_h4)
            )
            
            # Calculate indicators
            logger.info("Calculating indicators...")
            self._df_m15 = self.indicator_calc.calculate_all(self._df_m15, self.indicator_config)
            self._df_h1 = self.indicator_calc.calculate_all(self._df_h1, self.indicator_config)
            self._df_h4 = self.indicator_calc.calculate_all(self._df_h4, self.indicator_config)
            
            # Filter to date range (after indicator calculation for proper warmup)
            self._df_m15 = self._df_m15[
                (self._df_m15.index >= self.start_date) & 
                (self._df_m15.index <= self.end_date)
            ]
            self._df_h1 = self._df_h1[
                (self._df_h1.index >= self.start_date) & 
                (self._df_h1.index <= self.end_date)
            ]
            self._df_h4 = self._df_h4[
                (self._df_h4.index >= self.start_date) & 
                (self._df_h4.index <= self.end_date)
            ]
            
            logger.info(
                "Data preparation complete",
                m15_bars=len(self._df_m15),
                h1_bars=len(self._df_h1),
                h4_bars=len(self._df_h4),
                start=self._df_m15.index[0] if len(self._df_m15) > 0 else None,
                end=self._df_m15.index[-1] if len(self._df_m15) > 0 else None
            )
            
            return True
            
        except Exception as e:
            logger.exception("Failed to load data", error=str(e))
            return False
    
    def _load_from_csv(self) -> bool:
        """
        Try to load data from CSV files.
        
        Supports multiple CSV formats:
        - Standard comma-separated with OHLCV columns
        - Kaggle/MT5 semicolon-separated format
        
        Returns:
            True if CSV data was loaded, False otherwise
        """
        # Check for Kaggle datasets folder first (e.g., XAUUSD_datasets)
        kaggle_path = Path(self.data_path) / f"{self.symbol}_datasets"
        csv_path = Path(self.data_path) / self.symbol
        
        # Try Kaggle format first
        if kaggle_path.exists():
            return self._load_kaggle_csv(kaggle_path)
        
        # Fall back to standard CSV format
        if not csv_path.exists():
            return False
        
        csv_files = list(csv_path.glob("*.csv"))
        if not csv_files:
            return False
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        all_dfs = []
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                # Ensure timezone awareness
                if df.index.tz is None:
                    df.index = df.index.tz_localize(timezone.utc)
                all_dfs.append(df)
                logger.info(f"Loaded {len(df)} bars from {csv_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
        
        if not all_dfs:
            return False
        
        self._df_m1 = pd.concat(all_dfs)
        self._df_m1.sort_index(inplace=True)
        self._df_m1 = self._df_m1[~self._df_m1.index.duplicated(keep='last')]
        
        return True
    
    def _load_kaggle_csv(self, kaggle_path: Path) -> bool:
        """
        Load data from Kaggle XAUUSD dataset format.
        
        Format: Date;Open;High;Low;Close;Volume
        Date format: 2004.06.11 07:00
        
        We prefer to load 1-minute data and resample, but can use
        higher timeframes if M1 is not available.
        """
        logger.info(f"Loading from Kaggle datasets: {kaggle_path}")
        
        # Priority order: prefer smallest timeframe for accuracy
        timeframe_files = {
            '1m': 'XAU_1m_data.csv',
            '5m': 'XAU_5m_data.csv',
            '15m': 'XAU_15m_data.csv',
            '30m': 'XAU_30m_data.csv',
            '1h': 'XAU_1h_data.csv',
            '4h': 'XAU_4h_data.csv',
        }
        
        # Find the smallest available timeframe
        selected_file = None
        selected_tf = None
        for tf, filename in timeframe_files.items():
            filepath = kaggle_path / filename
            if filepath.exists():
                selected_file = filepath
                selected_tf = tf
                break
        
        if selected_file is None:
            logger.warning("No suitable Kaggle CSV file found")
            return False
        
        logger.info(f"Loading {selected_tf} data from {selected_file.name}")
        
        try:
            # Read CSV with semicolon separator
            df = pd.read_csv(
                selected_file,
                sep=';',
                names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                skiprows=1,  # Skip header
                dtype={
                    'open': float,
                    'high': float,
                    'low': float,
                    'close': float,
                    'volume': float
                }
            )
            
            # Parse datetime
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y.%m.%d %H:%M')
            df.set_index('datetime', inplace=True)
            df.index = df.index.tz_localize(timezone.utc)
            
            # Add missing columns expected by the system
            df['tick_volume'] = df['volume'].astype(int)
            df['spread'] = 35  # Typical gold spread in points
            
            # Filter to date range (with buffer for indicator warmup)
            warmup_start = self.start_date - timedelta(days=365)
            df = df[(df.index >= warmup_start) & (df.index <= self.end_date)]
            
            logger.info(f"Loaded {len(df)} bars from Kaggle {selected_tf} data")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            logger.info(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
            
            self._df_m1 = df
            return True
            
        except Exception as e:
            logger.exception(f"Failed to load Kaggle CSV: {e}")
            return False
    
    def _resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample M1 data to target timeframe"""
        # Pandas 3.0 uses lowercase frequency aliases
        rule_map = {
            '15min': '15min',
            '1H': '1h',
            '4H': '4h',
            '1D': '1D',
        }
        pandas_rule = rule_map.get(rule, rule.lower())
        
        df_resampled = df.resample(pandas_rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'tick_volume': 'sum',
            'spread': 'mean'
        }).dropna()
        
        return df_resampled
    
    def iterate_bars(self) -> Iterator[MultiTimeframeBar]:
        """
        Iterate through M15 bars with multi-timeframe context.
        
        Yields:
            MultiTimeframeBar for each 15-minute bar
        """
        if self._df_m15 is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        prev_h1_time = None
        prev_h4_time = None
        prev_h1_hull = None
        
        for idx, row in self._df_m15.iterrows():
            timestamp = idx.to_pydatetime()
            
            # Get swing points
            swing_high = row['swing_high_price'] if row['swing_high'] else None
            swing_low = row['swing_low_price'] if row['swing_low'] else None
            
            # Check for new H1 bar
            h1_time = timestamp.replace(minute=0, second=0, microsecond=0)
            h1_new_bar = prev_h1_time != h1_time
            prev_h1_time = h1_time
            
            # Get H1 data
            h1_close = None
            h1_ema_50 = None
            h1_hull_34 = None
            h1_hull_34_prev = None
            
            if h1_time in self._df_h1.index:
                h1_row = self._df_h1.loc[h1_time]
                h1_close = h1_row['close']
                h1_ema_50 = h1_row['ema_50']
                h1_hull_34 = h1_row['hull_34']
                h1_hull_34_prev = prev_h1_hull
                prev_h1_hull = h1_hull_34
            
            # Check for new H4 bar
            h4_hour = (timestamp.hour // 4) * 4
            h4_time = timestamp.replace(hour=h4_hour, minute=0, second=0, microsecond=0)
            h4_new_bar = prev_h4_time != h4_time
            prev_h4_time = h4_time
            
            # Get H4 data
            h4_close = None
            h4_ema_200 = None
            h4_hull_55 = None
            h4_atr = None
            
            if h4_time in self._df_h4.index:
                h4_row = self._df_h4.loc[h4_time]
                h4_close = h4_row['close']
                h4_ema_200 = h4_row['ema_200']
                h4_hull_55 = h4_row['hull_55']
                h4_atr = h4_row['atr']
            
            yield MultiTimeframeBar(
                timestamp=timestamp,
                m15_open=row['open'],
                m15_high=row['high'],
                m15_low=row['low'],
                m15_close=row['close'],
                m15_volume=row['volume'],
                m15_ema_21=row['ema_21'],
                m15_rsi=row['rsi'],
                m15_atr=row['atr'],
                m15_swing_high=swing_high,
                m15_swing_low=swing_low,
                h1_new_bar=h1_new_bar,
                h1_close=h1_close,
                h1_ema_50=h1_ema_50,
                h1_hull_34=h1_hull_34,
                h1_hull_34_prev=h1_hull_34_prev,
                h4_new_bar=h4_new_bar,
                h4_close=h4_close,
                h4_ema_200=h4_ema_200,
                h4_hull_55=h4_hull_55,
                h4_atr=h4_atr
            )
    
    def get_lookback_data(
        self,
        timestamp: datetime,
        timeframe: str,
        lookback: int
    ) -> Optional[pd.DataFrame]:
        """
        Get lookback data up to a specific timestamp.
        
        Used for strategy analysis that needs historical context.
        
        Args:
            timestamp: Current timestamp
            timeframe: Timeframe (M15, H1, H4)
            lookback: Number of bars to look back
            
        Returns:
            DataFrame with lookback data
        """
        df_map = {
            'M15': self._df_m15,
            'H1': self._df_h1,
            'H4': self._df_h4
        }
        
        df = df_map.get(timeframe)
        if df is None:
            return None
        
        # Get bars up to timestamp
        mask = df.index <= timestamp
        df_filtered = df[mask]
        
        # Return last N bars
        return df_filtered.tail(lookback).copy()
    
    @property
    def total_bars(self) -> int:
        """Total number of M15 bars in the backtest period"""
        return len(self._df_m15) if self._df_m15 is not None else 0
    
    @property
    def date_range(self) -> Tuple[datetime, datetime]:
        """Actual date range of loaded data"""
        if self._df_m15 is not None and len(self._df_m15) > 0:
            return (
                self._df_m15.index[0].to_pydatetime(),
                self._df_m15.index[-1].to_pydatetime()
            )
        return (self.start_date, self.end_date)
