"""
Test Data Generator

Generates realistic XAUUSD M1 data for backtesting when historical data
is not available or in incompatible format.

This module creates synthetic price data that mimics real market behavior:
- Trending periods
- Ranging periods
- Session-based volatility
- Realistic spreads and volumes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.core.logging_config import get_logger

logger = get_logger("test_data_generator")


class TestDataGenerator:
    """
    Generates realistic synthetic XAUUSD data for backtesting.
    
    Creates M1 data with:
    - Realistic price movements
    - Session-based patterns (Asian, London, NY)
    - Trending and ranging periods
    - Appropriate volatility for gold
    """
    
    def __init__(
        self,
        symbol: str = "XAUUSD",
        start_date: datetime = None,
        end_date: datetime = None,
        base_price: float = 2000.0,
        daily_volatility: float = 0.015  # 1.5% daily volatility
    ):
        """
        Initialize generator.
        
        Args:
            symbol: Symbol name
            start_date: Start date
            end_date: End date
            base_price: Starting price
            daily_volatility: Daily volatility as decimal
        """
        self.symbol = symbol
        self.start_date = start_date or datetime(2020, 1, 1, tzinfo=timezone.utc)
        self.end_date = end_date or datetime(2025, 12, 31, tzinfo=timezone.utc)
        self.base_price = base_price
        self.daily_volatility = daily_volatility
        
        # M1 volatility (approx daily_vol / sqrt(1440))
        self.m1_volatility = daily_volatility / np.sqrt(1440)
        
    def generate(self) -> pd.DataFrame:
        """
        Generate synthetic M1 data.
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(
            "Generating synthetic data",
            symbol=self.symbol,
            start=self.start_date.strftime("%Y-%m-%d"),
            end=self.end_date.strftime("%Y-%m-%d")
        )
        
        # Generate timestamps (M1 bars, excluding weekends)
        timestamps = []
        current = self.start_date
        
        while current <= self.end_date:
            # Skip weekends
            if current.weekday() < 5:  # Mon-Fri
                timestamps.append(current)
            current += timedelta(minutes=1)
            
            # Skip from Friday 22:00 to Sunday 22:00 (forex market closed)
            if current.weekday() == 4 and current.hour == 22:
                current = current + timedelta(days=2)
        
        n_bars = len(timestamps)
        logger.info(f"Generating {n_bars:,} M1 bars")
        
        if n_bars == 0:
            return pd.DataFrame()
        
        # Generate price series using geometric Brownian motion with trends
        prices = self._generate_price_series(n_bars, timestamps)
        
        # Generate OHLC from close prices
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        # Add noise for high/low
        high_noise = np.abs(np.random.normal(0, self.m1_volatility * 0.5, n_bars))
        low_noise = np.abs(np.random.normal(0, self.m1_volatility * 0.5, n_bars))
        
        highs = np.maximum(opens, prices) + high_noise * prices
        lows = np.minimum(opens, prices) - low_noise * prices
        
        # Generate volume (session-based)
        volumes = self._generate_volumes(timestamps)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes,
            'tick_volume': volumes,
            'spread': np.random.randint(15, 35, n_bars)  # Typical XAUUSD spread
        }, index=pd.DatetimeIndex(timestamps, tz=timezone.utc))
        
        logger.info(
            "Synthetic data generated",
            bars=len(df),
            start_price=f"${df['close'].iloc[0]:.2f}",
            end_price=f"${df['close'].iloc[-1]:.2f}"
        )
        
        return df
    
    def _generate_price_series(self, n_bars: int, timestamps: list) -> np.ndarray:
        """Generate price series with trends and mean reversion"""
        prices = np.zeros(n_bars)
        prices[0] = self.base_price
        
        # Define trend periods (changes every ~20 trading days)
        trend_period = 1440 * 20  # bars
        trend = 0.0  # Current trend strength
        
        for i in range(1, n_bars):
            # Update trend periodically
            if i % trend_period == 0:
                trend = np.random.uniform(-0.0003, 0.0003)  # Slight daily drift
            
            # Session-based volatility
            hour = timestamps[i].hour
            if 7 <= hour < 16:  # London session
                vol_mult = 1.2
            elif 12 <= hour < 21:  # NY session (overlaps)
                vol_mult = 1.3 if hour < 16 else 1.1
            else:  # Asian session
                vol_mult = 0.7
            
            # Random walk with trend
            change = (trend + np.random.normal(0, self.m1_volatility * vol_mult)) * prices[i-1]
            prices[i] = prices[i-1] + change
            
            # Mean reversion if price drifts too far
            drift_pct = (prices[i] - self.base_price) / self.base_price
            if abs(drift_pct) > 0.30:  # More than 30% from base
                prices[i] -= 0.001 * drift_pct * prices[i]
        
        return prices
    
    def _generate_volumes(self, timestamps: list) -> np.ndarray:
        """Generate realistic volume pattern"""
        n_bars = len(timestamps)
        volumes = np.zeros(n_bars)
        
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            
            # Base volume
            base = 100
            
            # Session multipliers
            if 7 <= hour < 9:  # London open
                mult = 3.0
            elif 12 <= hour < 14:  # NY open / London afternoon
                mult = 4.0
            elif 14 <= hour < 16:  # Peak overlap
                mult = 5.0
            elif 16 <= hour < 17:  # London close
                mult = 2.5
            elif 20 <= hour < 21:  # NY afternoon
                mult = 2.0
            elif 0 <= hour < 7:  # Asian session
                mult = 0.8
            else:
                mult = 1.5
            
            volumes[i] = base * mult * np.random.uniform(0.5, 1.5)
        
        return volumes.astype(int)
    
    def save_to_hcs(self, output_path: str = "data/history/TEST_XAUUSD") -> None:
        """
        Save generated data in a simple CSV format for testing.
        
        Note: This doesn't create actual HCS files but provides
        a compatible data format.
        """
        df = self.generate()
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split by year and save as CSV
        for year in df.index.year.unique():
            year_data = df[df.index.year == year]
            output_file = output_dir / f"{year}.csv"
            year_data.to_csv(output_file)
            logger.info(f"Saved {len(year_data)} bars to {output_file}")


def generate_test_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    base_price: float = 1900.0
) -> pd.DataFrame:
    """
    Convenience function to generate test data.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        base_price: Starting price
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    generator = TestDataGenerator(
        start_date=start_dt,
        end_date=end_dt,
        base_price=base_price
    )
    
    return generator.generate()


if __name__ == "__main__":
    # Generate and save test data
    print("Generating 5 years of synthetic XAUUSD data...")
    generator = TestDataGenerator(
        start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2025, 12, 31, tzinfo=timezone.utc),
        base_price=1900.0
    )
    df = generator.generate()
    print(f"\nGenerated {len(df):,} M1 bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['low'].min():.2f} to ${df['high'].max():.2f}")
    
    # Save summary stats
    print("\nMonthly statistics:")
    monthly = df.resample('ME').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    print(monthly.tail(12))
